import argparse
import yaml
import torch
import os
import sys
from thop import profile

# Add the project root to the Python path to allow imports from 'src'
sys.path.insert(0, os.getcwd())

# Import all model types
from src.models.pose_gcnn import PoseSelectiveSparse_ResNet44, SparseR2Conv
from src.models.mnist_architectures import PoseSelective_P4CNN_MNIST
from src.models.baseline_models import Baseline_ResNet44, Baseline_P4CNN_MNIST
from e2cnn import nn as enn

def get_model_and_shape(config, args):
    """Helper function to instantiate the correct model and get input shape."""
    model_name = config['model']
    in_channels = config['in_channels']
    
    if 'MNIST' in config['dataset'] or 'FashionMNIST' in config['dataset']:
        input_shape = (in_channels, 28, 28)
    elif 'CIFAR' in config['dataset'] or 'GTSRB' in config['dataset']:
        input_shape = (in_channels, 32, 32)
    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized.")

    # Instantiate the correct model architecture based on whether a checkpoint is provided
    is_baseline = args.checkpoint is None
    
    if 'P4CNN_MNIST' in model_name:
        ModelClass = Baseline_P4CNN_MNIST if is_baseline else PoseSelective_P4CNN_MNIST
        model = ModelClass(num_classes=config['num_classes'], in_channels=in_channels)
    elif 'ResNet44' in model_name:
        ModelClass = Baseline_ResNet44 if is_baseline else PoseSelectiveSparse_ResNet44
        model = ModelClass(
            n=config.get('resnet_n', 7),
            num_classes=config['num_classes'],
            in_channels=in_channels,
            group=config['group'],
            widths=config['widths']
        )
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
        
    return model, input_shape

def main(args):
    """
    This script loads a model and calculates its FLOPs using custom handlers
    for both sparse and baseline e2cnn layers to ensure accuracy.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model, input_shape = get_model_and_shape(config, args)

    if args.checkpoint:
        print(f"--- Calculating Pruned FLOPs for checkpoint: {args.checkpoint} ---")
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint file not found at {args.checkpoint}")
            return
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    else:
        print(f"--- Calculating Dense FLOPs for baseline model from: {args.config} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- Custom Handlers for e2cnn Layers ---
    # These handlers are forward hooks. They should NOT return any value.
    def dense_r2conv_handler(m: enn.R2Conv, x, y):
        output_dims = y.shape[2:]
        kernel_dims = m.kernel_size
        if isinstance(kernel_dims, int):
            kernel_dims = (kernel_dims, kernel_dims)
        
        in_channels_eff = m.in_type.size
        out_channels_eff = m.out_type.size
        
        macs = in_channels_eff * out_channels_eff * kernel_dims[0] * kernel_dims[1] * output_dims[0] * output_dims[1]
        flops = 2 * macs
        m.total_ops += torch.DoubleTensor([flops])

    def sparse_r2conv_handler(m: SparseR2Conv, x, y):
        # The hook for the inner m.conv (a dense R2Conv) has already run
        # and calculated its full FLOPs, storing them in m.conv.total_ops.
        with torch.no_grad():
            hard_mask = (m.gate.get_mask() > 0.5).float()
        active_ratio = torch.mean(hard_mask)

        dense_flops = m.conv.total_ops
        pruned_flops = dense_flops * active_ratio
        
        # Set the FLOPs for this sparse module to the pruned value
        m.total_ops = pruned_flops
        
        # Zero out the inner convolution's FLOPs to prevent thop from double-counting
        m.conv.total_ops = torch.DoubleTensor([0.])

    custom_ops = {
        SparseR2Conv: sparse_r2conv_handler,
        enn.R2Conv: dense_r2conv_handler,
    }   

    dummy_input = torch.randn(1, *input_shape).to(device)
    total_ops, total_params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
    
    gflops = total_ops / 1e9
    params_m = total_params / 1e6

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {params_m:.2f}M")
    print(f"GFLOPs: {gflops:.4f}G")
    print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified FLOPs calculator for sparse and baseline G-CNN models.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file for the experiment.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a trained model's .pth checkpoint file. If not provided, calculates baseline FLOPs.")
    args = parser.parse_args()
    main(args)
