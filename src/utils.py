import logging
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from functools import partial

def set_seed(seed: int):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def setup_logging(log_file: str):
    """Configures logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(results: dict, save_path: str):
    """Saves final results dictionary to a JSON file."""
    file_path = f"{save_path}/results.json"
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Final results saved to {file_path}")

def get_gate_weights(model: torch.nn.Module) -> dict:
    """
    Extracts and formats the current gate values (π_g) from 
    DifferentiableMaskGate modules.
    """
    from src.models.pose_gcnn import DifferentiableMaskGate # Local import to avoid circular dependency
    
    gate_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, DifferentiableMaskGate):
            # Get the deterministic mask value for logging
            with torch.no_grad():
                module.eval() # Use deterministic forward pass
                weights = module.get_mask().cpu().numpy()
                module.train() # Set back to train mode
            gate_weights[name] = [round(w, 4) for w in weights]
    return gate_weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def expected_calibration_error(conf, correct, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece  = torch.zeros(1, device=conf.device)
    for i in range(n_bins):
        mask   = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc   = correct[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece  += mask.float().mean() * (acc - bin_conf).abs()
    return ece.item()

@torch.no_grad()
def compute_ece(model, test_loader, device="cuda", n_bins=15):
    model.eval()
    all_conf, all_correct = [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                       # (batch, C)
        prob   = F.softmax(logits, dim=1)       # convert to probabilities
        conf, pred = prob.max(dim=1)            # highest prob per sample
        all_conf.append(conf)
        all_correct.append(pred.eq(y))          # Boolean tensor

    conf_tensor    = torch.cat(all_conf)        # shape (N,)
    correct_tensor = torch.cat(all_correct)     # shape (N,)

    return expected_calibration_error(conf_tensor,
                                      correct_tensor,
                                      n_bins=n_bins)


def calculate_gflops(model, input_shape, device):
    """
    Calculates the GFLOPs of a sparse model using a custom handler for thop
    to correctly account for pruned orientations.
    """
    try:
        from thop import profile
        from src.models.pose_gcnn import SparseR2Conv
        from e2cnn import nn as enn
    except ImportError:
        logging.warning("Required libraries for FLOPs calculation (`thop`, `e2cnn`) are not installed.")
        return -1

    dummy_input = torch.randn(1, *input_shape).to(device)
    
    def sparse_r2conv_handler(m: SparseR2Conv, x: torch.Tensor, y: torch.Tensor):
        # Get the hard 0/1 mask from the gate to determine the active ratio
        with torch.no_grad():
            hard_mask = (m.gate.get_mask() > 0.5).float()
        active_ratio = torch.mean(hard_mask)
        
        # Get the underlying dense convolution layer
        dense_conv = m.conv
        
        # Get the spatial dimensions of the output
        output_dims = y.shape[2:]
        
        # Handle integer vs. tuple kernel_size
        kernel_dims = dense_conv.kernel_size
        if isinstance(kernel_dims, int):
            kernel_dims = (kernel_dims, kernel_dims)

        # Get the effective number of input and output channels for the underlying planar convolution
        in_channels_eff = dense_conv.in_type.size
        out_channels_eff = dense_conv.out_type.size
        
        # Calculate the FLOPs for the equivalent DENSE G-Convolution
        # This is based on the standard formula for a 2D convolution, using the
        # effective channel counts that e2cnn uses internally.
        # FLOPs = 2 * (MACs) = 2 * (in_channels * out_channels * kernel_w * kernel_h * out_w * out_h)
        dense_macs = in_channels_eff * out_channels_eff * kernel_dims[0] * kernel_dims[1] * output_dims[0] * output_dims[1]
        dense_flops = 2 * dense_macs
        
        # Scale the total dense FLOPs by the ratio of active (non-pruned) output gates.
        # This correctly reflects the theoretical computational savings at inference time.
        pruned_flops = dense_flops * active_ratio
        
        m.total_ops += torch.DoubleTensor([pruned_flops])

    custom_ops = {
        SparseR2Conv: sparse_r2conv_handler,
    }

    try:
        model.eval()
        total_ops, _ = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
        gflops = total_ops / 1e9
    except Exception as e:
        logging.error(f"An error occurred during FLOPs calculation: {e}")
        gflops = -1
    
    return gflops
