import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_custom_ops_bn import BatchNormCustom

EPS, MOMENTUM = 1e-5, 0.1

# Every case runs in float64 and in float32.
# - float64 decides correctness: its round-off is far below the effect of any mistake in the formulas.
# - float32 only has to agree with PyTorch up to round-off.
# A difference passes if it is <= RTOL * max(1, largest |reference value|).
RTOL = {torch.float64: 1e-9, torch.float32: 1e-4}


def compare_with_pytorch(shape, training, dtype, device):
    """Forward + backward with BatchNormCustom and with nn.BatchNorm2d on the same values.

    Returns {quantity: (max abs difference, largest |reference value|)}.
    """
    torch.manual_seed(42)
    C = shape[1]
    kwargs = dict(device=device, dtype=dtype)

    # Non-trivial values everywhere, so that every term of the formulas matters
    input_custom = (torch.randn(shape, **kwargs) * 2.0 + 0.5).requires_grad_(True)
    gamma = (torch.rand(C, **kwargs) + 0.5).requires_grad_(True)
    beta = torch.randn(C, **kwargs).requires_grad_(True)
    running_mean = torch.randn(C, **kwargs) * 0.1
    running_var = torch.rand(C, **kwargs) + 0.5
    grad_output = torch.randn(shape, **kwargs)

    # PyTorch's BatchNorm on copies of the same values
    input_pytorch = input_custom.detach().clone().requires_grad_(True)
    bn_pytorch = nn.BatchNorm2d(C, eps=EPS, momentum=MOMENTUM, **kwargs)
    bn_pytorch.weight.data = gamma.detach().clone()
    bn_pytorch.bias.data = beta.detach().clone()
    bn_pytorch.running_mean.data = running_mean.clone()
    bn_pytorch.running_var.data = running_var.clone()
    bn_pytorch.train(training)

    output_custom = BatchNormCustom.apply(input_custom, gamma, beta, running_mean, running_var, training, MOMENTUM, EPS)
    output_pytorch = bn_pytorch(input_pytorch)

    output_custom.backward(grad_output)
    output_pytorch.backward(grad_output.clone())

    pairs = {
        "output": (output_custom, output_pytorch),
        "grad_input": (input_custom.grad, input_pytorch.grad),
        "grad_gamma": (gamma.grad, bn_pytorch.weight.grad),
        "grad_beta": (beta.grad, bn_pytorch.bias.grad),
        # the running statistics must be updated in place in training mode and untouched in inference mode
        "running_mean": (running_mean, bn_pytorch.running_mean),
        "running_var": (running_var, bn_pytorch.running_var),
        # inputs that are not listed in mutates_args must not be modified
        "input": (input_custom, input_pytorch),
        "gamma": (gamma, bn_pytorch.weight),
        "beta": (beta, bn_pytorch.bias),
    }
    return {
        name: (torch.abs(custom - reference).max().item(), reference.abs().max().item())
        for name, (custom, reference) in pairs.items()
    }


def run_case(title, shape, training, device):
    """Prints the differences of one case and returns the names of the quantities that failed."""
    print(f"\n{title}, input shape: {list(shape)}")
    print(f"   {'max abs difference':<20s}{'float64':>12s}{'float32':>12s}")

    results = {dtype: compare_with_pytorch(shape, training, dtype, device) for dtype in RTOL}
    failed = []
    for name in results[torch.float64]:
        row, passed = "", True
        for dtype, rtol in RTOL.items():
            diff, scale = results[dtype][name]
            row += f"{diff:12.2e}"
            passed = passed and diff <= rtol * max(1.0, scale)
        print(f"   {name:<20s}{row}   {'ok' if passed else 'FAIL'}")
        if not passed:
            failed.append(f"{title}: {name}")
    return failed


def test_custom_batchnorm():
    print("Testing Custom BatchNorm Implementation")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    failed = run_case("Training mode", (32, 64, 56, 56), True, device)
    failed += run_case("Training mode (small batch)", (4, 8, 1, 1), True, device)
    return failed


def test_inference_mode():
    print("\n" + "=" * 40)
    print("Testing Inference Mode")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return run_case("Inference mode", (8, 128, 28, 28), False, device)


if __name__ == "__main__":
    failed = test_custom_batchnorm() + test_inference_mode()

    print("\n" + "=" * 40)
    if failed:
        print("Test FAILED:")
        for name in failed:
            print(f"   {name}")
        sys.exit(1)
    print("All tests PASSED")
