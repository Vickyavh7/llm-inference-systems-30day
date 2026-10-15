import torch
import subprocess

def inspect_gpu():
    print("=== Step 1: Checking CUDA Availability ===")
    if not torch.cuda.is_available():
        print("No NVIDIA GPU detected by PyTorch. Ensure drivers & CUDA are installed.")
        return

    device_idx = 0
    props = torch.cuda.get_device_properties(device_idx)
    print(f"Device Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"Streaming Multiprocessors (SMs): {props.multi_processor_count}")

    print("\n=== Step 2: Querying nvidia-smi Telemetry ===")
    try:
        smi = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv"
        ]).decode("utf-8")
        print(smi)
    except Exception as e:
        print(f"Could not run nvidia-smi CLI: {e}")

    print("=== Step 3: Theoretical Compute Ceiling ===")
    # Each SM in modern NVIDIA architectures typically has 128 FP32 cores
    estimated_fp32_cores = props.multi_processor_count * 128
    print(f"Estimated FP32 CUDA Cores: ~{estimated_fp32_cores}")
    theoretical_tflops = (estimated_fp32_cores * 2 * 2.0e9) / 1e12
    print(f"Theoretical Peak FP32: ~{theoretical_tflops:.1f} TFLOPs")

if __name__ == "__main__":
    inspect_gpu()
