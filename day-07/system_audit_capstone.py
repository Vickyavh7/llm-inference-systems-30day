import torch
import time

def run_system_audit():
    print("=" * 60)
    print("WEEK 1 CAPSTONE: COMPREHENSIVE GPU SYSTEMS AUDIT")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Simulating audit template.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"1. Hardware: {props.name}")
    print(f"   VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"   SM Count: {props.multi_processor_count}")

    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = torch.matmul(a, a)
    torch.cuda.synchronize()
    avg_t = (time.perf_counter() - t0) / 20
    achieved_tflops = (2.0 * n**3 / avg_t) / 1e12
    print(f"2. Empirical Tensor Core Throughput: {achieved_tflops:.2f} TFLOPs")
    print("\nAudit complete! Use these empirical figures to write your Week 1 Report.")

if __name__ == "__main__":
    run_system_audit()
