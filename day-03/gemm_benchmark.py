import torch
import time

def benchmark_gemm():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    dims = [1024, 2048, 4096]
    print(f"{'Matrix Size (NxN)':<20}{'FP32 TFLOPS':<18}{'FP16 TFLOPS (Tensor Cores)':<30}{'Speedup':<10}")
    print("-" * 75)

    for n in dims:
        flops = 2.0 * (n ** 3)

        # FP32
        a_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        _ = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            _ = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()
        t_fp32 = (time.perf_counter() - t0) / iters
        tflops_fp32 = (flops / t_fp32) / 1e12

        # FP16 (Tensor Cores)
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        torch.cuda.synchronize()
        _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        t_fp16 = (time.perf_counter() - t0) / iters
        tflops_fp16 = (flops / t_fp16) / 1e12

        speedup = tflops_fp16 / t_fp32
        print(f"{f'{n}x{n}':<20}{tflops_fp32:<18.2f}{tflops_fp16:<30.2f}{speedup:<10.2f}x")

if __name__ == "__main__":
    benchmark_gemm()
