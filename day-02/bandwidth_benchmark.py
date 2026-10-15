import torch
import time

def benchmark_bandwidth():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    sizes_mb = [1, 10, 50, 100, 500, 1024]
    print(f"{'Size (MB)':<12}{'H2D Transfer (GB/s)':<22}{'D2D Bandwidth (GB/s)':<25}")
    print("-" * 60)

    for sz in sizes_mb:
        num_elements = (sz * 1024 * 1024) // 4
        # Pinned memory allows direct DMA transfer over PCIe without CPU staging
        host_tensor = torch.empty(num_elements, dtype=torch.float32, pin_memory=True)
        device_tensor_1 = torch.empty(num_elements, dtype=torch.float32, device="cuda")
        device_tensor_2 = torch.empty(num_elements, dtype=torch.float32, device="cuda")

        # Warmup
        device_tensor_1.copy_(host_tensor)
        torch.cuda.synchronize()

        # H2D
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            device_tensor_1.copy_(host_tensor)
        torch.cuda.synchronize()
        h2d_bw = (sz / 1024) / ((time.perf_counter() - t0) / iters)

        # D2D (internal VRAM)
        t0 = time.perf_counter()
        for _ in range(iters):
            device_tensor_2.copy_(device_tensor_1)
        torch.cuda.synchronize()
        d2d_bw = (sz / 1024) / ((time.perf_counter() - t0) / iters)

        print(f"{sz:<12}{h2d_bw:<22.2f}{d2d_bw:<25.2f}")

if __name__ == "__main__":
    benchmark_bandwidth()
