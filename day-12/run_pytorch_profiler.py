import torch

def profile_inference():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return
    x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            _ = torch.matmul(x, w)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    profile_inference()
