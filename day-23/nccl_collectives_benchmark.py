import torch
import torch.distributed as dist

def run_nccl_benchmark():
    print("=== NCCL Collective Operations Benchmark ===")
    print("To execute multi-GPU collective benchmark:")
    print("  torchrun --nproc_per_node=2 day-23/nccl_collectives_benchmark.py")

if __name__ == "__main__":
    run_nccl_benchmark()
