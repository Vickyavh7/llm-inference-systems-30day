import torch
import subprocess

def check_interconnects():
    print("=== Multi-GPU Topology Inspection ===")
    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs visible to PyTorch: {num_gpus}")

    try:
        topo = subprocess.check_output(["nvidia-smi", "topo", "-m"]).decode("utf-8")
        print("\nnvidia-smi topology matrix:\n" + topo)
    except Exception as e:
        print(f"nvidia-smi topo command unavailable: {e}")

    if num_gpus >= 2:
        print("\n=== Checking Peer-to-Peer (P2P) Direct Access ===")
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"GPU {i} -> GPU {j} Direct P2P Access: {'YES (NVLink/Direct PCIe)' if can_access else 'NO (Requires CPU Host RAM Staging)'}")
    else:
        print("\n[Note: Single GPU system detected].")
        print("Theoretical Architecture Comparison:")
        print(" - PCIe Gen4 x16: ~31.5 GB/s bidirectional peak")
        print(" - PCIe Gen5 x16: ~63.0 GB/s bidirectional peak")
        print(" - NVLink 4 (Hopper H100): ~900 GB/s bidirectional peak (28x faster than PCIe Gen4)")

if __name__ == "__main__":
    check_interconnects()
