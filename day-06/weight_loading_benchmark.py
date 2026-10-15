import torch
import time
import os
from safetensors.torch import save_file, load_file

def benchmark_weight_formats():
    print("Generating simulated model weight tensors (approx 1 GB)...")
    weights = {f"model.layers.{i}.weight": torch.randn(4096, 4096, dtype=torch.float16) for i in range(16)}

    pt_file = "weights_pytorch.pt"
    safe_file = "weights_safetensors.safetensors"

    print("Saving checkpoints to disk...")
    torch.save(weights, pt_file)
    save_file(weights, safe_file)

    pt_size_mb = os.path.getsize(pt_file) / (1024 * 1024)
    safe_size_mb = os.path.getsize(safe_file) / (1024 * 1024)
    print(f"File sizes -> PyTorch: {pt_size_mb:.2f} MB | Safetensors: {safe_size_mb:.2f} MB")

    t0 = time.perf_counter()
    loaded_pt = torch.load(pt_file, map_location="cpu")
    t_pt = time.perf_counter() - t0

    t0 = time.perf_counter()
    loaded_safe = load_file(safe_file)
    t_safe = time.perf_counter() - t0

    print(f"Load Time -> PyTorch: {t_pt:.3f}s | Safetensors: {t_safe:.3f}s")
    speedup = t_pt / t_safe if t_safe > 0 else 1.0
    print(f"Safetensors is {speedup:.2f}x faster at loading weights.")

    for f in [pt_file, safe_file]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    benchmark_weight_formats()
