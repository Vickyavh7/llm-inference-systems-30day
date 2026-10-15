import time
import torch

def simulate_autoregressive_loop(vocab_size=32000, steps=20):
    print(f"Simulating {steps} autoregressive decode iterations...")
    latencies = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for step in range(steps):
        t0 = time.perf_counter()
        logits = torch.randn(1, vocab_size, device=device)
        next_token = torch.argmax(logits, dim=-1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    print(f"Avg step latency: {sum(latencies)/len(latencies):.2f} ms")

if __name__ == "__main__":
    simulate_autoregressive_loop()
