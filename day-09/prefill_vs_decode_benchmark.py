import torch
import time

def compare_prefill_decode(hidden_dim=4096, prompt_len=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float16)

    # Prefill: GEMM
    prompt_tokens = torch.randn(prompt_len, hidden_dim, device=device, dtype=torch.float16)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.matmul(prompt_tokens, weight)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_time = (time.perf_counter() - t0) * 1000

    # Decode: 1 token GEMV
    single_token = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.matmul(single_token, weight)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    decode_time = (time.perf_counter() - t0) * 1000

    print(f"Prefill ({prompt_len} tokens) Latency: {prefill_time:.2f} ms ({prompt_len/prefill_time*1000:.0f} tok/s)")
    print(f"Decode (1 token) Latency: {decode_time:.2f} ms ({1/decode_time*1000:.0f} tok/s)")

if __name__ == "__main__":
    compare_prefill_decode()
