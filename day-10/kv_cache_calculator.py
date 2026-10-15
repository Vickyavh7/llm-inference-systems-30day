def calc_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_len, batch_size=1, dtype_bytes=2):
    total_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024**3)

if __name__ == "__main__":
    # Llama-3-8B specs: 32 layers, 8 KV heads (GQA), head_dim=128, FP16
    print("=== Llama-3-8B KV Cache Memory Footprint ===")
    for seq in [1024, 4096, 8192, 16384, 32768]:
        gb_b1 = calc_kv_cache_size(32, 8, 128, seq, batch_size=1, dtype_bytes=2)
        gb_b8 = calc_kv_cache_size(32, 8, 128, seq, batch_size=8, dtype_bytes=2)
        print(f"Context {seq:<6} tokens -> Batch=1: {gb_b1:.3f} GB | Batch=8: {gb_b8:.3f} GB")
