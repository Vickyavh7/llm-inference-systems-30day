# Day 09: Prefill (Prompt) vs. Decode (Generation) Compute Patterns

**Module:** `Stage 2: LLM Inference Mechanics`

---

## 1. Prerequisites & Foundational Context
Understanding GEMM (matrix-matrix multiply) vs GEMV (matrix-vector multiply).

---

## 2. Core Mental Model (Intuition First)
Reading vs Writing: When a human reads a 500-word prompt, they read all words together, digesting the full context in parallel. That is **Prefill** (compute-bound, massive parallel matrix math). When they write the response, they write one word at a time, pausing to recall knowledge for each word. That is **Decode** (memory-bound, sequential vector math).

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Prefill and Decode have completely opposite hardware characteristics! Prefill runs at thousands of tokens per second because it saturates Tensor Cores. Decode runs at 30 to 80 tokens per second because it is limited by how fast VRAM can stream weights. Co-locating both on the same GPU causes severe latency interference.

---

## 4. Today's Lab Objective
Benchmark a transformer layer executing prefill with prompt length 2048 vs. 128 decode steps of length 1, and compare their arithmetic intensity (FLOPs/byte).

---

## 5. Step-by-Step Code Walkthrough
We compare a $(2048 \times 4096) \times (4096 \times 4096)$ matrix multiplication against a $(1 \times 4096) \times (4096 \times 4096)$ operation.

```python
# To run today's experiment:
python3 prefill_vs_decode_benchmark.py
```

---

## 6. How to Interpret Your Results
Notice that processing 2048 prompt tokens in prefill takes only a few times longer than processing a single token in decode! This proves that batching tokens in parallel is vastly more efficient than processing them one by one.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Vary prompt length from 128 to 4096 tokens and observe how prefill throughput scales.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-09/prefill_vs_decode_roofline.md documenting arithmetic intensity differences.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Vary prompt length from 128 to 4096 tokens and observe how prefill throughput scales.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
