# Day 20: Quantization Strategies: AWQ, GPTQ, and FP8

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Understanding that FP16 uses 16 bits (2 bytes) per weight parameter.

---

## 2. Core Mental Model (Intuition First)
Imagine packing a suitcase. Full-precision FP16 weights are bulky winter coats. Quantization (INT4 or FP8) vacuums-packs those coats so they take up half or a quarter of the space. Because decode is memory-bandwidth bound, shrinking the weights from 2 bytes to 0.5 bytes means the GPU can read them 4x faster from VRAM!

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
A 70B parameter model in FP16 requires 140 GB of VRAM (at least two 80GB GPUs). In AWQ 4-bit, it fits into ~38 GB of VRAM (a single 48GB GPU or two 24GB GPUs), cutting infrastructure hosting costs by over 60%.

---

## 4. Today's Lab Objective
Benchmark an 8B model under FP16 vs AWQ 4-bit vs FP8 on vLLM and measure VRAM footprint, throughput, and generation latency.

---

## 5. Step-by-Step Code Walkthrough
We compare model memory consumption and generation tokens/sec across quantization formats.

```python
# To run today's experiment:
python3 quantization_comparison.py
```

---

## 6. How to Interpret Your Results
Notice that AWQ 4-bit drastically cuts memory and speeds up decode, while FP8 preserves almost 100% of mathematical precision while accelerating both prefill and decode on Ada/Hopper architectures.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare VRAM usage and tokens/sec across FP16, AWQ, and FP8.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-20/quantization_tradeoffs_matrix.md with a comprehensive performance scorecard.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare VRAM usage and tokens/sec across FP16, AWQ, and FP8.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
