# Day 03: CUDA Cores vs. Tensor Cores & Matrix Multiply (GEMM)

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Understanding that neural networks perform matrix multiplications ($Y = X \cdot W$).

---

## 2. Core Mental Model (Intuition First)
Standard CUDA cores are general-purpose calculators: each core multiplies two numbers and adds a third ($A \times B + C$) one at a time. Tensor Cores are specialized hardware macro-blocks. Instead of multiplying single numbers, a Tensor Core consumes entire $16 \times 16$ mini-matrices in a single clock cycle. It is the difference between counting by hand versus stamping an entire grid at once.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
LLMs spend >95% of their computational time executing GEMM operations. If your model runs in FP32 on standard CUDA cores, you leave 80% to 90% of your GPU's computing power on the table. Tensor Cores require lower-precision formats (FP16, BF16, FP8) and dimension alignments (multiples of 8/16). Knowing how to leverage them is what makes inference 5x to 10x faster.

---

## 4. Today's Lab Objective
Benchmark square matrix multiplications in FP32 (standard CUDA cores) vs. FP16 (Tensor Cores) and calculate the actual TFLOPs achieved versus your theoretical ceiling.

---

## 5. Step-by-Step Code Walkthrough
For an $N \times N$ matrix multiply, total floating-point operations equal $2 \times N^3$. We time the operation with `time.perf_counter()` and `torch.cuda.synchronize()` to calculate effective TFLOPs ($FLOPs / time / 10^{12}$).

```python
# To run today's experiment:
python3 gemm_benchmark.py
```

---

## 6. How to Interpret Your Results
You will observe a dramatic speedup (3x to 6x faster) moving from FP32 to FP16. Notice that larger matrices ($4096 \times 4096$) achieve much higher TFLOPs than smaller ones ($1024 \times 1024$) because large matrices provide enough parallel work to keep all Tensor Cores fully saturated.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Test non-aligned matrix dimensions ($N=4095$ vs $N=4096$). Notice how odd padding slightly harms Tensor Core efficiency.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-03/gemm_speedup_analysis.md documenting the speedup table and explaining why AI inference models are deployed in FP16/BF16/FP8 rather than FP32.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Test non-aligned matrix dimensions ($N=4095$ vs $N=4096$). Notice how odd padding slightly harms Tensor Core efficiency.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
