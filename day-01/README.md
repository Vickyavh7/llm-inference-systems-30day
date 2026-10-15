# Day 01: CPU vs. GPU Architecture & SIMT Fundamentals

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Basic familiarity with Linux terminal commands (`ls`, `cd`), Python syntax, and understanding that computers have a Central Processing Unit (CPU) and a Graphics Processing Unit (GPU).

---

## 2. Core Mental Model (Intuition First)
Imagine you need to move an entire beach of sand. A CPU is a Ferrari driven by an expert engineer: it moves incredibly fast, navigates complex traffic, and solves intricate logic problems, but it can only carry one small bucket of sand at a time. A GPU is a fleet of 10,000 slow dump trucks. Each truck moves at modest speed and only knows how to drive straight forward, but together they move the entire beach in a single trip. AI inference is moving that beach: multiplying billions of numbers simultaneously.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
If you try to run an LLM like Llama-3 on a CPU, generating a single paragraph can take 30 to 60 seconds because the CPU processes math sequentially across 8 to 32 cores. A GPU like an RTX 4090 or A100 has thousands of smaller cores executing in parallel, producing tokens in milliseconds. Understanding GPU thread scheduling (Warps of 32 threads and Streaming Multiprocessors) ensures you write code that keeps all cores busy rather than leaving 90% of your GPU idle.

---

## 4. Today's Lab Objective
Inspect your hardware environment, query the GPU's internal specs (compute capability, Streaming Multiprocessors, VRAM), and calculate the theoretical mathematical speed ceiling (Peak TFLOPs).

---

## 5. Step-by-Step Code Walkthrough
We use PyTorch device queries to inspect hardware properties from the CUDA driver, run `nvidia-smi` to examine operating power and memory limits, and calculate theoretical floating-point operations per second (FLOPs).

```python
# To run today's experiment:
python3 query_gpu_specs.py
```

---

## 6. How to Interpret Your Results
You will see your GPU model name, total VRAM, and SM count. The TFLOPs number is your hardware's absolute theoretical speed limit: no matter how optimized your software is, it can never perform more calculations per second than this ceiling.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare 32-bit float (FP32) vs 16-bit float (FP16). Why does cutting number precision in half double or quadruple the theoretical FLOPs on modern GPUs?`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-01/theoretical_specs_audit.md recording your GPU specs, SM count, and calculated theoretical peak TFLOPs.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare 32-bit float (FP32) vs 16-bit float (FP16). Why does cutting number precision in half double or quadruple the theoretical FLOPs on modern GPUs?`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
