# Day 22: Multi-GPU Scaling: Tensor Parallelism (TP) Architecture

**Module:** `Stage 5: GPU Networking & Parallelism`

---

## 1. Prerequisites & Foundational Context
Understanding matrix multiplication splits across devices.

---

## 2. Core Mental Model (Intuition First)
Imagine a massive calculation too big for one whiteboard. Tensor Parallelism splits the matrix itself across two whiteboards: Worker A multiplies the left half, Worker B multiplies the right half, and then they quickly call each other on the phone (`All-Reduce` collective) to sum their results and move to the next layer.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When a model is too large for a single GPU or when you need ultra-low latency for real-time voice, Tensor Parallelism shards every layer across multiple GPUs, cutting per-token generation latency almost in half per doubling of GPUs.

---

## 4. Today's Lab Objective
Configure vLLM with `--tensor-parallel-size 2` (or 4), verify memory distribution across devices, and measure scaling efficiency vs communication overhead.

---

## 5. Step-by-Step Code Walkthrough
We test multi-GPU sharding using Megatron-LM row/column parallel linear layers and measure the NCCL communication tax.

```python
# To run today's experiment:
python3 tensor_parallelism_benchmark.py
```

---

## 6. How to Interpret Your Results
On high-speed interconnects (NVLink), TP yields near-linear decode latency reductions. On slow interconnects (PCIe Gen3/4), communication overhead can eat away speedups.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare latency under $TP=1$ vs $TP=2$ vs $TP=4$.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-22/tensor_parallelism_scaling.md showing scaling efficiency and communication overhead.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare latency under $TP=1$ vs $TP=2$ vs $TP=4$.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
