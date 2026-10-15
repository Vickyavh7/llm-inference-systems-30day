# Day 23: Pipeline Parallelism (PP) & NCCL Collective Primitives

**Module:** `Stage 5: GPU Networking & Parallelism`

---

## 1. Prerequisites & Foundational Context
Understanding how models are composed of sequential transformer layers.

---

## 2. Core Mental Model (Intuition First)
Think of an assembly line. GPU 1 handles layers 1 to 16, then passes activations to GPU 2 for layers 17 to 32. Unlike Tensor Parallelism (which talks after every single matrix multiplication), Pipeline Parallelism only talks at the boundary between groups of layers.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When scaling across multiple physical nodes where network connections are slower than NVLink, Pipeline Parallelism minimizes communication frequency, making multi-node cluster serving viable.

---

## 4. Today's Lab Objective
Write a PyTorch distributed script running raw NCCL collective operations (`dist.all_reduce()`) across GPUs and measure transfer latency vs message size.

---

## 5. Step-by-Step Code Walkthrough
We benchmark raw NCCL collective communication times from 1 MB to 1 GB tensors.

```python
# To run today's experiment:
python3 nccl_collectives_benchmark.py
```

---

## 6. How to Interpret Your Results
Notice how All-Reduce latency scales with payload size and interconnect bandwidth.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare collective latency across tensor payload sizes from 1 MB to 512 MB.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-23/nccl_collectives_benchmark.md analyzing NCCL bandwidth saturation.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare collective latency across tensor payload sizes from 1 MB to 512 MB.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
