# Day 12: Profiling with PyTorch Profiler & Chrome Trace Inspection

**Module:** `Stage 3: Analyze Inference Performance`

---

## 1. Prerequisites & Foundational Context
Understanding how CPU launches CUDA kernels to GPU streams.

---

## 2. Core Mental Model (Intuition First)
Imagine watching a high-speed video recording of a construction crew. At normal speed, it looks like everyone is working constantly. In slow motion, you see the foreman spends 10 seconds shouting orders, and the crew only works for 2 seconds before waiting for the next order. PyTorch Profiler is that slow-motion camera: it exposes whether your CPU or GPU is the bottleneck.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Often, an inference script runs slowly not because the GPU is slow, but because Python is wasting time launching tiny kernels sequentially, causing CPU launch overhead and GPU idle bubbles.

---

## 4. Today's Lab Objective
Profile an inference forward pass using `torch.profiler`, identify top kernels by time, and export a Chrome tracing JSON file for visual timeline inspection.

---

## 5. Step-by-Step Code Walkthrough
We wrap operations in `torch.profiler.profile()`, recording CPU and CUDA activities. We inspect the sorted averages table.

```python
# To run today's experiment:
python3 run_pytorch_profiler.py
```

---

## 6. How to Interpret Your Results
In the table, find the kernel names (e.g. `gemm_kernel`, `elementwise_kernel`). Look at `Self CPU %` vs `Self CUDA %` to see where time is truly spent.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare batch size 1 vs batch size 4. Notice how kernel execution time increases while launch overhead stays constant.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-12/pytorch_profile_trace.md detailing the top 5 bottleneck kernels.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare batch size 1 vs batch size 4. Notice how kernel execution time increases while launch overhead stays constant.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
