# Day 13: Advanced Kernel Profiling with NVIDIA Nsight Systems (nsys)

**Module:** `Stage 3: Analyze Inference Performance`

---

## 1. Prerequisites & Foundational Context
Understanding OS threads and CUDA streams.

---

## 2. Core Mental Model (Intuition First)
While PyTorch profiler looks at Python operations, NVIDIA Nsight Systems (`nsys`) looks at the entire physical computer: CPU hardware threads, Linux kernel scheduling, PCIe data packets, and CUDA hardware execution queues.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When building production engines like vLLM or TensorRT-LLM, high-level Python profilers miss low-level GPU idle bubbles caused by thread locking, memory copies, or unoptimized kernel scheduling. `nsys` is the gold standard used by GPU systems performance engineers.

---

## 4. Today's Lab Objective
Learn how to invoke `nsys profile` to capture a full system trace and identify GPU synchronization bubbles.

---

## 5. Step-by-Step Code Walkthrough
We provide the production command line template for running `nsys` on inference workloads and explain how to interpret the timeline.

```python
# To run today's experiment:
python3 profile_nsys_command.sh
```

---

## 6. How to Interpret Your Results
On the timeline, look for gaps between CUDA kernels. Any white space on the GPU row represents lost throughput where cores sat idle waiting for CPU instructions.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare synchronous `torch.cuda.synchronize()` vs asynchronous execution.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-13/nsys_walkthrough.md with timeline analysis and bubble identification.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare synchronous `torch.cuda.synchronize()` vs asynchronous execution.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
