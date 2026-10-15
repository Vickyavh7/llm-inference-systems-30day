# Day 15: High-Performance Serving: Static vs. Continuous Batching

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Understanding naive sequential inference vs batching.

---

## 2. Core Mental Model (Intuition First)
Imagine a bus station (Static Batching). The bus waits until 20 passengers arrive, then drives them to the destination. But Passenger 1 gets off at Stop 1, while Passenger 20 stays until Stop 20. For the remaining 19 stops, the bus drives nearly empty seats! In **Continuous (Iteration-level) Batching**, the bus never stops: at every single intersection (decode iteration), finished passengers hop off and new passengers immediately hop into the empty seats.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Static batching wastes up to 70% of GPU compute because requests have different prompt and generation lengths. Continuous batching increases serving throughput by 3x to 5x on the exact same GPU hardware without changing model weights.

---

## 4. Today's Lab Objective
Deploy a model serving engine (e.g. vLLM) and benchmark continuous batching throughput under concurrent request load.

---

## 5. Step-by-Step Code Walkthrough
We use an asynchronous Python client (`asyncio` and `httpx`) to launch 10 simultaneous requests to the serving endpoint and measure total system throughput.

```python
# To run today's experiment:
python3 continuous_batching_client.py
```

---

## 6. How to Interpret Your Results
Notice that under continuous batching, requests that finish early do not block the GPU. Total system tokens/sec remains high even when individual request lengths vary widely.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare throughput under concurrency 1 vs 5 vs 20.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-15/batching_benchmark_comparison.md comparing static vs continuous batching throughput.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare throughput under concurrency 1 vs 5 vs 20.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
