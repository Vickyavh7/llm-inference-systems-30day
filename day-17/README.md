# Day 17: Request Scheduling, Queuing & Preemption Strategies

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Understanding how request queues operate in web servers.

---

## 2. Core Mental Model (Intuition First)
Imagine a popular restaurant. Tables (VRAM blocks) are fully occupied. When new guests arrive, do you make them wait outside (Waiting Queue), kick an existing diner out and throw away their food (Recompute Preemption), or ask an existing diner to pause and wait at the bar while saving their plate (Swapping Preemption)?

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When traffic spikes, KV cache will inevitably fill up. How your serving engine handles saturation determines whether requests crash with 500 errors or gracefully degrade with slight latency increases.

---

## 4. Today's Lab Objective
Induce queue saturation in a serving engine and compare the latency impact of recompute preemption versus swap preemption.

---

## 5. Step-by-Step Code Walkthrough
We simulate request burst spikes to observe queue transitions between Waiting, Running, and Swapped queues.

```python
# To run today's experiment:
python3 queue_saturation_test.py
```

---

## 6. How to Interpret Your Results
Monitor `num_requests_waiting` and `num_requests_swapped`. Recomputation saves host RAM but burns GPU FLOPs; swapping preserves FLOPs but requires PCIe host memory transfers.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare `--preemption-mode recompute` vs `--preemption-mode swap`.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-17/preemption_tradeoff_study.md evaluating recovery latency after memory saturation.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare `--preemption-mode recompute` vs `--preemption-mode swap`.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
