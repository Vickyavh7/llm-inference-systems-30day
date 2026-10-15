# Day 27: Autoscaling Architecture & Inference SLO Design

**Module:** `Stage 8: Production-Ready Systems`

---

## 1. Prerequisites & Foundational Context
Understanding Horizontal Pod Autoscaling (HPA).

---

## 2. Core Mental Model (Intuition First)
Traditional autoscalers add more servers when CPU utilization exceeds 80%. If you do that with LLMs, your autoscaler will go crazy! A single user generating text will cause the GPU to register near 100% utilization. Instead, autoscaling for LLMs must trigger on **Queue Depth** (are requests waiting?) and **KV Cache Saturation**.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Autoscaling on the wrong metric leads to runaway cloud costs or severe SLA violations. Knowing how to define Service Level Objectives (e.g. $p95$ TTFT < 200ms) and scale based on queue backlog saves tens of thousands of dollars in cloud spend.

---

## 4. Today's Lab Objective
Build an autoscaling simulation script that polls engine queue metrics and dynamically calculates desired worker replica counts.

---

## 5. Step-by-Step Code Walkthrough
We implement replica calculation logic based on target queue backlog per worker.

```python
# To run today's experiment:
python3 autoscaler_controller.py
```

---

## 6. How to Interpret Your Results
Notice how replica counts scale cleanly with request backlog, scaling down when queues empty.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Adjust target queue threshold and cooldown stabilization windows.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-27/autoscaler_controller.md analyzing autoscaling policies and SLO compliance.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Adjust target queue threshold and cooldown stabilization windows.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
