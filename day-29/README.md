# Day 29: End-to-End Stress Testing & Chaos Engineering

**Module:** `Stage 8: Production-Ready Systems`

---

## 1. Prerequisites & Foundational Context
Understanding load testing concepts.

---

## 2. Core Mental Model (Intuition First)
Before launching a bridge, engineers drive fleets of heavy trucks across it to find structural weaknesses. Today, you stress-test your inference cluster with realistic, bursty traffic to identify where it breaks before real users do.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Systems fail in unpredictable ways under load: socket exhaustion, memory leakage, preemption loops, and latency spikes. A controlled chaos stress test proves your system's resilience.

---

## 4. Today's Lab Objective
Build an asynchronous multi-client load test harness simulating 100 concurrent users with varied prompt lengths, measuring error rates and $p99$ TTFT under saturation.

---

## 5. Step-by-Step Code Walkthrough
We simulate bursty Poisson arrival rates and measure system degradation.

```python
# To run today's experiment:
python3 stress_test_harness.py
```

---

## 6. How to Interpret Your Results
Identify the exact Requests Per Second (RPS) threshold where latency begins to spike exponentially.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Vary RPS from 5 to 50 requests/sec and find the breaking point.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-29/stress_test_report.md capturing failure thresholds and saturation points.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Vary RPS from 5 to 50 requests/sec and find the breaking point.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
