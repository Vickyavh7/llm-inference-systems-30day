# Day 21: Weekly Capstone 3: High-Throughput Serving Benchmark

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Completion of Days 15 through 20.

---

## 2. Core Mental Model (Intuition First)
You have assembled all the engine components: continuous batching, PagedAttention, prompt caching, chunked prefill, and quantization. Today, you put the complete serving engine on the dyno and push it to maximum concurrency.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Production sizing requires knowing your engine's saturation curve: At what concurrency does throughput peak? At what request rate does latency degrade past your SLA? Today you answer those questions with hard data.

---

## 4. Today's Lab Objective
Run an automated load test sweeping concurrency from 1 to 64 concurrent requests, plotting throughput vs latency curves, and write `Week-3-Model-Serving-Engine-Report.md`.

---

## 5. Step-by-Step Code Walkthrough
Execute an automated load sweep collecting TTFT, ITL, and tokens/sec across concurrency steps.

```python
# To run today's experiment:
python3 serving_capstone_benchmark.py
```

---

## 6. How to Interpret Your Results
Plot Concurrency on the X-axis and Tokens/Sec on the Y-axis. Notice how throughput ramps up linearly and then plateaus as memory bandwidth saturates.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare serving throughput with and without prompt caching across concurrency levels.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `Portfolio Deliverable #3: `Week-3-Model-Serving-Engine-Report.md`.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare serving throughput with and without prompt caching across concurrency levels.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
