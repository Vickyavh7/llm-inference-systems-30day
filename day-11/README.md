# Day 11: Inference Latency Metrics: TTFT, TPOT, and ITL

**Module:** `Stage 3: Analyze Inference Performance`

---

## 1. Prerequisites & Foundational Context
Understanding streamed token output over WebSockets or Server-Sent Events (SSE).

---

## 2. Core Mental Model (Intuition First)
When a user chats with ChatGPT, they judge speed by two completely distinct feelings: 1) How long do they stare at a blank screen before the first word pops up? (Time to First Token, TTFT). 2) Once words start appearing, does it stream like a natural human speaking, or does it stutter and lag? (Inter-Token Latency, ITL).

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Inference SLAs and user retention are dictated by percentiles ($p50, p95, p99$), not averages. A single 500ms stutter ($p99$ ITL spike) makes a voice agent or chatbot feel broken even if the average tokens/sec is high.

---

## 4. Today's Lab Objective
Build a benchmark harness that instruments token timestamps, measures TTFT, computes per-token ITL, and calculates $p50, p90, p99$ percentiles.

---

## 5. Step-by-Step Code Walkthrough
We record high-resolution timestamps with `time.perf_counter()` for prompt reception, first token emission, and each subsequent token arrival.

```python
# To run today's experiment:
python3 latency_metrics_harness.py
```

---

## 6. How to Interpret Your Results
Notice the separation between TTFT and ITL. TTFT reflects prefill compute latency, while ITL reflects memory-bound decode step time plus streaming overhead.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Simulate network jitter by adding random sleep variations. How do $p95$ and $p99$ ITL diverge from the median $p50$?`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-11/latency_metrics_harness.md with an empirical latency distribution breakdown.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Simulate network jitter by adding random sleep variations. How do $p95$ and $p99$ ITL diverge from the median $p50$?`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
