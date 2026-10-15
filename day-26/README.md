# Day 26: Production Observability: Prometheus Metrics & Grafana

**Module:** `Stage 8: Production-Ready Systems`

---

## 1. Prerequisites & Foundational Context
Understanding time-series metrics and monitoring dashboards.

---

## 2. Core Mental Model (Intuition First)
Traditional web apps monitor CPU, RAM, and HTTP 500 rates. For LLMs, those metrics are misleading! A GPU sitting at 100% compute utilization might be serving requests efficiently or it might be thrashing and failing. You must monitor LLM-native metrics: Queue Depth, KV Cache Saturation, TTFT histograms, and Token Generation Rate.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When an LLM cluster slows down, you cannot fix it unless your dashboard tells you whether the bottleneck is KV cache exhaustion, queue starvation, or network latency.

---

## 4. Today's Lab Objective
Scrape Prometheus metrics from vLLM's `/metrics` endpoint and build a Grafana dashboard JSON configuration tracking latency and cache usage.

---

## 5. Step-by-Step Code Walkthrough
We inspect key vLLM Prometheus metrics: `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_factor`, and `vllm:time_to_first_token_seconds`.

```python
# To run today's experiment:
python3 metrics_collector_exporter.py
```

---

## 6. How to Interpret Your Results
Observe how `gpu_cache_usage_factor` approaching 1.0 warns of impending preemption before any request fails.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Simulate traffic bursts and observe metric spikes in real time.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-26/grafana_inference_dashboard.json and alerting rules for KV cache saturation.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Simulate traffic bursts and observe metric spikes in real time.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
