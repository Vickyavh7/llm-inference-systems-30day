# Day 04: Linux GPU Environment, nvidia-smi & Telemetry Automation

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Running background scripts in Python and file I/O basics.

---

## 2. Core Mental Model (Intuition First)
Imagine flying a high-performance jet without dials for fuel, engine temperature, or oil pressure. You would crash without warning! A GPU dynamically changes its clock frequencies based on temperature and power draw. If it gets too hot, it thermal-throttles and drops performance by 50% without throwing any error in your Python code.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
In production, an inference cluster might run out of memory (OOM), hit power caps, or throttle during peak hours. You cannot debug production issues without continuous telemetry. Programmatically querying and logging GPU metrics allows you to build health checks and autoscaling triggers.

---

## 4. Today's Lab Objective
Build a background telemetry logging daemon in Python that samples GPU utilization, memory usage, temperature, and power draw into a time-series CSV file while a stress workload is executing.

---

## 5. Step-by-Step Code Walkthrough
We use Python's `subprocess` to call `nvidia-smi` with `--query-gpu` flags, parse CSV output, run a concurrent matrix stress test, and log metrics over time.

```python
# To run today's experiment:
python3 telemetry_logger.py
```

---

## 6. How to Interpret Your Results
Inspect `gpu_telemetry.csv`. You will see `gpu_util_pct` jump to ~100%, `power_w` spike from idle (~25W) to peak (~250–450W), and `temp_c` rise. When the stress workload finishes, utilization drops to 0%. This is the raw heartbeat of GPU health.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Observe the power draw during idle vs stress. What is the power overhead of maintaining high clock speeds?`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-04/gpu_telemetry.csv and a summary of maximum power draw and temperature observed during the run.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Observe the power draw during idle vs stress. What is the power overhead of maintaining high clock speeds?`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
