# Day 14: Weekly Capstone 2: Prefill/Decode Performance Audit

**Module:** `Stage 3: Analyze Inference Performance`

---

## 1. Prerequisites & Foundational Context
Completion of Days 8 through 13.

---

## 2. Core Mental Model (Intuition First)
You now understand the anatomy of a request: prompt tokenization -> compute-heavy prefill -> memory-heavy decode with KV cache expansion -> token streaming. Today, you synthesize this knowledge into an end-to-end performance diagnostic report.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Before optimizing or serving a model with advanced frameworks, you must measure its naive baseline performance so you can prove the exact ROI of continuous batching and PagedAttention in Week 3.

---

## 4. Today's Lab Objective
Compile prefill throughput, decode latency, KV cache scaling, and profiling traces into your second major portfolio piece: `Week-2-Inference-Mechanics-Report.md`.

---

## 5. Step-by-Step Code Walkthrough
Execute the audit harness to aggregate latency curves across sequence lengths.

```python
# To run today's experiment:
python3 performance_audit_capstone.py
```

---

## 6. How to Interpret Your Results
Synthesize your measured TTFT and TPOT numbers across sequence lengths (256 to 4096 tokens).

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare model precision (FP16 vs INT8) across sequence lengths.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `Portfolio Deliverable #2: `Week-2-Inference-Mechanics-Report.md`.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare model precision (FP16 vs INT8) across sequence lengths.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
