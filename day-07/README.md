# Day 07: Weekly Capstone 1: Comprehensive GPU Systems Audit

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Completion of Days 1 through 6.

---

## 2. Core Mental Model (Intuition First)
Before a race car driver goes on the track, the mechanics conduct a complete systems diagnostic: engine dyno, tire grip, fuel pump flow rate, and telemetry. Today, you are the chief systems engineer conducting the full baseline audit of your GPU compute environment.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
In every AI engineering interview and production deployment, you need to know your system's exact operational envelope: What is your theoretical FLOPs? Your empirical memory bandwidth? Your PCIe speed? Your weight loading speed? Without these numbers, you cannot predict whether your system can meet a client's latency SLO.

---

## 4. Today's Lab Objective
Consolidate your findings from Days 1–6 into a unified systems profile script and write your first major portfolio piece: `Week-1-Systems-Foundation-Report.md`.

---

## 5. Step-by-Step Code Walkthrough
This capstone script runs a comprehensive diagnostic suite, executing device inspection, memory bandwidth benchmarking, GEMM TFLOP calculation, and saving the results into an audit summary.

```python
# To run today's experiment:
python3 system_audit_capstone.py
```

---

## 6. How to Interpret Your Results
This script outputs your hardware scorecard. Use these numbers to establish your hardware baseline.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare your empirical TFLOPs against your Day 1 theoretical ceiling. What percentage of the theoretical ceiling did your code achieve?`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `Portfolio Deliverable #1: `Week-1-Systems-Foundation-Report.md`.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare your empirical TFLOPs against your Day 1 theoretical ceiling. What percentage of the theoretical ceiling did your code achieve?`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
