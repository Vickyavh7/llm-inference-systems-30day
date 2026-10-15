# Day 24: Disaggregated Prefill and Decode Architecture

**Module:** `Stage 7: Architect Distributed Inference`

---

## 1. Prerequisites & Foundational Context
Understanding the fundamental conflict between Prefill and Decode from Days 9 and 19.

---

## 2. Core Mental Model (Intuition First)
Imagine a hospital where surgeons (compute-heavy prefill) and recovery nurses (memory-heavy decode) share the exact same room and equipment. It creates chaos and scheduling delays! **Disaggregated Serving** builds two separate wings: a Prefill Wing dedicated to churning through prompts at maximum TFLOPs, and a Decode Wing dedicated to streaming tokens smoothly. They transfer patient charts (KV cache) over high-speed networks.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Disaggregation (as seen in DistServe and Splitwise) is the cutting edge of large-scale LLM infrastructure in 2024–2026. It completely isolates prefill latency spikes from ongoing decode streams, guaranteeing strict $p99$ latency SLAs.

---

## 4. Today's Lab Objective
Model and simulate a disaggregated inference architecture: evaluate KV cache network transfer requirements and compare against unified serving.

---

## 5. Step-by-Step Code Walkthrough
We calculate network bandwidth requirements for transferring KV cache states over RDMA/RoCE between prefill and decode instances.

```python
# To run today's experiment:
python3 disaggregated_simulation.py
```

---

## 6. How to Interpret Your Results
Observe how network bandwidth (e.g. 100 Gbps vs 400 Gbps InfiniBand) determines the break-even point for disaggregated serving.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Calculate transfer latency across prompt lengths from 1k to 16k tokens over 100 Gbps vs 400 Gbps networks.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-24/disaggregated_architecture_spec.md detailing system design and latency guarantees.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Calculate transfer latency across prompt lengths from 1k to 16k tokens over 100 Gbps vs 400 Gbps networks.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
