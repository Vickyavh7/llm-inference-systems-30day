# Day 19: Chunked Prefill & Mitigating Decode Starvation

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Understanding the Prefill-Decode interference problem from Day 9.

---

## 2. Core Mental Model (Intuition First)
Imagine a highway where cars (decode iterations) are driving smoothly at 60 mph. Suddenly, an oversized cargo truck (a giant 8,000-token prompt prefill) enters the highway and blocks all lanes for 500 milliseconds. All the cars grind to a halt (stutter/jitter!). **Chunked Prefill** breaks that giant cargo truck into smaller pickup trucks (512-token chunks) so cars can keep flowing between them.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Without chunked prefill, a user happily streaming text will experience jarring 300–500ms freezes whenever another user submits a large prompt. Chunked prefill stabilizes Inter-Token Latency ($p99$ ITL) for a butter-smooth user experience.

---

## 4. Today's Lab Objective
Configure chunked prefill in vLLM (`--enable-chunked-prefill`) and measure the reduction in decode latency jitter while long prompts arrive.

---

## 5. Step-by-Step Code Walkthrough
We test ongoing token streaming while simultaneously firing long-prompt prefill requests in the background.

```python
# To run today's experiment:
python3 chunked_prefill_eval.py
```

---

## 6. How to Interpret Your Results
With chunked prefill enabled, $p99$ ITL remains steady and predictable even under heavy background prompt injection.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare `--enable-chunked-prefill` enabled vs disabled under bursty prompt arrivals.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-19/chunked_prefill_eval.md analyzing ITL jitter curves.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare `--enable-chunked-prefill` enabled vs disabled under bursty prompt arrivals.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
