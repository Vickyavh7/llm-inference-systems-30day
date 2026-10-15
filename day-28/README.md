# Day 28: Speculative Decoding & Advanced Serving Accelerations

**Module:** `Stage 8: Production-Ready Systems`

---

## 1. Prerequisites & Foundational Context
Understanding autoregressive generation from Day 8.

---

## 2. Core Mental Model (Intuition First)
Imagine a senior architect working with a fast junior apprentice. The apprentice quickly drafts 5 sentences of code. The architect reviews all 5 sentences in a single glance (one parallel verification step). If 4 sentences are good, the architect approves them and corrects the 5th. You just got 4 tokens in the time of 1 step! That is **Speculative Decoding**.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Speculative decoding achieves 2x to 3x wall-clock speedup for single-user generation without ANY loss of output quality or mathematical accuracy.

---

## 4. Today's Lab Objective
Deploy vLLM with speculative decoding (`--speculative-model`) and benchmark generation speedup across draft lookahead lengths.

---

## 5. Step-by-Step Code Walkthrough
We test a small draft model (e.g. Llama-3-1B) proposing tokens for a larger target model (Llama-3-8B).

```python
# To run today's experiment:
python3 speculative_decoding_test.py
```

---

## 6. How to Interpret Your Results
Observe the acceptance rate (e.g. 70-80%). High acceptance rate translates to 2x+ tokens/sec speedup.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare lookahead tokens $K=3$ vs $K=5$ across different prompt types.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-28/speculative_decoding_benchmark.md analyzing acceptance rates and effective speedup.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare lookahead tokens $K=3$ vs $K=5$ across different prompt types.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
