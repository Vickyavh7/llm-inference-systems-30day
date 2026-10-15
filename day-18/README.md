# Day 18: Prompt Caching & RadixAttention (SGLang Deep Dive)

**Module:** `Stage 4: Model Serving`

---

## 1. Prerequisites & Foundational Context
Understanding that many LLM requests share identical system prompts or few-shot examples.

---

## 2. Core Mental Model (Intuition First)
Imagine having a conversation where every time you ask a question, the person re-reads the entire company policy handbook before answering. That is what naive serving does. Prompt caching remembers the KV cache of the handbook so only your new question needs to be processed. RadixAttention (in SGLang) organizes these cached prefixes in a Radix tree for instant matching.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
For applications like customer support agents, coding assistants, or document Q&A, prompt prefixes can be 2,000 to 10,000 tokens long. Recomputing those tokens on every turn wastes GPU FLOPs. Prompt caching cuts TTFT by 80% to 90% and massively slashes serving costs.

---

## 4. Today's Lab Objective
Benchmark TTFT with prompt caching enabled vs disabled when sending requests with a shared 3,000-token system prompt.

---

## 5. Step-by-Step Code Walkthrough
We compare latency on the first request (cold cache) against subsequent requests (warm cache hit).

```python
# To run today's experiment:
python3 prefix_caching_benchmark.py
```

---

## 6. How to Interpret Your Results
On warm cache hits, TTFT drops from hundreds of milliseconds to under 30 milliseconds because the prefill phase is completely bypassed for the prefix.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare cache hit vs cache miss across prefix lengths from 500 to 4000 tokens.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-18/prefix_caching_speedup.md documenting TTFT reduction and cache hit ratios.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare cache hit vs cache miss across prefix lengths from 500 to 4000 tokens.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
