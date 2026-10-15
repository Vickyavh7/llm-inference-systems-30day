# Day 10: The KV Cache: Mathematical Mechanics & VRAM Footprint

**Module:** `Stage 2: LLM Inference Mechanics`

---

## 1. Prerequisites & Foundational Context
Understanding self-attention (Query, Key, Value matrices).

---

## 2. Core Mental Model (Intuition First)
Imagine calculating the balance of your bank account every day. On Day 100, you could recalculate all 100 days of transactions from scratch (naive inference: $O(N^2)$ math), or you could cache yesterday's final balance and simply add today's transaction (KV Cache: $O(1)$ per step). The catch? The notebook where you store past balances (VRAM) grows with every single day and every single client!

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
In production, the KV Cache—not the model weights—is usually what causes Out-of-Memory (OOM) crashes! For an 8B model with 32k context, a single user's KV cache can take several gigabytes. If 10 users send requests simultaneously, your 24GB GPU crashes instantly unless you know how to calculate and manage KV cache allocation.

---

## 4. Today's Lab Objective
Build a KV Cache memory calculator script and validate theoretical VRAM predictions against empirical PyTorch allocations.

---

## 5. Step-by-Step Code Walkthrough
We use the analytical formula: $2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times batch\_size \times bytes\_per\_elem$ and compare Multi-Head Attention (MHA) against Grouped-Query Attention (GQA).

```python
# To run today's experiment:
python3 kv_cache_calculator.py
```

---

## 6. How to Interpret Your Results
Notice that at 32k tokens, batch size 8 consumes ~16 GB of VRAM solely for KV cache! This demonstrates why Grouped-Query Attention (GQA) was invented to cut KV heads from 32 to 8, reducing memory by 75%.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare MHA (32 KV heads) vs GQA (8 KV heads) vs MQA (1 KV head).`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-10/kv_cache_scaling_analysis.md with scaling plots and memory formulas.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare MHA (32 KV heads) vs GQA (8 KV heads) vs MQA (1 KV head).`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
