# Day 25: AI Gateways: Load Balancing, Reverse Proxies & Routing

**Module:** `Stage 8: Production-Ready Systems`

---

## 1. Prerequisites & Foundational Context
Basic understanding of REST APIs, reverse proxies (like Nginx), and streaming HTTP responses.

---

## 2. Core Mental Model (Intuition First)
An AI Gateway is the air traffic controller of your LLM cluster. Users never talk directly to individual GPUs. Instead, they hit the Gateway, which checks API keys, enforces rate limits, inspects current GPU queue depths, and routes the request to the replica with the shortest wait time.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
In production, you run multiple vLLM instances behind a cluster. Without an intelligent AI Gateway, one worker might get overwhelmed with massive prompts while another sits idle.

---

## 4. Today's Lab Objective
Build an asynchronous AI Gateway in Python using FastAPI with Server-Sent Events (SSE) streaming and queue-aware load balancing across two mock backend workers.

---

## 5. Step-by-Step Code Walkthrough
We implement a reverse proxy router in FastAPI that tracks active connections per backend and routes to the least-busy worker.

```python
# To run today's experiment:
python3 gateway_router.py
```

---

## 6. How to Interpret Your Results
Notice how least-connections routing prevents traffic clumping and balances GPU utilization evenly.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare Round-Robin routing vs Least-Connections routing under uneven request sizes.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-25/gateway_router.py ready to front production vLLM servers.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare Round-Robin routing vs Least-Connections routing under uneven request sizes.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
