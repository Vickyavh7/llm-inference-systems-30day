# Day 02: GPU Memory Hierarchy & Bandwidth Limits

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Understanding Day 1's concept of GPU compute cores and VRAM.

---

## 2. Core Mental Model (Intuition First)
Think of GPU compute cores as master chefs in a kitchen. The chef can chop an onion in 1 millisecond. But where are the onions kept? The chef has a tiny cutting board (Registers & SRAM) right in front of them, but the main storage pantry (VRAM) is down the hallway. If the chef has to wait for assistants to bring onions from the pantry across a narrow doorway (Memory Bandwidth), the chef spends 90% of their time waiting! In LLMs, your GPU is almost always starved for memory bandwidth, not compute.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When generating text word by word (decode phase), an LLM has to read all 14 Gigabytes of model weights from VRAM to generate just ONE token. If your memory bandwidth is 1,000 GB/s, reading 14 GB takes 14 milliseconds. That means you can never exceed ~70 tokens per second on that GPU, no matter how many compute cores you have! Memory bandwidth is the primary bottleneck of LLM generation.

---

## 4. Today's Lab Objective
Write a PyTorch benchmark measuring real-world bandwidth for transferring data from Host CPU RAM to GPU VRAM (H2D) and within GPU VRAM itself (D2D).

---

## 5. Step-by-Step Code Walkthrough
We allocate tensors of increasing sizes (1 MB to 1 GB) and measure copy durations. Crucially, we use `torch.cuda.synchronize()` because PyTorch GPU operations are asynchronous: Python returns immediately while the GPU works in the background. `synchronize()` forces Python to wait until the transfer physically finishes so our timer is accurate.

```python
# To run today's experiment:
python3 bandwidth_benchmark.py
```

---

## 6. How to Interpret Your Results
Small buffer sizes (1 MB) show low bandwidth because kernel launch latency dominates. As size grows to 500 MB–1 GB, bandwidth saturates: PCIe Gen4 gives ~20–25 GB/s for H2D, while internal VRAM (D2D) reaches 500–900+ GB/s. This proves why moving data across the PCIe bus is 30x slower than keeping it inside VRAM.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare `pin_memory=True` vs `pin_memory=False` on the host tensor to see how non-pinned memory forces extra CPU buffering.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-02/bandwidth_results.md showing your measured H2D vs D2D bandwidth table and explaining why small transfers suffer from latency overhead.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare `pin_memory=True` vs `pin_memory=False` on the host tensor to see how non-pinned memory forces extra CPU buffering.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
