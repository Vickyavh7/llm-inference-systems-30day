# Day 08: Autoregressive Inference Lifecycle & Tokenization Pipeline

**Module:** `Stage 2: LLM Inference Mechanics`

---

## 1. Prerequisites & Foundational Context
Understanding how text is converted into integers (tokens) using a tokenizer.

---

## 2. Core Mental Model (Intuition First)
Imagine writing an essay one word at a time, but with a bizarre restriction: every time you want to write the next word, you must reread your entire essay from the very first word! That is autoregressive generation. The model predicts only ONE next token per step, appends it to the context, and repeats.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
Unlike traditional computer vision models that process an image in one shot, LLMs generate text token by token. Generating a 200-word response requires 200 sequential forward passes through the entire model! Understanding this loop explains why LLM latency scales directly with output length.

---

## 4. Today's Lab Objective
Write a minimal autoregressive generation loop in pure PyTorch without external frameworks and record the execution time of each step.

---

## 5. Step-by-Step Code Walkthrough
We simulate a vocabulary projection and sampling step. We measure how step latency behaves over 20 iterations.

```python
# To run today's experiment:
python3 autoregressive_loop.py
```

---

## 6. How to Interpret Your Results
Notice that every step takes a non-zero amount of time. In a real model, each step requires loading billions of weights from memory, which is why generation speed is measured in tokens per second.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare greedy sampling (`argmax`) vs top-p sampling with temperature.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-08/autoregressive_loop.md analyzing the step-by-step latency of autoregressive decoding.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare greedy sampling (`argmax`) vs top-p sampling with temperature.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
