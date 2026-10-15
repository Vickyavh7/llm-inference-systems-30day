# Day 06: GPU Storage Pipelines & Model Weight Loading (safetensors vs pickle)

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Understanding how model checkpoints are saved on disk.

---

## 2. Core Mental Model (Intuition First)
Imagine receiving a package. In the old format (PyTorch pickle `.pt`), the package is packed inside a puzzle box that requires running executable Python code to unpack every single part into CPU memory. In the modern format (Hugging Face `.safetensors`), the package is laid out in flat, contiguous memory blocks with a simple index header. The OS can memory-map (`mmap`) the file directly into RAM and DMA transfer it to GPU VRAM with zero parsing overhead.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When autoscaling LLMs in the cloud, cold-start latency is critical. Loading a 14 GB model via PyTorch pickle can take 45 seconds and spike CPU RAM to double the model size. Loading via `.safetensors` takes 3 to 5 seconds with zero CPU memory duplication, and prevents arbitrary code execution vulnerabilities.

---

## 4. Today's Lab Objective
Benchmark model weight serialization, file size, and deserialization load time between PyTorch `.pt` and Hugging Face `.safetensors` format.

---

## 5. Step-by-Step Code Walkthrough
We generate a dictionary of model weight tensors, save them in both formats, time the loading process using `time.perf_counter()`, and measure disk and memory efficiency.

```python
# To run today's experiment:
python3 weight_loading_benchmark.py
```

---

## 6. How to Interpret Your Results
`.safetensors` loads significantly faster and avoids unpickling overhead. Furthermore, `.safetensors` supports memory-mapping (`mmap`), allowing frameworks like vLLM to stream weights directly into GPU memory without allocating 14 GB of host CPU RAM first.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare cold read (clearing OS disk cache) vs warm read (file already in OS buffer cache).`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-06/weight_loading_benchmark.md analyzing why `.safetensors` has become the universal standard for LLM weights.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare cold read (clearing OS disk cache) vs warm read (file already in OS buffer cache).`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
