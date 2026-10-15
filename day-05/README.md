# Day 05: Interconnects: PCIe Gen4/5 vs. NVLink & P2P Memory Access

**Module:** `Stage 1: GPU & Systems Foundation`

---

## 1. Prerequisites & Foundational Context
Understanding that multi-GPU clusters communicate across hardware interconnects.

---

## 2. Core Mental Model (Intuition First)
Imagine two offices on opposite sides of a city. If Office A sends paper documents to Office B via the standard postal system (PCIe bus via the CPU Host RAM), documents take hours. But if you build a private high-speed pneumatic tube directly connecting their windows (NVLink Peer-to-Peer P2P), documents transfer instantaneously without touching the postal system.

---

## 3. Why We Are Doing This (The "Why This Matters" in Production)
When models are larger than one GPU (e.g. Llama-3-70B requires multiple GPUs), the GPUs must exchange intermediate tensor calculations after every single transformer layer. If they communicate over PCIe Gen4 (32 GB/s), network overhead dominates. If they communicate over NVLink (600–900 GB/s), communication latency drops by 20x.

---

## 4. Today's Lab Objective
Inspect intra-node topology using `nvidia-smi topo -m` and write a PyTorch script to verify Peer-to-Peer (P2P) access and evaluate direct device memory transfer capability.

---

## 5. Step-by-Step Code Walkthrough
The script queries the number of GPUs. If 2 or more exist, it queries `torch.cuda.can_device_access_peer(device_a, device_b)`. On a single-GPU system, it prints the theoretical topology comparison and explains PCIe vs NVLink differences.

```python
# To run today's experiment:
python3 p2p_check.py
```

---

## 6. How to Interpret Your Results
In `nvidia-smi topo -m`, you will see legend codes: `SYS` (traverses CPU host bridge, slowest), `PIX` (traverses single PCIe switch), `NV#` (direct NVLink, fastest). If P2P access is True, GPUs directly read each other's VRAM without CPU intervention.

---

## 7. The Daily Controlled Experiment
* **Variable to Change:** `Compare transfer latency: P2P direct transfer vs copying to CPU host memory first and then to the target GPU.`
* **Hypothesis Question:** What happens to latency, bandwidth, or throughput when you alter this parameter?

---

## 8. Daily Portfolio Deliverable
* **Target File:** `day-05/interconnect_topology_report.md explaining why multi-GPU inference requires high-speed interconnects like NVLink.`
* **Write-up Template:**
  1. **Hypothesis:** Expected outcome when changing `Compare transfer latency: P2P direct transfer vs copying to CPU host memory first and then to the target GPU.`.
  2. **Experimental Setup:** Hardware, driver, VRAM, precision, and software versions.
  3. **Measured Data:** Tabular metrics and numbers from your run.
  4. **Bottleneck Analysis:** Did you hit compute limits, memory bandwidth ceilings, or interconnect transfer bottlenecks?
  5. **Production Trade-off:** What was gained vs what was sacrificed?
