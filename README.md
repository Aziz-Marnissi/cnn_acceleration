# 🧠⚡ CNN Hardware Accelerator — LeNet-5 on FPGA (PYNQ-Z2)

> **A hand-built, quantized, block-by-block HLS pipeline that runs MNIST inference 74× faster than a CPU.**

---

## 🎯 Project Principle

Modern CNN inference on CPU is bottlenecked by sequential, general-purpose compute. This project explores a **hardware-first approach**: instead of running LeNet-5 as one monolithic model, the network is **decomposed into 3 independent HLS hardware blocks**, each synthesized, quantized, and validated as its own IP core on a **Xilinx PYNQ-Z2 (XC7Z020)** FPGA.

The core idea: **specialize the silicon to the math**. Convolutions become deeply pipelined MAC arrays, pooling becomes pure combinational comparator trees, and fully-connected layers exploit massive parallel unrolling — all impossible to fully exploit on a general-purpose CPU core.

---

## 🏗️ CNN Architecture — LeNet-5

```
Input Image (28×28, grayscale)
        │
        ▼
┌───────────────────────────────┐
│  🧩 BLOCK 1                    │
│  Conv1: 6 filters × 5×5        │  →  ReLU  →  24×24×6
│  MaxPool 2×2                   │  →         12×12×6
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  🧩 BLOCK 2                    │
│  Conv2: 16 filters × 5×5×6     │  →  ReLU  →  8×8×16
│  MaxPool 2×2                   │  →         4×4×16
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  🧩 BLOCK 3                    │
│  Flatten                       │  →  256
│  FC1: 256 → 120                │  →  ReLU
│  FC2: 120 → 84                 │  →  ReLU
│  FC3: 84  → 10                 │  →  raw logits
│  Argmax                        │  →  predicted class (0-9)
└───────────────────────────────┘
        │
        ▼
   🔢 Predicted Digit
```

**Total parameters:** ~44,400 (Conv1: 156 · Conv2: 2,416 · FC1: 30,840 · FC2: 10,164 · FC3: 850)

---

## 📐 Mathematical Foundations

### Convolution (Block 1 & 2)
Each output feature map is computed as:

```
out[co][i][j] = ReLU( b[co] + Σ_ci Σ_ki Σ_kj  in[ci][i+ki][j+kj] · W[co][ci][ki][kj] )
```

This is a **multi-channel cross-correlation**, not true mathematical convolution (kernel is not flipped) — standard convention in deep learning frameworks.

### Max Pooling
Non-linear downsampling, preserves the strongest activation in each 2×2 window:

```
out[i][j] = max( in[2i][2j], in[2i][2j+1], in[2i+1][2j], in[2i+1][2j+1] )
```

### Fully-Connected Layers
Classic matrix-vector product:

```
out[o] = ReLU( b[o] + Σ_i  in[i] · W[o][i] )
```

Final layer (FC3) skips ReLU — raw logits are fed directly into **argmax**, since softmax normalization is unnecessary when only the *arg*-max class is needed (monotonic transformation preserves ranking).

### Fixed-Point Quantization
All activations/weights use **`ap_fixed<8,4>`** (Q4.4 format — 4 integer bits, 4 fractional bits), replacing 32-bit floats:

```
value_fixed = round(value_float × 2^4) / 2^4
```

Accumulators are widened (`ap_fixed<20,10>` to `ap_fixed<26,16>`) with **`AP_SAT`** saturation mode to prevent overflow during long dot-product reductions — critical since FC1 alone sums 256 products per output neuron.

---

## ⚙️ Hardware Design Strategy

| Block | Technique | Purpose |
|-------|-----------|---------|
| 🧱 Block 1 | Sliding-window line buffer | Streams pixels with `II=1`, avoids re-reading image memory |
| 🧱 Block 2 | Full `UNROLL` on kernel dims + `ARRAY_PARTITION` | Maximizes DSP parallelism for the heaviest conv layer |
| 🧱 Block 3 | Grouped partial-sum accumulation (`cyclic` partition) | Balances FC layer's massive 256/120/84-wide dot products across parallel adders |

Each block is deployed as an **independent bitstream/IP**, validated via:
- ✅ AXI-Lite register inspection (start/done handshake)
- ✅ Sentinel-value pre-filling (distinguishing "nothing computed" from "zeros computed")
- ✅ MD5-verified bitstream integrity before deployment

---

## 📊 Speedup Results (MNIST test[0], label = 7)

| Metric | CPU (float32) | FPGA (INT8, HLS) |
|--------|:---:|:---:|
| Block 1 (Conv+Pool) | — | 1.220 ms |
| Block 2 (Conv+Pool) | — | 1.316 ms |
| Block 3 (FC+Argmax) | — | 1.374 ms |
| **Total compute time** | **287.64 ms** | **3.910 ms** |
| **Speedup** | | **🚀 ~74×** |

> ⚠️ Note: latencies are measured **per kernel** via AXI-Lite timing, then summed — this reflects **compute-time speedup**, not necessarily a single continuous end-to-end hardware run.

**Accuracy under quantization:**
- Block 1 output MSE vs. quantization-matched float reference: `0.002`
- Bit-exact match rate (±1 LSB): `92.7%`
- **CPU/FPGA final classification agreement: 100%** on validated samples — INT8 quantization noise never flipped a prediction.

---

## 🐞 Hardware Debugging Lessons

- ⚡ **`AP_START` needs an explicit 0→1 transition** — reusing IP objects without this silently no-ops every call after the first.
- 🚧 **Missing AXI burst pragma** blocked input reads (`gmem0`) despite a clean C-simulation — csim passing ≠ hardware working.
- 🌊 **Fixed-point overflow without `AP_SAT`** caused silent wraparound instead of proper saturation.
- 🔄 **Stale IP reuse in Vivado** — "Export RTL" set to *RTL only* instead of *IP Catalog* meant synthesis passed but the bitstream didn't reflect the latest source.

> **Biggest takeaway:** when C-sim passes but hardware doesn't, the bug is almost always in the **AXI protocol layer** — bit-exact software replicas + sentinel testing beat intuition every time.

---

## 📁 Repository Structure

```
cnn_acceleration/
├── block_1/     # Conv1 + Pool1
├── block_2/     # Conv2 + Pool2
├── block_3/     # FC1 + FC2 + FC3 + Argmax
├── scripts/     # Weight extraction / generation utilities
└── results/     # Benchmark logs
```

---

## 🛠️ Tech Stack

`Vitis HLS` · `Vivado` · `C/C++` · `Python` · `PYNQ-Z2 (XC7Z020)` · `ap_fixed quantization`

---

## 👤 Author

**Aziz Marnissi** — Electrical Engineering Student, ENIT
[LinkedIn](https://linkedin.com/in/aziz-marnissi) · [GitHub](https://github.com/Aziz-Marnissi)
