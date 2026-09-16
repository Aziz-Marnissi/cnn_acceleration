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

### 🌀 Convolution (Block 1 & 2)

Each output feature map is computed as a multi-channel cross-correlation:

$$
\text{out}[c_o][i][j] = \text{ReLU}\left( b[c_o] + \sum_{c_i} \sum_{k_i=0}^{K-1} \sum_{k_j=0}^{K-1} \text{in}[c_i][i+k_i][j+k_j] \cdot W[c_o][c_i][k_i][k_j] \right)
$$

where $K$ is the kernel size ($K = 5$ for both Conv1 and Conv2). This is a **cross-correlation**, not true mathematical convolution — the kernel is not flipped, following standard deep learning convention.

### 🔲 Max Pooling

Non-linear downsampling, preserving the strongest activation in each 2×2 window:

$$
\text{out}[i][j] = \max\Big(\text{in}[2i][2j],\ \text{in}[2i][2j{+}1],\ \text{in}[2i{+}1][2j],\ \text{in}[2i{+}1][2j{+}1]\Big)
$$

### 🔗 Fully-Connected Layers

Classic affine transformation followed by non-linearity:

$$
\text{out}[o] = \text{ReLU}\left( b[o] + \sum_{i=0}^{N-1} \text{in}[i] \cdot W[o][i] \right)
$$

The final layer (FC3) **skips ReLU**, producing raw logits directly consumed by argmax:

$$
z[o] = b[o] + \sum_{i=0}^{N-1} \text{in}[i] \cdot W[o][i], \qquad
\hat{y} = \arg\max_{o} \; z[o]
$$

Softmax normalization is unnecessary here, since:

$$
\arg\max_{o} \; z[o] \;=\; \arg\max_{o} \; \text{softmax}(z)_o
$$

Softmax is **strictly monotonic** — it preserves the ranking of logits, so skipping it saves an exponential + division per inference with zero impact on the predicted class.

### 🔢 Fixed-Point Quantization

All activations and weights use **`ap_fixed<8,4>`** — Q4.4 format (4 integer bits, 4 fractional bits) — replacing 32-bit floats:

$$
x_{\text{fixed}} = \frac{\text{round}\left(x_{\text{float}} \cdot 2^{4}\right)}{2^{4}}
$$

with representable range:

$$
x_{\text{fixed}} \in \left[-2^{3},\ 2^{3} - 2^{-4}\right] = [-8,\ 7.9375]
$$

### ⚡ Compute Speedup

$$
\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{FPGA}}} = \frac{287.64\ \text{ms}}{3.910\ \text{ms}} \approx 73.6\times
$$

where the FPGA total is the sum of per-block AXI-Lite-measured compute times:

$$
T_{\text{FPGA}} = T_{\text{block1}} + T_{\text{block2}} + T_{\text{block3}} = 1.220 + 1.316 + 1.374 = 3.910\ \text{ms}
$$

### 🎯 Quantization Error

Mean squared error between the FPGA's fixed-point output and a quantization-matched float reference:

$$
\text{MSE} = \frac{1}{n}\sum_{k=1}^{n} \left(y_{\text{fpga}}^{(k)} - y_{\text{ref}}^{(k)}\right)^2 = 0.002
$$

Bit-exact match rate (±1 LSB tolerance, where $1\ \text{LSB} = 2^{-4} = 0.0625$):

$$
\text{Match Rate} = \frac{1}{n}\sum_{k=1}^{n} \mathbb{1}\left[\left|y_{\text{fpga}}^{(k)} - y_{\text{ref}}^{(k)}\right| \leq 1\ \text{LSB}\right] = 92.7\%
$$

Despite local quantization noise, **final classification agreement between CPU and FPGA is 100%** — the argmax decision is robust to sub-LSB perturbations in intermediate activations.

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
