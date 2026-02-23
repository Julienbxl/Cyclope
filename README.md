# ⚡ Cyclope — Bitcoin Puzzle Solver (CUDA + CRT Stride Attack)

Cyclope is a GPU-accelerated Bitcoin puzzle solver built on top of [CUDACyclone](https://github.com/Dookoo2/CUDACyclone) by Dookoo2.

It adds a **stride/offset attack strategy** based on the Chinese Remainder Theorem (CRT), allowing targeted sampling of the key space based on patterns observed in previously solved puzzles.

> ⚠️ This project is for **educational purposes** — learning CUDA, Python, and number theory applied to cryptography.  
> The Bitcoin Puzzle Challenge is a public learning exercise. No realistic expectation of finding a key is implied.

---

## 🔑 What's new compared to CUDACyclone

CUDACyclone does brute-force over a range. Cyclope adds a **mathematical sampling layer**:

- Instead of scanning every key, Cyclope scans keys matching `k ≡ rᵢ (mod pᵢ)` for several primes `pᵢ`
- The stride and offset are computed via **CRT** (Chinese Remainder Theorem) from a set of "ingredients" — prime factors and residues observed in previously solved puzzles
- This is orchestrated by **Cyclope Commander** (`cyclope_commander.py`), a Python script that generates strides, plans missions, and supervises GPU execution end-to-end

---

## 🏗️ Architecture

```
cyclope_commander.py
├── SOUP      → generates (stride, offset) pairs via CRT
├── PLANNER   → filters pairs by estimated execution time per puzzle
└── SUPERVISOR → launches Cyclope GPU missions one by one, detects victories
```

---

## 🚀 Requirements

- NVIDIA GPU (CUDA-capable)
- CUDA Toolkit 13.1+ (for sm_120 / RTX 5000 series)  
  CUDA 12.0+ works with sm_89 compatibility mode for older cards
- `libssl-dev` for the Telegram notification feature
- Python 3.8+ for the Commander scripts

```bash
sudo apt-get install libssl-dev
sudo apt install cuda-toolkit-13-1
```

---

## 🛠️ Build

```bash
git clone https://github.com/<your-username>/Cyclope.git
cd Cyclope
make
```

> **GPU Architecture** : the Makefile is set to `sm_120` (RTX 5000 series / Blackwell).  
> Edit `GENCODE` in the Makefile for your GPU:
> | GPU Series | sm |
> |---|---|
> | RTX 5000 (Blackwell) | sm_120 |
> | RTX 4000 (Ada Lovelace) | sm_89 |
> | RTX 3000 (Ampere) | sm_86 |
> | RTX 2000 (Turing) | sm_75 |

---

## ▶️ Usage

### Direct (single mission)

```bash
./Cyclope -range=71 -target=1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU -stride=484790845027 -offset=123456789
```

### Tests

You can run 
```bash
./python3 testpk.py
```
to check ECC and you can run
```bash
./python3 trap.py
```
to verify no key is missed

| Parameter | Description |
|---|---|
| `-range=N` | Puzzle number (key space = [2^(N-1), 2^N)) |
| `-target=<addr>` | Bitcoin address to search for |
| `-stride=<val>` | Step between tested keys |
| `-offset=<val>` | Starting offset (0 is valid) |

### Via Commander (recommended)

```bash
# Full flow: generate strides → plan missions → run
python3 commander.py

# Step by step
python3 commander.py --soup     # generate strides.txt
python3 commander.py --plan     # generate missions.txt
python3 commander.py --run      # execute missions
```

---

## 🔔 Telegram Notifications

Cyclope can send a Telegram alert when a key is found. Set your credentials via environment variables (never hardcode them):

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

---

## ⚡ Performance

| GPU | Speed |
|---|---|
| RTX 5060 (sm_120) | ~1400 MK/s |

---

## 📜 Credits

Cyclope is based on **[CUDACyclone](https://github.com/Dookoo2/CUDACyclone)** by [Dookoo2](https://github.com/Dookoo2).  
The secp256k1 math originates from **[JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch)** and **[FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul)**.  
Special thanks to Jean-Luc Pons for his foundational contributions to the cryptographic community.
