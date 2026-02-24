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
- This is orchestrated by **Commander** (`commander.py`), a Python script that generates strides, plans missions, and supervises GPU execution end-to-end

---

## 🏗️ Architecture

```
commander.py
├── SOUP       → generates (stride, offset) pairs via CRT
├── PLANNER    → filters pairs by estimated execution time per puzzle
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
>
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
# Sequential scan — no stride/offset needed, scans every key from range_min
./Cyclope -range=71 -target=1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU

# Sequential scan on an explicit hex sub-range
./Cyclope -range=796F00000000000000:7975FFFFFFFFFFFFFF -target=1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU

# CRT stride attack — puzzle number mode
./Cyclope -range=71 -target=1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU -stride=484790845027 -offset=22

# CRT stride attack — explicit hex range
./Cyclope -range=796F00000000000000:7975FFFFFFFFFFFFFF -target=1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU -stride=484790845027 -offset=22
```

| Parameter | Description |
|---|---|
| `-range=N` | Puzzle number — scans `[2^(N-1), 2^N)` |
| `-range=MIN:MAX` | Explicit hex range — scans `[MIN, MAX]` (hex values, no `0x` prefix) |
| `-target=<addr>` | Bitcoin address to search for |
| `-stride=<val>` | Step between tested keys (default: `1` — sequential scan) |
| `-offset=<val>` | Starting offset, default: `0` (`0` is valid) |

> If `-stride` is omitted, Cyclope defaults to `stride=1, offset=0` — a plain sequential scan from the start of the range.  
> The hex range mode is useful to target a specific sub-zone of a puzzle instead of the full range.

### Via Commander (recommended)

```bash
# Full flow: generate strides → plan missions → run
python3 commander.py

# Step by step
python3 commander.py --soup     # generate strides.txt via CRT
python3 commander.py --plan     # generate missions.txt (filtered & sorted)
python3 commander.py --run      # execute missions one by one
python3 commander.py --run --file=custom.txt  # run a custom mission file
```

---

## 🧪 Testing & Validation

Two test utilities are included to verify Cyclope's correctness before running real missions.

### `testpk.py` — ECC & Hash160 correctness

Generates random private keys, computes the expected Bitcoin address in Python (using `fastecdsa` as reference), then calls Cyclope's `-testpk` mode and compares results.

```bash
pip install fastecdsa base58
python3 testpk.py
```

Expected output:
```
============================================================
  🚀 CYCLOPE TEST: Validation ECC & Hash (50 tests) 🚀
============================================================
  Test 1/50 ...  ✅ OK
  Test 2/50 ...  ✅ OK
  ...
  [Résultat] ✅ SUCCÈS TOTAL : 50 clés validées !
```

### `trap.py` — No key skipping

Verifies that Cyclope does not miss any key in a given range. Plants known keys (traps) in the search space, runs Cyclope, and checks that every trap was found.

```bash
python3 trap.py
```

Expected output:
```
All traps found ✅ — no key skipping detected.
```

> Run both tests after any recompilation or modification to the CUDA kernel.

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
