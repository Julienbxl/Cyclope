import subprocess
import re
import random
import time
import sys
import hashlib

# Check dependencies
try:
    from fastecdsa.keys import get_public_key
    from fastecdsa.curve import secp256k1
    import base58
except ImportError:
    print("Error: fastecdsa or base58 not found. Please install them: pip install fastecdsa base58")
    sys.exit(1)

# --- CONSTANTS ---
PUZZLE_ID = 71

RANGE_MIN = 1 << (PUZZLE_ID - 1)
RANGE_MAX = 1 << PUZZLE_ID

# Plages pour les paramètres aléatoires
STRIDE_MIN = 1_000_000_000   # 1 milliard
STRIDE_MAX = 5_000_000_000 # 1 milliard
OFFSET_MAX = 10_000_000    # 10 millions (toujours << RANGE_MIN, donc aligned_start tombe bien dans le range)

def get_hash160_str(priv_key):
    pub_key = get_public_key(priv_key, secp256k1)
    prefix = b'\x02' if (pub_key.y % 2 == 0) else b'\x03'
    pk_compressed = prefix + pub_key.x.to_bytes(32, 'big')
    sha = hashlib.sha256(pk_compressed).digest()
    ripemd = hashlib.new('ripemd160', sha).digest()
    return base58.b58encode_check(b'\x00' + ripemd).decode('utf-8')

def run_test(iteration, total_tests):
    # 1. Tirage aléatoire des paramètres
    stride = random.randint(STRIDE_MIN, STRIDE_MAX)
    offset = random.randint(0, OFFSET_MAX)

    print(f"\n[{iteration}/{total_tests}] Generating Trap...")
    print(f"Stride: {stride}  |  Offset: {offset}")

    # 2. Calcul de l'aligned_start (même logique que Cyclope)
    dist = 0
    if RANGE_MIN > offset:
        dist = RANGE_MIN - offset

    n_steps_start = (dist + stride - 1) // stride
    aligned_start = offset + n_steps_start * stride

    # 3. Nombre de clés dans ce range avec ce stride
    total_keys = (RANGE_MAX - aligned_start) // stride

    if total_keys <= 0:
        print("  (stride trop grand pour ce range, on re-tire)")
        return None  # On signale qu'il faut retenter

    # 4. Position aléatoire dans le range
    random_step = random.randint(0, total_keys)
    percent = (random_step / total_keys) * 100.0

    trap_key = aligned_start + (random_step * stride)
    trap_address = get_hash160_str(trap_key)

    print(f"Target:   Puzzle {PUZZLE_ID}")
    print(f"Trap Key: {hex(trap_key)}")
    print(f"Position: {percent:.2f}% of range  ({total_keys:,} keys total)")
    print(f"Address:  {trap_address}")

    # 5. Lancement de Cyclope
    cmd = [
        "./Cyclope",
        f"-range={PUZZLE_ID}",
        f"-target={trap_address}",
        f"-stride={stride}",
        f"-offset={offset}",
    ]

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    found_key = None
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            if "[Speed]" in line or "VICTORY" in line or "PRIVATE KEY" in line:
                sys.stdout.write(line)
            match = re.search(r"PRIVATE KEY\s*:\s*(?:0x)?0*([0-9a-fA-F]+)", line)
            if match:
                found_key = int(match.group(1), 16)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds.")

    if found_key:
        if found_key == trap_key:
            print("✅ SUCCESS! Key matches.")
            return True
        else:
            print(f"❌ FAILURE! Keys mismatch.\n  Expected: {hex(trap_key)}\n  Found:    {hex(found_key)}")
            return False
    else:
        print("❌ FAILURE! No key found.")
        return False

def main():
    total_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"Starting {total_tests} automated trap tests on Puzzle {PUZZLE_ID}...")
    print(f"Stride range : [{STRIDE_MIN:,} – {STRIDE_MAX:,}]")
    print(f"Offset range : [0 – {OFFSET_MAX:,}]")

    success_count = 0
    i = 1
    while i <= total_tests:
        result = run_test(i, total_tests)
        if result is None:
            # Stride trop grand, on retente sans compter l'itération
            continue
        if result:
            success_count += 1
        else:
            print("Aborting tests due to failure.")
            break
        i += 1

    print(f"\n=== SUMMARY: {success_count}/{total_tests} tests passed ===")

if __name__ == "__main__":
    main()