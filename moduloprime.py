import random
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANGE_CHECK = 1000000      # Rayon de recherche (+/- 1 Million)
MIN_FACTOR_SIZE = 10       # On ignore les petits facteurs
EXCLUDE_IDS = [20, 130]    # On exclut le 20 (Source) et le 130 (Trop grand/Hors sujet)
MIN_PUZZLES = 2            # Filtre pour l'affichage général
MAGIC_REMAINDER = 863317   # LE RESTE A TRACKER SPECIFIQUEMENT
OUTPUT_FILE = "scan_results.txt" # Fichier de sortie

RAW_DATA = """
00000000000000000000000000000000000000000000000000000000000d2c55 20
00000000000000000000000000000000000000000000000000000000001ba534 21
00000000000000000000000000000000000000000000000000000000002de40f 22
0000000000000000000000000000000000000000000000000000000000556e52 23
0000000000000000000000000000000000000000000000000000000000dc2a04 24
0000000000000000000000000000000000000000000000000000000001fa5ee5 25
000000000000000000000000000000000000000000000000000000000340326e 26
0000000000000000000000000000000000000000000000000000000006ac3875 27
000000000000000000000000000000000000000000000000000000000d916ce8 28
00000000000000000000000000000000000000000000000000000000017e2551e 29
0000000000000000000000000000000000000000000000000000000003d94cd64 30
0000000000000000000000000000000000000000000000000000000007d4fe747 31
000000000000000000000000000000000000000000000000000000000b862a62e 32
0000000000000000000000000000000000000000000000000000000001a96ca8d8 33
00000000000000000000000000000000000000000000000000000000034a65911d 34
0000000000000000000000000000000000000000000000000000000004aed21170 35
0000000000000000000000000000000000000000000000000000000009de820a7c 36
0000000000000000000000000000000000000000000000000000000001757756a93 37
00000000000000000000000000000000000000000000000000000000022382facd0 38
0000000000000000000000000000000000000000000000000000000004b5f8303e9 39
000000000000000000000000000000000000000000000000000000000e9ae4933d6 40
0000000000000000000000000000000000000000000000000000000153869acc5b 41
00000000000000000000000000000000000000000000000000000002a221c58d8f 42
00000000000000000000000000000000000000000000000000000006bd3b27c591 43
0000000000000000000000000000000000000000000000000000000e02b35a358f 44
0000000000000000000000000000000000000000000000000000000122fca143c05 45
00000000000000000000000000000000000000000000000000000002ec18388d544 46
00000000000000000000000000000000000000000000000000000006cd610b53cba 47
0000000000000000000000000000000000000000000000000000000ade6d7ce3b9b 48
0000000000000000000000000000000000000000000000000000000174176b015f4d 49
000000000000000000000000000000000000000000000000000000022bd43c2e9354 50
000000000000000000000000000000000000000000000000000000075070a1a009d4 51
0000000000000000000000000000000000000000000000000000000efae164cb9e3c 52
0000000000000000000000000000000000000000000000000000000180788e47e326c 53
0000000000000000000000000000000000000000000000000000000236fb6d5ad1f43 54
00000000000000000000000000000000000000000000000000000006abe1f9b67e114 55
00000000000000000000000000000000000000000000000000000009d18b63ac4ffdf 56
00000000000000000000000000000000000000000000000000000001eb25c90795d61c 57
00000000000000000000000000000000000000000000000000000002c675b852189a21 58
00000000000000000000000000000000000000000000000000000007496cbb87cab44f 59
0000000000000000000000000000000000000000000000000000000fc07a1825367bbe 60
00000000000000000000000000000000000000000000000000000013c96a3742f64906 61
000000000000000000000000000000000000000000000000000000363d541eb611abee 62
0000000000000000000000000000000000000000000000000000007cce5efdaccf6808 63
000000000000000000000000000000000000000000000000000000f7051f27b09112d4 64
000000000000000000000000000000000000000000000000000001a838b13505b26867 65
0000000000000000000000000000000000000000000000000000002832ed74f2b5e35ee 66
000000000000000000000000000000000000000000000000000000730fc235c1942c1ae 67
000000000000000000000000000000000000000000000000000000bebb3940cd0fc1491 68
000000000000000000000000000000000000000000000000000000101d83275fb2bc7e0c 69
000000000000000000000000000000000000000000000000000000349b84b6431a6c4ef1 70
0000000000000000000000000000000000000000000000000000004c5ce114686a1336e07 75
0000000000000000000000000000000000000000000000000000ea1a5c66dcc11b5ad180 80
00000000000000000000000000000000000000000000000000011720c4f018d51b8cebba8 85
0000000000000000000000000000000000000000000000000002ce00bb2136a445c71e85bf 90
000000000000000000000000000000000000000000000000000527a792b183c7f64a0e8b1f4 95
0000000000000000000000000000000000000000000000000af55fc59c335c8ec67ed24826 100
000000000000000000000000000000000000000000000000016f14fc2054cd87ee6396b33df3 105
000000000000000000000000000000000000000000000000035c0d7234df7deb0f20cf7062444 110
000000000000000000000000000000000000000000000000060f4d11574f5deee49961d9609ac6 115
0000000000000000000000000000000000000000000000000b10f22572c497a836ea187f2e1fc23 120
00000000000000000000000000000000000000000000000001c533b6bb7f0804e09960225e44877ac 125
000000000000000000000000000000000000000000000000033e7665705359f04f28b88cf897c603c9 130
"""

def parse():
    items = []
    for line in RAW_DATA.strip().splitlines():
        if not line: continue
        parts = line.split()
        if len(parts) >= 2:
            pid = int(parts[1])
            # EXCLUSION DU 20 ET DU 130
            if pid in EXCLUDE_IDS: continue
            items.append({"id": pid, "key": int(parts[0], 16)})
    return sorted(items, key=lambda x: x["id"])

# ==============================================================================
# MATH CORE
# ==============================================================================
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1: res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def is_prime(n, k=3):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = power(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n, max_iter=2000):
    if n % 2 == 0: return 2
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    g = 1
    i = 0
    while g == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        g = gcd(abs(x - y), n)
        i += 1
        if i > max_iter: return None
        if g == n: return None
    return g

def factorize(n):
    factors = []
    # Fast trial division
    for p in [2, 3, 5, 7, 11, 13, 17, 19]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1: return factors
    
    stack = [n]
    while stack:
        curr = stack.pop()
        if curr == 1: continue
        if is_prime(curr):
            factors.append(curr)
            continue
        f = pollard_rho(curr)
        if f:
            stack.append(f)
            stack.append(curr // f)
        else:
            factors.append(curr)
    return factors

# ==============================================================================
# WORKER FUNCTION
# ==============================================================================
def process_puzzle(args):
    """
    Retourne un dict simple: { (factor, remainder) : {puzzles...} }
    """
    item, range_check, min_factor = args
    local_results = defaultdict(set)
    base = item['key']
    puzzle_id = item['id']
    
    for delta in range(-range_check, range_check + 1):
        if delta == 0: continue
        val = base + delta
        if val < 2: continue
        
        facts = factorize(val)
        
        for f in set(facts): 
            if f >= min_factor:
                # Key = -Delta (mod F)
                # Correction pour avoir un reste positif standard
                remainder = (-delta) % f
                local_results[(f, remainder)].add(puzzle_id)
    
    # Convert defaultdict to standard dict for pickling
    return dict(local_results)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"--- PARALLEL MODULO COMPRESSOR & MAGIC HUNTER ---")
    print(f"Saving to: {OUTPUT_FILE}")
    print(f"Using {cpu_count()} CPU cores.")
    print("-" * 60)
    
    items = parse()
    TOTAL_PUZZLES = len(items)
    args_list = [(item, RANGE_CHECK, MIN_FACTOR_SIZE) for item in items]
    
    # Global Aggregation
    global_data = defaultdict(set)
    start_time = time.time()
    
    print("Starting workers...")
    
    with Pool(processes=cpu_count()) as pool:
        for i, result in enumerate(pool.imap_unordered(process_puzzle, args_list), 1):
            for (factor, rem), p_set in result.items():
                global_data[(factor, rem)].update(p_set)
            
            elapsed = time.time() - start_time
            if i % 1 == 0:
                print(f"Processed {i}/{TOTAL_PUZZLES} puzzles... ({elapsed:.1f}s)", end='\r')
    
    print("\nProcessing complete. Writing to file...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # 2. GENERAL REPORT (EXHAUSTIVE)
        f.write("\n" + "=" * 130 + "\n")
        f.write("🏆 EXHAUSTIVE ANOMALIES REPORT (Factors appearing in >= 2 puzzles)\n")
        f.write("Sorted by Frequency (Count) -> then by Factor Size\n")
        f.write(f"{'Factor':<35} | {'Remainder':<12} | {'Count':<8} | {'Coverage %':<12} | {'Puzzles'}\n")
        f.write("-" * 130 + "\n")
        
        general_stats = []
        for (factor, rem), p_set in global_data.items():
            count = len(p_set)
            # On garde tout ce qui respecte le minimum (2 par défaut)
            if count >= MIN_PUZZLES:
                pct = (count / TOTAL_PUZZLES) * 100
                general_stats.append((factor, rem, count, pct, p_set))
        
        # Tri : D'abord par nombre d'occurrences (desc), ensuite par taille du facteur (desc)
        # Cela mettra les 9/9 en haut, et les gros facteurs doublet ensuite.
        general_stats.sort(key=lambda x: (x[2], x[0]), reverse=True)
        
        # ON BOUCLE SUR TOUTE LA LISTE (Pas de [:100])
        for factor, rem, count, pct, p_set in general_stats:
            p_list = sorted(list(p_set))
            # J'ai élargi la colonne Factor à 35 caractères pour que ça soit propre
            f.write(f"{factor:<35} | {rem:<12} | {count:<8} | {pct:6.2f}%       | {str(p_list)}\n")

    print(f"\n✅ Done! Check {OUTPUT_FILE} for exhaustive results.")

if __name__ == "__main__":
    main()