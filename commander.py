#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║           CYCLOPE COMMANDER — Soupe → Planification → Attaque       ║
╚══════════════════════════════════════════════════════════════════════╝

Flow complet en un seul script :
  1. SOUPE      : Génère des (stride, offset) via CRT à partir des ingrédients
  2. PLANIFICATEUR : Filtre les combinaisons selon la durée estimée par puzzle
  3. SUPERVISOR : Lance Cyclope mission par mission, log les victoires

Usage :
  python cyclope_commander.py            → lance le flow complet
  python cyclope_commander.py --soup     → génère strides.txt seulement
  python cyclope_commander.py --plan     → planifie depuis strides.txt existant
  python cyclope_commander.py --run      → lance depuis missions.txt existant
  python cyclope_commander.py --run --file=fast.txt  → lance un fichier spécifique
"""

import sys
import os
import math
import time
import subprocess
from itertools import product, combinations
from datetime import datetime

# ==============================================================================
# ██████████████████████  SECTION 1 : CONFIGURATION  ██████████████████████████
# ==============================================================================

# --- La Soupe : tes ingrédients (facteur, [restes possibles]) ---
# Trier du plus "rare" (count faible) au plus "fréquent" en tête
# pour que les combinaisons les plus discriminantes soient générées en premier.
INGREDIENTS = [
    (367,  [58]),
    (107,  [44]),
    (37,   [27]),
    (19,   [0]),
    (17,   [7]),
    (1033, [410, 783]),
    (13,   [0, 4]),
    (11,   [9, 8]),
]

# Nombre de facteurs à combiner à chaque fois (4 à 7 recommandé)
# Plus c'est grand → stride plus grand → moins de clés à tester → plus rapide
# Mais aussi moins de combinaisons générées
BRICKS = 6

# --- Puzzles cibles ---
PUZZLES = {
    71:  "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU",
    72:  "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",
    73:  "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4",
    74:  "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",
    76:  "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",
    77:  "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE",
    78:  "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb",
    79:  "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8",
    81:  "15qsCm78whspNQFydGJQk5rexzxTQopnHZ",
    82:  "13zYrYhhJxp6Ui1VV7pqa5WDhNWM45ARAC",
    83:  "14MdEb4eFcT3MVG5sPFG4jGLuHJSnt1Dk2",
    84:  "1CMq3SvFcVEcpLMuuH8PUcNiqsK1oicG2D",
    86:  "1K3x5L6G57Y494fDqBfrojD28UJv4s5JcK",
    87:  "1PxH3K1Shdjb7gSEoTX7UPDZ6SH4qGPrvq",
    88:  "16AbnZjZZipwHMkYKBSfswGWKDmXHjEpSf",
    89:  "19QciEHbGVNY4hrhfKXmcBBCrJSBZ6TaVt",
    91:  "1EzVHtmbN4fs4MiNk3ppEnKKhsmXYJ4s74",
    92:  "1AE8NzzgKE7Yhz7BWtAcAAxiFMbPo82NB5",
    93:  "17Q7tuG2JwFFU9rXVj3uZqRtioH3mx2Jad",
    94:  "1K6xGMUbs6ZTXBnhw1pippqwK6wjBWtNpL",
    96:  "15ANYzzCp5BFHcCnVFzXqyibpzgPLWaD8b",
    97:  "18ywPwj39nGjqBrQJSzZVq2izR12MDpDr8",
    98:  "1CaBVPrwUxbQYYswu32w7Mj4HR4maNoJSX",
    99:  "1JWnE6p6UN7ZJBN7TtcbNDoRcjFtuDWoNL",
    101: "1CKCVdbDJasYmhswB6HKZHEAnNaDpK7W4n",
    102: "1PXv28YxmYMaB8zxrKeZBW8dt2HK7RkRPX",
    103: "1AcAmB6jmtU6AiEcXkmiNE9TNVPsj9DULf",
    104: "1EQJvpsmhazYCcKX5Au6AZmZKRnzarMVZu",
    106: "18KsfuHuzQaBTNLASyj15hy4LuqPUo1FNB",
    107: "15EJFC5ZTs9nhsdvSUeBXjLAuYq3SWaxTc",
    108: "1HB1iKUqeffnVsvQsbpC6dNi1XKbyNuqao",
    109: "1GvgAXVCbA8FBjXfWiAms4ytFeJcKsoyhL",
    111: "1824ZJQ7nKJ9QFTRBqn7z7dHV5EGpzUpH3",
    112: "18A7NA9FTsnJxWgkoFfPAFbQzuQxpRtCos",
    113: "1NeGn21dUDDeqFQ63xb2SpgUuXuBLA4WT4",
    114: "174SNxfqpdMGYy5YQcfLbSTK3MRNZEePoy",
    116: "1MnJ6hdhvK37VLmqcdEwqC3iFxyWH2PHUV",
    117: "1KNRfGWw7Q9Rmwsc6NT5zsdvEb9M2Wkj5Z",
    118: "1PJZPzvGX19a7twf5HyD2VvNiPdHLzm9F6",
    119: "1GuBBhf61rnvRe4K8zu8vdQB3kHzwFqSy7",
    121: "1GDSuiThEV64c166LUFC9uDcVdGjqkxKyh",
    122: "1Me3ASYt5JCTAK2XaC32RMeH34PdprrfDx",
    123: "1CdufMQL892A69KXgv6UNBD17ywWqYpKut",
    124: "1BkkGsX9ZM6iwL3zbqs7HWBV7SvosR6m8N",
    126: "1AWCLZAjKbV1P7AHvaPNCKiB7ZWVDMxFiz",
    127: "1G6EFyBRU86sThN3SSt3GrHu1sA7w7nzi4",
    128: "1MZ2L1gFrCtkkn6DnTT2e4PFUTHw9gNwaj",
    129: "1Hz3uv3nNZzBVMXLGadCucgjiCs5W9vaGz",
    131: "16zRPnT8znwq42q7XeMkZUhb1bKqgRogyy",
    132: "1KrU4dHE5WrW8rhWDsTRjR21r8t3dsrS3R",
    133: "17uDfp5r4n441xkgLFmhNoSW1KWp6xVLD",
    134: "13A3JrvXmvg5w9XGvyyR4JEJqiLz8ZySY3",
    135: "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v",
    136: "1UDHPdovvR985NrWSkdWQDEQ1xuRiTALq",
    137: "15nf31J46iLuK1ZkTnqHo7WgN5cARFK3RA",
    138: "1Ab4vzG6wEQBDNQM1B2bvUz4fqXXdFk2WT",
    139: "1Fz63c775VV9fNyj25d9Xfw3YHE6sKCxbt",
    140: "1QKBaU6WAeycb3DbKbLBkX7vJiaS8r42Xo",
    141: "1CD91Vm97mLQvXhrnoMChhJx4TP9MaQkJo",
    142: "15MnK2jXPqTMURX4xC3h4mAZxyCcaWWEDD",
    143: "13N66gCzWWHEZBxhVxG18P8wyjEWF9Yoi1",
    144: "1NevxKDYuDcCh1ZMMi6ftmWwGrZKC6j7Ux",
    145: "19GpszRNUej5yYqxXoLnbZWKew3KdVLkXg",
    146: "1M7ipcdYHey2Y5RZM34MBbpugghmjaV89P",
    147: "18aNhurEAJsw6BAgtANpexk5ob1aGTwSeL",
    148: "1FwZXt6EpRT7Fkndzv6K4b4DFoT4trbMrV",
    149: "1CXvTzR6qv8wJ7eprzUKeWxyGcHwDYP1i2",
    150: "1MUJSJYtGPVGkBCTqGspnxyHahpt5Te8jy",
    151: "13Q84TNNvgcL3HJiqQPvyBb9m4hxjS3jkV",
    152: "1LuUHyrQr8PKSvbcY1v1PiuGuqFjWpDumN",
    153: "18192XpzzdDi2K11QVHR7td2HcPS6Qs5vg",
    154: "1NgVmsCCJaKLzGyKLFJfVequnFW9ZvnMLN",
    155: "1AoeP37TmHdFh8uN72fu9AqgtLrUwcv2wJ",
    156: "1FTpAbQa4h8trvhQXjXnmNhqdiGBd1oraE",
    157: "14JHoRAdmJg3XR4RjMDh6Wed6ft6hzbQe9",
    158: "19z6waranEf8CcP8FqNgdwUe1QRxvUNKBG",
    159: "14u4nA5sugaswb6SZgn5av2vuChdMnD9E5",
    160: "1NBC8uXJy1GiJ6drkiZa1WuKn51ps7EPTv",
}

# --- Timing ---
GPU_SPEED        = 1_400_000_000   # MK/s de ton GPU
LIMIT_MAX_SEC    = 600             # On garde uniquement les missions < 10 min
CYCLOPE_BIN      = "./Cyclope"

# --- Fichiers ---
STRIDES_FILE     = "strides.txt"
MISSIONS_FILE    = "missions.txt"
LOG_FILE         = "commander_log.txt"
VICTORY_FILE     = "FOUND_KEYS.txt"


# ==============================================================================
# ██████████████████████  SECTION 2 : MATHS CRT  ██████████████████████████████
# ==============================================================================

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x, y = extended_gcd(b % a, a)
    return gcd, y - (b // a) * x, x

def modinv(a, m):
    gcd, x, _ = extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"Pas d'inverse modulaire pour {a} mod {m}")
    return x % m

def solve_crt(constraints):
    """Retourne (offset, stride) via le théorème chinois des restes."""
    # 1. On convertit l'itérateur en liste pour pouvoir le lire plusieurs fois
    constraints = list(constraints)
    
    # 2. Le reste du code fonctionne maintenant parfaitement
    M = math.prod(mod for mod, _ in constraints)
    result = 0
    for mod, rem in constraints:
        Mi  = M // mod
        yi  = modinv(Mi, mod)
        result += rem * Mi * yi
    return result % M, M

# ==============================================================================
# ██████████████████████  SECTION 3 : SOUPE  ██████████████████████████████████
# ==============================================================================

def generate_strides(output_file=STRIDES_FILE, bricks=BRICKS):
    """Génère toutes les combinaisons CRT et écrit dans output_file."""
    print(f"\n{'='*60}")
    print(f"  SOUPE — {len(INGREDIENTS)} ingrédients, {bricks} briques par combinaison")
    print(f"{'='*60}")

    n_combos    = math.comb(len(INGREDIENTS), bricks)
    avg_stride  = math.prod(sorted([i[0] for i in INGREDIENTS], reverse=True)[:bricks])
    print(f"  Combinaisons de facteurs : {n_combos:,}")
    print(f"  Stride moyen estimé      : {avg_stride:,}")
    print(f"  Fichier de sortie        : {output_file}")
    print()

    count = 0
    with open(output_file, "w") as f:
        for chosen in combinations(INGREDIENTS, bricks):
            mods     = [item[0] for item in chosen]
            rem_lists = [item[1] for item in chosen]
            for rems in product(*rem_lists):
                offset, stride = solve_crt(zip(mods, rems))
                f.write(f"-stride={stride} -offset={offset}\n")
                count += 1
                if count % 5000 == 0:
                    print(f"  {count:,} strides générés...", end="\r")

    print(f"  ✅ {count:,} strides écrits dans {output_file}        ")
    return count


# ==============================================================================
# ██████████████████████  SECTION 4 : PLANIFICATEUR  ██████████████████████████
# ==============================================================================

def plan_missions(strides_file=STRIDES_FILE, missions_file=MISSIONS_FILE):
    """Lit strides_file, croise avec les puzzles, écrit missions.txt."""
    print(f"\n{'='*60}")
    print(f"  PLANIFICATEUR")
    print(f"{'='*60}")

    if not os.path.exists(strides_file):
        print(f"  ❌ {strides_file} introuvable. Lance --soup d'abord.")
        return 0

    # Lecture des (stride, offset)
    candidates = []
    with open(strides_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                s = int(parts[0].split("=")[1])
                o = int(parts[1].split("=")[1])
                candidates.append((s, o))
            except Exception:
                continue

    print(f"  Candidats chargés : {len(candidates):,}")
    print(f"  Puzzles cibles    : {len(PUZZLES)}")
    print(f"  Limite temps      : {LIMIT_MAX_SEC}s ({LIMIT_MAX_SEC//60} min)")

    missions = []
    skipped  = 0

    for puzzle_id, addr in sorted(PUZZLES.items()):
        range_size = 2 ** (puzzle_id - 1)
        for stride, offset in candidates:
            keys_to_check = range_size / stride
            time_sec      = keys_to_check / GPU_SPEED
            if time_sec <= LIMIT_MAX_SEC:
                missions.append((time_sec, puzzle_id, addr, stride, offset))
            else:
                skipped += 1

    # Trier : puzzles petits d'abord, puis missions rapides en tête
    missions.sort(key=lambda x: (x[1], x[0]))

    with open(missions_file, "w") as f:
        for time_sec, pid, addr, stride, offset in missions:
            f.write(f"-range={pid} -target={addr} -stride={stride} -offset={offset}\n")

    total_time_h = sum(t for t, *_ in missions) / 3600
    print(f"  ✅ {len(missions):,} missions écrites dans {missions_file}")
    print(f"  ⏭  {skipped:,} combinaisons ignorées (trop longues)")
    print(f"  🕐 Durée totale estimée : {total_time_h:.1f}h")
    return len(missions)


# ==============================================================================
# ██████████████████████  SECTION 5 : SUPERVISOR  █████████████████████████████
# ==============================================================================

def log(msg):
    ts   = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_victory(output_str, cmd):
    with open(VICTORY_FILE, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"VICTORY !\nCommande : {cmd}\nDate : {datetime.now()}\n")
        f.write("-" * 30 + "\n")
        f.write(output_str + "\n")
        f.write("=" * 70 + "\n")

def run_missions(missions_file=MISSIONS_FILE):
    """Lance les missions une par une depuis missions_file."""
    print(f"\n{'='*60}")
    print(f"  SUPERVISOR")
    print(f"{'='*60}")

    if not os.path.exists(missions_file):
        print(f"  ❌ {missions_file} introuvable. Lance --plan d'abord.")
        return

    commands = []
    with open(missions_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)

    total = len(commands)
    log(f"SUPERVISOR démarré — {total} missions dans {missions_file}")

    start_global = time.time()

    for i, params in enumerate(commands):
        cmd = f"{CYCLOPE_BIN} {params}"
        step = i + 1

        elapsed = time.time() - start_global
        avg     = elapsed / i if i > 0 else 0
        eta_sec = avg * (total - i)
        eta_str = f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m"

        print(f"\n{'─'*60}")
        log(f"Mission [{step}/{total}] | ETA restant : {eta_str}")
        log(f"  {cmd}")

        try:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )

            full_output = ""
            victory     = False

            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    # Affiche uniquement les lignes utiles pour ne pas spammer
                    stripped = line.rstrip()
                    if any(k in stripped for k in ["[Speed]", "VICTORY", "PRIVATE KEY", "Puzzle", "Running"]):
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    full_output += line
                    if "VICTORY" in line:
                        victory = True

            proc.wait()

            if victory:
                log(f"🏆 VICTOIRE mission {step} ! Voir {VICTORY_FILE}")
                save_victory(full_output, cmd)
                # Décommente la ligne suivante pour s'arrêter après la première trouvaille :
                # break
            else:
                log(f"Mission {step} terminée — not found.")

        except KeyboardInterrupt:
            log("Supervisor interrompu par l'utilisateur (Ctrl+C).")
            break
        except Exception as e:
            log(f"Erreur mission {step} : {e}")

    elapsed_total = time.time() - start_global
    log(f"SUPERVISOR terminé — {elapsed_total/3600:.2f}h écoulées.")


# ==============================================================================
# ██████████████████████  SECTION 6 : MAIN  ████████████████████████████████████
# ==============================================================================

def main():
    args = sys.argv[1:]

    # Fichier de missions personnalisé ?
    missions_file = MISSIONS_FILE
    for a in args:
        if a.startswith("--file="):
            missions_file = a.split("=", 1)[1]

    mode_soup = "--soup" in args
    mode_plan = "--plan" in args
    mode_run  = "--run"  in args
    mode_all  = not (mode_soup or mode_plan or mode_run)  # défaut = tout

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              CYCLOPE COMMANDER  —  Bitcoin Puzzle Hunter             ║
╚══════════════════════════════════════════════════════════════════════╝""")
    print(f"  Ingrédients : {len(INGREDIENTS)}  |  Briques : {BRICKS}  |  GPU : {GPU_SPEED//1_000_000} MK/s")
    print(f"  Puzzles cibles : {sorted(PUZZLES.keys())}")

    if mode_soup or mode_all:
        generate_strides()

    if mode_plan or mode_all:
        n = plan_missions(missions_file=missions_file)
        if n == 0:
            print("  Aucune mission générée, abandon.")
            return

    if mode_run or mode_all:
        run_missions(missions_file=missions_file)

if __name__ == "__main__":
    main()