import subprocess
import re
import os
import hashlib
import base58
import random
# Plus besoin de Keccak ni PyCryptodome car Cyclone = Bitcoin only
from fastecdsa.keys import get_public_key
from fastecdsa.curve import secp256k1

# --- CONFIGURATION ---
CYCLONE_EXECUTABLE = "./Cyclope"
NUM_TESTS_KEYS = 50

def python_ecc_hash(priv_key_int):
    """Calcule et retourne l'adresse BTC (Compressed) uniquement."""
    if priv_key_int == 0 or priv_key_int >= secp256k1.q:
        return "InvalidKey"

    # 1. Calcul ECC (Fastecdsa)
    pub_key = get_public_key(priv_key_int, secp256k1)
    
    # 2. Format Compressed (02/03 + X)
    prefix = b'\x02' if (pub_key.y % 2 == 0) else b'\x03'
    pk_compressed = prefix + pub_key.x.to_bytes(32, 'big')
    
    # 3. Hash160 (SHA256 -> RIPEMD160)
    sha = hashlib.sha256(pk_compressed).digest()
    ripemd = hashlib.new('ripemd160', sha).digest()
    
    # 4. Base58Check Encoding (Version 0x00 pour Mainnet)
    versioned_ripemd = b'\x00' + ripemd
    checksum = hashlib.sha256(hashlib.sha256(versioned_ripemd).digest()).digest()[:4]
    btc_address = base58.b58encode(versioned_ripemd + checksum).decode('utf-8')
    
    return btc_address

def parse_cyclone_output(output):
    """Extrait l'adresse BTC de la sortie de CUDACyclone."""
    # Exemple de sortie Cyclone :
    # Originale
    #   Private : 000...01
    #   Public X: ...
    #   BTC Addr: 19vkiEajf1kG9RuLLvKHQ8F7H... (Hash160: ...)
    
    results = {}
    
    # Regex pour capturer la clé privée et l'adresse BTC
    # On cherche "Private : <HEX>" puis plus loin "BTC Addr: <ADDR>"
    pk_match = re.search(r'Key\s*:\s*([0-9a-fA-F]+)', output)
    btc_match = re.search(r'Addr\s*:\s*([13][a-km-zA-HJ-NP-Z1-9]{25,34})', output)
    
    if pk_match and btc_match:
        pk = pk_match.group(1)
        btc = btc_match.group(1)
        return pk, btc
    return None, None

def run_test_campaign():
    print("="*60)
    print(f"  🚀 CYCLONE TEST: Validation ECC & Hash ({NUM_TESTS_KEYS} tests) 🚀")
    print("="*60)

    ecc_failures = 0
    
    for i in range(NUM_TESTS_KEYS):
        # Génération d'une clé privée aléatoire valide
        random_pk_int = int.from_bytes(os.urandom(32), 'big')
        if random_pk_int == 0: random_pk_int = 1
        if random_pk_int >= secp256k1.q: random_pk_int = secp256k1.q - 1
        
        random_pk_hex = f"{random_pk_int:064x}"
        
        # Calcul de référence (Python)
        expected_btc = python_ecc_hash(random_pk_int)

        print(f"  Test {i+1}/{NUM_TESTS_KEYS} ... ", end="", flush=True)
        
        try:
            # Appel de CUDACyclone en mode test
            process = subprocess.run([CYCLONE_EXECUTABLE, "-testpk", random_pk_hex], capture_output=True, text=True)
            
            if process.returncode != 0:
                print(" ❌ CRASH Cyclone (Return Code != 0)")
                # Affiche stderr pour debug
                print(f"     STDERR: {process.stderr.strip()}")
                ecc_failures += 1
                continue

            # Parsing de la sortie
            cyclone_pk, cyclone_btc = parse_cyclone_output(process.stdout)
            
            if not cyclone_btc:
                print(" ❌ Parsing Error (Adresse introuvable dans la sortie)")
                print(f"     STDOUT: {process.stdout.strip()[:100]}...")
                ecc_failures += 1
                continue

            # Comparaison
            if cyclone_btc == expected_btc:
                print(" ✅ OK")
            else:
                print(f" ❌ FAILED pour clé {random_pk_hex[:10]}...")
                print(f"    -> BTC Attendu (Python) : {expected_btc}")
                print(f"    -> BTC Obtenu (Cyclone) : {cyclone_btc}")
                ecc_failures += 1
        
        except Exception as e:
            print(f" ❌ ERREUR Script: {e}")
            ecc_failures += 1

    print("-" * 60)
    if ecc_failures == 0: 
        print(f"\n  [Résultat] ✅ SUCCÈS TOTAL : {NUM_TESTS_KEYS} clés validées !")
        print("  Le moteur ECC (Lazy) et le Hash (Zero Stack) sont corrects.")
    else: 
        print(f"\n  [Résultat] ❌ ÉCHEC : {ecc_failures}/{NUM_TESTS_KEYS} erreurs détectées.")

if __name__ == "__main__":
    if not os.path.exists(CYCLONE_EXECUTABLE):
        print(f"❌ ERREUR: Exécutable '{CYCLONE_EXECUTABLE}' non trouvé.")
        print("   Compilez le projet d'abord : make ...")
    else:
        run_test_campaign()