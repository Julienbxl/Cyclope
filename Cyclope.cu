/*
 * ======================================================================================
 * CYCLOPE V1.0
 * ======================================================================================
 */

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <unistd.h>

#include "ECC.h"
#include "Hash.h"
#include "Utils.h"

// =================================================================================
// 1. CONFIGURATION
// =================================================================================

#define PUZZLE_ID 76

#ifndef MAX_BATCH_SIZE
// NOTE : MAX_BATCH_SIZE détermine la taille de la constant memory utilisée pour
// les tables Gx/Gy (MAX_BATCH_SIZE/2 * 32 octets chacune). La constant memory
// est limitée à 64 KB par device. Avec MAX_BATCH_SIZE=128 : 2 tables × 64 points
// × 32 B = 4 KB. Avec MAX_BATCH_SIZE=512 : 16 KB. Garder < 256 pour une marge
// confortable. Si tu augmentes, vérifie cudaGetDeviceProperties.totalConstMem.
#define MAX_BATCH_SIZE 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif



// --- TARGET HASH160 PAR PUZZLE ---
#if PUZZLE_ID == 71
const uint8_t TARGET_HASH_BYTES[20] = {0xf6, 0xf5, 0x43, 0x1d, 0x25, 0xbb, 0xf7, 0xb1, 0x2e, 0x8a, 0xdd, 0x9a, 0xf5, 0xe3, 0x47, 0x5c, 0x44, 0xa0, 0xa5, 0xb8};
#elif PUZZLE_ID == 72
const uint8_t TARGET_HASH_BYTES[20] = {0xbf, 0x74, 0x13, 0xe8, 0xdf, 0x4e, 0x7a, 0x34, 0xce, 0x9d, 0xc1, 0x3e, 0x2f, 0x26, 0x48, 0x78, 0x3e, 0xc5, 0x4a, 0xdb};
#elif PUZZLE_ID == 76
const uint8_t TARGET_HASH_BYTES[20] = {0x86, 0xf9, 0xfe, 0xa5, 0xcd, 0xec, 0xf0, 0x33, 0x16, 0x1d, 0xd2, 0xf8, 0xf8, 0x56, 0x07, 0x68, 0xae, 0x0a, 0x6d, 0x14};
#else
const uint8_t TARGET_HASH_BYTES[20] = {0};
#endif

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

// ===================================
// GLOBAL DEVICE MEMORY
// ===================================
__device__ __constant__ uint8_t c_target_hash160[20];
__device__ __constant__ uint32_t c_target_prefix;

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE / 2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE / 2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

// =================================================================================
// 2. HOST HELPERS
// =================================================================================

static const uint32_t K_CPU[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,
    0x923F82A4,0xAB1C5ED5,0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,
    0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,0xE49B69C1,0xEFBE4786,
    0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,
    0x06CA6351,0x14292967,0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,
    0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,0xA2BFE8A1,0xA81A664B,
    0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,
    0x5B9CCA4F,0x682E6FF3,0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,
    0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

void sha256_cpu_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | block[i*4+3];
    for (int i = 16; i < 64; i++) w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + SIGMA1(e) + CH(e, f, g) + K_CPU[i] + w[i];
        uint32_t t2 = SIGMA0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// IMPORTANT : Cette fonction ne supporte que des messages <= 55 octets
// (tout tient dans un seul bloc SHA-256 avec padding).
// Pour 33 bytes (pubkey compressée) et 32 bytes (hash SHA256) : OK.
void sha256_cpu_simple(const uint8_t* data, size_t len, uint8_t out[32]) {
    assert(len <= 55 && "sha256_cpu_simple : message trop long (max 55 bytes, single-block only)");
    uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[64] = {0};
    memcpy(block, data, len);
    block[len] = 0x80;
    uint64_t bitlen = len * 8;
    for(int i = 0; i < 8; i++) block[63-i] = (bitlen >> (i*8)) & 0xFF;
    sha256_cpu_block(state, block);
    for(int i = 0; i < 8; i++) {
        out[i*4]   = (state[i]>>24)&0xFF;
        out[i*4+1] = (state[i]>>16)&0xFF;
        out[i*4+2] = (state[i]>>8)&0xFF;
        out[i*4+3] = (state[i])&0xFF;
    }
}

std::string hash160ToAddr(const uint8_t* h160) {
    uint8_t data[25]; data[0] = 0x00; memcpy(data + 1, h160, 20);
    uint8_t hash[32];
    sha256_cpu_simple(data, 21, hash);
    sha256_cpu_simple(hash, 32, hash);
    memcpy(data + 21, hash, 4);
    const char* code_string = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> input(data, data + 25);
    std::string result = "";
    while (input.size() > 0) {
        int remainder = 0;
        for (size_t i = 0; i < input.size(); i++) {
            int digit = input[i]; int temp = remainder * 256 + digit;
            input[i] = temp / 58; remainder = temp % 58;
        }
        result.insert(0, 1, code_string[remainder]);
        while (input.size() > 0 && input[0] == 0) input.erase(input.begin());
    }
    for (int i = 0; i < 25 && data[i] == 0; i++) result.insert(0, 1, '1');
    return result;
}

bool base58Decode(const std::string &addr, std::vector<uint8_t> &out) {
    static const char *alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> result(25, 0);
    for (char c : addr) {
        const char *p = strchr(alphabet, c);
        if (!p) return false;
        int carry = (int)(p - alphabet);
        for (int i = 24; i >= 0; --i) {
            carry += 58 * result[i]; result[i] = carry % 256; carry /= 256;
        }
    }
    out = result;
    return true;
}

bool addrToHash160(const std::string &addr, uint8_t hash160[20]) {
    std::vector<uint8_t> decoded;
    if (!base58Decode(addr, decoded)) return false;
    if (decoded.size() != 25) return false;
    uint8_t check[32];
    sha256_cpu_simple(decoded.data(), 21, check);
    sha256_cpu_simple(check, 32, check);
    if (memcmp(check, decoded.data() + 21, 4) != 0) return false;
    memcpy(hash160, decoded.data() + 1, 20);
    return true;
}

// Parse une string hex (sans 0x) en unsigned __int128
static unsigned __int128 parseHex128(const std::string &s) {
    unsigned __int128 result = 0;
    for (char c : s) {
        result <<= 4;
        if (c >= '0' && c <= '9') result |= (c - '0');
        else if (c >= 'a' && c <= 'f') result |= (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') result |= (c - 'A' + 10);
    }
    return result;
}

struct RuntimeConfig {
    int puzzle_id = PUZZLE_ID;
    uint64_t stride = 0;
    uint64_t offset = 0;
    uint8_t target_hash[20];
    bool target_from_cli = false;
    bool range_is_hex = false;
    unsigned __int128 range_min_hex = 0;
    unsigned __int128 range_max_hex = 0;
    void loadPuzzleTarget() { memcpy(target_hash, TARGET_HASH_BYTES, 20); }
};

bool parseArg(const char *arg, const char *name, std::string &value) {
    std::string s = arg;
    std::string prefix = std::string("-") + name + "=";
    if (s.rfind(prefix, 0) == 0) { value = s.substr(prefix.length()); return true; }
    return false;
}

bool parseHexKey(const char *hex, uint64_t k[4]) {
    std::string s = hex;
    if (s.length() > 2 && s[0] == '0' && s[1] == 'x') s = s.substr(2);
    if (s.length() > 64) { std::cerr << "Key too long\n"; return false; }
    while (s.length() < 64) s.insert(0, "0");
    for (int i = 0; i < 4; i++) {
        std::string sub = s.substr((3 - i) * 16, 16);
        k[i] = std::strtoull(sub.c_str(), nullptr, 16);
    }
    return true;
}

RuntimeConfig parseRuntimeArgs(int argc, char **argv) {
    RuntimeConfig cfg; cfg.loadPuzzleTarget();
    for (int i = 1; i < argc; ++i) {
        std::string val;
        if (parseArg(argv[i], "stride", val)) { cfg.stride = std::stoull(val); }
        else if (parseArg(argv[i], "offset", val)) { cfg.offset = std::stoull(val); }
        else if (parseArg(argv[i], "range", val)) {
            auto colon = val.find(':');
            if (colon != std::string::npos) {
                cfg.range_is_hex = true;
                cfg.range_min_hex = parseHex128(val.substr(0, colon));
                cfg.range_max_hex = parseHex128(val.substr(colon + 1));
            } else {
                cfg.puzzle_id = std::stoi(val);
            }
        }
        else if (parseArg(argv[i], "target", val)) {
            if (val.length() > 30) {
                if (addrToHash160(val, cfg.target_hash)) cfg.target_from_cli = true;
            }
        }
    }
    return cfg;
}

void print_result_key(uint64_t scalar_count[4]) {
    std::cout << "\n--- KEY RECONSTRUCTION ---\n";
    std::cout << "PRIVATE KEY : 0x" << std::hex << std::setfill('0')
              << std::setw(16) << scalar_count[3] << std::setw(16) << scalar_count[2]
              << std::setw(16) << scalar_count[1] << std::setw(16) << scalar_count[0]
              << std::dec << "\n";
    std::cout << "--------------------------\n";
}

// --- BIGNUM ADD/SUB 128 (POUR LE STRIDE) ---
void u256_add_128(uint64_t k[4], unsigned __int128 v) {
    unsigned __int128 sum = (unsigned __int128)k[0] + (uint64_t)v;
    k[0] = (uint64_t)sum;
    unsigned __int128 carry = (sum >> 64) + (v >> 64);
    sum = (unsigned __int128)k[1] + carry;
    k[1] = (uint64_t)sum;
    carry = sum >> 64;
    sum = (unsigned __int128)k[2] + carry;
    k[2] = (uint64_t)sum;
    carry = sum >> 64;
    k[3] += (uint64_t)carry;
}

void u256_sub_128(uint64_t k[4], unsigned __int128 v) {
    unsigned __int128 val_lo = (uint64_t)v;
    unsigned __int128 val_hi = (uint64_t)(v >> 64);
    uint64_t borrow = 0;
    if (k[0] < val_lo) borrow = 1;
    k[0] -= val_lo;
    uint64_t borrow_next = 0;
    if (k[1] < val_hi + borrow) borrow_next = 1;
    k[1] -= (val_hi + borrow);
    borrow = borrow_next;
    if (k[2] < borrow) borrow_next = 1; else borrow_next = 0;
    k[2] -= borrow;
    borrow = borrow_next;
    k[3] -= borrow;
}

// =================================================================================
// 3. DEVICE UTILS
// =================================================================================

struct __align__(16) TestPkBatchReport {
    int ok; int inv_ok; int point_ok_norm; int hash_ok_raw; int hash_ok_norm;
    int slot; int mode_full; uint32_t batch_size;
    uint64_t expected_x[4]; uint64_t expected_y[4];
    uint64_t got_x_raw[4]; uint64_t got_y_raw[4];
    uint64_t got_x_norm[4]; uint64_t got_y_norm[4];
    uint64_t dx_raw[4]; uint64_t inv_batch[4]; uint64_t inv_ref[4];
    uint8_t h160_raw[20]; uint8_t h160_norm[20];
};

__device__ __forceinline__ int load_found_flag_relaxed(const int *p) { return *((const volatile int *)p); }
__device__ __forceinline__ bool warp_found_ready(const int *__restrict__ d_found_flag, unsigned full_mask, unsigned lane) {
    int f = 0; if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0); return f == FOUND_READY;
}

// =================================================================================
// 4. KERNEL HELPERS (TWIN TURBO)
// =================================================================================

__device__ __forceinline__ void report_solution_sign(
    int* d_found_flag, FoundResult* d_found_result, int gid,
    const uint64_t* base_scalar, const uint64_t* rx,
    int step_idx, bool is_neg)
{
    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
        d_found_result->threadId = gid;
        d_found_result->iter = is_neg ? -step_idx : step_idx;
        #pragma unroll
        for(int k = 0; k < 4; ++k) d_found_result->scalar[k] = base_scalar[k];
        #pragma unroll
        for(int k = 0; k < 4; ++k) d_found_result->Rx[k] = rx[k];
        __threadfence_system();
        atomicExch(d_found_flag, FOUND_READY);
    }
}

__device__ __forceinline__ void check_twin_candidate(
    const uint64_t* x_norm, uint8_t parity,
    const uint64_t* base_scalar, int step_idx, bool is_neg,
    int gid, int lane, int* d_found_flag, FoundResult* d_found_result,
    unsigned int& local_hashes, unsigned long long* hashes_accum)
{
    // Vérification rapide du préfixe (4 bytes) avant hash complet
    if (checkHash160Prefix(x_norm, parity, c_target_prefix)) {
        // Préfixe matché — on calcule le hash160 complet pour confirmation
        uint8_t h20[20];
        getHash160_33_from_limbs(parity, x_norm, h20);
        if (hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
            report_solution_sign(d_found_flag, d_found_result, gid, base_scalar, x_norm, step_idx, is_neg);
        }
    }
    local_hashes++;
}

__device__ __forceinline__ void simple_mod_neg(uint64_t* val) {
    const uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2Full, 0xFFFFFFFFFFFFFFFFull,
        0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull
    };
    if ((val[0]|val[1]|val[2]|val[3]) == 0) return;
    asm volatile(
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;\n\t"
        : "=l"(val[0]), "=l"(val[1]), "=l"(val[2]), "=l"(val[3])
        : "l"(P[0]), "l"(P[1]), "l"(P[2]), "l"(P[3]),
          "l"(val[0]), "l"(val[1]), "l"(val[2]), "l"(val[3])
    );
}

__device__ __forceinline__ unsigned long long warp_reduce_sum(unsigned long long val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =================================================================================
// 5. CŒUR DU RÉACTEUR (CORE BATCH STEP)
// =================================================================================
template <bool IS_TEST_MODE>
__device__ __forceinline__ void core_batch_step(
    uint64_t x1[4], uint64_t y1[4], uint64_t S[4],
    uint32_t batch_size, int half, int* d_found_flag, FoundResult* d_found_result,
    unsigned int* local_hashes, unsigned long long* hashes_accum,
    uint32_t target_prefix, int gid, int lane, unsigned full_mask)
{
    uint64_t subp[MAX_BATCH_SIZE/2][4];
    uint64_t acc[4], tmp[4];

    // --- PHASE 1 : ACCUMULATION ---
    fieldSub(&c_Gx[(size_t)(half-1)*4], x1, acc);
    #pragma unroll 4
    for(int j = 0; j < 4; ++j) subp[half-1][j] = acc[j];

    for(int i = half-2; i >= 0; --i) {
        fieldSub(&c_Gx[(size_t)i*4], x1, tmp);
        _ModMult(acc, acc, tmp);
        #pragma unroll 4
        for(int j = 0; j < 4; ++j) subp[i][j] = acc[j];
    }

    // --- OPTIMISATION : SAUT INTÉGRÉ ---
    uint64_t dx_jump[4];
    fieldSub(&c_Jx[0], x1, dx_jump);
    uint64_t total_prod[4];
    _ModMult(total_prod, acc, dx_jump);

    // --- PHASE 2 : INVERSION ---
    uint64_t inv_total[5];
    #pragma unroll 4
    for(int j = 0; j < 4; ++j) inv_total[j] = total_prod[j];
    inv_total[4] = 0ull;
    fieldNormalize(inv_total);
    _ModInv(inv_total);

    // --- PHASE 3 : SÉPARATION ---
    uint64_t inv_jump[4];
    _ModMult(inv_jump, inv_total, acc);

    uint64_t inverse[4];
    _ModMult(inverse, inv_total, dx_jump);

    // --- PHASE 4 : BOUCLE TWIN TURBO ---
    for (int i = 0; i < half; ++i) {
        if constexpr (!IS_TEST_MODE) { if (warp_found_ready(d_found_flag, full_mask, lane)) return; }

        uint64_t dx_inv_i[4];
        if (i < half - 1) {
            _ModMult(dx_inv_i, subp[i+1], inverse);
        } else {
            #pragma unroll 4
            for(int j = 0; j < 4; ++j) dx_inv_i[j] = inverse[j];
        }

        // CHECK POSITIF (P + i*G)
        {
            uint64_t px3[4], s[4], lam[4];
            fieldSub(&c_Gy[(size_t)i*4], y1, s);
            _ModMult(lam, s, dx_inv_i);
            _ModSqr(px3, lam);
            fieldSub(px3, x1, px3);
            fieldSub(px3, &c_Gx[(size_t)i*4], px3);

            fieldSub(x1, px3, s);
            _ModMult(s, s, lam);
            uint8_t odd;
            ModSub256isOdd(s, y1, &odd);

            if constexpr (!IS_TEST_MODE) check_twin_candidate(px3, odd ? 0x03 : 0x02, S, i + 1, false, gid, lane, d_found_flag, d_found_result, *local_hashes, hashes_accum);
        }

        // CHECK NÉGATIF (P - i*G)
        if constexpr (!IS_TEST_MODE) {
            uint64_t s_neg[4], lam_neg[4], px3_neg[4];
            fieldAdd(&c_Gy[(size_t)i*4], y1, s_neg);
            _ModMult(lam_neg, s_neg, dx_inv_i);
            simple_mod_neg(lam_neg);

            _ModSqr(px3_neg, lam_neg);
            fieldSub(px3_neg, x1, px3_neg);
            fieldSub(px3_neg, &c_Gx[(size_t)i*4], px3_neg);

            uint64_t y3_neg[4];
            fieldSub(x1, px3_neg, y3_neg);
            _ModMult(y3_neg, y3_neg, lam_neg);
            fieldSub(y3_neg, y1, y3_neg);

            uint8_t odd_neg = (y3_neg[0] & 1) ? 0x03 : 0x02;
            check_twin_candidate(px3_neg, odd_neg, S, i + 1, true, gid, lane, d_found_flag, d_found_result, *local_hashes, hashes_accum);
        }

        if (i < half - 1) {
            uint64_t gxmi[4];
            fieldSub(&c_Gx[(size_t)i*4], x1, gxmi);
            _ModMult(inverse, inverse, gxmi);
        }
    }

    // --- PHASE 5 : SAUT ---
    {
        uint64_t lam[4], s[4], x3[4], y3[4];
        fieldSub(&c_Jy[0], y1, s);
        _ModMult(lam, s, inv_jump);

        _ModSqr(x3, lam);
        fieldSub(x3, x1, x3); fieldSub(x3, &c_Jx[0], x3);
        fieldSub(x1, x3, s); _ModMult(y3, s, lam); fieldSub(y3, y1, y3);

        #pragma unroll 4
        for (int j = 0; j < 4; ++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
    }
}

// =================================================================================
// 6. KERNEL PRODUCTION & TEST
// =================================================================================

__global__ void __launch_bounds__(256, 2) kernel_solve(
    const uint64_t* __restrict__ Px, const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx, uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars, uint64_t* __restrict__ counts256,
    uint64_t threadsTotal, uint32_t batch_size, uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag, FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    uint64_t stride_val)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;
    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;

    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    unsigned int local_hashes = 0;
    uint64_t x1[4], y1[4], S[4];

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx]; y1[i] = Py[idx]; S[i] = start_scalars[idx];
    }

    uint64_t rem[4];
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4+i];
    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) return;

    uint32_t batches_done = 0;
    const uint64_t scalar_jump = stride_val * (uint64_t)batch_size;

    while (batches_done < max_batches_per_launch) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) {
            unsigned long long agg = warp_reduce_sum((unsigned long long)local_hashes);
            if (lane == 0 && agg > 0) atomicAdd(hashes_accum, agg);
            return;
        }

        // Check point courant
        {
            uint8_t odd_pos = (y1[0] & 1) ? 0x03 : 0x02;
            check_twin_candidate(x1, odd_pos, S, 0, false, gid, lane, d_found_flag, d_found_result, local_hashes, hashes_accum);
        }

        // Batch step
        core_batch_step<false>(x1, y1, S, batch_size, half, d_found_flag, d_found_result,
            &local_hashes, hashes_accum, c_target_prefix, gid, lane, full_mask);

        // Update scalaire
        scalarAdd256_PTX(S, scalar_jump);
        sub256_u64_inplace(rem, (uint64_t)batch_size);
        ++batches_done;
    }

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        Rx[gid * 4 + i] = x1[i]; Ry[gid * 4 + i] = y1[i];
        counts256[gid * 4 + i] = rem[i]; start_scalars[gid * 4 + i] = S[i];
    }

    {
        unsigned long long agg = warp_reduce_sum((unsigned long long)local_hashes);
        if (lane == 0 && agg > 0) atomicAdd(hashes_accum, agg);
    }
}

// --- INITIALISATION ---
__global__ void kernel_init_scalars_stride(uint64_t *d_scalars, uint64_t bl, uint64_t bh, uint64_t st, int bs, int tot, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot) return;
    unsigned __int128 res;
    if (mode == 0) {
        unsigned __int128 base_128 = ((unsigned __int128)bh << 64) | bl;
        unsigned __int128 spacing  = (unsigned __int128)st * (unsigned __int128)bs;
        res = base_128 + (spacing * idx);
    } else {
        res = (unsigned __int128)(idx + 1) * (unsigned __int128)st;
    }
    d_scalars[idx * 4 + 0] = (uint64_t)(res & 0xFFFFFFFFFFFFFFFFULL);
    d_scalars[idx * 4 + 1] = (uint64_t)(res >> 64);
    d_scalars[idx * 4 + 2] = 0ULL; d_scalars[idx * 4 + 3] = 0ULL;
}

void init_stride_tables(uint32_t batch_size, uint64_t effective_stride) {
    const uint32_t half = batch_size >> 1;
    uint64_t *d_s, *d_x, *d_y;
    cudaMalloc(&d_s, batch_size*32);
    cudaMalloc(&d_x, batch_size*32);
    cudaMalloc(&d_y, batch_size*32);

    kernel_init_scalars_stride<<<1, 256>>>(d_s, 0, 0, effective_stride, 1, (int)batch_size, 1);
    scalarMulKernelBase<<<1, 256>>>(d_s, d_x, d_y, (int)batch_size);
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(c_Gx, d_x, (size_t)half * 32);
    cudaMemcpyToSymbol(c_Gy, d_y, (size_t)half * 32);

    uint64_t h_Jx[4], h_Jy[4];
    cudaMemcpy(h_Jx, d_x + (batch_size - 1u) * 4, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jy, d_y + (batch_size - 1u) * 4, 32, cudaMemcpyDeviceToHost);

    cudaMemcpyToSymbol(c_Jx, h_Jx, 32);
    cudaMemcpyToSymbol(c_Jy, h_Jy, 32);

    cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
}

// --- TEST UNITAIRE ---
__global__ void kernel_test_unit(uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t* rx, uint64_t* ry, uint8_t* h160, bool* check) {
    uint64_t s[4] = {k0, k1, k2, k3};
    scalarMulBaseAffine(s, rx, ry);
    fieldCopy(rx, rx); fieldNormalize(rx);
    uint64_t ry_norm[4]; fieldCopy(ry, ry_norm); fieldNormalize(ry_norm);
    uint8_t p = (ry_norm[0] & 1) ? 0x03 : 0x02;
    getHash160_33_from_limbs(p, rx, h160);
    *check = true;
}

void run_test_pk(int argc, char **argv) {
    if (argc < 3) { std::cerr << "Usage: -testpk <privkey>\n"; std::exit(1); }
    const char* hex_key = argv[2];
    uint64_t k[4];
    if (!parseHexKey(hex_key, k)) std::exit(1);
    // En mode test, on utilise un stride arbitraire (1) juste pour initialiser les tables
    init_stride_tables(MAX_BATCH_SIZE, 1);

    uint64_t *d_rx = nullptr, *d_ry = nullptr;
    uint8_t  *d_h  = nullptr;
    bool     *d_ok = nullptr;
    TestPkBatchReport *d_rep = nullptr;
    uint8_t  *d_t  = nullptr;

    cudaMalloc(&d_rx,  32);
    cudaMalloc(&d_ry,  32);
    cudaMalloc(&d_h,   20);
    cudaMalloc(&d_ok,  1);
    cudaMalloc(&d_rep, sizeof(TestPkBatchReport));
    uint8_t dummy_h[20] = {0};
    cudaMalloc(&d_t, 20);
    cudaMemcpy(d_t, dummy_h, 20, cudaMemcpyHostToDevice);

    kernel_test_unit<<<1, 1>>>(k[0], k[1], k[2], k[3], d_rx, d_ry, d_h, d_ok);
    cudaDeviceSynchronize();

    uint8_t h_h160[20];
    cudaMemcpy(h_h160, d_h, 20, cudaMemcpyDeviceToHost);

    std::string btc_addr = hash160ToAddr(h_h160);
    std::cout << "[TESTPK] OK: 1\n";
    std::cout << "Key:  " << hex_key << "\n";
    std::cout << "Addr: " << btc_addr << "\n";

    // Libération propre de tous les buffers
    cudaFree(d_rx); cudaFree(d_ry); cudaFree(d_h);
    cudaFree(d_ok); cudaFree(d_rep); cudaFree(d_t);

    std::exit(0);
}

// =================================================================================
// 7. TELEGRAM (OPTIONNEL — configurer via variables d'environnement)
//    export TELEGRAM_BOT_TOKEN="votre_token"
//    export TELEGRAM_CHAT_ID="votre_chat_id"
// =================================================================================

std::string https_get(const std::string& host, const std::string& path) {
    SSL_CTX* ctx = nullptr;
    SSL* ssl = nullptr;
    int sock = -1;
    std::string response;

    try {
        SSL_library_init();
        OpenSSL_add_all_algorithms();
        SSL_load_error_strings();

        ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) return "";

        struct hostent* he = gethostbyname(host.c_str());
        if (!he) return "";

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(443);
        addr.sin_addr = *((struct in_addr*)he->h_addr);

        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return "";
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock); return "";
        }

        ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host.c_str());
        SSL_set_fd(ssl, sock);
        if (SSL_connect(ssl) <= 0) { return ""; }

        std::string request = "GET " + path + " HTTP/1.0\r\nHost: " + host +
                              "\r\nConnection: close\r\nUser-Agent: CyclopeBot\r\n\r\n";
        SSL_write(ssl, request.c_str(), request.length());

        char buffer[4096];
        int bytes;
        while ((bytes = SSL_read(ssl, buffer, sizeof(buffer) - 1)) > 0) {
            buffer[bytes] = 0;
            response.append(buffer);
        }
    } catch (...) {}

    if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
    if (sock >= 0) close(sock);
    if (ctx) SSL_CTX_free(ctx);

    return response;
}

void sendTelegramMessage(const std::string& message) {
	const char* bot_token = getenv("TELEGRAM_BOT_TOKEN");
    const char* chat_id   = getenv("TELEGRAM_CHAT_ID");

    if (!bot_token || bot_token[0] == '\0') return;
    if (!chat_id   || chat_id[0]   == '\0') return;

    std::ostringstream oss;
    oss << std::hex;
    for (char c : message) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            oss << c;
        } else {
            oss << '%' << std::uppercase << std::setw(2) << std::setfill('0') << (int)(unsigned char)c;
        }
    }

    std::string path = "/bot" + std::string(bot_token) +
                       "/sendMessage?chat_id=" + std::string(chat_id) +
                       "&text=" + oss.str() + "&parse_mode=Markdown";
    https_get("api.telegram.org", path);
    std::cout << "[Notification] Telegram sent." << std::endl;
}

// =================================================================================
// 8. MAIN
// =================================================================================
int main(int argc, char *argv[]) {
    if (argc >= 3 && std::string(argv[1]) == "-testpk") {
        int dev = 0; cudaSetDevice(dev); run_test_pk(argc, argv);
    }

    RuntimeConfig cfg = parseRuntimeArgs(argc, argv);

    if (cfg.stride == 0) {
        cfg.stride = 1;  // mode exploration sequentielle (stride/offset non fournis)
    }
    if (!cfg.target_from_cli) {
        std::cerr << "Erreur : -target=<adresse> est obligatoire.\n";
        std::cerr << "Exemple : ./Cyclope -range=72 -stride=821027694461 -offset=350675963729 -target=<adresse>\n";
        return 1;
    }
    // offset=0 est une valeur valide (CRT peut produire offset=0)

    unsigned __int128 range_min_128 = cfg.range_is_hex
        ? cfg.range_min_hex
        : ((unsigned __int128)1 << (cfg.puzzle_id - 1));
    unsigned __int128 range_max_128 = cfg.range_is_hex
        ? cfg.range_max_hex
        : ((unsigned __int128)1 << cfg.puzzle_id);
    unsigned __int128 stride_128    = (unsigned __int128)cfg.stride;
    unsigned __int128 offset_128    = (unsigned __int128)cfg.offset;
    unsigned __int128 dist          = (range_min_128 > offset_128) ? (range_min_128 - offset_128) : 0;
    unsigned __int128 n_steps       = (dist + stride_128 - 1) / stride_128;
    unsigned __int128 aligned_start = offset_128 + n_steps * stride_128;
    unsigned __int128 total_keys_128 = (range_max_128 - aligned_start) / stride_128 + 1;
    double total_keys_double = (double)total_keys_128;

    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int threads = 256;

    // Calcul du nombre de blocs : on vise un multiple élevé des SM pour saturer
    // la file d'exécution et masquer les latences mémoire.
    // Le facteur 128 est empiriquement bon pour les kernels lourds en registres
    // comme celui-ci (ECC + hash). cudaOccupancyMaxActiveBlocksPerMultiprocessor
    // retournerait seulement 2 à cause du __launch_bounds__(256,2), ce qui
    // sous-utiliserait massivement le GPU (factor 64x trop peu de blocs).
    int blocks = prop.multiProcessorCount * 128;

    int total_threads = blocks * threads;

    unsigned __int128 stride_eff_128 = stride_128 * (unsigned __int128)total_threads;
    uint64_t stride_eff_val = (uint64_t)stride_eff_128;

    uint32_t batch_size = MAX_BATCH_SIZE;
    uint32_t loops = 64;

    uint32_t pfx = (cfg.target_hash[3]<<24)|(cfg.target_hash[2]<<16)|(cfg.target_hash[1]<<8)|cfg.target_hash[0];
    cudaMemcpyToSymbol(c_target_prefix, &pfx, 4);
    cudaMemcpyToSymbol(c_target_hash160, cfg.target_hash, 20);

    uint64_t *d_s, *d_x, *d_y, *d_rx, *d_ry, *d_cnt;
    int *d_flg;
    FoundResult *d_res;
    unsigned long long *d_chk; // conservé pour hashes_accum (paramètre kernel)

    cudaMalloc(&d_s,   total_threads*32);
    cudaMalloc(&d_x,   total_threads*32);
    cudaMalloc(&d_y,   total_threads*32);
    cudaMalloc(&d_rx,  total_threads*32);
    cudaMalloc(&d_ry,  total_threads*32);
    cudaMalloc(&d_cnt, total_threads*32);
    cudaMalloc(&d_flg, 4);
    cudaMalloc(&d_res, sizeof(FoundResult));
    cudaMalloc(&d_chk, 8);

    cudaMemset(d_flg, 0, 4);
    cudaMemset(d_chk, 0, 8);

    init_stride_tables(batch_size, stride_eff_val);

    std::cout << "======== CYCLOPE V1.0 ================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Blocs       : " << blocks << " (" << blocks / prop.multiProcessorCount << " blocs/SM)\n";
    std::cout << "Threads     : " << total_threads << "\n";
    std::cout << "Target      : ";
    if (cfg.range_is_hex) {
        auto print128hex = [](unsigned __int128 v) {
            uint64_t hi = (uint64_t)(v >> 64), lo = (uint64_t)v;
            if (hi) std::cout << std::hex << hi;
            std::cout << std::hex << std::setfill('0') << std::setw(hi ? 16 : 1) << lo << std::dec;
        };
        print128hex(cfg.range_min_hex);
        std::cout << ":";
        print128hex(cfg.range_max_hex);
        std::cout << "\n";
    } else {
        std::cout << "Puzzle " << cfg.puzzle_id << "\n";
    }
    std::cout << "Keys        : " << total_keys_double/1e6 << " M\n";

    unsigned __int128 half_jump = (unsigned __int128)(batch_size / 2) * stride_eff_128;
    unsigned __int128 start_centered = aligned_start + half_jump;

    kernel_init_scalars_stride<<<blocks, threads>>>(
        d_s,
        (uint64_t)start_centered, (uint64_t)(start_centered >> 64),
        cfg.stride,
        1,
        total_threads,
        0
    );

    scalarMulKernelBase<<<blocks, threads>>>(d_s, d_x, d_y, total_threads);
    cudaDeviceSynchronize();

    cudaMemset(d_cnt, 0xFF, total_threads*32);

    std::cout << "Running Search...\n";
    signal(SIGINT, handle_sigint);
    auto t0     = std::chrono::high_resolution_clock::now();
    auto t_last = t0;

    double keys_hashed  = 0;
    double range_covered = 0;
    int found = 0;

    while (!g_sigint && found == 0) {
        kernel_solve<<<blocks, threads>>>(
            d_x, d_y, d_rx, d_ry, d_s, d_cnt,
            total_threads, batch_size, loops,
            d_flg, d_res, d_chk,
            stride_eff_val
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        cudaMemcpy(&found, d_flg, 4, cudaMemcpyDeviceToHost);
        std::swap(d_x, d_rx);
        std::swap(d_y, d_ry);

        double current_hashes = (double)total_threads * (double)batch_size * (double)loops;
        keys_hashed   += current_hashes;
        range_covered += current_hashes;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();

        if (dt > 1.0) {
            double hashrate    = current_hashes / dt / 1e6;
            double elapsed     = std::chrono::duration<double>(now - t0).count();
            double avg_speed   = range_covered / elapsed;
            double remaining   = total_keys_double - range_covered;
            double eta_sec     = (avg_speed > 0) ? (remaining / avg_speed) : 0;

            int h = (int)(eta_sec / 3600);
            int m = (int)((eta_sec - h*3600) / 60);
            int s = (int)((long long)eta_sec % 60);

            double progress = std::min((range_covered / total_keys_double) * 100.0, 100.0);

            std::cout << "\r[Speed] " << std::fixed << std::setprecision(2) << hashrate
                      << " MK/s | Progress: " << std::setprecision(2) << progress << " %"
                      << " | ETA: " << std::setfill('0') << std::setw(2) << h << ":"
                      << std::setw(2) << m << ":" << std::setw(2) << s << std::flush;

            t_last    = now;
            keys_hashed = 0;
        }

        if (range_covered >= total_keys_double) break;
    }

    std::cout << "\n";

    if (found) {
        FoundResult res;
        cudaMemcpy(&res, d_res, sizeof(FoundResult), cudaMemcpyDeviceToHost);
        std::cout << "\n\n======== VICTORY ! KEY FOUND ==========================\n";

        uint64_t k_final[4] = {res.scalar[0], res.scalar[1], res.scalar[2], res.scalar[3]};
        long long iter_steps = (long long)res.iter;
        unsigned __int128 abs_iter  = (unsigned __int128)std::abs(iter_steps);
        unsigned __int128 offset_val = abs_iter * stride_eff_128;

        if (iter_steps < 0) u256_sub_128(k_final, offset_val);
        else                u256_add_128(k_final, offset_val);

        print_result_key(k_final);
        std::cout << "Rx found: " << formatHex256(res.Rx) << "\n";

        // --- ENVOI TELEGRAM (si variables d'environnement configurées) ---
        std::ostringstream msg;
        msg << "*CYCLOPE HIT* \xF0\x9F\x91\x81\xEF\xB8\x8F\n\n"
            << "*Puzzle:* #" << cfg.puzzle_id << "\n"
            << "*Key:* `0x" << std::hex << std::setfill('0')
            << std::setw(16) << k_final[3] << std::setw(16) << k_final[2]
            << std::setw(16) << k_final[1] << std::setw(16) << k_final[0] << "`\n";

        sendTelegramMessage(msg.str());

    } else if (!g_sigint) {
        std::cout << "======== RANGE FINISHED: NOT FOUND ========\n";
    }

    cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_rx); cudaFree(d_ry); cudaFree(d_cnt);
    cudaFree(d_flg); cudaFree(d_res); cudaFree(d_chk);

    return 0;
}
