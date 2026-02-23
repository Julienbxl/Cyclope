#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>

// =================================================================================
// 1. STRUCTURES & CONSTANTS
// =================================================================================

#define WARP_SIZE 32

// Flags
#define FOUND_NONE 0
#define FOUND_LOCK 1
#define FOUND_READY 2

struct FoundResult {
  int threadId;
  int iter;           // Nombre de pas (steps)
  uint64_t scalar[4]; // Clé de base (start_scalar)
  uint64_t Rx[4];     // Pour vérification visuelle
  // uint64_t Ry[4];  // Optionnel, décommenter si besoin
};

struct MatchResult {
  int found;
  uint8_t publicKey[33];
  uint8_t sha256[32];
  uint8_t ripemd160[20];
};

// Global device constants
extern __device__ __constant__ uint8_t c_target_hash160[20];
extern __device__ __constant__ uint32_t c_target_prefix;

// Error checking macro
#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    cudaError_t err = ans;                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// =================================================================================
// 2. DEVICE UTILS
// =================================================================================

__device__ __forceinline__ uint32_t bfe(uint32_t x, uint32_t bit,
                                        uint32_t num_bits) {
  uint32_t ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(num_bits));
  return ret;
}

// ---------------------------------------------------------------------
// is_inf: check if point is infinity (0,0) - optimized
// ---------------------------------------------------------------------
__device__ __forceinline__ int is_inf(const uint64_t *x, const uint64_t *y) {
  return ((x[0] | x[1] | x[2] | x[3] | y[0] | y[1] | y[2] | y[3]) == 0);
}

// ---------------------------------------------------------------------
// load/store 256-bit
// ---------------------------------------------------------------------
__device__ __forceinline__ void load256(const uint64_t *src, uint64_t *dst) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}

__device__ __forceinline__ void store256(uint64_t *dst, const uint64_t *src) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}

// ---------------------------------------------------------------------
// HASH UTILS
// ---------------------------------------------------------------------
static __device__ __forceinline__ uint32_t load_u32_le(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static __device__ __forceinline__ bool
hash160_matches_prefix_then_full(const uint8_t *__restrict__ h,
                                 const uint8_t *__restrict__ target,
                                 const uint32_t target_prefix_le) {
  if (load_u32_le(h) != target_prefix_le)
    return false;
#pragma unroll
  for (int k = 4; k < 20; ++k) {
    if (h[k] != target[k])
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------
// Comparison functions
// ---------------------------------------------------------------------
__device__ __forceinline__ bool eq4_u64(const uint64_t *a, const uint64_t *b) {
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

__device__ __forceinline__ bool eq20_u8(const uint8_t *a, const uint8_t *b) {
  for (int i = 0; i < 20; ++i)
    if (a[i] != b[i])
      return false;
  return true;
}

// ---------------------------------------------------------------------
// WARP UTILS
// ---------------------------------------------------------------------
__device__ __forceinline__ unsigned long long
warp_reduce_add_ull(unsigned long long val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

// ---------------------------------------------------------------------
// COMPARISON 256
// ---------------------------------------------------------------------
__device__ __forceinline__ bool ge256_u64(const uint64_t *a, uint64_t b_low64) {
  // Check if a >= b where b is a 64-bit value extended to 256 bits
  // a[3], a[2], a[1] must be 0 for a to be < b (if b is 64 bit)
  if (a[3] > 0 || a[2] > 0 || a[1] > 0)
    return true;
  return a[0] >= b_low64;
}

// ---------------------------------------------------------------------
// ARITHMETIC UTILS (PTX)
// ---------------------------------------------------------------------

// Incrémentation 256 bits (x = x + 1) en PTX natif
__device__ __forceinline__ void inc256_device(uint64_t *x) {
  asm volatile("add.cc.u64      %0, %0, 1;\n\t"
               "addc.cc.u64     %1, %1, 0;\n\t"
               "addc.cc.u64     %2, %2, 0;\n\t"
               "addc.u64        %3, %3, 0;\n\t"
               : "+l"(x[0]), "+l"(x[1]), "+l"(x[2]), "+l"(x[3]));
}

// Soustraction 256 bits (x = x - v) où v est un uint64_t
__device__ __forceinline__ void sub256_u64_inplace(uint64_t *x, uint64_t v) {
  asm volatile("sub.cc.u64      %0, %0, %4;\n\t"
               "subc.cc.u64     %1, %1, 0;\n\t"
               "subc.cc.u64     %2, %2, 0;\n\t"
               "subc.u64        %3, %3, 0;\n\t"
               : "+l"(x[0]), "+l"(x[1]), "+l"(x[2]), "+l"(x[3])
               : "l"(v));
}

// Addition 256 bits simple
__device__ __forceinline__ void add256(uint64_t *r, const uint64_t *a,
                                       const uint64_t *b) {
  asm volatile("add.cc.u64      %0, %4, %8;\n\t"
               "addc.cc.u64     %1, %5, %9;\n\t"
               "addc.cc.u64     %2, %6, %10;\n\t"
               "addc.u64        %3, %7, %11;\n\t"
               : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
               : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b[0]),
                 "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

// Soustraction 256 bits simple
__device__ __forceinline__ void sub256(uint64_t *r, const uint64_t *a,
                                       const uint64_t *b) {
  asm volatile("sub.cc.u64      %0, %4, %8;\n\t"
               "subc.cc.u64     %1, %5, %9;\n\t"
               "subc.cc.u64     %2, %6, %10;\n\t"
               "subc.u64        %3, %7, %11;\n\t"
               : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
               : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b[0]),
                 "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

// ---------------------------------------------------------------------
// HOST HELPERS
// ---------------------------------------------------------------------
inline std::string formatHex256(const uint64_t *val) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  // val est en little endian (u64[0] est le poids faible), on affiche du poids
  // fort au faible
  for (int i = 3; i >= 0; --i) {
    ss << std::setw(16) << val[i];
  }
  return ss.str();
}

#endif // UTILS_H
