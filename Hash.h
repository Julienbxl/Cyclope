#ifndef HASH_H
#define HASH_H

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// MatchResult defined in Utils.h

// Prototypes
__device__ void getSHA256_33bytes(const uint8_t *pubkey33, uint8_t sha[32]);
__device__ void getRIPEMD160_32bytes(const uint8_t *sha, uint8_t ripemd[20]);
__device__ void getHash160_33bytes(const uint8_t *pubkey33, uint8_t *hash20);

__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8],
                                            uint8_t ripemd20[20]);
__device__ void getHash160_33_from_limbs(uint8_t prefix02_03,
                                         const uint64_t x_be_limbs[4],
                                         uint8_t out20[20]);

// Helpers Inline (Zero Stack)
__device__ __forceinline__ uint32_t ror32_i(uint32_t x, int n) {
#if __CUDA_ARCH__ >= 350
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << (32 - n));
#endif
}
__device__ __forceinline__ uint32_t bigS0_i(uint32_t x) {
  return ror32_i(x, 2) ^ ror32_i(x, 13) ^ ror32_i(x, 22);
}
__device__ __forceinline__ uint32_t bigS1_i(uint32_t x) {
  return ror32_i(x, 6) ^ ror32_i(x, 11) ^ ror32_i(x, 25);
}
__device__ __forceinline__ uint32_t smallS0_i(uint32_t x) {
  return ror32_i(x, 7) ^ ror32_i(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t smallS1_i(uint32_t x) {
  return ror32_i(x, 17) ^ ror32_i(x, 19) ^ (x >> 10);
}
__device__ __forceinline__ uint32_t Ch_i(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t Maj_i(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) | (x & z) | (y & z);
}

static const __device__ __constant__ uint32_t K_SHA_INLINE[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
    0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
    0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
    0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
    0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};

#define ROL_OPT(x, n) __funnelshift_l((x), (x), (n))
#define F1_OPT(x, y, z) ((x) ^ (y) ^ (z))
#define F2_OPT(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define F3_OPT(x, y, z) (((x) | ~(y)) ^ (z))
#define F4_OPT(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define F5_OPT(x, y, z) ((x) ^ ((y) | ~(z)))
#define R_STEP_OPT(f, a, b, c, d, e, x, s, k)                                  \
  {                                                                            \
    a += f(b, c, d) + x + k;                                                   \
    a = ROL_OPT(a, s) + e;                                                     \
    c = ROL_OPT(c, 10);                                                        \
  }

// --- CHECK PREFIX OPTIMISÉ ---
__device__ __forceinline__ bool checkHash160Prefix(const uint64_t x_limbs[4],
                                                   uint64_t y_limb0,
                                                   uint32_t target_prefix) {
  uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
  uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
  uint32_t w[16];

  // Load Big Endian
  uint64_t v3 = x_limbs[3];
  uint64_t v2 = x_limbs[2];
  uint64_t v1 = x_limbs[1];
  uint64_t v0 = x_limbs[0];
  uint32_t x7 = (uint32_t)(v3 >> 32);
  uint32_t x6 = (uint32_t)v3;
  uint32_t x5 = (uint32_t)(v2 >> 32);
  uint32_t x4 = (uint32_t)v2;
  uint32_t x3 = (uint32_t)(v1 >> 32);
  uint32_t x2 = (uint32_t)v1;
  uint32_t x1 = (uint32_t)(v0 >> 32);
  uint32_t x0 = (uint32_t)v0;

  uint8_t prefix = ((uint32_t)y_limb0 & 1) ? 0x03 : 0x02;

  w[0] = ((uint32_t)prefix << 24) | (x7 >> 8);
  w[1] = (x7 << 24) | (x6 >> 8);
  w[2] = (x6 << 24) | (x5 >> 8);
  w[3] = (x5 << 24) | (x4 >> 8);
  w[4] = (x4 << 24) | (x3 >> 8);
  w[5] = (x3 << 24) | (x2 >> 8);
  w[6] = (x2 << 24) | (x1 >> 8);
  w[7] = (x1 << 24) | (x0 >> 8);
  w[8] = (x0 << 24) | 0x00800000;
  w[9] = 0;
  w[10] = 0;
  w[11] = 0;
  w[12] = 0;
  w[13] = 0;
  w[14] = 0;
  w[15] = 33 * 8;

#pragma unroll
  for (int i = 0; i < 64; i++) {
    uint32_t w_curr;
    if (i < 16)
      w_curr = w[i];
    else {
      uint32_t w_t2 = w[(i - 2) & 15];
      uint32_t w_t7 = w[(i - 7) & 15];
      uint32_t w_t15 = w[(i - 15) & 15];
      uint32_t w_t16 = w[(i - 16) & 15];
      w_curr = smallS1_i(w_t2) + w_t7 + smallS0_i(w_t15) + w_t16;
      w[i & 15] = w_curr;
    }
    uint32_t temp1 = h + bigS1_i(e) + Ch_i(e, f, g) + K_SHA_INLINE[i] + w_curr;
    uint32_t temp2 = bigS0_i(a) + Maj_i(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  uint32_t X[16];
  X[0] = __byte_perm(0x6a09e667 + a, 0, 0x0123);
  X[1] = __byte_perm(0xbb67ae85 + b, 0, 0x0123);
  X[2] = __byte_perm(0x3c6ef372 + c, 0, 0x0123);
  X[3] = __byte_perm(0xa54ff53a + d, 0, 0x0123);
  X[4] = __byte_perm(0x510e527f + e, 0, 0x0123);
  X[5] = __byte_perm(0x9b05688c + f, 0, 0x0123);
  X[6] = __byte_perm(0x1f83d9ab + g, 0, 0x0123);
  X[7] = __byte_perm(0x5be0cd19 + h, 0, 0x0123);
  X[8] = 0x00000080;
  X[9] = 0;
  X[10] = 0;
  X[11] = 0;
  X[12] = 0;
  X[13] = 0;
  X[14] = 256;
  X[15] = 0;

  uint32_t al = 0x67452301, bl = 0xEFCDAB89, cl = 0x98BADCFE, dl = 0x10325476,
           el = 0xC3D2E1F0;
  uint32_t ar = 0x67452301, br = 0xEFCDAB89, cr = 0x98BADCFE, dr = 0x10325476,
           er = 0xC3D2E1F0;

  R_STEP_OPT(F1_OPT, al, bl, cl, dl, el, X[0], 11, 0);
  R_STEP_OPT(F1_OPT, el, al, bl, cl, dl, X[1], 14, 0);
  R_STEP_OPT(F1_OPT, dl, el, al, bl, cl, X[2], 15, 0);
  R_STEP_OPT(F1_OPT, cl, dl, el, al, bl, X[3], 12, 0);
  R_STEP_OPT(F1_OPT, bl, cl, dl, el, al, X[4], 5, 0);
  R_STEP_OPT(F1_OPT, al, bl, cl, dl, el, X[5], 8, 0);
  R_STEP_OPT(F1_OPT, el, al, bl, cl, dl, X[6], 7, 0);
  R_STEP_OPT(F1_OPT, dl, el, al, bl, cl, X[7], 9, 0);
  R_STEP_OPT(F1_OPT, cl, dl, el, al, bl, X[8], 11, 0);
  R_STEP_OPT(F1_OPT, bl, cl, dl, el, al, X[9], 13, 0);
  R_STEP_OPT(F1_OPT, al, bl, cl, dl, el, X[10], 14, 0);
  R_STEP_OPT(F1_OPT, el, al, bl, cl, dl, X[11], 15, 0);
  R_STEP_OPT(F1_OPT, dl, el, al, bl, cl, X[12], 6, 0);
  R_STEP_OPT(F1_OPT, cl, dl, el, al, bl, X[13], 7, 0);
  R_STEP_OPT(F1_OPT, bl, cl, dl, el, al, X[14], 9, 0);
  R_STEP_OPT(F1_OPT, al, bl, cl, dl, el, X[15], 8, 0);

  const uint32_t K1R = 0x50A28BE6;
  R_STEP_OPT(F5_OPT, ar, br, cr, dr, er, X[5], 8, K1R);
  R_STEP_OPT(F5_OPT, er, ar, br, cr, dr, X[14], 9, K1R);
  R_STEP_OPT(F5_OPT, dr, er, ar, br, cr, X[7], 9, K1R);
  R_STEP_OPT(F5_OPT, cr, dr, er, ar, br, X[0], 11, K1R);
  R_STEP_OPT(F5_OPT, br, cr, dr, er, ar, X[9], 13, K1R);
  R_STEP_OPT(F5_OPT, ar, br, cr, dr, er, X[2], 15, K1R);
  R_STEP_OPT(F5_OPT, er, ar, br, cr, dr, X[11], 15, K1R);
  R_STEP_OPT(F5_OPT, dr, er, ar, br, cr, X[4], 5, K1R);
  R_STEP_OPT(F5_OPT, cr, dr, er, ar, br, X[13], 7, K1R);
  R_STEP_OPT(F5_OPT, br, cr, dr, er, ar, X[6], 7, K1R);
  R_STEP_OPT(F5_OPT, ar, br, cr, dr, er, X[15], 8, K1R);
  R_STEP_OPT(F5_OPT, er, ar, br, cr, dr, X[8], 11, K1R);
  R_STEP_OPT(F5_OPT, dr, er, ar, br, cr, X[1], 14, K1R);
  R_STEP_OPT(F5_OPT, cr, dr, er, ar, br, X[10], 14, K1R);
  R_STEP_OPT(F5_OPT, br, cr, dr, er, ar, X[3], 12, K1R);
  R_STEP_OPT(F5_OPT, ar, br, cr, dr, er, X[12], 6, K1R);

  const uint32_t K2L = 0x5A827999;
  R_STEP_OPT(F2_OPT, el, al, bl, cl, dl, X[7], 7, K2L);
  R_STEP_OPT(F2_OPT, dl, el, al, bl, cl, X[4], 6, K2L);
  R_STEP_OPT(F2_OPT, cl, dl, el, al, bl, X[13], 8, K2L);
  R_STEP_OPT(F2_OPT, bl, cl, dl, el, al, X[1], 13, K2L);
  R_STEP_OPT(F2_OPT, al, bl, cl, dl, el, X[10], 11, K2L);
  R_STEP_OPT(F2_OPT, el, al, bl, cl, dl, X[6], 9, K2L);
  R_STEP_OPT(F2_OPT, dl, el, al, bl, cl, X[15], 7, K2L);
  R_STEP_OPT(F2_OPT, cl, dl, el, al, bl, X[3], 15, K2L);
  R_STEP_OPT(F2_OPT, bl, cl, dl, el, al, X[12], 7, K2L);
  R_STEP_OPT(F2_OPT, al, bl, cl, dl, el, X[0], 12, K2L);
  R_STEP_OPT(F2_OPT, el, al, bl, cl, dl, X[9], 15, K2L);
  R_STEP_OPT(F2_OPT, dl, el, al, bl, cl, X[5], 9, K2L);
  R_STEP_OPT(F2_OPT, cl, dl, el, al, bl, X[2], 11, K2L);
  R_STEP_OPT(F2_OPT, bl, cl, dl, el, al, X[14], 7, K2L);
  R_STEP_OPT(F2_OPT, al, bl, cl, dl, el, X[11], 13, K2L);
  R_STEP_OPT(F2_OPT, el, al, bl, cl, dl, X[8], 12, K2L);

  const uint32_t K2R = 0x5C4DD124;
  R_STEP_OPT(F4_OPT, er, ar, br, cr, dr, X[6], 9, K2R);
  R_STEP_OPT(F4_OPT, dr, er, ar, br, cr, X[11], 13, K2R);
  R_STEP_OPT(F4_OPT, cr, dr, er, ar, br, X[3], 15, K2R);
  R_STEP_OPT(F4_OPT, br, cr, dr, er, ar, X[7], 7, K2R);
  R_STEP_OPT(F4_OPT, ar, br, cr, dr, er, X[0], 12, K2R);
  R_STEP_OPT(F4_OPT, er, ar, br, cr, dr, X[13], 8, K2R);
  R_STEP_OPT(F4_OPT, dr, er, ar, br, cr, X[5], 9, K2R);
  R_STEP_OPT(F4_OPT, cr, dr, er, ar, br, X[10], 11, K2R);
  R_STEP_OPT(F4_OPT, br, cr, dr, er, ar, X[14], 7, K2R);
  R_STEP_OPT(F4_OPT, ar, br, cr, dr, er, X[15], 7, K2R);
  R_STEP_OPT(F4_OPT, er, ar, br, cr, dr, X[8], 12, K2R);
  R_STEP_OPT(F4_OPT, dr, er, ar, br, cr, X[12], 7, K2R);
  R_STEP_OPT(F4_OPT, cr, dr, er, ar, br, X[4], 6, K2R);
  R_STEP_OPT(F4_OPT, br, cr, dr, er, ar, X[9], 15, K2R);
  R_STEP_OPT(F4_OPT, ar, br, cr, dr, er, X[1], 13, K2R);
  R_STEP_OPT(F4_OPT, er, ar, br, cr, dr, X[2], 11, K2R);

  const uint32_t K3L = 0x6ED9EBA1;
  R_STEP_OPT(F3_OPT, dl, el, al, bl, cl, X[3], 11, K3L);
  R_STEP_OPT(F3_OPT, cl, dl, el, al, bl, X[10], 13, K3L);
  R_STEP_OPT(F3_OPT, bl, cl, dl, el, al, X[14], 6, K3L);
  R_STEP_OPT(F3_OPT, al, bl, cl, dl, el, X[4], 7, K3L);
  R_STEP_OPT(F3_OPT, el, al, bl, cl, dl, X[9], 14, K3L);
  R_STEP_OPT(F3_OPT, dl, el, al, bl, cl, X[15], 9, K3L);
  R_STEP_OPT(F3_OPT, cl, dl, el, al, bl, X[8], 13, K3L);
  R_STEP_OPT(F3_OPT, bl, cl, dl, el, al, X[1], 15, K3L);
  R_STEP_OPT(F3_OPT, al, bl, cl, dl, el, X[2], 14, K3L);
  R_STEP_OPT(F3_OPT, el, al, bl, cl, dl, X[7], 8, K3L);
  R_STEP_OPT(F3_OPT, dl, el, al, bl, cl, X[0], 13, K3L);
  R_STEP_OPT(F3_OPT, cl, dl, el, al, bl, X[6], 6, K3L);
  R_STEP_OPT(F3_OPT, bl, cl, dl, el, al, X[13], 5, K3L);
  R_STEP_OPT(F3_OPT, al, bl, cl, dl, el, X[11], 12, K3L);
  R_STEP_OPT(F3_OPT, el, al, bl, cl, dl, X[5], 7, K3L);
  R_STEP_OPT(F3_OPT, dl, el, al, bl, cl, X[12], 5, K3L);

  const uint32_t K3R = 0x6D703EF3;
  R_STEP_OPT(F3_OPT, dr, er, ar, br, cr, X[15], 9, K3R);
  R_STEP_OPT(F3_OPT, cr, dr, er, ar, br, X[5], 7, K3R);
  R_STEP_OPT(F3_OPT, br, cr, dr, er, ar, X[1], 15, K3R);
  R_STEP_OPT(F3_OPT, ar, br, cr, dr, er, X[3], 11, K3R);
  R_STEP_OPT(F3_OPT, er, ar, br, cr, dr, X[7], 8, K3R);
  R_STEP_OPT(F3_OPT, dr, er, ar, br, cr, X[14], 6, K3R);
  R_STEP_OPT(F3_OPT, cr, dr, er, ar, br, X[6], 6, K3R);
  R_STEP_OPT(F3_OPT, br, cr, dr, er, ar, X[9], 14, K3R);
  R_STEP_OPT(F3_OPT, ar, br, cr, dr, er, X[11], 12, K3R);
  R_STEP_OPT(F3_OPT, er, ar, br, cr, dr, X[8], 13, K3R);
  R_STEP_OPT(F3_OPT, dr, er, ar, br, cr, X[12], 5, K3R);
  R_STEP_OPT(F3_OPT, cr, dr, er, ar, br, X[2], 14, K3R);
  R_STEP_OPT(F3_OPT, br, cr, dr, er, ar, X[10], 13, K3R);
  R_STEP_OPT(F3_OPT, ar, br, cr, dr, er, X[0], 13, K3R);
  R_STEP_OPT(F3_OPT, er, ar, br, cr, dr, X[4], 7, K3R);
  R_STEP_OPT(F3_OPT, dr, er, ar, br, cr, X[13], 5, K3R);

  const uint32_t K4L = 0x8F1BBCDC;
  R_STEP_OPT(F4_OPT, cl, dl, el, al, bl, X[1], 11, K4L);
  R_STEP_OPT(F4_OPT, bl, cl, dl, el, al, X[9], 12, K4L);
  R_STEP_OPT(F4_OPT, al, bl, cl, dl, el, X[11], 14, K4L);
  R_STEP_OPT(F4_OPT, el, al, bl, cl, dl, X[10], 15, K4L);
  R_STEP_OPT(F4_OPT, dl, el, al, bl, cl, X[0], 14, K4L);
  R_STEP_OPT(F4_OPT, cl, dl, el, al, bl, X[8], 15, K4L);
  R_STEP_OPT(F4_OPT, bl, cl, dl, el, al, X[12], 9, K4L);
  R_STEP_OPT(F4_OPT, al, bl, cl, dl, el, X[4], 8, K4L);
  R_STEP_OPT(F4_OPT, el, al, bl, cl, dl, X[13], 9, K4L);
  R_STEP_OPT(F4_OPT, dl, el, al, bl, cl, X[3], 14, K4L);
  R_STEP_OPT(F4_OPT, cl, dl, el, al, bl, X[7], 5, K4L);
  R_STEP_OPT(F4_OPT, bl, cl, dl, el, al, X[15], 6, K4L);
  R_STEP_OPT(F4_OPT, al, bl, cl, dl, el, X[14], 8, K4L);
  R_STEP_OPT(F4_OPT, el, al, bl, cl, dl, X[5], 6, K4L);
  R_STEP_OPT(F4_OPT, dl, el, al, bl, cl, X[6], 5, K4L);
  R_STEP_OPT(F4_OPT, cl, dl, el, al, bl, X[2], 12, K4L);

  const uint32_t K4R = 0x7A6D76E9;
  R_STEP_OPT(F2_OPT, cr, dr, er, ar, br, X[8], 15, K4R);
  R_STEP_OPT(F2_OPT, br, cr, dr, er, ar, X[6], 5, K4R);
  R_STEP_OPT(F2_OPT, ar, br, cr, dr, er, X[4], 8, K4R);
  R_STEP_OPT(F2_OPT, er, ar, br, cr, dr, X[1], 11, K4R);
  R_STEP_OPT(F2_OPT, dr, er, ar, br, cr, X[3], 14, K4R);
  R_STEP_OPT(F2_OPT, cr, dr, er, ar, br, X[11], 14, K4R);
  R_STEP_OPT(F2_OPT, br, cr, dr, er, ar, X[15], 6, K4R);
  R_STEP_OPT(F2_OPT, ar, br, cr, dr, er, X[0], 14, K4R);
  R_STEP_OPT(F2_OPT, er, ar, br, cr, dr, X[5], 6, K4R);
  R_STEP_OPT(F2_OPT, dr, er, ar, br, cr, X[12], 9, K4R);
  R_STEP_OPT(F2_OPT, cr, dr, er, ar, br, X[2], 12, K4R);
  R_STEP_OPT(F2_OPT, br, cr, dr, er, ar, X[13], 9, K4R);
  R_STEP_OPT(F2_OPT, ar, br, cr, dr, er, X[9], 12, K4R);
  R_STEP_OPT(F2_OPT, er, ar, br, cr, dr, X[7], 5, K4R);
  R_STEP_OPT(F2_OPT, dr, er, ar, br, cr, X[10], 15, K4R);
  R_STEP_OPT(F2_OPT, cr, dr, er, ar, br, X[14], 8, K4R);

  const uint32_t K5L = 0xA953FD4E;
  R_STEP_OPT(F5_OPT, bl, cl, dl, el, al, X[4], 9, K5L);
  R_STEP_OPT(F5_OPT, al, bl, cl, dl, el, X[0], 15, K5L);
  R_STEP_OPT(F5_OPT, el, al, bl, cl, dl, X[5], 5, K5L);
  R_STEP_OPT(F5_OPT, dl, el, al, bl, cl, X[9], 11, K5L);
  R_STEP_OPT(F5_OPT, cl, dl, el, al, bl, X[7], 6, K5L);
  R_STEP_OPT(F5_OPT, bl, cl, dl, el, al, X[12], 8, K5L);
  R_STEP_OPT(F5_OPT, al, bl, cl, dl, el, X[2], 13, K5L);
  R_STEP_OPT(F5_OPT, el, al, bl, cl, dl, X[10], 12, K5L);
  R_STEP_OPT(F5_OPT, dl, el, al, bl, cl, X[14], 5, K5L);
  R_STEP_OPT(F5_OPT, cl, dl, el, al, bl, X[1], 12, K5L);
  R_STEP_OPT(F5_OPT, bl, cl, dl, el, al, X[3], 13, K5L);
  R_STEP_OPT(F5_OPT, al, bl, cl, dl, el, X[8], 14, K5L);
  R_STEP_OPT(F5_OPT, el, al, bl, cl, dl, X[11], 11, K5L);
  R_STEP_OPT(F5_OPT, dl, el, al, bl, cl, X[6], 8, K5L);
  R_STEP_OPT(F5_OPT, cl, dl, el, al, bl, X[15], 5, K5L);
  R_STEP_OPT(F5_OPT, bl, cl, dl, el, al, X[13], 6, K5L);

  const uint32_t K5R = 0x00000000;
  R_STEP_OPT(F1_OPT, br, cr, dr, er, ar, X[12], 8, K5R);
  R_STEP_OPT(F1_OPT, ar, br, cr, dr, er, X[15], 5, K5R);
  R_STEP_OPT(F1_OPT, er, ar, br, cr, dr, X[10], 12, K5R);
  R_STEP_OPT(F1_OPT, dr, er, ar, br, cr, X[4], 9, K5R);
  R_STEP_OPT(F1_OPT, cr, dr, er, ar, br, X[1], 12, K5R);
  R_STEP_OPT(F1_OPT, br, cr, dr, er, ar, X[5], 5, K5R);
  R_STEP_OPT(F1_OPT, ar, br, cr, dr, er, X[8], 14, K5R);
  R_STEP_OPT(F1_OPT, er, ar, br, cr, dr, X[7], 6, K5R);
  R_STEP_OPT(F1_OPT, dr, er, ar, br, cr, X[6], 8, K5R);
  R_STEP_OPT(F1_OPT, cr, dr, er, ar, br, X[2], 13, K5R);
  R_STEP_OPT(F1_OPT, br, cr, dr, er, ar, X[13], 6, K5R);
  R_STEP_OPT(F1_OPT, ar, br, cr, dr, er, X[14], 5, K5R);
  R_STEP_OPT(F1_OPT, er, ar, br, cr, dr, X[0], 15, K5R);
  R_STEP_OPT(F1_OPT, dr, er, ar, br, cr, X[3], 13, K5R);
  R_STEP_OPT(F1_OPT, cr, dr, er, ar, br, X[9], 11, K5R);
  R_STEP_OPT(F1_OPT, br, cr, dr, er, ar, X[11], 11, K5R);

  // Résultat h160[0]
  uint32_t h0 = 0xEFCDAB89 + cl + dr;

  // CORRECTION : Pas de __byte_perm ! Comparaison Little Endian directe.
  return (h0 == target_prefix);
}
#endif