// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;


__device__
const float SOFTENING_FACTOR_SQ = 0.000001f;

__device__
const float TAU = 6.283185307179586f;

// 1/sqrt(pi)
__device__
// const float INV_SQRT_PI = 1.0f / sqrtf(CUDART_PI_F);
const float INV_SQRT_PI = 0.5641895835477563f;

// __device__
// const float EPS_DIV0 = 0.00000000001f;

// Vector operations for float3
__device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator/(const float3 &a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator*(const float3 &a, const float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

extern "C" __global__ void zero_f32(float* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

extern "C" __global__ void zero_f64(double* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}

// For returning both from a function.
struct ForceEnergy {
    float3 force;
    float energy;
};

// Apparently normally adding to output can cause race conditions.
__device__ __forceinline__ void atomicAddFloat3(float3* addr, const float3 v) {
    atomicAdd(&addr->x, v.x);
    atomicAdd(&addr->y, v.y);
    atomicAdd(&addr->z, v.z);
}

__device__ inline float3 min_image(float3 ext, float3 dv) {
    dv.x -= rintf(dv.x / ext.x) * ext.x;
    dv.y -= rintf(dv.y / ext.y) * ext.y;
    dv.z -= rintf(dv.z / ext.z) * ext.z;

    return dv;
}

// These params includes inv_r due to it being shared with LJ.
__device__
ForceEnergy coulomb_force_spme_short_range(
    float r,
    float inv_r,
    float3 dir,
    float q_0,
    float q_1,
    float cutoff_dist,
    float alpha
) {
    ForceEnergy result;

    // Outside cutoff: no short-range contribution
    if (r >= cutoff_dist) {
        result.force  = make_float3(0.f, 0.f, 0.f);
        result.energy = 0.f;
        return result;
    }

    const float alpha_r = alpha * r;
    const float erfc_term = erfcf(alpha_r);
    const float charge_term = q_0 * q_1;

    const float exp_term  = __expf(-(alpha_r * alpha_r));

    const float inv_r_sq = inv_r * inv_r;
    const float coef = 2.0f * alpha * exp_term * INV_SQRT_PI;
    // mul_add
    const float force_mag = charge_term * fmaf(erfc_term, inv_r_sq, coef * inv_r);

    result.force = dir * force_mag;
    result.energy = charge_term * inv_r * erfc_term;
    return result;
}

// This assumes diff (and dir) is in order tgt - src.
// Different API.
__device__
ForceEnergy lj_force_v2(
    float3 diff,
    float r,
    float inv_r,
    float3 dir,
    float sigma,
    float eps
) {
    const float sr = sigma * inv_r;
    const float sr2 = sr * sr;
    const float sr4 = sr2 * sr2;
    const float sr6 = sr4 * sr2;
    const float sr12 = sr6 * sr6;

    // Optimized mul_add.
    const float mag = 24.0f * eps * fmaf(2.f, sr12, -sr6) * inv_r;

    ForceEnergy result;
    result.force = dir * mag;
    result.energy = 4.f * eps * (sr12 - sr6);

    return result;
}

// Helpers to unflatten positions
// todo: QC these.
__device__ inline float3 ld3(const float* a, uint32_t i) {
    // a is flattened [x0 y0 z0 x1 y1 z1 ...], i = atom index
    const uint32_t j = 3u * i;
    return make_float3(a[j+0], a[j+1], a[j+2]);
}

__device__ inline float3 load_pos(
    uint8_t atom_type, uint8_t water_site, uint32_t idx,
    const float* pos_dyn,
    const float* pos_w_o, const float* pos_w_m,
    const float* pos_w_h0, const float* pos_w_h1
){
    if (atom_type == 0) return ld3(pos_dyn, idx);
    // water
    switch (water_site) {
        case 1: return ld3(pos_w_o,  idx);
        case 2: return ld3(pos_w_m,  idx);
        case 3: return ld3(pos_w_h0, idx);
        case 4: return ld3(pos_w_h1, idx);
        default: return make_float3(0.f,0.f,0.f); // shouldn't happen
    }
}

// Helpers to avoid atomic adds on energies
__inline__ __device__ double warp_sum(double x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

__inline__ __device__ double block_sum(double x) {
    static __shared__ double smem[32]; // one lane-0 per warp
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    x = warp_sum(x);
    if (lane == 0) smem[wid] = x;
    __syncthreads();

    x = (threadIdx.x < blockDim.x / 32) ? smem[lane] : 0.0;
    if (wid == 0) x = warp_sum(x);
    return x;
}
