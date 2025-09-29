#include <cstdint>
#include <initializer_list>
#include <math_constants.h> // CUDART_PI_F

#include "util.cu"


// Handles LJ and Coulomb force, pairwise.
// This assumes inputs have already been organized and flattened. All inputs share the same index.
// Amber 1-2 and 1-3 exclusions are handled upstream.
extern "C" __global__
void nonbonded_force_kernel(
    // Out arrays and values
    float3* out_dyn,
    float3* out_water_o,
    float3* out_water_m,
    float3* out_water_h0,
    float3* out_water_h1,
    //
    double* out_virial,  // Virial pair sum, used for the barostat.
    double* out_energy,
    // Atom posits (not pair-wise)
    // Note: These are float*, not float3*, due to the way our helpers are set up.
    const float* pos_dyn,
    const float* pos_water_o,
    const float* pos_water_m,
    const float* pos_water_h0,
    const float* pos_water_h1,
    // Pair-wise inputs
    const uint32_t* tgt_is,
    const uint32_t* src_is,
    //
    const float* sigmas,
    const float* epss,
    const float* qs_tgt,
    const float* qs_src,
    // We use these two indices to know which output array to assign
    // forces to.
    const uint8_t* atom_types_tgt,
    const uint8_t* water_types_tgt,
    // For symmetric application
    const uint8_t* atom_types_src,
    const uint8_t* water_types_src,
    const uint8_t* scale_14s,
    const uint8_t* calc_ljs,
    const uint8_t* calc_coulombs,
    const uint8_t* symmetric,
    // Non-array inputs
    float3 cell_extent,
    float cutoff_ewald,
    float alpha_ewald,
    uint32_t N
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Per-thread energy accumulators.
    double e_acc = 0.0;
    double w_acc = 0.0;

    // todo: When you apply this to water, you must use the unit cell
    // todo to take a min image of the diff, vice using it directly.

    for (size_t i = index; i < N; i += stride) {
        const uint32_t it = tgt_is[i];
        const uint32_t is = src_is[i];

        const float3 posit_tgt = load_pos(
            atom_types_tgt[i], water_types_tgt[i], it,
            pos_dyn, pos_water_o, pos_water_m, pos_water_h0, pos_water_h1
        );
        const float3 posit_src = load_pos(
            atom_types_src[i], water_types_src[i], is,
            pos_dyn, pos_water_o, pos_water_m, pos_water_h0, pos_water_h1
        );

        const float sigma = sigmas[i];
        const float eps = epss[i];
        const uint8_t scale_14 = scale_14s[i];

        float3 diff = posit_tgt - posit_src;
        diff = min_image(cell_extent, diff);

        // We set up r and its variants like this to share between the Coulomb and LJ
        // functions.
        const float r_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        // Protect against r ~ 0 (also skip exact self if arrays alias)
        if (r_sq < 1e-16f) {
            continue;
        }

        // `rsqrtf` is a fast/approximate CUDA function. Maybe worth revisiting if it introduces
        // errors.
        const float inv_r = rsqrtf(r_sq);
        const float r = r_sq * inv_r;
        const float3 dir = diff * inv_r;

        ForceEnergy f_lj;
        f_lj.force = make_float3(0.f, 0.f, 0.f);
        f_lj.energy = 0.f;

        if (calc_ljs[i]) {
            f_lj = lj_force_v2(diff, r, inv_r, dir, sigma, eps);
        }

        const float q_tgt = qs_tgt[i];
        const float q_src = qs_src[i];

        ForceEnergy f_coulomb;
        f_coulomb.force = make_float3(0.f, 0.f, 0.f);
        f_coulomb.energy = 0.f;

        if (calc_coulombs[i]) {
            f_coulomb = coulomb_force_spme_short_range(
                r,
                inv_r,
                dir,
                q_tgt,
                q_src,
                cutoff_ewald,
                alpha_ewald
            );
        }

        if (scale_14) {
            f_lj.force = f_lj.force * 0.5f;
            f_lj.energy = f_lj.energy * 0.5f;

            f_coulomb.force = f_coulomb.force * 0.833333333f;
            f_coulomb.energy = f_coulomb.energy * 0.833333333f;
        }

        const float3 f = f_lj.force + f_coulomb.force;
        const double e_pair = (double)f_lj.energy + (double)f_coulomb.energy;

        const double virial_pair =
            (double)diff.x * (double)f.x +
            (double)diff.y * (double)f.y +
            (double)diff.z * (double)f.z;
        w_acc += virial_pair;
        if (atom_types_tgt[i] == 0) { e_acc += e_pair; }


        const uint32_t out_i = tgt_is[i];

        if (atom_types_tgt[i] == 0) {
            atomicAddFloat3(&out_dyn[out_i], f);
//             atomicAdd(out_energy, e_pair);
        } else {
            if (water_types_tgt[i] == 1) {
                atomicAddFloat3(&out_water_o[out_i], f);
            } else if (water_types_tgt[i] == 2) {
                atomicAddFloat3(&out_water_m[out_i], f);
            } else if (water_types_tgt[i] == 3) {
                atomicAddFloat3(&out_water_h0[out_i], f);
            } else {
                atomicAddFloat3(&out_water_h1[out_i], f);
            }
        }

        if (symmetric[i]) {
            const uint32_t out_i_s = src_is[i];
            const float3 f_s = f * -1.0f;

            if (atom_types_src[i] == 0) {
                atomicAddFloat3(&out_dyn[out_i_s], f_s);
            } else {
                if (water_types_src[i] == 1) {
                    atomicAddFloat3(&out_water_o[out_i_s], f_s);
                } else if (water_types_src[i] == 2) {
                    atomicAddFloat3(&out_water_m[out_i_s], f_s);
                } else if (water_types_src[i] == 3) {
                    atomicAddFloat3(&out_water_h0[out_i_s], f_s);
                } else {
                    atomicAddFloat3(&out_water_h1[out_i_s], f_s);
                }
            }
        }
    }

    double e_blk = block_sum(e_acc);
    double w_blk = block_sum(w_acc);
    if (threadIdx.x == 0) {
        atomicAdd(out_energy, e_blk);
        atomicAdd(out_virial, w_blk);
    }
}