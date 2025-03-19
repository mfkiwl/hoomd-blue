// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __TWOSTEP_NPT_MTK_GPU_CUH__
#define __TWOSTEP_NPT_MTK_GPU_CUH__

#include <hip/hip_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file TwoStepNPTMTKGPU.cuh
    \brief Declares GPU kernel code for NPT integration on the GPU using the Martyna-Tobias-Klein
   (MTK) equations. Used by TwoStepNPTMTKGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the the first step of the computation
hipError_t gpu_npt_rescale_step_one(Scalar4* d_pos,
                                    Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar thermo_rescale,
                                    Scalar* mat_exp_v,
                                    Scalar* mat_exp_r,
                                    Scalar* mat_exp_r_int,
                                    Scalar deltaT,
                                    bool rescale_all,
                                    const unsigned int block_size);

//! Kernel driver for wrapping particles back in the box (part of first step)
hipError_t gpu_npt_rescale_wrap(const unsigned int N,
                                Scalar4* d_pos,
                                int3* d_image,
                                const BoxDim& box,
                                const unsigned int block_size);

//! Kernel driver for the the second step of the computation called by NPTUpdaterGPU
hipError_t gpu_npt_rescale_step_two(Scalar4* d_vel,
                                    Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar4* d_net_force,
                                    Scalar* mat_exp_v,
                                    Scalar deltaT,
                                    Scalar thermo_rescale,
                                    const unsigned int block_size,
                                    unsigned int n_dimensions);

//! Rescale all positions
void gpu_npt_rescale_rescale(const unsigned int N,
                             Scalar4* d_postype,
                             Scalar mat_exp_r_xx,
                             Scalar mat_exp_r_xy,
                             Scalar mat_exp_r_xz,
                             Scalar mat_exp_r_yy,
                             Scalar mat_exp_r_yz,
                             Scalar mat_exp_r_zz,
                             const unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
