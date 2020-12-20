// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "EvaluatorConstraintManifold.h"

/*! \file ActiveVicsekForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveVicsekForceComputeGPU.
*/

#ifndef __ACTIVE_VICSEK_FORCE_COMPUTE_GPU_CUH__
#define __ACTIVE_VICSEK_FORCE_COMPUTE_GPU_CUH__


cudaError_t gpu_compute_active_vicsek_force_set_mean_velocity(const unsigned int group_size,
                                           		unsigned int *d_rtag,
                                           		unsigned int *d_groupTags,
                                                       Scalar3 *d_f_actVec,
                                                       const Scalar3 *d_f_actVec_backup,
                                                       const unsigned int *d_n_neigh,
                                                       const unsigned int *d_nlist,
                                                       const unsigned int *d_head_list,
                                                       const Scalar4 *d_pos,
                                                       const BoxDim& box,
                                                       const Scalar r_dist_sq,
                                                       unsigned int block_size);



#endif
