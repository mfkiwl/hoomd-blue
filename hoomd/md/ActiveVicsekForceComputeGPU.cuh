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


cudaError_t gpu_compute_vicsek_active_force_set_mean_velocity(const unsigned int group_size,
                                                       Scalar3 *d_f_actVec,
                                                       const Scalar3 *d_f_actVec_backup,
                                                       const unsigned int *d_n_neigh,
                                                       const unsigned int *d_nlist,
                                                       const unsigned int *d_head_list,
                                                       EvaluatorConstraintManifold manifold,
                                                       unsigned int block_size);



#endif
