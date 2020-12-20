// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveVicsekForceComputeGPU.cuh"
#include "hoomd/RandomNumbers.h"
#include "EvaluatorConstraintManifold.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;

#include <assert.h>


//! Kernel for adjusting active force vectors to align parallel to an ellipsoid surface constraint on the GPU
/*! \param group_size number of particles
    \param d_rtag convert global tag to global index
    \param d_groupTags stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_actVec particle active force unit vector
    \param d_t_actVec particle active force unit vector
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
*/
__global__ void gpu_compute_active_vicsek_force_set_mean_velocity_kernel(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   Scalar3 *d_f_actVec,
                                                   const Scalar3 *d_f_actVec_backup,
                                                   const unsigned int *d_n_neigh,
                                                   const unsigned int *d_nlist,
                                                   const unsigned int *d_head_list,
                                                   const Scalar4 *d_pos,
                                                   const BoxDim& box,
                                                   const Scalar r_dist_sq)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int tag = d_groupTags[group_idx];
    unsigned int idx = d_rtag[tag];

    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of our particle.
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

    unsigned int my_head = d_head_list[idx];
    unsigned int cur_j = 0;

    unsigned int next_j(0);
    next_j = threadIdx.x%tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x%tpp) : 0;

    
    Scalar3 mean_vel = d_f_actVec_backup[idx];
    unsigned int nneigh = 1;
    for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
        {
        // read the current neighbor index
        cur_j = next_j;
        if (neigh_idx+tpp < n_neigh)
            {
            next_j = __ldg(d_nlist + my_head + neigh_idx+tpp);
            }
        // get the neighbor's position
        Scalar4 postypej = __ldg(d_pos + cur_j);
        Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

        // calculate dr (with periodic boundary conditions)
        Scalar3 dx = posi - posj;

        // apply periodic boundary conditions
        dx = box.minImage(dx);

        // calculate r squared
        Scalar rsq = dot(dx, dx);
        
        if(rsq < r_dist_sq)
            {
            mean_vel += d_f_actVec_backup[j];
            nneigh++;
        }
    mean_vel /= nneigh;

    Scalar new_norm = Scalar(1.0)/slow::sqrt(mean_vel.x*mean_vel.x + mean_vel.y*mean_vel.y + mean_vel.z*mean_vel.z);

    mean_vel *= new_norm;

    d_f_actVec[idx].x = mean_vel.x;
    d_f_actVec[idx].y = mean_vel.y;
    d_f_actVec[idx].z = mean_vel.z;
    }




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
                                                   unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_compute_active_vicsek_force_set_mean_velocity_kernel<<< grid, threads>>>(group_size,
                                                                    d_rtag,
                                                                    d_groupTags,
                                                                    d_f_actVec,
                                                                    d_f_actVec_backup,
                                                                    d_n_neigh,
                                                                    d_nlist,
                                                                    d_head_list,
                                                                    d_pos,
                                                                    box,
                                                                    r_dist_sq);
    return cudaSuccess;
    }

