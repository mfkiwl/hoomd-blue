// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveVicsekForceComputeGPU.cuh"
#include "hoomd/RandomNumbers.h"
#include "EvaluatorConstraintManifold.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;

#include <assert.h>

//! Kernel for applying rotational diffusion to active force vectors on the GPU
/*! \param group_size number of particles
    \param d_rtag convert global tag to global index
    \param d_groupTags stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_actVec particle active force unit vector
    \param d_t_actVec particle active torque unit vector
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param is2D check if simulation is 2D or 3D
    \param rotationDiff particle rotational diffusion constant
    \param seed seed for random number generator
*/
__global__ void gpu_compute_active_vicsek_force_rotational_diffusion_kernel(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   const Scalar4 *d_pos,
                                                   Scalar3 *d_f_actVec,
                                                   const Scalar3 *d_f_actVec_backup,
                                                   Scalar3 *d_t_actVec,
                                               	   const unsigned int *d_n_neigh,
                                                   const unsigned int *d_nlist,
                                                   const unsigned int *d_head_list,
                                                   const BoxDim box,
                                                   EvaluatorConstraintManifold manifold,
                                                   bool constraint,
                                                   bool is2D,
                                                   const Scalar rotationDiff,
                                                   const unsigned int timestep,
                                                   const Scalar r_dist_sq,
                                                   const Scalar coupling,
                                                   const int seed)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int tag = d_groupTags[group_idx];
    unsigned int idx = d_rtag[tag];
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::ActiveForceCompute, seed, tag, timestep);

    Scalar3 pos_i = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

    unsigned int my_head = d_head_list[idx];
    const unsigned int size = d_n_neigh[idx];

    if (is2D) // 2D
        {
        Scalar delta_theta; // rotational diffusion angle
        delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);
        Scalar theta; // angle on plane defining orientation of active force vector
        theta = atan2(d_f_actVec[tag].y, d_f_actVec[tag].x);

        Scalar mean_delta_theta = 0;
	for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = d_nlist[my_head + k];
            if(tag == j) continue;

            Scalar3 pos_j = make_scalar3(d_pos[j].x, d_pos[j].y, d_pos[j].z);
            // apply periodic boundary conditions
            Scalar3 dx = box.minImage(pos_i-pos_j);
            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            if (rsq < r_dist_sq)
           	    {
                Scalar theta_n; 
	        theta_n = atan2(d_f_actVec_backup[j].y, d_f_actVec_backup[j].x);
                //mean_delta_theta += slow::sin(theta_n-theta);
                mean_delta_theta += slow::sin(0.5*(theta_n-theta));
                }
            }
        theta += delta_theta + coupling*mean_delta_theta;
        d_f_actVec[tag].x = cos(theta);
        d_f_actVec[tag].y = sin(theta);
        // in 2D there is only one meaningful direction for torque
        }
    else // 3D: Following Stenhammar, Soft Matter, 2014
        {
        if (!constraint) // if no constraint
            {
            hoomd::SpherePointGenerator<Scalar> unit_vec;
            vec3<Scalar> rand_vec;
            unit_vec(rng, rand_vec);

            vec3<Scalar> aux_vec;
            aux_vec.x = d_f_actVec[tag].y * rand_vec.z - d_f_actVec[tag].z * rand_vec.y;
            aux_vec.y = d_f_actVec[tag].z * rand_vec.x - d_f_actVec[tag].x * rand_vec.z;
            aux_vec.z = d_f_actVec[tag].x * rand_vec.y - d_f_actVec[tag].y * rand_vec.x;
            Scalar aux_vec_mag = sqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
            aux_vec.x /= aux_vec_mag;
            aux_vec.y /= aux_vec_mag;
            aux_vec.z /= aux_vec_mag;

            vec3<Scalar> current_vec;
            current_vec.x = d_f_actVec[tag].x;
            current_vec.y = d_f_actVec[tag].y;
            current_vec.z = d_f_actVec[tag].z;

            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);
            d_f_actVec[tag].x = cos(delta_theta)*current_vec.x + sin(delta_theta)*aux_vec.x;
            d_f_actVec[tag].y = cos(delta_theta)*current_vec.y + sin(delta_theta)*aux_vec.y;
            d_f_actVec[tag].z = cos(delta_theta)*current_vec.z + sin(delta_theta)*aux_vec.z;

            // torque vector rotates rigidly along with force vector
            d_t_actVec[tag].x = cos(delta_theta)*current_vec.x + sin(delta_theta)*aux_vec.x;
            d_t_actVec[tag].y = cos(delta_theta)*current_vec.y + sin(delta_theta)*aux_vec.y;
            d_t_actVec[tag].z = cos(delta_theta)*current_vec.z + sin(delta_theta)*aux_vec.z;

            }
        else // if constraint
            {
            Scalar3 current_pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

   	    Scalar3 norm_scalar3 = manifold.evalNormal(current_pos);; // the normal vector to which the particles are confined.
	    Scalar nNorm =  slow::sqrt(norm_scalar3.x*norm_scalar3.x + norm_scalar3.y*norm_scalar3.y + norm_scalar3.z*norm_scalar3.z);
	    norm_scalar3.x /= nNorm;
	    norm_scalar3.y /= nNorm;
	    norm_scalar3.z /= nNorm;
            vec3<Scalar> norm;
            norm = vec3<Scalar> (norm_scalar3);

            vec3<Scalar> current_vec;
            current_vec.x = d_f_actVec[tag].x;
            current_vec.y = d_f_actVec[tag].y;
            current_vec.z = d_f_actVec[tag].z;
            vec3<Scalar> aux_vec = cross(current_vec, norm); // aux vec for defining direction that active force vector rotates towards.
            
            nNorm =  slow::sqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
	    aux_vec.x /= nNorm;
	    aux_vec.y /= nNorm;
	    aux_vec.z /= nNorm;

            Scalar mean_delta_theta = 0;
	    for (unsigned int k = 0; k < size; k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = d_nlist[my_head + k];

                if(tag == j) continue;

                Scalar3 pos_j = make_scalar3(d_pos[j].x, d_pos[j].y, d_pos[j].z);
                // apply periodic boundary conditions
                Scalar3 dx = box.minImage(pos_i-pos_j);
                // calculate r_ij squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);

                if (rsq < r_dist_sq)
	            {
            	    vec3<Scalar> current_f_n;
            	    current_f_n.x = d_f_actVec_backup[j].x;
            	    current_f_n.y = d_f_actVec_backup[j].y;
            	    current_f_n.z = d_f_actVec_backup[j].z;
            	    vec3<Scalar> aux_n = cross(current_f_n, norm);

            	    nNorm =  slow::sqrt(aux_n.x*aux_n.x + aux_n.y*aux_n.y + aux_n.z*aux_n.z);
	    	    aux_n.x /= nNorm;
	    	    aux_n.y /= nNorm;
	    	    aux_n.z /= nNorm;

                    Scalar theta_n = dot(aux_n,aux_vec); 
	    	    theta_n = 1-theta_n*theta_n;
	    	    //theta_n = (1-theta_n)/2;
                    if (theta_n < 0)
			theta_n = 0;
	    	    else{ 
	    	    	theta_n = slow::sqrt(theta_n);
			if(dot(aux_n,current_vec) > 0)
	    	    		theta_n = -theta_n;
		    }
                    mean_delta_theta += theta_n;
                    }

                }

            Scalar delta_theta; // rotational diffusion angle
            delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);

	    delta_theta  = delta_theta + mean_delta_theta*coupling;

            d_f_actVec[tag].x = cos(delta_theta) * current_vec.x + sin(delta_theta) * aux_vec.x;
            d_f_actVec[tag].y = cos(delta_theta) * current_vec.y + sin(delta_theta) * aux_vec.y;
            d_f_actVec[tag].z = cos(delta_theta) * current_vec.z + sin(delta_theta) * aux_vec.z;

            // torque vector rotates rigidly along with force vector
            d_t_actVec[tag].x = cos(delta_theta) * current_vec.x + sin(delta_theta) * aux_vec.x;
            d_t_actVec[tag].y = cos(delta_theta) * current_vec.y + sin(delta_theta) * aux_vec.y;
            d_t_actVec[tag].z = cos(delta_theta) * current_vec.z + sin(delta_theta) * aux_vec.z;

            }
        }
    }

cudaError_t gpu_compute_active_vicsek_force_rotational_diffusion(const unsigned int group_size,
                                                       unsigned int *d_rtag,
                                                       unsigned int *d_groupTags,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_force,
                                                       Scalar4 *d_torque,
                                                       Scalar3 *d_f_actVec,
                                                       const Scalar3 *d_f_actVec_backup,
                                                       Scalar3 *d_t_actVec,
                                               	       const unsigned int *d_n_neigh,
                                                       const unsigned int *d_nlist,
                                                       const unsigned int *d_head_list,
                                                       const BoxDim box,
                                                       const EvaluatorConstraintManifold manifold,
                                                       bool constraint,
                                                       bool is2D,
                                                       const Scalar rotationDiff,
                                                       const unsigned int timestep,
                                                       const Scalar r_dist_sq,
                                                       const Scalar coupling,
                                                       const int seed,
                                                       unsigned int block_size)
    {
    // setup the grid to run the kernel

    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_compute_active_vicsek_force_rotational_diffusion_kernel<<< grid, threads>>>(group_size,
                                                                    d_rtag,
                                                                    d_groupTags,
                                                                    d_pos,
                                                                    d_f_actVec,
                                                                    d_f_actVec_backup,
                                                                    d_t_actVec,
                                                                    d_n_neigh,
                                                                    d_nlist,
                                                                    d_head_list,
                                                                    box,
                                                                    manifold,
								    constraint,
                                                                    is2D,
                                                                    rotationDiff,
                                                                    timestep,
                                                                    r_dist_sq,
                                                                    coupling,
                                                                    seed);
    return cudaSuccess;
    }
