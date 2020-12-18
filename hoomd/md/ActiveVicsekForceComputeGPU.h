// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveVicsekForceCompute.h"
#include "EvaluatorConstraintManifold.h"

/*! \file ActiveVicsekForceComputeGPU.h
    \brief Declares a class for computing active forces on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ACTIVEVICSEKFORCECOMPUTE_GPU_H__
#define __ACTIVEVICSEKFORCECOMPUTE_GPU_H__

//! Adds an active force to a number of particles on the GPU
/*! \ingroup computes
*/
class PYBIND11_EXPORT ActiveVicsekForceComputeGPU : public ActiveVicsekForceCompute
    {
    public:
        //! Constructs the compute
        ActiveVicsekForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<NeighborList> nlist,
                             int seed, pybind11::list f_lst, pybind11::list t_lst,
                             bool orientation_link, bool orientation_reverse_link, Scalar rotation_diff);

        void addManifold(std::shared_ptr<Manifold> manifold);

    protected:
        unsigned int m_block_size;  //!< block size to execute on the GPU

        //! Set forces for particles
        virtual void setForces();

        //! Orientational diffusion for spherical particles
        virtual void rotationalDiffusion(unsigned int timestep);

        //! Set constraints if particles confined to a surface
        virtual void setConstraint();

        //! Set constraints if particles confined to a surface
        virtual void setMeanVelocity();

        GPUArray<unsigned int>  m_groupTags; //! Stores list converting group index to global tag
    };

//! Exports the ActiveVicsekForceComputeGPU Class to python
void export_ActiveVicsekForceComputeGPU(pybind11::module& m);
#endif
