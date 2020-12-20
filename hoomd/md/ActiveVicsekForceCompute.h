// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include <memory>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Manifold.h"
#include "NeighborList.h"

/*! \file ActiveForceCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ACTIVEVICSEKFORCECOMPUTE_H__
#define __ACTIVEVICSEKFORCECOMPUTE_H__

//! Adds an active force to a number of particles
/*! \ingroup computes
*/
class PYBIND11_EXPORT ActiveVicsekForceCompute : public ActiveForceCompute
    {
    public:
        //! Constructs the compute
        ActiveVicsekForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<NeighborList> nlist,
			                 Scalar r_dist,
                             int seed, pybind11::list f_lst, pybind11::list t_lst,
                             bool orientation_link, bool orientation_reverse_link,
                             Scalar rotation_diff);

        //! Destructor
        ~ActiveVicsekForceCompute();

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Set constraints if particles confined to a surface
        virtual void setMeanVelocity(unsigned int timestep);

        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation

        Scalar m_r_dist_sq;

        GPUArray<Scalar3> m_f_activeVec_backup;             //!< hold backup copy of particle f_activeVec
    };

//! Exports the ActiveVicsekForceComputeClass to python
void export_ActiveVicsekForceCompute(pybind11::module& m);
#endif
