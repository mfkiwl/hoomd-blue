// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file ReverseNonequilibriumShearFlow.h
 * \brief Declaration of Reverse nonequilibrium shear flow
 */

#ifndef MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_
#define MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Reverse nonequilibrium shear flow updater
/*!
 * A flow is induced by swapping velocities in x direction based on particle position in
 * y-direction.
 */
class PYBIND11_EXPORT ReverseNonequilibriumShearFlow : public Updater
    {
    public:
    //! Constructor
    ReverseNonequilibriumShearFlow(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Trigger> trigger,
                                   unsigned int num_swap,
                                   Scalar slab_width,
                                   Scalar target_momentum);

    //! Destructor
    virtual ~ReverseNonequilibriumShearFlow();

    //! Apply velocity swaps
    virtual void update(uint64_t timestep);

    //! Get max number of swaps
    Scalar getNumSwap() const
        {
        return m_num_swap;
        }

    //! Set the maximum number of swapped pairs
    void setNumSwap(unsigned int num_swap);

    //! Get slab width
    Scalar getSlabWidth() const
        {
        return m_slab_width;
        }

    //! Set the slab width
    void setSlabWidth(Scalar slab_width);

    //! Get target momentum
    Scalar getTargetMomentum() const
        {
        return m_target_momentum;
        }

    //! Set the target momentum
    void setTargetMomentum(Scalar target_momentum);

    //! Get summed exchanged momentum
    Scalar getSummedExchangedMomentum() const
        {
        return m_summed_momentum_exchange;
        }

    protected:
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data
    unsigned int m_num_swap;                          //!< maximum number of swaps
    Scalar m_slab_width;                              //!< width of slabs
    Scalar m_summed_momentum_exchange;                //!< summed momentum excange between slabs
    Scalar2 m_pos_lo;                                 //!< position of bottom slab in box
    Scalar2 m_pos_hi;                                 //!< position of top slab in box
    unsigned int m_num_lo;                            //!< number of particles in bottom slab
    GPUArray<Scalar2> m_particles_lo; //!< List of all particles (indices,momentum) in bottom slab
                                      //!< sorted by momentum closest to +m_target_momentum
    unsigned int m_num_hi;            //!< number of particles in top slab
    GPUArray<Scalar2> m_particles_hi; //!< List of all particles (indices,momentum) in top slab
                                      //!< sorted by momentum closest to -m_target_momentum
    unsigned int m_num_staged;        //!< number of particles staged for swapping
    GPUArray<Scalar2> m_particles_staged; //!< List of all particles staged for swapping
    Scalar m_target_momentum;             //!< target momentum for particles in the slabs

    //! Find candidate particles for swapping in the slabs
    virtual void findSwapParticles();

    //! Stage particle momentum for swapping
    void stageSwapParticles();

    //! Swaps momentum between the slabs
    virtual void swapParticleMomentum();

    private:
    bool m_update_slabs; //!< If true, update the slab positions

    //! Request to check box on next update
    void requestUpdateSlabs()
        {
        m_update_slabs = true;
        }

    //! Sets the slab positions in the box and validates them
    void setSlabs();
    };

    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_
