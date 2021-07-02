// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/Manifold.h"

/*! \file TorusManifold.h
    \brief Declares the implicit function of a torus.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TORUS_MANIFOLD_H__
#define __TORUS_MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT TorusManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param r_r The radius of the ring.
            \param r_c The radius of the cylinder.
            \param P The location of the torus.
        */
        TorusManifold(std::shared_ptr<SystemDefinition> sysdef,
                  Scalar r_r, 
                  Scalar r_c, 
                  Scalar3 P);

        //! Destructor
        virtual ~TorusManifold();

        //! Return the value of the implicit surface function of the torus.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

	Scalar3 returnL(){return m_P;};

	Scalar3 returnR(){return make_scalar3(m_rr, m_rc, 0);};

    protected:
        Scalar m_rr; //! The radius of the ring.
        Scalar m_rc; //! The radius of the cylinder.
        Scalar3 m_P; //! The center of the torus.

    private:
        //! Validate that the torus is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the TorusManifold class to python
void export_TorusManifold(pybind11::module& m);

#endif
