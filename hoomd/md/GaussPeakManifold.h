// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/Manifold.h"

/*! \file GaussPeakManifold.h
    \brief Declares the implicit function of a gauss peak.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __GAUSS_PEAK_MANIFOLD_H__
#define __GAUSS_PEAK_MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT GaussPeakManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param a multiplication factor
            \param sigma standard deviation.
        */
        GaussPeakManifold(std::shared_ptr<SystemDefinition> sysdef,
                  Scalar a, 
                  Scalar sigma);

        //! Destructor
        virtual ~GaussPeakManifold();

        //! Return the value of the implicit surface function of the gauss peak.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

	Scalar3 returnL(){return make_scalar3(0.0,0.0,0.0);};

	Scalar3 returnR(){return make_scalar3( m_a, m_invsigmasq, 0.0);};

    protected:
        Scalar m_a; //! multiplicatiion factor.
        Scalar m_invsigmasq; //! inverse of the standard deviation squared

    private:
        //! Validate that the gauss peak is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the GaussPeakManifold class to python
void export_GaussPeakManifold(pybind11::module& m);

#endif
