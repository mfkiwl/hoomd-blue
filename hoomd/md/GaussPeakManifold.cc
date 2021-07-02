// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "GaussPeakManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file GaussPeakManifold.cc
    \brief Contains code for the GaussPeakManifold class
*/

/*!
            \param a multiplication factor
            \param sigma standard deviation.
*/
GaussPeakManifold::GaussPeakManifold(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar a,
                               Scalar sigma)
  : Manifold(sysdef), m_a(a), m_invsigmasq(1.0/(sigma*sigma)) 
       {
    m_exec_conf->msg->notice(5) << "Constructing GaussPeakManifold" << endl;
    m_surf = 7;
    validate();
       }

GaussPeakManifold::~GaussPeakManifold() 
       {
    m_exec_conf->msg->notice(5) << "Destroying GaussPeakManifold" << endl;
       }

        //! Return the value of the implicit surface function of the gauss peak.
        /*! \param point The position to evaluate the function.
        */
Scalar GaussPeakManifold::implicit_function(Scalar3 point)
       {
       return m_a*fast::exp(-0.5 * (point.x*point.x + point.y*point.y) * m_invsigmasq ) - point.z;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 GaussPeakManifold::derivative(Scalar3 point)
       {
       Scalar3 delta = make_scalar3(0.0,0.0,-1.0);
       
       Scalar factor = -m_invsigmasq*m_a*fast::exp(-0.5 * (point.x*point.x + point.y*point.y) * m_invsigmasq);
       delta.x = point.x*factor;
       delta.y = point.y*factor;
       return delta;
       }

void GaussPeakManifold::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_a > hi.z || m_a < lo.z)
        {
        m_exec_conf->msg->warning() << "constrain.gauss_peak_manifold: GaussPeak manifold is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        }
    }

//! Exports the GaussPeakManifold class to python
void export_GaussPeakManifold(pybind11::module& m)
    {
    py::class_< GaussPeakManifold, std::shared_ptr<GaussPeakManifold> >(m, "GaussPeakManifold", py::base<Manifold>())
    .def(py::init< std::shared_ptr<SystemDefinition>,Scalar, Scalar >())
    .def("implicit_function", &GaussPeakManifold::implicit_function)
    .def("derivative", &GaussPeakManifold::derivative)
    .def("returnL", &GaussPeakManifold::returnL)
    .def("returnR", &GaussPeakManifold::returnR)
    ;
    }
