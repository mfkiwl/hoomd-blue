// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "TorusManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file TorusManifold.cc
    \brief Contains code for the TorusManifold class
*/

/*! 
   \param r_r The radius of the ring.
   \param r_c The radius of the cylinder.
   \param P The location of the torus.
*/
TorusManifold::TorusManifold(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar r_r,
                               Scalar r_c,
                               Scalar3 P)
  : Manifold(sysdef), m_rr(r_r), m_rc(r_c), m_P(P) 
       {
    m_exec_conf->msg->notice(5) << "Constructing TorusManifold" << endl;
    m_surf = 7;
    validate();
       }

TorusManifold::~TorusManifold() 
       {
    m_exec_conf->msg->notice(5) << "Destroying TorusManifold" << endl;
       }

        //! Return the value of the implicit surface function of the torus.
        /*! \param point The position to evaluate the function.
        */
Scalar TorusManifold::implicit_function(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       Scalar c = m_rr - fast::sqrt(delta.x*delta.x+delta.y*delta.y);
       return c*c + delta.z*delta.z - m_rc*m_rc;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 TorusManifold::derivative(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       Scalar c = m_rr - fast::sqrt(delta.x*delta.x+delta.y*delta.y);

       delta.x = -delta.x*c/(m_rr-c);
       delta.y = -delta.y*c/(m_rr-c);
       return 2*delta;
      }
 

void TorusManifold::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_P.x + m_rr + m_rc > hi.x || m_P.x - m_rr - m_rc < lo.x ||
        m_P.y + m_rr + m_rc > hi.y || m_P.y - m_rr - m_rc < lo.y ||
        m_P.z + m_rc > hi.z || m_P.z - m_rc < lo.z)
        {
        m_exec_conf->msg->warning() << "constrain.torus_manifold: Torus manifold is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        }
    }

//! Exports the TorusManifold class to python
void export_TorusManifold(pybind11::module& m)
    {
    py::class_< TorusManifold, std::shared_ptr<TorusManifold> >(m, "TorusManifold", py::base<Manifold>())
    .def(py::init< std::shared_ptr<SystemDefinition>,Scalar, Scalar, Scalar3 >())
    .def("implicit_function", &TorusManifold::implicit_function)
    .def("derivative", &TorusManifold::derivative)
    .def("returnL", &TorusManifold::returnL)
    .def("returnR", &TorusManifold::returnR)
    ;
    }
