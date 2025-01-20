// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TriangleAreaConservationMeshParameters.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

/*! \file TriangleAreaConservationMeshForceCompute.h
    \brief Declares a class for computing triangle area conservation forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_H__
#define __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Computes triangle area conservation forces on the mesh
/*! Triangle Area Conservation forces are computed on every triangle in a mesh.
    \ingroup computes
*/
class PYBIND11_EXPORT TriangleAreaConservationMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    TriangleAreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~TriangleAreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, const triangle_area_conservation_param_t& params);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual pybind11::array_t<Scalar> getArea()
        {
        ArrayHandle<Scalar> h_area(m_area, access_location::host, access_mode::read);
        return pybind11::array(m_mesh_data->getMeshTriangleData()->getNTypes(), h_area.data);
        };

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    GPUArray<triangle_area_conservation_param_t> m_params; //!< Parameters
    GPUArray<Scalar> m_area;                               //!< memory space for area

    std::shared_ptr<MeshDefinition>
        m_mesh_data; //!< Mesh data to use in computing area conservation energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the TriangleAreaConservationMeshForceCompute class to python
void export_TriangleAreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
