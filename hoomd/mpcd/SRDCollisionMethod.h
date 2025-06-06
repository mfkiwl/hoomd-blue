// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SRDCollisionMethod.h
 * \brief Declaration of mpcd::SRDCollisionMethod
 */

#ifndef MPCD_SRD_COLLISION_METHOD_H_
#define MPCD_SRD_COLLISION_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellThermoCompute.h"
#include "CollisionMethod.h"

#include "hoomd/Variant.h"

namespace hoomd
    {
namespace mpcd
    {
class PYBIND11_EXPORT SRDCollisionMethod : public mpcd::CollisionMethod
    {
    public:
    //! Constructor
    SRDCollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int cur_timestep,
                       unsigned int period,
                       int phase,
                       Scalar angle);

    //! Destructor
    virtual ~SRDCollisionMethod();

    //! Start autotuning kernel launch parameters
    void startAutotuning() override;

    //! Check if kernel autotuning is complete
    bool isAutotuningComplete() override;

    void setCellList(std::shared_ptr<mpcd::CellList> cl) override;

    //! Get the MPCD rotation angles
    Scalar getRotationAngle() const
        {
        return m_angle;
        }

    //! Set the MPCD rotation angle
    /*!
     * \param angle MPCD rotation angle in degrees
     */
    void setRotationAngle(Scalar angle)
        {
        m_angle = angle;
        }

    //! Get the cell rotation vectors from the last call
    const GPUVector<double3>& getRotationVectors() const
        {
        return m_rotvec;
        }

    //! Get the cell-level rescale factors for the temperature
    const GPUVector<double>& getScaleFactors() const
        {
        return m_factors;
        }

    //! Get the temperature
    std::shared_ptr<Variant> getTemperature() const
        {
        return m_T;
        }

    //! Set the temperature
    void setTemperature(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    //! Get the requested thermo flags
    mpcd::detail::ThermoFlags getRequestedThermoFlags() const
        {
        mpcd::detail::ThermoFlags flags;
        if (m_T)
            flags[mpcd::detail::thermo_options::energy] = 1;

        return flags;
        }

    protected:
    std::shared_ptr<mpcd::CellThermoCompute> m_thermo; //!< Cell thermo
    GPUVector<double3> m_rotvec;                       //!< MPCD rotation vectors
    Scalar m_angle;                                    //!< MPCD rotation angle (degrees)

    std::shared_ptr<Variant> m_T; //!< Temperature for thermostat
    GPUVector<double> m_factors;  //!< Cell-level rescale factors

    //! Implementation of the collision rule
    void rule(uint64_t timestep) override;

    //! Randomly draw cell rotation vectors
    virtual void drawRotationVectors(uint64_t timestep);

    //! Apply rotation matrix to velocities
    virtual void rotate(uint64_t timestep);

    //! Attach callback signals
    void attachCallbacks();

    //! Detach callback signals
    void detachCallbacks();
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SRD_COLLISION_METHOD_H_
