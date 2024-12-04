// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file ReverseNonequilibriumShearFlowUtilities.h
 * \brief Helper functions for Reverse nonequilibrium shear flow
 */

#ifndef MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_UTILITIES_H_
#define MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_UTILITIES_H_

#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {

class CompareMomentumToTarget
    {
    public:
    CompareMomentumToTarget(Scalar target_momentum, const unsigned int* tags_)
        : p(target_momentum), tags(tags_)
        {
        }

    bool operator()(const Scalar2& in0, const Scalar2& in1) const
        {
        const Scalar dp0 = std::fabs(in0.y - p);
        const Scalar dp1 = std::fabs(in1.y - p);
        if (dp0 < dp1)
            {
            // 0 is closer to target than 1
            return true;
            }
        else if (dp0 > dp1)
            {
            // 0 is farther from target than 1
            return false;
            }
        else
            {
            // both are equal distant, break tie using lower tag
            return tags[__scalar_as_int(in0.x)] < tags[__scalar_as_int(in1.x)];
            }
        }

    private:
    const Scalar p;                 //!< Momentum target
    const unsigned int* const tags; //!< Tag array
    };

class MaximumMomentum
    {
    public:
    MaximumMomentum(const unsigned int* tags_) : tags(tags_) { }

    bool operator()(const Scalar2& in0, const Scalar2& in1) const
        {
        const Scalar p0 = in0.y;
        const Scalar p1 = in1.y;
        if (p0 > p1)
            {
            // particle 0 has higher momemtum than 1 so should be selected first
            return true;
            }
        else if (p0 < p1)
            {
            // particle 0 has lower momentum than 1
            return false;
            }
        else
            {
            // both are equal distant, break tie using lower tag
            return tags[__scalar_as_int(in0.x)] < tags[__scalar_as_int(in1.x)];
            }
        }

    private:
    const unsigned int* const tags; //!< Tag array
    };

class MinimumMomentum
    {
    public:
    MinimumMomentum(const unsigned int* tags_) : tags(tags_) { }

    bool operator()(const Scalar2& in0, const Scalar2& in1) const
        {
        const Scalar p0 = in0.y;
        const Scalar p1 = in1.y;
        if (p0 < p1)
            {
            // particle 0 is lower in momemtum than 1 so should be selected first
            return true;
            }
        else if (p0 > p1)
            {
            // partilce 0 has higher momentum than 1
            return false;
            }
        else
            {
            // both are equal distant, break tie using lower tag
            return tags[__scalar_as_int(in0.x)] < tags[__scalar_as_int(in1.x)];
            }
        }

    private:
    const unsigned int* const tags; //!< Tag array
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_UTILITIES_H_
