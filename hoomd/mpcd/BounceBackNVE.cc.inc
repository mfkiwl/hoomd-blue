// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackNVE@_geometry@.cc
 * \brief Template instantation for BounceBackNVE with @_geometry@.
 *
 * The geometry class is filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/BounceBackNVE.h"

#define GEOMETRY_CLASS @_geometry@
#define EXPORT_FUNCTION export_BounceBackNVE@_geometry@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {

// Explicit instantiation of CPU class
template class BounceBackNVE<GEOMETRY_CLASS>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_BounceBackNVE<GEOMETRY_CLASS>(m);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
