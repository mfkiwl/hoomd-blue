// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialTersoff.h"
#include "hoomd/md/Evaluator@_evaluator@.h"

#define EVALUATOR_CLASS Evaluator@_evaluator@
#define EXPORT_FUNCTION export_Potential@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialTersoff<EVALUATOR_CLASS>(m, "Potential@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
