// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "hoomd/md/PairModulator.h"
#include "hoomd/md/PatchEnvelope.h"

#include "hoomd/md/EvaluatorPair@_evaluator@.h"
#define EVALUATOR_CLASS EvaluatorPair@_evaluator@
#define EXPORT_FUNCTION export_AnisoPotentialPairPatchy@_evaluator@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<PairModulator<EVALUATOR_CLASS, PatchEnvelope>>(
        m,
        "AnisoPotentialPairPatchy@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
