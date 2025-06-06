// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairGB.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairGBGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairGB>(m, "AnisoPotentialPairGBGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
