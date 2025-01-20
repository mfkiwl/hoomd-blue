// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <deque>

#include <pybind11/pybind11.h>

#include "GSDDumpWriter.h"

namespace hoomd
    {
class PYBIND11_EXPORT GSDDequeWriter : public GSDDumpWriter
    {
    public:
    GSDDequeWriter(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<Trigger> trigger,
                   const std::string& fname,
                   std::shared_ptr<ParticleGroup> group,
                   pybind11::object logger,
                   int queue_size,
                   std::string mode,
                   bool write_on_init,
                   bool clear_whole_buffer_after_dump,
                   uint64_t timestep);
    ~GSDDequeWriter() = default;

    void analyze(uint64_t timestep) override;

    void dump(long int start, long int end);

    int getMaxQueueSize() const;
    void setMaxQueueSize(int new_max_size);
    bool getClearWholeBufferAfterDump() const;
    void setClearWholeBufferAfterDump(bool clear_whole_buffer_after_dump);

    size_t getCurrentQueueSize() const;

    protected:
    int m_queue_size;
    bool m_clear_whole_buffer_after_dump;
    std::deque<GSDDumpWriter::GSDFrame> m_frame_queue;
    std::deque<pybind11::dict> m_log_queue;
    };

namespace detail
    {
void export_GSDDequeWriter(pybind11::module& m);
    } // namespace detail
    } // namespace hoomd
