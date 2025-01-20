// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "GSDDequeWriter.h"
#include "hoomd/GSDDumpWriter.h"

namespace hoomd
    {
GSDDequeWriter::GSDDequeWriter(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<Trigger> trigger,
                               const std::string& fname,
                               std::shared_ptr<ParticleGroup> group,
                               pybind11::object logger,
                               int queue_size,
                               std::string mode,
                               bool write_at_init,
                               bool clear_whole_buffer_after_dump,
                               uint64_t timestep)
    : GSDDumpWriter(sysdef, trigger, fname, group, mode), m_queue_size(queue_size)
    {
    setLogWriter(logger);
    bool file_empty = true;
    m_clear_whole_buffer_after_dump = true;
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        if (m_exec_conf->isRoot())
            {
            file_empty = gsd_get_nframes(&m_handle) == 0;
            }
        bcast(file_empty, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        file_empty = gsd_get_nframes(&m_handle) == 0;
        }
    if (file_empty)
        {
        if (!write_at_init)
            {
            throw std::runtime_error("Must set write_at_start to write to a new file.");
            }
        else
            {
            analyze(timestep);
            dump(0, -1);
            }
        }
    setClearWholeBufferAfterDump(clear_whole_buffer_after_dump);
    }

void GSDDequeWriter::analyze(uint64_t timestep)
    {
    m_frame_queue.emplace_front();
    populateLocalFrame(m_frame_queue.front(), timestep);
    m_log_queue.push_front(getLogData());
    if (m_queue_size != -1 && m_frame_queue.size() > static_cast<size_t>(m_queue_size))
        {
        m_frame_queue.pop_back();
        m_log_queue.pop_back();
        }
    }

void GSDDequeWriter::dump(long int start, long int end)
    {
    auto buffer_length = static_cast<long int>(m_frame_queue.size());
    if (end > buffer_length)
        {
        throw std::runtime_error("Burst.dump's end index is out of range.");
        }
    if (start < 0 || start > buffer_length)
        {
        throw std::runtime_error("Burst.dump's start index is out of range.");
        }
    long int iterator_start, iterator_end;
    if (end < 0)
        {
        iterator_end = buffer_length - start;
        iterator_start = 0;
        }
    else
        {
        iterator_end = buffer_length - start;
        iterator_start = buffer_length - end;
        }
    for (auto i = iterator_end - 1; i >= iterator_start; --i)
        {
        write(m_frame_queue[i], m_log_queue[i]);
        }
    if (m_clear_whole_buffer_after_dump)
        {
        m_frame_queue.clear();
        m_log_queue.clear();
        }
    else
        {
        m_frame_queue.erase(m_frame_queue.begin() + iterator_start, m_frame_queue.end());
        m_log_queue.erase(m_log_queue.begin() + iterator_start, m_log_queue.end());
        }
    }

int GSDDequeWriter::getMaxQueueSize() const
    {
    return m_queue_size;
    }

size_t GSDDequeWriter::getCurrentQueueSize() const
    {
    return m_frame_queue.size();
    }

void GSDDequeWriter::setMaxQueueSize(int new_max_size)
    {
    m_queue_size = new_max_size;
    if (m_queue_size == -1)
        {
        return;
        }
    while (static_cast<size_t>(m_queue_size) < m_frame_queue.size())
        {
        m_frame_queue.pop_back();
        m_log_queue.pop_back();
        }
    }

bool GSDDequeWriter::getClearWholeBufferAfterDump() const
    {
    return m_clear_whole_buffer_after_dump;
    }

void GSDDequeWriter::setClearWholeBufferAfterDump(bool clear_whole_buffer_after_dump)
    {
    m_clear_whole_buffer_after_dump = clear_whole_buffer_after_dump;
    }

namespace detail
    {
void export_GSDDequeWriter(pybind11::module& m)
    {
    pybind11::class_<GSDDequeWriter, GSDDumpWriter, std::shared_ptr<GSDDequeWriter>>(
        m,
        "GSDDequeWriter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::string,
                            std::shared_ptr<ParticleGroup>,
                            pybind11::object,
                            int,
                            std::string,
                            bool,
                            bool,
                            uint64_t>())
        .def_property("max_burst_size",
                      &GSDDequeWriter::getMaxQueueSize,
                      &GSDDequeWriter::setMaxQueueSize)
        .def_property("clear_whole_buffer_after_dump",
                      &GSDDequeWriter::getClearWholeBufferAfterDump,
                      &GSDDequeWriter::setClearWholeBufferAfterDump)
        .def("__len__", &GSDDequeWriter::getCurrentQueueSize)
        .def("dump", &GSDDequeWriter::dump);
    }
    } // namespace detail
    } // namespace hoomd
