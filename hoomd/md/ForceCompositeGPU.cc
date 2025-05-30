// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ForceCompositeGPU.h"
#include "hoomd/VectorMath.h"

#include "ForceCompositeGPU.cuh"

/*! \file ForceCompositeGPU.cc
    \brief Contains code for the ForceCompositeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
 */
ForceCompositeGPU::ForceCompositeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : ForceComposite(sysdef)
    {
    // Initialize autotuners.
    m_tuner_force.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                          AutotunerBase::getTppListPow2(m_exec_conf)},
                                         m_exec_conf,
                                         "force_composite"));

    m_tuner_virial.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                           AutotunerBase::getTppListPow2(m_exec_conf)},
                                          m_exec_conf,
                                          "virial_composite"));

    m_tuner_update.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "update_composite"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_force, m_tuner_virial, m_tuner_update});

    GPUArray<uint2> flag(1, m_exec_conf);
    std::swap(m_flag, flag);

        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::overwrite);
        *h_flag.data = make_uint2(0, 0);
        }
    GPUVector<unsigned int> rigid_center(m_exec_conf);
    m_rigid_center.swap(rigid_center);

    GPUVector<unsigned int> lookup_center(m_exec_conf);
    m_lookup_center.swap(lookup_center);
    }

ForceCompositeGPU::~ForceCompositeGPU() { }

//! Compute the forces and torques on the central particle
void ForceCompositeGPU::computeForces(uint64_t timestep)
    {
    // If no rigid bodies exist return early. This also prevents accessing arrays assuming that this
    // is non-zero.
    if (m_n_molecules_global == 0)
        {
        return;
        }
    // access local molecule data (need to move this on top because of GPUArray scoping issues)
    const Index2D& molecule_indexer = getMoleculeIndexer();
    unsigned int nmol = molecule_indexer.getH();

    const GPUVector<unsigned int>& molecule_list = getMoleculeList();
    const GPUVector<unsigned int>& molecule_length = getMoleculeLengths();

    ArrayHandle<unsigned int> d_molecule_length(molecule_length,
                                                access_location::device,
                                                access_mode::read);
    ArrayHandle<unsigned int> d_molecule_list(molecule_list,
                                              access_location::device,
                                              access_mode::read);
    ArrayHandle<unsigned int> d_molecule_idx(getMoleculeIndex(),
                                             access_location::device,
                                             access_mode::read);

    // access particle data
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(),
                                     access_location::device,
                                     access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                      access_location::device,
                                      access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(),
                                     access_location::device,
                                     access_mode::readwrite);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

    // access rigid body definition
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation,
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rigid_center(m_rigid_center,
                                             access_location::device,
                                             access_mode::read);

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

        {
        ArrayHandle<uint2> d_flag(m_flag, access_location::device, access_mode::overwrite);

        // reset force and torque
        m_exec_conf->setDevice();

        unsigned int nelem = m_pdata->getN();

        if (nelem != 0)
            {
            hipMemsetAsync(d_force.data, 0, sizeof(Scalar4) * nelem);
            hipMemsetAsync(d_torque.data, 0, sizeof(Scalar4) * nelem);
            }

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_force->begin();
        auto param = m_tuner_force->getParam();
        unsigned int block_size = param[0];
        unsigned int n_bodies_per_block = param[1];

        // launch GPU kernel
        kernel::gpu_rigid_force(d_force.data,
                                d_torque.data,
                                d_molecule_length.data,
                                d_molecule_list.data,
                                d_molecule_idx.data,
                                d_rigid_center.data,
                                molecule_indexer,
                                d_postype.data,
                                d_orientation.data,
                                m_body_idx,
                                d_body_pos.data,
                                d_body_orientation.data,
                                d_body_len.data,
                                d_body.data,
                                d_tag.data,
                                d_flag.data,
                                d_net_force.data,
                                d_net_torque.data,
                                nmol,
                                m_pdata->getN(),
                                n_bodies_per_block,
                                block_size,
                                m_exec_conf->dev_prop,
                                !compute_virial,
                                m_n_rigid);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    uint2 flag;
        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::read);
        flag = *h_flag.data;
        }

    if (flag.x)
        {
        std::ostringstream s;
        s << "Composite particle with body tag " << flag.x - 1 << " incomplete" << std::endl
          << std::endl;
        throw std::runtime_error(s.str());
        }

    m_tuner_force->end();

    if (compute_virial)
        {
        // reset virial
        unsigned int nelem = m_pdata->getN();

        if (nelem != 0)
            {
            hipMemsetAsync(d_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
            }

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_virial->begin();
        auto param = m_tuner_virial->getParam();
        unsigned int block_size = param[0];
        unsigned int n_bodies_per_block = param[1];

        // launch GPU kernel
        kernel::gpu_rigid_virial(d_virial.data,
                                 d_molecule_length.data,
                                 d_molecule_list.data,
                                 d_molecule_idx.data,
                                 d_rigid_center.data,
                                 molecule_indexer,
                                 d_postype.data,
                                 d_orientation.data,
                                 m_body_idx,
                                 d_body_pos.data,
                                 d_body_orientation.data,
                                 d_net_force.data,
                                 d_net_virial.data,
                                 d_body.data,
                                 d_tag.data,
                                 nmol,
                                 m_pdata->getN(),
                                 n_bodies_per_block,
                                 m_pdata->getNetVirial().getPitch(),
                                 m_virial_pitch,
                                 block_size,
                                 m_exec_conf->dev_prop,
                                 m_n_rigid);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_virial->end();
        }
    }

void ForceCompositeGPU::updateCompositeParticles(uint64_t timestep)
    {
    // If no rigid bodies exist return early. This also prevents accessing arrays assuming that this
    // is non-zero.
    if (m_n_molecules_global == 0)
        {
        return;
        }

    // access molecule order
    const GPUArray<unsigned int>& molecule_length = getMoleculeLengths();

    ArrayHandle<unsigned int> d_molecule_order(getMoleculeOrder(),
                                               access_location::device,
                                               access_mode::read);
    ArrayHandle<unsigned int> d_molecule_len(molecule_length,
                                             access_location::device,
                                             access_mode::read);
    ArrayHandle<unsigned int> d_molecule_idx(getMoleculeIndex(),
                                             access_location::device,
                                             access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::device,
                                  access_mode::read);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    // access body positions, orientations, and types
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation,
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_body_types(m_body_types,
                                           access_location::device,
                                           access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);

    // lookup table
    ArrayHandle<unsigned int> d_lookup_center(m_lookup_center,
                                              access_location::device,
                                              access_mode::read);

        {
        ArrayHandle<uint2> d_flag(m_flag, access_location::device, access_mode::overwrite);

        m_exec_conf->setDevice();

        m_tuner_update->begin();
        unsigned int block_size = m_tuner_update->getParam()[0];

        kernel::gpu_update_composite(m_pdata->getN(),
                                     m_pdata->getNGhosts(),
                                     d_postype.data,
                                     d_velocity.data,
                                     d_orientation.data,
                                     d_angmom.data,
                                     d_inertia.data,
                                     m_body_idx,
                                     d_lookup_center.data,
                                     d_body_pos.data,
                                     d_body_orientation.data,
                                     d_body_types.data,
                                     d_body_len.data,
                                     d_molecule_order.data,
                                     d_molecule_len.data,
                                     d_molecule_idx.data,
                                     d_image.data,
                                     m_pdata->getBox(),
                                     m_pdata->getGlobalBox(),
                                     block_size,
                                     d_flag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_update->end();
        }

    uint2 flag;
        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::read);
        flag = *h_flag.data;
        }

    if (flag.x)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                         access_location::host,
                                         access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        unsigned int idx = flag.x - 1;
        unsigned int body_id = h_body.data[idx];
        unsigned int tag = h_tag.data[idx];

        std::ostringstream s;
        s << "Particle " << tag << " part of composite body " << body_id
          << " is missing central particle" << std::endl
          << std::endl;
        throw std::runtime_error(s.str());
        }

    if (flag.y)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                         access_location::host,
                                         access_mode::read);

        unsigned int idx = flag.y - 1;
        unsigned int body_id = h_body.data[idx];

        std::ostringstream s;
        s << "Composite particle with body id " << body_id << " incomplete" << std::endl
          << std::endl;
        throw std::runtime_error(s.str());
        }
    }

void ForceCompositeGPU::findRigidCenters()
    {
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);

    m_rigid_center.resize(m_pdata->getN() + m_pdata->getNGhosts());

    m_lookup_center.resize(m_pdata->getN() + m_pdata->getNGhosts());

    ArrayHandle<unsigned int> d_rigid_center(m_rigid_center,
                                             access_location::device,
                                             access_mode::overwrite);
    ArrayHandle<unsigned int> d_lookup_center(m_lookup_center,
                                              access_location::device,
                                              access_mode::overwrite);

    kernel::gpu_find_rigid_centers(d_body.data,
                                   d_tag.data,
                                   d_rtag.data,
                                   m_pdata->getN(),
                                   m_pdata->getNGhosts(),
                                   d_rigid_center.data,
                                   d_lookup_center.data,
                                   m_n_rigid);
    }

namespace detail
    {
void export_ForceCompositeGPU(pybind11::module& m)
    {
    pybind11::class_<ForceCompositeGPU, ForceComposite, std::shared_ptr<ForceCompositeGPU>>(
        m,
        "ForceCompositeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
