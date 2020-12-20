// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ActiveVicsekForceCompute.h"

#include <vector>

using namespace std;
using namespace hoomd;
namespace py = pybind11;

/*! \file ActiveVicsekForceCompute.cc
    \brief Contains code for the ActiveVicsekForceCompute class
*/

/*! \param seed required user-specified seed number for random number generator.
    \param f_lst An array of (x,y,z) tuples for the active force vector for each particle.
    \param t_lst An array of (xyz) tuples for the active torque vector for each particle
    \param orientation_link if True then forces and torques are applied in the particle's reference frame. If false, then the box reference fra    me is used. Only relevant for non-point-like anisotropic particles.
    /param orientation_reverse_link When True, the particle's orientation is set to match the active force vector. Useful for
    for using a particle's orientation to log the active force vector. Not recommended for anisotropic particles
    \param rotation_diff rotational diffusion constant for all particles.
    \param constraint specifies a constraint surface, to which particles are confined,
    such as update.constraint_ellipsoid.
*/
ActiveVicsekForceCompute::ActiveVicsekForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<ParticleGroup> group,
                                        std::shared_ptr<NeighborList> nlist,
                                        int seed,
                                        py::list f_lst,
                                        py::list t_lst,
                                        bool orientation_link,
                                        bool orientation_reverse_link,
                                        Scalar rotation_diff)
        : ActiveForceCompute(sysdef,group,seed,f_lst,t_lst,orientation_link,orientation_reverse_link,rotation_diff), m_nlist(nlist)
    {
    assert(m_nlist);
    }

ActiveVicsekForceCompute::~ActiveVicsekForceCompute()
    {
    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar3>(MaxN, m_exec_conf).swap(m_f_activeVec_backup);

    m_exec_conf->msg->notice(5) << "Destroying ActiveVicsekForceCompute" << endl;
    }

/*! this function adds a mainfold constraint to the active particles
*/



/*! This function sets an ellipsoid surface constraint for all active particles. Torque is not considered here
*/
void ActiveVicsekForceCompute::setMeanVelocity(unsigned int timestep)
    {
    //  array handles

    m_nlist->compute(timestep);
    
    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::readwrite);

    unsigned int N_backup = m_pdata->getN();
        {
        ArrayHandle<Scalar3> h_f_actVec_backup(m_f_activeVec_backup, access_location::host, access_mode::overwrite);
        memcpy(h_f_actVec_backup.data, h_f_actVec.data, sizeof(Scalar4) * N_backup);
        }

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    //Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_f_actVec_backup(m_f_activeVec_backup, access_location::host, access_mode::read);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        
        Scalar3 mean_vel = h_f_actVec_backup.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());
            mean_vel += h_f_actVec_backup.data[j];
            }
        mean_vel /= (size+1);

        Scalar new_norm = Scalar(1.0)/slow::sqrt(mean_vel.x*mean_vel.x + mean_vel.y*mean_vel.y + mean_vel.z*mean_vel.z);

        mean_vel *= new_norm;

        h_f_actVec.data[i].x = mean_vel.x;
        h_f_actVec.data[i].y = mean_vel.y;
        h_f_actVec.data[i].z = mean_vel.z;
        }
    }

/*! This function applies constraints, rotational diffusion, and sets forces for all active particles
    \param timestep Current timestep
*/
void ActiveVicsekForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "ActiveVicsekForceCompute");

    if (last_computed != timestep)
        {
        m_rotationConst = slow::sqrt(2.0 * m_rotationDiff * m_deltaT);

        last_computed = timestep;

        setMeanVelocity(timestep);

        if (m_constraint)
            {
            setConstraint(); // apply surface constraints to active particles active force vectors
            }

        if (m_rotationDiff != 0)
            {
            rotationalDiffusion(timestep); // apply rotational diffusion to active particles
            }
        setForces(); // set forces for particles
        }

    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);

    }


void export_ActiveVicsekForceCompute(py::module& m)
    {
    py::class_< ActiveVicsekForceCompute, std::shared_ptr<ActiveVicsekForceCompute> >(m, "ActiveVicsekForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, int, py::list, py::list,  bool, bool, Scalar>())
    ;
    }
