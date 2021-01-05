// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ActiveVicsekForceCompute.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

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
			                Scalar r_dist,
			                Scalar coupling,
                                        int seed,
                                        py::list f_lst,
                                        py::list t_lst,
                                        bool orientation_link,
                                        bool orientation_reverse_link,
                                        Scalar rotation_diff)
        : ActiveForceCompute(sysdef,group,seed,f_lst,t_lst,orientation_link,orientation_reverse_link,rotation_diff), m_nlist(nlist), m_r_dist_sq(r_dist*r_dist)
    {

    m_coupling = m_deltaT*coupling;
    assert(m_nlist);
    }

ActiveVicsekForceCompute::~ActiveVicsekForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveVicsekForceCompute" << endl;
    }


/*! This function applies rotational diffusion to all active particles. The orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
void ActiveVicsekForceCompute::rotationalDiffusion(unsigned int timestep)
    {
    //  array handles
    //
    m_nlist->compute(timestep);

    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_f_actVec_backup(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_t_actVec(m_t_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::ActiveForceCompute, m_seed, tag, timestep);

        Scalar3 pos_i = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];

        if (m_sysdef->getNDimensions() == 2) // 2D
            {
            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng); // rotational diffusion angle

            Scalar theta; // angle on plane defining orientation of active force vector
            theta = atan2(h_f_actVec.data[i].y, h_f_actVec.data[i].x);
            
            Scalar mean_delta_theta = 0;
	    for (unsigned int k = 0; k < size; k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = h_nlist.data[myHead + k];
                assert(j < m_pdata->getN() + m_pdata->getNGhosts());

                if(i == j) continue;

                Scalar3 pos_j = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
                // apply periodic boundary conditions
                Scalar3 dx = box.minImage(pos_i-pos_j);
                // calculate r_ij squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);

                if (rsq < m_r_dist_sq)
               	    {
                    Scalar theta_n; 
		    theta_n = atan2(h_f_actVec_backup.data[j].y, h_f_actVec_backup.data[j].x);
                    mean_delta_theta += m_coupling*slow::sin(theta_n-theta);
                    }
                }
            theta += (delta_theta+mean_delta_theta);
            h_f_actVec.data[i].x = slow::cos(theta);
            h_f_actVec.data[i].y = slow::sin(theta);
            // In 2D, the only meaningful torque vector is out of plane and should not change
            }
        else // 3D: Following Stenhammar, Soft Matter, 2014
            {
            if (m_constraint) // if constraint exists
                {
                Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
                Scalar3 norm_scalar3 = m_manifold->derivative(current_pos); // the normal vector to which the particles are confined.
		Scalar norm_normal = slow::rsqrt(dot(norm_scalar3,norm_scalar3));
		
		norm_scalar3 *= norm_normal;

                vec3<Scalar> norm;
                norm = vec3<Scalar> (norm_scalar3);

                vec3<Scalar> current_f_vec;
                current_f_vec.x = h_f_actVec.data[i].x;
                current_f_vec.y = h_f_actVec.data[i].y;
                current_f_vec.z = h_f_actVec.data[i].z;

                vec3<Scalar> current_t_vec;
                current_t_vec.x = h_t_actVec.data[i].x;
                current_t_vec.y = h_t_actVec.data[i].y;
                current_t_vec.z = h_t_actVec.data[i].z;

                vec3<Scalar> aux_vec = cross(current_f_vec, norm); // aux vec for defining direction that active force vector rotates towards. Torque ignored

		norm_normal = slow::rsqrt(dot(aux_vec,aux_vec));

		aux_vec *= norm_normal;

                Scalar mean_delta_theta = 0;
	        for (unsigned int k = 0; k < size; k++)
                    {
                    // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                    unsigned int j = h_nlist.data[myHead + k];
                    assert(j < m_pdata->getN() + m_pdata->getNGhosts());

                    if(i == j) continue;

                    Scalar3 pos_j = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
                    // apply periodic boundary conditions
                    Scalar3 dx = box.minImage(pos_i-pos_j);
                    // calculate r_ij squared (FLOPS: 5)
                    Scalar rsq = dot(dx, dx);

                    if (rsq < m_r_dist_sq)
		        {
                	vec3<Scalar> current_f_n;
                	current_f_n.x = h_f_actVec_backup.data[j].x;
                	current_f_n.y = h_f_actVec_backup.data[j].y;
                	current_f_n.z = h_f_actVec_backup.data[j].z;
                	vec3<Scalar> aux_n = cross(current_f_n, norm);

			norm_normal = slow::rsqrt(dot(aux_n,aux_n));
			aux_n *= norm_normal;

                        Scalar theta_n = dot(aux_n,aux_vec); 
			theta_n = slow::sqrt(1-theta_n*theta_n);
			if(dot(aux_n,current_f_vec) > 0)
				theta_n *= -1;
                        mean_delta_theta += (m_coupling*theta_n);
                        }
                    }

                // rotational diffusion angle
                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);

		delta_theta += mean_delta_theta;

                h_f_actVec.data[i].x = slow::cos(delta_theta)*current_f_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_f_actVec.data[i].y = slow::cos(delta_theta)*current_f_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_f_actVec.data[i].z = slow::cos(delta_theta)*current_f_vec.z + slow::sin(delta_theta)*aux_vec.z;

                h_t_actVec.data[i].x = slow::cos(delta_theta)*current_t_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_t_actVec.data[i].y = slow::cos(delta_theta)*current_t_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_t_actVec.data[i].z = slow::cos(delta_theta)*current_t_vec.z + slow::sin(delta_theta)*aux_vec.z;

                }
            else // if constraint not exists
                {
                hoomd::SpherePointGenerator<Scalar> unit_vec;
                vec3<Scalar> rand_vec;
                unit_vec(rng, rand_vec);

                vec3<Scalar> aux_vec;
                aux_vec.x = h_f_actVec.data[i].y * rand_vec.z - h_f_actVec.data[i].z * rand_vec.y;
                aux_vec.y = h_f_actVec.data[i].z * rand_vec.x - h_f_actVec.data[i].x * rand_vec.z;
                aux_vec.z = h_f_actVec.data[i].x * rand_vec.y - h_f_actVec.data[i].y * rand_vec.x;
                Scalar aux_vec_mag = slow::rsqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
                aux_vec *= aux_vec_mag;

                vec3<Scalar> current_f_vec;
                current_f_vec.x = h_f_actVec.data[i].x;
                current_f_vec.y = h_f_actVec.data[i].y;
                current_f_vec.z = h_f_actVec.data[i].z;

                vec3<Scalar> current_t_vec;
                current_t_vec.x = h_t_actVec.data[i].x;
                current_t_vec.y = h_t_actVec.data[i].y;
                current_t_vec.z = h_t_actVec.data[i].z;

                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);
                h_f_actVec.data[i].x = slow::cos(delta_theta)*current_f_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_f_actVec.data[i].y = slow::cos(delta_theta)*current_f_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_f_actVec.data[i].z = slow::cos(delta_theta)*current_f_vec.z + slow::sin(delta_theta)*aux_vec.z;

                h_t_actVec.data[i].x = slow::cos(delta_theta)*current_t_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_t_actVec.data[i].y = slow::cos(delta_theta)*current_t_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_t_actVec.data[i].z = slow::cos(delta_theta)*current_t_vec.z + slow::sin(delta_theta)*aux_vec.z;

                }
            }
        }
    }

void export_ActiveVicsekForceCompute(py::module& m)
    {
    py::class_< ActiveVicsekForceCompute, std::shared_ptr<ActiveVicsekForceCompute> >(m, "ActiveVicsekForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, Scalar, Scalar, int, py::list, py::list,  bool, bool, Scalar>())
    ;
    }
