// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParticleData.h
 * \brief Definition of mpcd::ParticleData
 */

#include "ParticleData.h"

#include "hoomd/CachedAllocator.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif // ENABLE_MPI

#include <pybind11/stl.h>

#include <algorithm>
#include <iomanip>
#include <random>
using namespace std;

namespace hoomd
    {
// 9/8 factor for amortized growth
const float mpcd::ParticleData::resize_factor = 9. / 8.;

/*!
 * \param N Number of MPCD particles
 * \param local_box Local simulation box
 * \param kT Temperature
 * \param seed Seed to pseudo-random number generator
 * \param ndimensions Dimensionality of the system
 * \param exec_conf Execution configuration
 * \param decomposition Domain decomposition
 *
 * MPCD particles are randomly initialized in the box with velocities
 * equipartitioned at kT.
 */
mpcd::ParticleData::ParticleData(unsigned int N,
                                 std::shared_ptr<const BoxDim> local_box,
                                 Scalar kT,
                                 unsigned int seed,
                                 unsigned int ndimensions,
                                 std::shared_ptr<ExecutionConfiguration> exec_conf,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : m_N(0), m_N_virtual(0), m_N_global(0), m_N_max(0), m_exec_conf(exec_conf), m_mass(1.0),
      m_valid_cell_cache(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD ParticleData" << endl;

    // set domain decomposition
    unsigned int my_seed = seed;
#ifdef ENABLE_MPI
    setupMPI(decomposition);
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Bcast(&my_seed, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
        my_seed
            += m_exec_conf->getRank(); // each rank must get a different seed value for C++11 PRNG
        }
#endif // ENABLE_MPI

    initializeRandom(N, local_box, kT, my_seed, ndimensions);
    }

/*!
 * \param snapshot Snapshot of the MPCD particle data
 * \param global_box Global simulation box
 * \param exec_conf Execution configuration
 * \param decomposition Domain decomposition
 */
mpcd::ParticleData::ParticleData(const mpcd::ParticleDataSnapshot& snapshot,
                                 std::shared_ptr<const BoxDim> global_box,
                                 std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : m_N(0), m_N_virtual(0), m_N_global(0), m_N_max(0), m_exec_conf(exec_conf), m_mass(1.0),
      m_valid_cell_cache(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD ParticleData" << endl;

// set domain decomposition
#ifdef ENABLE_MPI
    setupMPI(decomposition);
#endif

    initializeFromSnapshot(snapshot, global_box);
    }

mpcd::ParticleData::~ParticleData()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD ParticleData" << endl;
#ifdef ENABLE_MPI
    MPI_Type_free(&m_mpi_pdata_element);
#endif // ENABLE_MPI
    }

/*!
 * \param snapshot mpcd::ParticleDataSnapshot to read on the root (0) rank
 * \param global_box Current global box
 * \post The MPCD particle data has been initialized across all ranks with
 *       the contents of \a snapshot.
 *
 * This is a collective call, requiring participation from all ranks even if
 * the snapshot is only valid on the root rank.
 */
void mpcd::ParticleData::initializeFromSnapshot(const mpcd::ParticleDataSnapshot& snapshot,
                                                std::shared_ptr<const BoxDim> global_box)
    {
    m_exec_conf->msg->notice(4) << "MPCD ParticleData: initializing from snapshot" << std::endl;

    if (!checkSnapshot(snapshot))
        {
        m_exec_conf->msg->error() << "Invalid MPCD particle data snapshot" << std::endl;
        throw std::runtime_error("Error initializing MPCD particle data.");
        }

    if (!checkInBox(snapshot, global_box))
        {
        m_exec_conf->msg->error() << "Not all MPCD particles were found inside the box" << endl;
        throw std::runtime_error("Error initializing MPCD particle data");
        }

    // global number of particles
    unsigned int N_global(0);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        unsigned int n_ranks = m_exec_conf->getNRanks();
        unsigned int rank = m_exec_conf->getRank();
        const unsigned int root = 0;

        // assign each particle to a rank
        unsigned int num_types = 0;
        std::vector<int> num_per_rank;       // number per rank
        std::vector<int> rank_displacements; // displacement of particles per rank
        std::vector<mpcd::detail::pdata_element> particles;
        if (rank == root)
            {
            const Index3D& di = m_decomposition->getDomainIndexer();
            ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                                   access_location::host,
                                                   access_mode::read);

            // global particle number is snapshot size, also allocate temporary data for sending
            N_global = snapshot.size;
            particles.resize(N_global);

            // temporary data for counts per rank
            num_per_rank.resize(n_ranks);
            rank_displacements.resize(n_ranks);
            std::fill(num_per_rank.begin(), num_per_rank.end(), 0);

            // loop over particles in snapshot, place them into domains
            for (auto it = snapshot.position.begin(); it != snapshot.position.end(); ++it)
                {
                unsigned int snap_idx = (unsigned int)(it - snapshot.position.begin());

                // determine domain the particle is placed into
                Scalar3 pos = vec_to_scalar3(*it);
                Scalar3 f = global_box->makeFraction(pos);
                int i = int(f.x * ((Scalar)di.getW()));
                int j = int(f.y * ((Scalar)di.getH()));
                int k = int(f.z * ((Scalar)di.getD()));

                // wrap particles that are exactly on a boundary
                char3 flags = make_char3(0, 0, 0);
                if (i == (int)di.getW())
                    {
                    i = 0;
                    flags.x = 1;
                    }

                if (j == (int)di.getH())
                    {
                    j = 0;
                    flags.y = 1;
                    }

                if (k == (int)di.getD())
                    {
                    k = 0;
                    flags.z = 1;
                    }
                uchar3 periodic = make_uchar3(flags.x, flags.y, flags.z);
                auto wrapping_box = *global_box;
                wrapping_box.setPeriodic(periodic);
                int3 img = make_int3(0, 0, 0);
                wrapping_box.wrap(pos, img, flags);

                // place particle into the computational domain
                unsigned int rank
                    = m_decomposition->placeParticle(*global_box, pos, h_cart_ranks.data);
                if (rank >= n_ranks)
                    {
                    m_exec_conf->msg->error()
                        << "init.*: Particle " << snap_idx << " out of bounds." << std::endl;
                    m_exec_conf->msg->error() << "Cartesian coordinates: " << std::endl;
                    m_exec_conf->msg->error()
                        << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl;
                    m_exec_conf->msg->error() << "Fractional coordinates: " << std::endl;
                    m_exec_conf->msg->error()
                        << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl;
                    Scalar3 lo = global_box->getLo();
                    Scalar3 hi = global_box->getHi();
                    m_exec_conf->msg->error() << "Global box lo: (" << lo.x << ", " << lo.y << ", "
                                              << lo.z << ")" << std::endl;
                    m_exec_conf->msg->error() << "           hi: (" << hi.x << ", " << hi.y << ", "
                                              << hi.z << ")" << std::endl;

                    throw std::runtime_error("Error initializing from snapshot.");
                    }

                // pack particle into pdata element
                mpcd::detail::pdata_element particle;
                particle.pos
                    = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(snapshot.type[snap_idx]));
                const auto vel = snapshot.velocity[snap_idx];
                particle.vel
                    = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
                particle.tag = snap_idx;
                particle.comm_flag = rank; // hijack the comm flag to sort by rank

                // fill up per-processor data structures
                particles[snap_idx] = particle;
                ++num_per_rank[rank];
                }

            // mass is set equal for all particles
            m_mass = snapshot.mass;

            // assign type map from snapshot
            m_type_mapping = snapshot.type_mapping;
            num_types = static_cast<unsigned int>(m_type_mapping.size());

            // exclusive scan of displacements
            rank_displacements[0] = 0;
            for (size_t i = 1; i < num_per_rank.size(); ++i)
                {
                rank_displacements[i] = rank_displacements[i - 1] + num_per_rank[i - 1];
                }

            // sort particles by rank
            std::sort(particles.begin(),
                      particles.end(),
                      [](const mpcd::detail::pdata_element& a, const mpcd::detail::pdata_element& b)
                      { return a.comm_flag < b.comm_flag; });
            }

        // broadcast global number of particles
        MPI_Bcast(&N_global, 1, MPI_UNSIGNED, root, mpi_comm);

        // broadcast the particle mass
        MPI_Bcast(&m_mass, 1, MPI_HOOMD_SCALAR, root, mpi_comm);

        // broadcast type mapping
        MPI_Bcast(&num_types, 1, MPI_UNSIGNED, root, mpi_comm);
        if (rank != root)
            {
            m_type_mapping.resize(num_types);
            }
        bcast(m_type_mapping, root, mpi_comm);

        // scatter the number of particles on each rank and allocate
        MPI_Scatter(num_per_rank.data(), 1, MPI_UNSIGNED, &m_N, 1, MPI_UNSIGNED, root, mpi_comm);
        allocate(std::max(1u, m_N));
        std::vector<mpcd::detail::pdata_element> recv_particles(m_N);

        // distribute particle data to processors
        MPI_Scatterv(particles.data(),
                     num_per_rank.data(),
                     rank_displacements.data(),
                     m_mpi_pdata_element,
                     recv_particles.data(),
                     m_N,
                     m_mpi_pdata_element,
                     root,
                     mpi_comm);

        // Fill-up particle data arrays
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_comm_flag(m_comm_flags,
                                              access_location::host,
                                              access_mode::overwrite);
        for (unsigned int idx = 0; idx < m_N; idx++)
            {
            const mpcd::detail::pdata_element particle = recv_particles[idx];
            h_pos.data[idx] = particle.pos;
            h_vel.data[idx] = particle.vel;
            h_tag.data[idx] = particle.tag;
            h_comm_flag.data[idx] = 0; // initialize with zero by default
            }
        }
    else
#endif // ENABLE_MPI
        {
        // number of local particles is the global number
        N_global = snapshot.size;
        allocate(N_global);
        m_N = N_global;

        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);

        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            const auto pos = snapshot.position[idx];
            h_pos.data[idx]
                = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(snapshot.type[idx]));

            const auto vel = snapshot.velocity[idx];
            h_vel.data[idx]
                = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));

            h_tag.data[idx] = idx;
            }

        // mass is equal for all particles
        m_mass = snapshot.mass;

        // initialize type mapping
        m_type_mapping = snapshot.type_mapping;
        }

    setNGlobal(N_global);
    }

/*!
 * \param N Global number of particles (global in MPI)
 * \param local_box Local simulation box
 * \param kT Temperature (in energy units)
 * \param seed Random seed
 * \param ndimensions Dimensionality
 */
void mpcd::ParticleData::initializeRandom(unsigned int N,
                                          std::shared_ptr<const BoxDim> local_box,
                                          Scalar kT,
                                          unsigned int seed,
                                          unsigned int ndimensions)
    {
    // only one particle type is supported for this construction method
    m_type_mapping.clear();
    m_type_mapping.push_back("A");
    // default particle mass is 1.0
    m_mass = Scalar(1.0);

    // figure out how many local particles I should own
    setNGlobal(N);
    unsigned int tag_start;
#ifdef ENABLE_MPI
    const unsigned int nrank = m_exec_conf->getNRanks();
    if (nrank > 1)
        {
        const unsigned int rank = m_exec_conf->getRank();
        m_N = N / nrank;
        tag_start = rank * m_N;

        // distribute remainder of particles to low ranks
        // this number is usually small, so it doesn't really matter
        const unsigned int Nleft = N - m_N * nrank;
        if (rank < Nleft)
            {
            ++m_N;
            // need to offset 1 for every rank above 0 that I am
            tag_start += rank;
            }
        else
            {
            // offset but total number of extra particles below me
            tag_start += Nleft;
            }
        }
    else
#endif // ENABLE_MPI
        {
        m_N = N;
        tag_start = 0;
        }

    // center of box
    const Scalar3 lo = local_box->getLo();
    const Scalar3 hi = local_box->getHi();

    // random number generator
    std::mt19937 mt(seed);
    std::uniform_real_distribution<Scalar> pos_x(lo.x, hi.x);
    std::uniform_real_distribution<Scalar> pos_y(lo.y, hi.y);
    std::uniform_real_distribution<Scalar> pos_z(lo.z, hi.z);
    std::normal_distribution<Scalar> vel(0.0, fast::sqrt(kT / m_mass));

    // allocate and fill up with random values
    allocate(m_N);
    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);
    double3 vel_cm = make_double3(0, 0, 0);
    for (unsigned int i = 0; i < m_N; ++i)
        {
        h_pos.data[i] = make_scalar4(pos_x(mt),
                                     pos_y(mt),
                                     (ndimensions == 3) ? pos_z(mt) : Scalar(0.0),
                                     __int_as_scalar(0));
        h_vel.data[i] = make_scalar4(vel(mt),
                                     vel(mt),
                                     (ndimensions == 3) ? vel(mt) : Scalar(0.0),
                                     __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[i] = tag_start + i;

        // add up total velocity
        vel_cm.x += h_vel.data[i].x;
        vel_cm.y += h_vel.data[i].y;
        vel_cm.z += h_vel.data[i].z;
        }

// compute average velocity per-particle to remove
#ifdef ENABLE_MPI
    if (nrank > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &vel_cm,
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI
    if (N > 0)
        {
        vel_cm.x /= N;
        vel_cm.y /= N;
        vel_cm.z /= N;
        }

    // subtract center-of-mass velocity
    for (unsigned int i = 0; i < m_N; ++i)
        {
        h_vel.data[i].x -= vel_cm.x;
        h_vel.data[i].y -= vel_cm.y;
        h_vel.data[i].z -= vel_cm.z;
        }
    }

/*!
 * \param snapshot mpcd::ParticleDataSnapshot to fill
 * \param global_box Current global box
 * \post \a snapshot holds the current MPCD particle data on the root (0) rank
 */
void mpcd::ParticleData::takeSnapshot(mpcd::ParticleDataSnapshot& snapshot,
                                      const shared_ptr<const BoxDim> global_box) const
    {
    m_exec_conf->msg->notice(4) << "MPCD ParticleData: taking snapshot" << std::endl;

    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        const unsigned int n_ranks = m_exec_conf->getNRanks();
        const unsigned int rank = m_exec_conf->getRank();
        const unsigned int root = 0;

        /*
         * Stage particles to send on each rank. The root rank gets enough memory to hold ALL
         * particles later even though we only fill with the local number for now.
         */
        std::vector<detail::pdata_element> particles((rank != root) ? m_N : m_N_global);
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            mpcd::detail::pdata_element particle;
            particle.pos = h_pos.data[idx];
            particle.vel = h_vel.data[idx];
            particle.tag = h_tag.data[idx];
            particle.comm_flag = 0;
            particles[idx] = particle;
            }

        // size particles per rank and displacements
        std::vector<int> num_per_rank(n_ranks);       // number per rank
        std::vector<int> rank_displacements(n_ranks); // displacement of particles per rank
        MPI_Gather(&m_N, 1, MPI_UNSIGNED, num_per_rank.data(), 1, MPI_UNSIGNED, root, mpi_comm);
        rank_displacements[0] = 0;
        for (size_t i = 1; i < num_per_rank.size(); ++i)
            {
            rank_displacements[i] = rank_displacements[i - 1] + num_per_rank[i - 1];
            }

        // gather data back to root rank
        MPI_Gatherv((rank != root) ? particles.data() : MPI_IN_PLACE,
                    m_N,
                    m_mpi_pdata_element,
                    particles.data(),
                    num_per_rank.data(),
                    rank_displacements.data(),
                    m_mpi_pdata_element,
                    root,
                    mpi_comm);

        if (rank == root)
            {
            /*
             * First sort the particles into tag order. This is probably a slow step, but could be
             * sped up by sorting on each rank first then merging the sorted segments here.
             */
            std::sort(particles.begin(),
                      particles.end(),
                      [](const mpcd::detail::pdata_element& a, const mpcd::detail::pdata_element& b)
                      { return a.tag < b.tag; });

            // allocate memory in snapshot
            snapshot.resize(m_N_global);

            // unpack into snapshot
            for (unsigned int idx = 0; idx < m_N_global; ++idx)
                {
                const auto particle = particles[idx];

                // wrapped position
                const Scalar4 postype = particle.pos;
                Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                int3 img = make_int3(0, 0, 0);
                global_box->wrap(pos, img);
                snapshot.position[idx] = vec3<Scalar>(pos);

                // typeid
                snapshot.type[idx] = __scalar_as_int(postype.w);

                // velocity
                const Scalar4 velcell = particle.vel;
                const Scalar3 vel = make_scalar3(velcell.x, velcell.y, velcell.z);
                snapshot.velocity[idx] = vec3<Scalar>(vel);
                }
            }
        }
    else
#endif
        {
        // allocate memory in snapshot
        snapshot.resize(m_N);

        // sort by tags
        std::vector<std::pair<unsigned int, unsigned int>> tag_index(m_N);
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            tag_index[idx] = std::make_pair(h_tag.data[idx], idx);
            }
        std::sort(tag_index.begin(),
                  tag_index.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // unpack into snapshot
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            const unsigned int pidx = tag_index[idx].second;

            // wrapped position
            const Scalar4 postype = h_pos.data[pidx];
            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
            int3 img = make_int3(0, 0, 0);
            global_box->wrap(pos, img);
            snapshot.position[idx] = vec3<Scalar>(pos);

            // typeid
            snapshot.type[idx] = __scalar_as_int(postype.w);

            // velocity
            const Scalar4 velcell = h_vel.data[pidx];
            const Scalar3 vel = make_scalar3(velcell.x, velcell.y, velcell.z);
            snapshot.velocity[idx] = vec3<Scalar>(vel);
            }
        }

    // set the type mapping and mass, which is synced across processors
    snapshot.type_mapping = m_type_mapping;
    snapshot.mass = m_mass;
    }

/*!
 * \param snapshot Snapshot to validate
 *
 * This is a convenience wrapper to broadcast the result of snapshot validation
 * from the root rank. Note that this is a collective call that must be made
 * from all ranks.
 *
 * \sa mpcd::ParticleDataSnapshot::validate
 */
bool mpcd::ParticleData::checkSnapshot(const mpcd::ParticleDataSnapshot& snapshot)
    {
    bool valid_snapshot = true;
    if (m_exec_conf->getRank() == 0)
        {
        valid_snapshot = snapshot.validate();
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        MPI_Bcast(&valid_snapshot, 1, MPI_CXX_BOOL, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    return valid_snapshot;
    }

/*!
 * \param snapshot Snapshot to validate
 * \param box Global simulation box
 *
 * Checks that all particles lie within the global simulation box on initialization
 * on the root rank, and broadcasts the results to the other ranks. Note that
 * this is a collective call that must be made from all ranks.
 */
bool mpcd::ParticleData::checkInBox(const mpcd::ParticleDataSnapshot& snapshot,
                                    std::shared_ptr<const BoxDim> box)
    {
    bool in_box = true;
    if (m_exec_conf->getRank() == 0)
        {
        Scalar3 lo = box->getLo();
        Scalar3 hi = box->getHi();

        const Scalar tol = Scalar(1e-5);

        for (unsigned int i = 0; i < snapshot.size; ++i)
            {
            Scalar3 f = box->makeFraction(vec_to_scalar3(snapshot.position[i]));
            if (f.x < -tol || f.x > Scalar(1.0) + tol || f.y < -tol || f.y > Scalar(1.0) + tol
                || f.z < -tol || f.z > Scalar(1.0) + tol)
                {
                m_exec_conf->msg->warning()
                    << "pos " << i << ":" << setprecision(12) << snapshot.position[i].x << " "
                    << snapshot.position[i].y << " " << snapshot.position[i].z << endl;
                m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " "
                                            << f.y << " " << f.z << endl;
                m_exec_conf->msg->warning() << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
                m_exec_conf->msg->warning() << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
                in_box = false;
                break;
                }
            }
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        MPI_Bcast(&in_box, 1, MPI_CXX_BOOL, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    return in_box;
    }

/*!
 * \param nglobal Global number of particles
 */
void mpcd::ParticleData::setNGlobal(unsigned int nglobal)
    {
    assert(m_N <= nglobal);
    m_N_global = nglobal;

    // TODO: any signaling if needed to subscribers
    }

/*!
 * \param N_max maximum number of particles that can be held in allocation
 *
 * A clean allocation of all particle data arrays is made to accommodate \a N_max
 * total particles.
 */
void mpcd::ParticleData::allocate(unsigned int N_max)
    {
    m_N_max = N_max;

    //! Allocate the particle data
    GPUArray<Scalar4> pos(N_max, m_exec_conf);
    m_pos.swap(pos);

    GPUArray<Scalar4> vel(N_max, m_exec_conf);
    m_vel.swap(vel);

    GPUArray<unsigned int> tag(N_max, m_exec_conf);
    m_tag.swap(tag);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        GPUArray<unsigned int> comm_flags(N_max, m_exec_conf);
        m_comm_flags.swap(comm_flags);
        }
#endif // ENABLE_MPI

    // Allocate the alternate data
    GPUArray<Scalar4> pos_alt(N_max, m_exec_conf);
    m_pos_alt.swap(pos_alt);

    GPUArray<Scalar4> vel_alt(N_max, m_exec_conf);
    m_vel_alt.swap(vel_alt);

    GPUArray<unsigned int> tag_alt(N_max, m_exec_conf);
    m_tag_alt.swap(tag_alt);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        GPUArray<unsigned int> comm_flags_alt(N_max, m_exec_conf);
        m_comm_flags_alt.swap(comm_flags_alt);

        GPUArray<unsigned int> remove_ids(N_max, m_exec_conf);
        m_remove_ids.swap(remove_ids);

#ifdef ENABLE_HIP
        GPUFlags<unsigned int> num_remove(m_exec_conf);
        m_num_remove.swap(num_remove);

        // this array is used for particle migration
        GPUArray<unsigned char> remove_flags(N_max, m_exec_conf);
        m_remove_flags.swap(remove_flags);
#endif // ENABLE_HIP
        }
#endif // ENABLE_MPI
    }

/*!
 * \param N_max maximum number of particles that can be held in allocation
 *
 * A resize of all particle data arrays is made to accommodate \a N_max particles.
 * This has the behavior defined by GPUArray for resizing.
 */
void mpcd::ParticleData::reallocate(unsigned int N_max)
    {
    m_N_max = N_max;

    // Reallocate the particle data
    m_pos.resize(N_max);
    m_vel.resize(N_max);
    m_tag.resize(N_max);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        m_comm_flags.resize(N_max);
        }
#endif // ENABLE_MPI

    // Reallocate the alternate data
    m_pos_alt.resize(N_max);
    m_vel_alt.resize(N_max);
    m_tag_alt.resize(N_max);
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        m_comm_flags_alt.resize(N_max);
        m_remove_ids.resize(N_max);

#ifdef ENABLE_HIP
        m_remove_flags.resize(N_max);
#endif // ENABLE_HIP
        }
#endif // ENABLE_MPI
    }

/*!
 * \param N New number of particles held by the data
 *
 * A new size for the particle data arrays is chosen using amortized growth
 * (set by growth_factor), and the particle data is resized. If N is less than
 * m_N_max, then nothing needs to be reallocated, and the current number of
 * owned particles is simply changed.
 */
void mpcd::ParticleData::resize(unsigned int N)
    {
    // compute the new size of the array using amortized growth
    unsigned int N_max = m_N_max;
    if (N > N_max)
        {
        while (N > N_max)
            {
            N_max = ((unsigned int)(((float)N_max) * resize_factor)) + 1;
            }
        reallocate(N_max);
        }
    m_N = N;
    }

/*!
 * \param mass New particle mass
 *
 * This is a collective call requiring the participation of all ranks. The mass
 * will be broadcast from rank 0 (root) to ensure it is synced across all ranks
 * in MPI. The mass cannot be set independently between ranks because this is not meaningful.
 */
void mpcd::ParticleData::setMass(Scalar mass)
    {
    assert(mass > Scalar(0.));
    m_mass = mass;

#ifdef ENABLE_MPI
    // in mpi, the mass must be synced between all ranks
    if (m_decomposition)
        {
        MPI_Bcast(&m_mass, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI
    }

/*!
 * \param name Type name to get the index of
 * \return Type index of the corresponding type name
 * \throw runtime_error if the type name is not found
 */
unsigned int mpcd::ParticleData::getTypeByName(const std::string& name) const
    {
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); ++i)
        {
        if (m_type_mapping[i] == name)
            return i;
        }

    m_exec_conf->msg->error() << "MPCD particle type " << name << " not found!" << endl;
    throw runtime_error("Error mapping MPCD type name");
    }

/*!
 * \param type Type index to get the name of
 * \returns Type name of the requested type
 * \throw runtime_error if the requested type index is out of bounds
 *
 * Type indices must range from 0 to getNTypes.
 */
std::string mpcd::ParticleData::getNameByType(unsigned int type) const
    {
    // check for an invalid request
    if (type >= m_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Requesting name for non-existent MPCD particle type " << type
                                  << endl;
        throw runtime_error("Error mapping MPCD type name");
        }

    // return the name
    return m_type_mapping[type];
    }

/*!
 * \param idx Local particle index
 * \returns Position of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
Scalar3 mpcd::ParticleData::getPosition(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx
                                  << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    const Scalar4 postype = h_pos.data[idx];
    return make_scalar3(postype.x, postype.y, postype.z);
    }

/*!
 * \param idx Local particle index
 * \returns Type of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
unsigned int mpcd::ParticleData::getType(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx
                                  << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    const Scalar4 postype = h_pos.data[idx];
    return __scalar_as_int(postype.w);
    }

/*!
 * \param idx Local particle index
 * \returns Velocity of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
Scalar3 mpcd::ParticleData::getVelocity(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx
                                  << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
    const Scalar4 velcell = h_vel.data[idx];
    return make_scalar3(velcell.x, velcell.y, velcell.z);
    }

/*!
 * \param idx Local particle index
 * \returns Tag of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
unsigned int mpcd::ParticleData::getTag(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx
                                  << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);
    return h_tag.data[idx];
    }

/*!
 * \param N Allocate space for \a N additional virtual particles in the particle data arrays
 * \returns The first index of the new virtual particles in the arrays.
 */
unsigned int mpcd::ParticleData::addVirtualParticles(unsigned int N)
    {
    const unsigned int first_idx = m_N + m_N_virtual;
    if (N == 0)
        {
        return first_idx;
        }

    // increase number of virtual particles
    m_N_virtual += N;

    // minimum size of new arrays must accommodate current particles plus virtual
    const unsigned int N_min = m_N + m_N_virtual;

    // compute the new size of the array using amortized growth
    unsigned int N_max = m_N_max;
    if (N_min > N_max)
        {
        while (N_min > N_max)
            {
            N_max = ((unsigned int)(((float)N_max) * resize_factor)) + 1;
            }
        reallocate(N_max);
        }

    notifyNumVirtual();

    return first_idx;
    }

#ifdef ENABLE_MPI

MPI_Datatype mpcd::ParticleData::getElementMPIDatatype() const
    {
    return m_mpi_pdata_element;
    }

/*!
 * \param out Buffer into which particle data is packed
 * \param mask Mask for \a m_comm_flags to determine if communication is necessary
 * \param timestep Current timestep
 *
 * Packs all particles where the communication flags are bitwise AND against \a mask
 * into a buffer and removes them from the particle data arrays. The output buffer
 * is automatically resized to accommodate the data.
 *
 * \post The particle data arrays remain compact, but is not guaranteed to retain its current order.
 */
void mpcd::ParticleData::removeParticles(GPUVector<mpcd::detail::pdata_element>& out,
                                         unsigned int mask,
                                         uint64_t timestep)
    {
    if (m_N_virtual > 0)
        {
        m_exec_conf->msg->error() << "MPCD particles cannot be removed with virtual particles set."
                                  << std::endl;
        throw std::runtime_error("MPCD particles cannot be removed with virtual particles set");
        }

    // partition the remove / keep particle indexes
    // this makes it so that all particles we remove are at the front in the order they were in the
    // arrays and all particles to be removed are at the end of the array in reverse order of their
    // original sorting
    unsigned int n_remove(0);
        {
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags,
                                               access_location::host,
                                               access_mode::read);
        ArrayHandle<unsigned int> h_remove_ids(m_remove_ids,
                                               access_location::host,
                                               access_mode::overwrite);
        unsigned int keep_addr = m_N;
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            if (h_comm_flags.data[idx] & mask)
                {
                h_remove_ids.data[n_remove++] = idx;
                }
            else
                {
                h_remove_ids.data[--keep_addr] = idx;
                }
            }
        }

    // resize buffer and remove the particles, using backfilling of holes popped off end of arrays
    out.resize(n_remove);
        {
        ArrayHandle<mpcd::detail::pdata_element> h_out(out,
                                                       access_location::host,
                                                       access_mode::overwrite);
        ArrayHandle<unsigned int> h_remove_idx(m_remove_ids,
                                               access_location::host,
                                               access_mode::read);

        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags,
                                               access_location::host,
                                               access_mode::readwrite);

        for (unsigned int idx = 0; idx < n_remove; ++idx)
            {
            // pack the current particle
            const unsigned int remove_pid = h_remove_idx.data[idx];
            mpcd::detail::pdata_element p;
            p.pos = h_pos.data[remove_pid];
            p.vel = h_vel.data[remove_pid];
            p.tag = h_tag.data[remove_pid];
            p.comm_flag = h_comm_flags.data[remove_pid];
            h_out.data[idx] = p;

            // backfill with a keep particle from the end of the array
            const unsigned int fill_idx = idx + n_remove;
            if (fill_idx < m_N)
                {
                const unsigned int fill_pid = h_remove_idx.data[fill_idx];
                h_pos.data[remove_pid] = h_pos.data[fill_pid];
                h_vel.data[remove_pid] = h_vel.data[fill_pid];
                h_tag.data[remove_pid] = h_tag.data[fill_pid];
                h_comm_flags.data[remove_pid] = h_comm_flags.data[fill_pid];
                }
            }
        }

    // resize self down (just changes value of m_N since removing)
    const unsigned int n_keep = m_N - n_remove;
    resize(n_keep);

    notifySort(timestep);
    }

/*!
 * \param in List of particle data elements to fill the particle data with
 * \param mask Bitmask for direction send occurred
 * \param timestep Current timestep
 */
void mpcd::ParticleData::addParticles(const GPUVector<mpcd::detail::pdata_element>& in,
                                      unsigned int mask,
                                      uint64_t timestep)
    {
    if (m_N_virtual > 0)
        {
        m_exec_conf->msg->error() << "MPCD particles cannot be added with virtual particles set."
                                  << std::endl;
        throw std::runtime_error("MPCD particles cannot be added with virtual particles set");
        }

    unsigned int num_add_ptls = (unsigned int)in.size();

    unsigned int old_nparticles = m_N;
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // resize particle data using amortized O(1) array resizing
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags,
                                               access_location::host,
                                               access_mode::readwrite);

        // add new particles at the end
        ArrayHandle<mpcd::detail::pdata_element> h_in(in, access_location::host, access_mode::read);
        unsigned int n = old_nparticles;
        for (unsigned int i = 0; i < num_add_ptls; ++i)
            {
            const mpcd::detail::pdata_element& p = h_in.data[i];
            h_pos.data[n] = p.pos;
            h_vel.data[n] = p.vel;
            h_tag.data[n] = p.tag;
            h_comm_flags.data[n] = p.comm_flag & ~mask; // unset the bitmask after communication
            n++;
            }
        }

    // cache is invalid because particles migrated, sort signal is tripped because adding particles
    // is like reordering
    invalidateCellCache();
    notifySort(timestep);
    }

#ifdef ENABLE_HIP
/*!
 * \param out Buffer into which particle data is packed
 * \param mask Mask for \a m_comm_flags to determine if communication is necessary
 * \param timestep Current timestep
 *
 * Packs all particles where the communication flags are bitwise AND against \a mask
 * into a buffer and removes them from the particle data arrays using the GPU.
 * The output buffer is automatically resized to accommodate the data.
 *
 * \post The particle data arrays remain compact.
 */
void mpcd::ParticleData::removeParticlesGPU(GPUVector<mpcd::detail::pdata_element>& out,
                                            unsigned int mask,
                                            uint64_t timestep)
    {
    if (m_N_virtual > 0)
        {
        m_exec_conf->msg->error() << "MPCD particles cannot be removed with virtual particles set."
                                  << std::endl;
        throw std::runtime_error("MPCD particles cannot be removed with virtual particles set");
        }

    // quit early if there are no particles to remove
    if (m_N == 0)
        {
        out.resize(0);
        return;
        }

        // flag particles that have left
        {
        ArrayHandle<unsigned char> d_remove_flags(m_remove_flags,
                                                  access_location::device,
                                                  access_mode::overwrite);
        ArrayHandle<unsigned int> d_comm_flags(m_comm_flags,
                                               access_location::device,
                                               access_mode::read);

        m_mark_tuner->begin();
        mpcd::gpu::mark_removed_particles(d_remove_flags.data,
                                          d_comm_flags.data,
                                          mask,
                                          m_N,
                                          m_mark_tuner->getParam()[0]);
        m_mark_tuner->end();
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

        // use cub to partition the particle indexes and count the number to remove
        {
        ArrayHandle<unsigned char> d_remove_flags(m_remove_flags,
                                                  access_location::device,
                                                  access_mode::read);
        ArrayHandle<unsigned int> d_remove_ids(m_remove_ids,
                                               access_location::device,
                                               access_mode::overwrite);

        // size temporary storage
        void* d_tmp = NULL;
        size_t tmp_bytes = 0;
        mpcd::gpu::partition_particles(d_tmp,
                                       tmp_bytes,
                                       d_remove_flags.data,
                                       d_remove_ids.data,
                                       m_num_remove.getDeviceFlags(),
                                       m_N);

        // partition particles to keep
        ScopedAllocation<unsigned char> d_tmp_alloc(m_exec_conf->getCachedAllocator(),
                                                    (tmp_bytes > 0) ? tmp_bytes : 1);
        d_tmp = (void*)d_tmp_alloc();
        mpcd::gpu::partition_particles(d_tmp,
                                       tmp_bytes,
                                       d_remove_flags.data,
                                       d_remove_ids.data,
                                       m_num_remove.getDeviceFlags(),
                                       m_N);

        // check for errors after the partitioning is completed
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // resize the output buffer large enough to hold the returned result
    const unsigned int n_remove = m_num_remove.readFlags();
    const unsigned int n_keep = m_N - n_remove;
    out.resize(n_remove);

        // remove the particles and compact down the current array
        {
        // access output array
        ArrayHandle<mpcd::detail::pdata_element> d_out(out,
                                                       access_location::device,
                                                       access_mode::overwrite);

        // access particle data arrays to read from
        ArrayHandle<Scalar4> d_pos(m_pos, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_vel, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_tag, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_comm_flags(m_comm_flags,
                                               access_location::device,
                                               access_mode::readwrite);

        ArrayHandle<unsigned int> d_remove_ids(m_remove_ids,
                                               access_location::device,
                                               access_mode::read);

        m_remove_tuner->begin();
        mpcd::gpu::remove_particles(d_out.data,
                                    d_pos.data,
                                    d_vel.data,
                                    d_tag.data,
                                    d_comm_flags.data,
                                    d_remove_ids.data,
                                    n_remove,
                                    m_N,
                                    m_remove_tuner->getParam()[0]);
        m_remove_tuner->end();
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    resize(n_keep);

    notifySort(timestep);
    }

/*!
 * \param in List of particle data elements to fill the particle data with
 * \param mask Bitmask for direction send occurred
 * \param timestep Current timestep
 */
void mpcd::ParticleData::addParticlesGPU(const GPUVector<mpcd::detail::pdata_element>& in,
                                         unsigned int mask,
                                         uint64_t timestep)
    {
    if (m_N_virtual > 0)
        {
        m_exec_conf->msg->error() << "MPCD particles cannot be added with virtual particles set."
                                  << std::endl;
        throw std::runtime_error("MPCD particles cannot be added with virtual particles set");
        }

    unsigned int old_nparticles = m_N;
    unsigned int num_add_ptls = (unsigned int)in.size();
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // amortized resizing of particle data
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> d_pos(m_pos, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_vel, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_tag, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_comm_flags(m_comm_flags,
                                               access_location::device,
                                               access_mode::readwrite);

        // Access input array
        ArrayHandle<mpcd::detail::pdata_element> d_in(in,
                                                      access_location::device,
                                                      access_mode::read);

        // add new particles on GPU
        m_add_tuner->begin();
        mpcd::gpu::add_particles(old_nparticles,
                                 num_add_ptls,
                                 d_pos.data,
                                 d_vel.data,
                                 d_tag.data,
                                 d_comm_flags.data,
                                 d_in.data,
                                 mask,
                                 m_add_tuner->getParam()[0]);
        m_add_tuner->end();
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // cache is invalid because particles migrated, sort signal is tripped because adding particles
    // is like reordering
    invalidateCellCache();
    notifySort(timestep);
    }
#endif // ENABLE_HIP

void mpcd::ParticleData::setupMPI(std::shared_ptr<DomainDecomposition> decomposition)
    {
    // set domain decomposition
    if (decomposition)
        m_decomposition = decomposition;

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_mark_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_pdata_mark"));
        m_remove_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                              m_exec_conf,
                                              "mpcd_pdata_remove"));
        m_add_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                           m_exec_conf,
                                           "mpcd_pdata_add"));
        m_autotuners.insert(m_autotuners.end(), {m_mark_tuner, m_remove_tuner, m_add_tuner});
        }
#endif // ENABLE_HIP

        // create new data type for the pdata_element
        {
        const MPI_Datatype mpi_scalar4 = m_exec_conf->getMPIConfig()->getScalar4Datatype();
        int blocklengths[] = {1, 1, 1, 1};
        MPI_Datatype types[] = {mpi_scalar4, mpi_scalar4, MPI_UNSIGNED, MPI_UNSIGNED};
        MPI_Aint offsets[] = {offsetof(mpcd::detail::pdata_element, pos),
                              offsetof(mpcd::detail::pdata_element, vel),
                              offsetof(mpcd::detail::pdata_element, tag),
                              offsetof(mpcd::detail::pdata_element, comm_flag)};
        MPI_Datatype tmp;
        MPI_Type_create_struct(4, blocklengths, offsets, types, &tmp);
        MPI_Type_create_resized(tmp, 0, sizeof(mpcd::detail::pdata_element), &m_mpi_pdata_element);
        MPI_Type_commit(&m_mpi_pdata_element);
        }
    }
#endif // ENABLE_MPI

/*!
 * \param m Python module to export to
 */
namespace mpcd
    {
namespace detail
    {
void export_ParticleData(pybind11::module& m)
    {
    pybind11::class_<mpcd::ParticleData, std::shared_ptr<mpcd::ParticleData>>(m, "MPCDParticleData")
        .def(pybind11::init<unsigned int,
                            std::shared_ptr<const BoxDim>,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            std::shared_ptr<ExecutionConfiguration>>())
        .def(pybind11::init<unsigned int,
                            std::shared_ptr<const BoxDim>,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            std::shared_ptr<ExecutionConfiguration>,
                            std::shared_ptr<DomainDecomposition>>())
        .def_property_readonly("N", &mpcd::ParticleData::getN)
        .def_property_readonly("N_global", &mpcd::ParticleData::getNGlobal)
        .def("getPosition", &mpcd::ParticleData::getPosition)
        .def("getType", &mpcd::ParticleData::getType)
        .def("getVelocity", &mpcd::ParticleData::getVelocity)
        .def("getTag", &mpcd::ParticleData::getTag)
        .def_property_readonly("n_types", &mpcd::ParticleData::getNTypes)
        .def_property_readonly("types", &mpcd::ParticleData::getTypeNames)
        .def("getNameByType", &mpcd::ParticleData::getNameByType)
        .def("getTypeByName", &mpcd::ParticleData::getTypeByName)
        .def_property("mass", &mpcd::ParticleData::getMass, &mpcd::ParticleData::setMass);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
