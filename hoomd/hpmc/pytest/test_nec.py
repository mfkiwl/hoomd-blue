# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd

def test_nec(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=["A"])
    simulation = simulation_factory(snap)
    simulation.operations.integrator = hoomd.hpmc.nec.integrate.Sphere()
    simulation.run(10)

