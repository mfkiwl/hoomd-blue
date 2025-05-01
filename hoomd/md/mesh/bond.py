# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Mesh bond force classes apply a force and virial between every mesh vertex
particle and their neighbors based on the given mesh triangulation.

.. math::

    U_\mathrm{mesh bond} = \sum_{j \in \mathrm{mesh}} \sum_{k \in
    \mathrm{Neigh}(j)}U_{jk}(r)

The connectivity and, thus the neighbor set :math:`\mathrm{Neigh}` of each
vertex particle :math:`j` is defined by a mesh triangulation.

See Also:
   See the documentation in `hoomd.mesh.Mesh` for more information on the
   initialization of the mesh object.

In the mesh bond (j,k), :math:`r` is the length of the edge between the
mesh vertex particles :math:`r= |\mathrm{minimum\_image}(\vec{r}_k
- \vec{r}_j)|`.

Note:
   The mesh bond forces are computed over the mesh data structure and not the
   separate bond data structure. Hence, the mesh bonds are defined exclusively
   by the mesh triangulation as HOOMD-blue automatically constructs the mesh
   bond pairs based on ``triangulation`` in the `hoomd.mesh.Mesh` object.
   The bonds should **not** be defined separately in the `hoomd.State` member
   ``bond_group``!

.. rubric Per-particle energies and virials

Mesh bond force classes assign 1/2 of the potential energy to each of the
particles in the bond group:

.. math::

    U_i = \frac{1}{2} \sum_{k \in \mathrm{Neigh}(i)}U_{ik}(r)

and similarly for virials.

.. invisible-code-block: python

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(type_ids = [0,0,0,0],
          triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]])
"""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import inspect


class Harmonic(MeshPotential):
    r"""Harmonic bond potential.

    Args:
        mesh (hoomd.mesh.Mesh): Mesh data structure constraint.

    `Harmonic` computes forces, virials, and energies on all mesh bonds
    in ``mesh`` with the harmonic potential (see `hoomd.md.bond.Harmonic`).

    .. rubric:: Example:

    .. code-block:: python

        harmonic = hoomd.md.mesh.bond.Harmonic(mesh)
        harmonic.params["mesh"] = dict(k=10.0, r0=1.0)

    {inherited}

    ----------

    **Members defined in** `Harmonic`:

    Attributes:
        params (TypeParameter[``mesh name``,dict]):
            The parameter of the harmonic bonds for the defined mesh.
            The mesh type name defaults to "mesh". The dictionary has
            the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - rest length
              :math:`[\mathrm{length}]`
    """

    _cpp_class_name = "PotentialMeshBondHarmonic"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshPotential._doc_inherited))

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, r0=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh)


class FENEWCA(MeshPotential):
    r"""FENE and WCA bond potential.

    Args:
        mesh (hoomd.mesh.Mesh): Mesh data structure constraint.

    `FENEWCA` computes forces, virials, and energies on all mesh bonds
    in ``mesh`` with the harmonic potential (see `hoomd.md.bond.FENEWCA`).

    .. rubric:: Example:

    .. code-block:: python

        bond_potential = hoomd.md.mesh.bond.FENEWCA(mesh)
        bond_potential.params["mesh"] = dict(
            k=10.0, r0=1.0, epsilon=0.8, sigma=1.2, delta=0.0
        )

    {inherited}

    ----------

    **Members defined in** `FENEWCA`:

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the FENEWCA potential bonds.
            The mesh type name defaults to "mesh". The dictionary has
            the following keys:

            * ``k`` (`float`, **required**) - attractive force strength
              :math:`k` :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.

            * ``r0`` (`float`, **required**) - size parameter
              :math:`r_0` :math:`[\mathrm{length}]`.

            * ``epsilon`` (`float`, **required**) - repulsive force strength
              :math:`\varepsilon` :math:`[\mathrm{energy}]`.

            * ``sigma`` (`float`, **required**) - repulsive force interaction
              width :math:`\sigma` :math:`[\mathrm{length}]`.

            * ``delta`` (`float`, **required**) - radial shift :math:`\Delta`
              :math:`[\mathrm{length}]`.
    """

    _cpp_class_name = "PotentialMeshBondFENE"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshPotential._doc_inherited))

    def __init__(self, mesh):
        params = TypeParameter(
            "params",
            "types",
            TypeParameterDict(
                k=float, r0=float, epsilon=float, sigma=float, delta=float, len_keys=1
            ),
        )
        self._add_typeparam(params)

        super().__init__(mesh)


class Tether(MeshPotential):
    r"""Tethering bond potential.

    Args:
        mesh (hoomd.mesh.Mesh): Mesh data structure constraint.

    `Tether` computes forces, virials, and energies on all mesh bonds
    in ``mesh`` with the harmonic potential (see `hoomd.md.bond.Tether`).

    .. rubric:: Example:

    .. code-block:: python

        bond_potential = hoomd.md.mesh.bond.Tether(mesh)
        bond_potential.params["mesh"] = dict(
            k_b=10.0, l_min=0.9, l_c1=1.2, l_c0=1.8, l_max=2.1
        )

    {inherited}

    ----------

    **Members defined in** `Tether`:

    Attributes:
        params (TypeParameter[``mesh name``,dict]):
            The parameter of the Tether bonds for the defined mesh.
            The mesh type name defaults to "mesh". The dictionary has
            the following keys:

            * ``k_b`` (`float`, **required**) - bond stiffness
              :math:`[\mathrm{energy}]`

            * ``l_min`` (`float`, **required**) - minimum bond length
              :math:`[\mathrm{length}]`

            * ``l_c1`` (`float`, **required**) - cutoff distance of repulsive
              part :math:`[\mathrm{length}]`

            * ``l_c0`` (`float`, **required**) - cutoff distance of attractive
              part :math:`[\mathrm{length}]`

            * ``l_max`` (`float`, **required**) - maximum bond length
              :math:`[\mathrm{length}]`
    """

    _cpp_class_name = "PotentialMeshBondTether"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshPotential._doc_inherited))

    def __init__(self, mesh):
        params = TypeParameter(
            "params",
            "types",
            TypeParameterDict(
                k_b=float, l_min=float, l_c1=float, l_c0=float, l_max=float, len_keys=1
            ),
        )
        self._add_typeparam(params)

        super().__init__(mesh)


__all__ = [
    "FENEWCA",
    "Harmonic",
    "Tether",
]
