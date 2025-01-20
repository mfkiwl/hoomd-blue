# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Mesh bending force classes apply a force and virial to every mesh vertex
particle based on the local curvature :math:`K` of the given mesh triangulation.

.. math::

    U_\mathrm{mesh bending} = \sum_{j \in \mathrm{mesh}}
    U_{j}(K(\mathbf{r}_j))

The curvature at each vertex particle :math:`j` is determined at its position
:math:`\mathbf{r}_j`.

See Also:
   See the documentation in `hoomd.mesh.Mesh` for more information on the
   initialization of the mesh object.

.. invisible-code-block: python

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(type_ids = [0,0,0,0],
          triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]])
"""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.error import MPINotAvailableError


class BendingRigidity(MeshPotential):
    r"""Bending potential.

    Args:
        mesh (`hoomd.mesh.Mesh`): Mesh data structure constraint.

    `BendingRigidity` specifies a bending energy applied to
    all mesh triangles in ``mesh``.

    .. math::

        U(i) = \frac{1}{2} k \sum_{j \in \mathrm{Neigh}(i)}
        ( 1 - cos(\theta_{ij}))

    with :math:`\theta_{ij}` is the angle between the two normal
    directors of the bordering triangles of bond :math:`i` and :math:`j`.

    .. rubric:: Example:

    .. code-block:: python

        bending_potential = hoomd.md.mesh.bending.BendingRigidity(mesh)
        bending_potential.params["mesh"] = dict(k=10.0)

    {inherited}

    ----------

    **Members defined in** `BendingRigidity`:

    Attributes:
        params (TypeParameter[``mesh name``,dict]):
            The parameter of the bending energy for the defined mesh.
            The mesh type name defaults to "mesh". The dictionary has
            the following keys:

            * ``k`` (`float`, **required**) - bending stiffness
              :math:`[\mathrm{energy}]`
    """

    _cpp_class_name = "BendingRigidityMeshForceCompute"
    __doc__ = __doc__.replace("{inherited}", MeshPotential._doc_inherited)

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh)


class Helfrich(MeshPotential):
    r"""Helfrich bending potential.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    `Helfrich` specifies a Helfrich bending energy applied to
    all particles within the mesh.

    .. math::

        U(i) = \frac{1}{2} k \frac{1}{\sigma_i}\left( \sum_{j \in
        \mathrm{Neigh}(i)} \frac{\sigma_{ij}}{l_{ij}} (\mathbf{r}_j
        - \mathbf{r}_k) \right)^2

    with the area of the dual cell of vertex i
    :math:`\sigma_i=(\sum_{j \in \mathrm{Neigh}(i)}\sigma_{ij})/4`, the
    length of the bond in the dual lattice  :math:`\sigma_{ij}=
    r_{ij}(\text{cot}\theta_1+\text{cot}\theta_2)/2` and the angles
    :math:`\theta_1` and :math:`\theta_2` opposite to the shared bond of
    vertex :math:`i` and :math:`j`.

    See Also:
        * `Gompper and Kroll 1996 <https://doi.org/10.1051/jp1:1996246>`__
        * `Helfrich 1973 <https://doi.org/10.1515/znc-1973-11-1209>`__

    Attention:
        `Helfrich` is NOT implemented for MPI parallel execution!

    .. rubric:: Example:

    .. skip: next if(hoomd.version.mpi_enabled)

    .. code-block:: python

        helfrich_potential = hoomd.md.mesh.bending.Helfrich(mesh)
        helfrich_potential.params["mesh"] = dict(k=10.0)

    {inherited}

    ----------

    **Members defined in** `Helfrich`:

    Attributes:
        params (TypeParameter[dict]):
            The parameter of the Helfrich energy for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - bending stiffness
              :math:`[\mathrm{energy}]`

    """

    _cpp_class_name = "HelfrichMeshForceCompute"
    __doc__ = __doc__.replace("{inherited}", MeshPotential._doc_inherited)

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh)

    def _attach_hook(self):
        if self._simulation.device.communicator.num_ranks == 1:
            super()._attach_hook()
        else:
            raise MPINotAvailableError("Helfrich is not implemented for MPI")


__all__ = [
    "BendingRigidity",
    "Helfrich",
]
