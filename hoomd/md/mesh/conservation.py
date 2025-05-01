# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Mesh conservation force classes apply a force and virial to every mesh vertex
particle based on a global or local quantity :math:`A` of the given mesh
triangulation :math:`T`.

.. math::

    U_\mathrm{conservation} = U(A(T))

See Also:
   See the documentation in `hoomd.mesh.Mesh` for more information on the
   initialization of the mesh object.

.. invisible-code-block: python

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(type_ids = [0,0,0,0],
          triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]])
"""

from hoomd.md.mesh.potential import MeshPotential, MeshConservationPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.logging import log
import inspect


class Area(MeshConservationPotential):
    r"""Area conservation potential.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure.
        ignore_type (`bool`, *optional*): ignores mesh type if set to `True`
            and calculates the conservation energy considering all triangles in
            the mesh. Defaults to `False`.

    `Area` specifies a global area conservation energy for each
    mesh type.

    .. math::

        U_t = k \frac{ (A_t - A_{0,t} )^2}{2 \cdot A_0}

    with :math:`A_t` is the instantaneous total area of all triangles
    of type :math:`t`.

    .. rubric:: Example:

    .. code-block:: python

        area_conservation_potential = hoomd.md.mesh.conservation.Area(mesh)
        area_conservation_potential.params["mesh"] = dict(k=10.0, A0=250)

    {inherited}

    ----------

    **Members defined in** `Area`:

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the area conservation for each mesh type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - bending stiffness
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``A0`` (`float`, **required**) - targeted global surface area
              :math:`[\mathrm{length}]^2]`
    """

    _cpp_class_name = "AreaConservationMeshForceCompute"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshConservationPotential._doc_inherited))

    def __init__(self, mesh, ignore_type=False):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, A0=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh, ignore_type)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh triangulation."""
        return self._cpp_obj.getArea()


class TriangleArea(MeshPotential):
    r"""Triangle Area conservation potential.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure.

    `TriangleArea` specifies an area conservation energy by
    applying an area constraint to each triangle of the mesh.

    .. math::

        U(r) = k \sum_{i,j,k \in \mathrm{mesh{ }triangle}}
        \frac{( N_\mathrm{tri} \cdot A_{ijk} - A_0 )^2}
        {2 \cdot A_0 \cdot N_\mathrm{tri}}

    The potential sums over the area :math:`A_{ijk}` of all triplets
    :math:`i`, :math:`j` and :math:`k` that make up a triangle in the
    mesh. The number of triangles in the mesh are symbolized by
    :math:`N_\mathrm{tri}`.

    .. rubric:: Example:

    .. code-block:: python

        tringle_area_conservation_potential = \
            hoomd.md.mesh.conservation.TriangleArea(mesh)
        tringle_area_conservation_potential.params["mesh"] = dict(k=10.0, A0=250)

    {inherited}

    ----------

    **Members defined in** `TriangleArea`:

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the triangle area conservation for each mesh type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - area conservation coefficient
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``A0`` (`float`, **required**) - target surface area
              of a single triangle in the mesh
              :math:`[\mathrm{length}]^2`
    """

    _cpp_class_name = "TriangleAreaConservationMeshForceCompute"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshConservationPotential._doc_inherited))

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, A0=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh triangulation."""
        return self._cpp_obj.getArea()


class Volume(MeshConservationPotential):
    r"""Volume conservation potential.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure.
        ignore_type (`bool`, *optional*): ignores mesh type if set to `True`
            and calculates the conservation energy considering all triangles in
            the mesh. Defaults to `False`.

    :py:class:`Volume` specifies a volume constraint on the whole mesh
    surface:

    .. math::

        U(r) = k \frac{( V(r) - V_0 )^2}{2 \cdot V_0}

    .. rubric:: Example:

    .. code-block:: python

        volume = hoomd.md.mesh.conservation.Volume(mesh)
        volume.params["mesh"] = dict(k=10.0, V0=100)

    {inherited}

    ----------

    **Members defined in** `Volume`:

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the volume constraint for the defined mesh.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-3}]`

            * ``V0`` (`float`, **required**) - target volume
              :math:`[\mathrm{length}^{3}]`
    """

    _cpp_class_name = "VolumeConservationMeshForceCompute"
    __doc__ = inspect.cleandoc(__doc__).replace("{inherited}", inspect.cleandoc(MeshConservationPotential._doc_inherited))

    def __init__(self, mesh, ignore_type=False):
        params = TypeParameter(
            "params", "types", TypeParameterDict(k=float, V0=float, len_keys=1)
        )
        self._add_typeparam(params)

        super().__init__(mesh, ignore_type)

    @log(requires_run=True)
    def volume(self):
        """Volume of the mesh triangulation."""
        return self._cpp_obj.getVolume()


__all__ = [
    "Area",
    "MeshConservationPotential",
    "TriangleArea",
    "Volume",
]
