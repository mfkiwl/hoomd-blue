# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh potential base class."""

from hoomd.md import _md
from hoomd.mesh import Mesh
from hoomd.md.force import Force
from hoomd.data.typeconverter import OnlyTypes
import hoomd
import warnings
import copy
import inspect

validate_mesh = OnlyTypes(Mesh)


class MeshPotential(Force):
    """Constructs the potential applied to a mesh.

    `MeshPotential` is the base class for all potentials applied to meshes.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    {inherited}

    ----------

    **Members defined in** `MeshPotential`:
    """

    __doc__ = inspect.cleandoc(__doc__.replace("{inherited}", inspect.cleandoc(Force._doc_inherited)))
    _doc_inherited = (
        Force._doc_inherited
        + """
    ----------

    **Members inherited from**
    `MeshPotential <hoomd.md.mesh.MeshPotential>`:

    .. py:attribute:: mesh

        Mesh data structure used to compute the bond potential.
        `Read more... <hoomd.md.mesh.MeshPotential.mesh>`
    """
    )

    def __init__(self, mesh):
        self._mesh = validate_mesh(mesh)

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if self._mesh._attached and self._simulation != self._mesh._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent mesh structure."
                f" This is happending since the force is moving to a new "
                f"simulation. To suppress the warning explicitly set new mesh.",
                RuntimeWarning,
            )
            self._mesh = copy.deepcopy(self._mesh)
        self.mesh._attach(self._simulation)

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def, self._mesh._cpp_obj
        )

    def _detach_hook(self):
        self._mesh._detach()

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, self.mesh)
            except ValueError as err:
                raise err.__class__(
                    f"For {type(self)} in TypeParameter {typeparam.name} " f"{err!s}"
                )

    @property
    def mesh(self):
        """Mesh data structure used to compute the bond potential."""
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if self._attached:
            raise RuntimeError("mesh cannot be set after calling Simulation.run().")
        mesh = validate_mesh(value)
        self._mesh = mesh


class MeshConservationPotential(MeshPotential):
    """Constructs the conservation potential applied to a mesh.

    `MeshConservationPotential` is the base class for global conservation
    potentials applied to meshes.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    __doc__ += MeshPotential._doc_inherited

    def __init__(self, mesh, ignore_type):
        super().__init__(mesh)
        self._ignore_type = ignore_type

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if self._mesh._attached and self._simulation != self._mesh._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent mesh structure."
                f" This is happending since the force is moving to a new "
                f"simulation. To suppress the warning explicitly set new mesh.",
                RuntimeWarning,
            )
            self._mesh = copy.deepcopy(self._mesh)
        self.mesh._attach(self._simulation)

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def, self._mesh._cpp_obj, self._ignore_type
        )
