# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Apply external fields to HPMC simulations."""

from hoomd.operation import _HOOMDBaseObject


class ExternalField(_HOOMDBaseObject):
    """Base class external field.

    Provides common methods for all external field subclasses.

    Note:
        Users should use the subclasses and not instantiate `ExternalField`
        directly.
    """
