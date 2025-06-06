# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define ParticleFilter.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    filter = hoomd.filter.All()
    other = hoomd.filter.Tags([0])
"""

from hoomd import _hoomd


class ParticleFilter(_hoomd.ParticleFilter):
    """Base class for all particle filters.

    This class provides methods common to all particle filters.

    Note:
        Users should instantiate one of the subclasses.
    """

    _doc_inherited = """
    ----------

    **Members inherited from** `ParticleFilter <hoomd.filter.ParticleFilter>`:

    .. py:method:: __hash__

        Return a hash of the filter parameters.
        `Read more... <hoomd.filter.ParticleFilter.__hash__>`

    .. py:method:: __eq__

        Test for equality between two particle filters.
        `Read more... <hoomd.filter.ParticleFilter.__eq__>`

    .. py:method:: __str__

        Format a human readable string describing the filter.
        `Read more... <hoomd.filter.ParticleFilter.__str__>`

    .. py:method:: __call__

        Evaluate the filter.
        `Read more... <hoomd.filter.ParticleFilter.__call__>`
    """

    def __hash__(self):
        """Return a hash of the filter parameters.

        .. rubric:: Example:

        .. code-block:: python

            hash(filter)
        """
        return NotImplementedError("Must implement hash for ParticleFilters.")

    def __eq__(self, other):
        """Test for equality between two particle filters.

        .. rubric:: Example:

        .. code-block:: python

            if filter == other:
                pass
        """
        raise NotImplementedError(
            "Equality between {} is not defined.".format(self.__class__)
        )

    def __str__(self):
        """Format a human readable string describing the filter.

        .. rubric:: Example:

        .. code-block:: python

            print(str(filter))
        """
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        """Evaluate the filter.

        Returns:
            list[int]: The particle tags selected by this filter.

        Note:
            This method may return tags that are only present on the local MPI
            rank. The full set of particles selected is the combination of
            these the lists across ranks with a set union operation.

        .. rubric:: Example:

        .. code-block:: python

            locally_selected_tags = filter(simulation.state)
        """
        return self._get_selected_tags(state._cpp_sys_def)
