# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD updaters.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

from . import _mpcd
from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
from hoomd.data.typeconverter import OnlyTypes, positive_real

import math


class ReverseNonequilibriumShearFlow(Updater):
    r"""MPCD reverse nonequilibrium shear flow.

    Args:
        trigger (hoomd.trigger.trigger_like): Trigger to swap momentum.

        num_swaps (int): Maximum number of times to swap momentum per update.

        slab_width (float): Width of momentum-exchange slabs.

        target_momentum (float): Target momentum for swapped particles. This
            argument has a default value of infinity but can be
            redefined with positive real numbers only.

    This updater generates a bidirectional shear flow in *x* by imposing a
    momentum flux on the system in *y*.
    There are two exchange slabs separated by a distance of :math:`L_y/2`. The
    edges of these slabs are located at (:math:`-L_y/2`, :math:`-L_y/2` +
    `slab_width`) and (:math:`0.0`, `slab_width`) along the *y*-direction
    (gradient direction) of the simulation box. On each `trigger`, particles
    are sorted into the exchange slabs based on their positions in the box.
    Particles whose *x*-component momenta are near `target_momentum`
    in the lower slab, and those near -`target_momentum` in the upper
    slab, are selected for a pairwise momentum swap. Up to `num_swaps` swaps
    are executed per update.

    The amount of momentum transferred from the lower slab to the upper slab is
    known. Therefore `summed_exchanged_momentum`, which returns the accumulated
    momentum exchanged till the current timestep, is used to calculate the
    momentum flux, :math:`j_{xy}`. The shear rate, :math:`\dot{\gamma}`, is
    also extracted as the gradient of the linear velocity profile developed
    from the flow. The viscosity can then be computed from these two quantities
    as:

    .. math::

        \eta (\dot{\gamma}) = \frac{j_{xy}}{\dot{\gamma}}.

    .. rubric:: Examples:

    In the original implementation by
    `Müller-Plathe <https://doi.org/10.1103/PhysRevE.59.4894>`_,
    only the fastest particle and the slowest particle are swapped. This is
    achieved by setting `num_swaps` to 1 and keeping `target_momentum` at its
    default value of infinity.

    .. code-block:: python

            flow = hoomd.mpcd.update.ReverseNonequilibriumShearFlow(
                trigger=1, num_swaps=1, slab_width=1
            )
            simulation.operations.updaters.append(flow)

    An alternative approach proposed by
    `Tenney and Maginn <https://doi.org/10.1063/1.3276454>`_ swaps particles
    that are instead closest to the `target_momentum`, typically requiring more
    swaps per update.

    .. code-block:: python

            flow = hoomd.mpcd.update.ReverseNonequilibriumShearFlow(
                trigger=1, num_swaps=10, slab_width=1, target_momentum=5
            )
            simulation.operations.updaters.append(flow)

    {inherited}

    ----------

    **Members defined in** `ReverseNonequilibriumShearFlow`:

    Attributes:
        trigger (hoomd.trigger.trigger_like): Trigger to swap momentum.

            .. rubric:: Example:

            .. code-block:: python

                flow.trigger = 1

        num_swaps (int): Maximum number of times to swap momentum per update.

            .. rubric:: Example:

            .. code-block:: python

                flow.num_swaps = 10

        slab_width (float): Width of momentum-exchange slabs.

            .. rubric:: Example:

            .. code-block:: python

                flow.slab_width = 1

        target_momentum (float): Target momentum for swapped particles.

            .. rubric:: Example:

            .. code-block:: python

                flow.target_momentum = 5

    """

    __doc__ = __doc__.replace("{inherited}", Updater._doc_inherited)

    # Constructor
    def __init__(self, trigger, num_swaps, slab_width, target_momentum=math.inf):
        # Call the parent constructor with the trigger parameter
        super().__init__(trigger)

        # Create a ParameterDict with the given parameters
        param_dict = ParameterDict(
            num_swaps=int(num_swaps),
            slab_width=float(slab_width),
            target_momentum=OnlyTypes(float, preprocess=positive_real),
        )
        param_dict["target_momentum"] = target_momentum

        # Update the internal parameter dictionary created by the parent class
        self._param_dict.update(param_dict)

    # Use the _attach_hook method to create the C++ version of the object
    def _attach_hook(self):
        sim = self._simulation
        self._cpp_obj = _mpcd.ReverseNonequilibriumShearFlow(
            sim.state._cpp_sys_def,
            self.trigger,
            self.num_swaps,
            self.slab_width,
            self.target_momentum,
        )

        super()._attach_hook()

    @log(category="scalar", requires_run=True)
    def summed_exchanged_momentum(self):
        r"""float: Total momentum exchanged.

        This quantity logs the total momentum exchanged between all swapped
        particle pairs. The value reported is the total exchanged momentum
        accumulated till the current timestep.

        """
        return self._cpp_obj.summed_exchanged_momentum


__all__ = [
    "ReverseNonequilibriumShearFlow",
]
