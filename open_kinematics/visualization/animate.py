import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from open_kinematics.visualization import styles
from open_kinematics.visualization.plot_2d import plot_2d
from open_kinematics.visualization.plot_3d import plot_3d

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot


# Local dispatch table to avoid coupling visualization -> api.py
_PLOT_DISPATCH = {PlanarRobot: plot_2d, ScaraRobot: plot_2d, ArticulatedRobot: plot_3d}

def animate(robot, trajectory, ax=None, interval=styles.ANIMATION_INTERVAL_MS, **plot_kwargs):
    """
    Animate a robot following a sequence of joint configurations.

    The animation is created by repeatedly calling the existing visualization functions (``plot_2d()`` or ``plot_3d()``),
    ensuring that all rendering logic is reused rather than duplicated.

    :param robot: Robot instance.
    :param trajectory: Non-empty sequence of joint configurations.
    :param ax: Existing Matplotlib axes. If ``None``, new axes are created.
    :param interval: Delay between animation frames in milliseconds.
    :param plot_kwargs: Additional keyword arguments forwarded to ``plot_2d()`` or ``plot_3d()``.
    :return: Tuple ``(fig, ax, anim)``.
    :raises TypeError: If ``robot`` is not a supported robot type, or propagated from ``plot_2d()`` / ``plot_3d()``
        if a trajectory configuration contains invalid data types.
    :raises ValueError: If ``trajectory`` is empty, or propagated from ``plot_2d()`` / ``plot_3d()``
        if a trajectory configuration has an invalid number of joint values.
    :raises JointLimitViolation: Propagated from ``plot_2d()`` / ``plot_3d()``
        if any trajectory configuration exceeds the configured joint limits.

    .. note::
        Keep a reference to the returned ``anim`` object.
        Otherwise, Matplotlib may garbage-collect it before playback.
    """

    if len(trajectory) == 0:
        raise ValueError("trajectory must contain at least one joint configuration.")

    plot_function = None

    for robot_type, renderer in _PLOT_DISPATCH.items():
        if isinstance(robot, robot_type):
            plot_function = renderer
            break

    if plot_function is None:
        raise TypeError("animate() supports only PlanarRobot, ScaraRobot, and ArticulatedRobot.")

    # Draw initial frame.
    fig, ax = plot_function(robot, trajectory[0], ax=ax, **plot_kwargs)

    def update(frame_index):
        ax.clear()
        plot_function(robot, trajectory[frame_index], ax=ax, **plot_kwargs)

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)

    return fig, ax, anim