import numpy as np
import matplotlib.pyplot as plt

from open_kinematics.visualization import styles

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot

def sample_workspace(robot, num_samples=8000, seed=None):
    """
    Sample the reachable workspace of a robot by randomly generating
    joint configurations within the configured joint limits.

    This function is robot-agnostic. It relies only on the public
    ``joint_limits`` attribute and ``forward_kinematics()`` method
    provided by every BaseRobot subclass.

    :param robot: Robot instance supporting ``joint_limits`` and ``forward_kinematics()``.
    :param num_samples: Number of random joint configurations to sample.
    :param seed: Optional random seed for reproducible sampling.
    :return: NumPy array of shape ``(num_samples, 3)``, where each row contains the end-effector position ``[x, y, z]``.
    :raises ValueError: If ``num_samples`` is not a positive integer.
    """

    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    rng = np.random.default_rng(seed)

    # Generate random joint values within configured joint limits.
    joint_samples = np.column_stack([
        rng.uniform(lower, upper, num_samples)
        for lower, upper in robot.joint_limits
    ])

    workspace_points = np.empty((num_samples, 3), dtype=np.float64)

    # Intentionally robot-agnostic.
    # Every BaseRobot subclass exposes forward_kinematics(), so no
    # isinstance() dispatch is required here.
    for i, joint_values in enumerate(joint_samples):
        transform = robot.forward_kinematics(joint_values)
        workspace_points[i] = transform[:3, 3]

    return workspace_points

def plot_workspace(robot, joint_values=None, num_samples=8000, seed=None, ax=None):
    """
    Visualize the reachable workspace of a robot.

    The workspace is estimated by randomly sampling joint configurations
    within the configured joint limits.

    :param robot: PlanarRobot, ScaraRobot, or ArticulatedRobot instance.
    :param joint_values: Reserved for future workspace-overlay support. Currently unused.
    :param num_samples: Number of random workspace samples.
    :param seed: Optional random seed for reproducible sampling.
    :param ax: Existing Matplotlib axes. If ``None``, a new figure and axes are created.
    :return: Tuple containing the Matplotlib figure and axes.
    :raises TypeError: If ``robot`` is not a supported robot type.
    :raises ValueError: Propagated from ``sample_workspace()`` if ``num_samples`` is not a positive integer.
    """

    _ = joint_values  # Reserved for future workspace overlay support.

    if not isinstance(robot, (PlanarRobot, ScaraRobot, ArticulatedRobot)):
        raise TypeError("plot_workspace() supports only PlanarRobot, ScaraRobot, and ArticulatedRobot.")

    points = sample_workspace(robot, num_samples=num_samples, seed=seed)

    # Planar / SCARA Workspace
    if isinstance(robot, (PlanarRobot, ScaraRobot)):
        if ax is None:
            fig, ax = plt.subplots(figsize=styles.DEFAULT_FIGSIZE_2D)
        else:
            fig = ax.figure

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=styles.WORKSPACE_MARKER_SIZE,
            facecolors=styles.WORKSPACE_FACE_COLOR,
            edgecolors=styles.WORKSPACE_EDGE_COLOR,
            alpha=styles.WORKSPACE_ALPHA,
            label="Workspace",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(styles.GRID_ENABLED, alpha=styles.GRID_ALPHA, linestyle=styles.GRID_LINESTYLE)

    # Articulated Workspace
    else: # isinstance(robot, ArticulatedRobot)
        if ax is None:
            fig = plt.figure(figsize=styles.DEFAULT_FIGSIZE_3D)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=styles.WORKSPACE_MARKER_SIZE,
            facecolors=styles.WORKSPACE_FACE_COLOR,
            edgecolors=styles.WORKSPACE_EDGE_COLOR,
            alpha=styles.WORKSPACE_ALPHA,
            label="Workspace",
        )

        x_range = max(float(np.ptp(points[:, 0])), 1e-6)
        y_range = max(float(np.ptp(points[:, 1])), 1e-6)
        z_range = max(float(np.ptp(points[:, 2])), 1e-6)

        ax.set_box_aspect((x_range, y_range, z_range))

        preset = styles.VIEW_PRESETS[styles.DEFAULT_VIEW]
        ax.view_init(elev=preset["elev"], azim=preset["azim"])
        ax.set_proj_type(styles.PROJECTION_TYPE)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(styles.GRID_ENABLED, alpha=styles.GRID_ALPHA, linestyle=styles.GRID_LINESTYLE)

    ax.legend(fontsize=styles.LEGEND_FONTSIZE)

    return fig, ax