import numpy as np
import matplotlib.pyplot as plt
from open_kinematics.visualization import styles
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.robots.articulated import ArticulatedRobot

def plot_3d(robot, joint_values, ax=None, view=styles.DEFAULT_VIEW, trajectory=None):
    """
    Visualize an articulated robot configuration in three dimensions.

    :param robot: ArticulatedRobot instance to visualize.
    :param joint_values: Joint values describing the robot configuration.
    :param ax: Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    :param view: Camera preset name defined in ``styles.VIEW_PRESETS``.
    :param trajectory: Optional sequence of joint configurations.
        If provided, the end-effector trajectory is drawn.
    :return: Tuple containing the Matplotlib figure and axes.
    :raises TypeError: If ``robot`` is not an ``ArticulatedRobot`` instance, or propagated from ``forward_kinematics()``
        if ``joint_values`` has an invalid type.
    :raises ValueError: If ``view`` is not a recognized preset in ``styles.VIEW_PRESETS``,
        or propagated from ``forward_kinematics()`` if the number of supplied joint values is incorrect.
    :raises JointLimitViolation: Propagated from ``forward_kinematics()`` if a joint exceeds its configured limits.
    """

    if not isinstance(robot, ArticulatedRobot):
        raise TypeError("plot_3d() only supports Articulated Robot.")

    if view not in styles.VIEW_PRESETS:
        raise ValueError(
            f"Unknown view preset '{view}'. "
            f"Expected one of: {list(styles.VIEW_PRESETS.keys())}"
        )

    transforms = robot.forward_kinematics(joint_values, return_all=True)
    augmented_transforms = [identity_matrix(4)] + list(transforms)

    xs = []
    ys = []
    zs = []

    for transform in augmented_transforms:
        origin = transform[:3, 3]

        xs.append(origin[0])
        ys.append(origin[1])
        zs.append(origin[2])

    # Create axes if needed
    if ax is None:
        fig = plt.figure(figsize=styles.DEFAULT_FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if trajectory:
        ee_x = []
        ee_y = []
        ee_z = []

        for config in trajectory:
            traj_transforms = robot.forward_kinematics(config, return_all=True)

            ee_origin = traj_transforms[-1][:3, 3]

            ee_x.append(ee_origin[0])
            ee_y.append(ee_origin[1])
            ee_z.append(ee_origin[2])

        # Trajectory plotting
        ax.plot(ee_x, ee_y, ee_z, color=styles.TRAJECTORY_COLOR, linewidth=styles.TRAJECTORY_LINEWIDTH, linestyle=styles.TRAJECTORY_LINESTYLE, label="Trajectory")

    # Draw robot links
    ax.plot(xs, ys, zs, color=styles.LINK_COLOR, linewidth=styles.LINK_LINEWIDTH, zorder=1, label="Links")

    # Draw base
    ax.plot(xs[0], ys[0], zs[0], marker=styles.BASE_MARKER, markersize=styles.BASE_MARKERSIZE, color=styles.BASE_COLOR, linestyle="None", zorder=2, label="Base")

    # Revolute joints
    if len(xs) > 2:
        ax.plot(xs[1:-1], ys[1:-1], zs[1:-1], linestyle="None", marker=styles.JOINT_REVOLUTE_MARKER, markersize=styles.JOINT_MARKERSIZE, color=styles.JOINT_REVOLUTE_COLOR, zorder=2, label="Revolute Joint")

    # Highlight end-effector
    ax.plot(xs[-1], ys[-1], zs[-1], linestyle="None", marker=styles.END_EFFECTOR_MARKER, markersize=styles.END_EFFECTOR_MARKERSIZE, color=styles.END_EFFECTOR_COLOR, zorder=3, label="End Effector")

    # Compute coordinate-frame arrow scale
    link_lengths = []

    for i in range(len(augmented_transforms) - 1):
        p1 = augmented_transforms[i][:3, 3]
        p2 = augmented_transforms[i + 1][:3, 3]

        distance = np.linalg.norm(p2 - p1)

        if distance > 1e-9:
            link_lengths.append(distance)

    if link_lengths:
        scale = 0.15 * (sum(link_lengths) / len(link_lengths))
    else:
        scale = 0.1

    # Draw coordinate frames
    for index, transform in enumerate(augmented_transforms):
        origin = transform[:3, 3]

        x_axis = transform[:3, 0]
        y_axis = transform[:3, 1]
        z_axis = transform[:3, 2]

        linewidth = styles.FRAME_LINEWIDTH
        alpha = styles.FRAME_ALPHA

        if index == len(augmented_transforms) - 1:
            linewidth = styles.END_EFFECTOR_FRAME_LINEWIDTH
            alpha = 1.0

        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=scale, normalize=True, color=styles.FRAME_X_COLOR, linewidth=linewidth, alpha=alpha) # X-axis (Red)
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=scale, normalize=True, color=styles.FRAME_Y_COLOR, linewidth=linewidth, alpha=alpha) # Y-axis (Green)
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=scale, normalize=True, color=styles.FRAME_Z_COLOR, linewidth=linewidth, alpha=alpha) # Z-axis (Blue)

    # Camera preset
    preset = styles.VIEW_PRESETS[view]
    ax.view_init(elev=preset["elev"], azim=preset["azim"])
    ax.set_proj_type(styles.PROJECTION_TYPE)

    # Equal aspect ratio
    MIN_AXIS_RANGE = 1e-6

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    z_range = max(zs) - min(zs)

    ax.set_box_aspect((max(x_range, MIN_AXIS_RANGE), max(y_range, MIN_AXIS_RANGE), max(z_range, MIN_AXIS_RANGE)))

    # Grid and legend
    ax.grid(styles.GRID_ENABLED, alpha=styles.GRID_ALPHA, linestyle=styles.GRID_LINESTYLE)
    ax.legend(fontsize=styles.LEGEND_FONTSIZE)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax