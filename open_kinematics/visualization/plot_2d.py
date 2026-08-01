import matplotlib.pyplot as plt
from open_kinematics.visualization import styles
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot

def plot_2d(robot, joint_values, ax=None, trajectory=None):
    """
    Visualize a planar or SCARA robot configuration in two dimensions.

    :param robot: PlanarRobot or ScaraRobot instance to visualize.
    :param joint_values: Joint values describing the robot configuration.
    :param ax: Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    :param trajectory: Optional sequence of joint configurations.
        If provided, the end-effector trajectory is drawn.
    :return: Tuple containing the Matplotlib figure and axes.
    :raises TypeError: If ``robot`` is not a supported robot type, or propagated from ``get_joint_positions()``
        if ``joint_values`` has an invalid type.
    :raises ValueError: Propagated from ``get_joint_positions()`` if the number of supplied joint values is incorrect.
    :raises JointLimitViolation: Propagated from ``get_joint_positions()`` if a joint exceeds its configured limits.
    """

    if not isinstance(robot, (PlanarRobot, ScaraRobot)):
        raise TypeError("plot_2d() supports only PlanarRobot and ScaraRobot.")

    # Get joint positions from the robot
    points = robot.get_joint_positions(joint_values)

    # Project to XY plane
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=styles.DEFAULT_FIGSIZE_2D)
    else:
        fig = ax.figure

    if trajectory:
        ee_x = []
        ee_y = []

        for config in trajectory:
            traj_points = robot.get_joint_positions(config)

            ee_x.append(traj_points[-1][0])
            ee_y.append(traj_points[-1][1])

        # Trajectory plotting
        ax.plot(ee_x, ee_y, color=styles.TRAJECTORY_COLOR, linewidth=styles.TRAJECTORY_LINEWIDTH, linestyle=styles.TRAJECTORY_LINESTYLE, label="Trajectory")

    # Draw links
    ax.plot(xs, ys, linewidth=styles.LINK_LINEWIDTH, color=styles.LINK_COLOR, label="Links")

    # Draw base
    ax.plot(xs[0], ys[0], marker=styles.BASE_MARKER, markersize=styles.BASE_MARKERSIZE, color=styles.BASE_COLOR, linestyle="None", label="Base")

    # Draw joints according to joint type
    # points[0] is the base origin.
    # joint_types[0] corresponds to points[1].
    revolute_label_added = False
    prismatic_label_added = False
    for i, joint_type in enumerate(robot.joint_types):
        x = xs[i + 1]
        y = ys[i + 1]
        # Skip end-effector point (drawn separately below)
        if i + 1 == len(points) - 1:
            continue

        if joint_type == "revolute":
            label = None
            if not revolute_label_added:
                label = "Revolute Joint"
                revolute_label_added = True
            ax.plot(x, y, marker=styles.JOINT_REVOLUTE_MARKER, markersize=styles.JOINT_MARKERSIZE, color=styles.JOINT_REVOLUTE_COLOR, linestyle="None", label=label)

        elif joint_type == "prismatic":
            label = None
            if not prismatic_label_added:
                label = "Prismatic Joint"
                prismatic_label_added = True
            ax.plot(x, y, marker=styles.JOINT_PRISMATIC_MARKER, markersize=styles.JOINT_MARKERSIZE, color=styles.JOINT_PRISMATIC_COLOR, linestyle="None", label=label)

    # Highlight end-effector
    ax.plot(xs[-1], ys[-1], marker=styles.END_EFFECTOR_MARKER, markersize=styles.END_EFFECTOR_MARKERSIZE, color=styles.END_EFFECTOR_COLOR, linestyle="None", label="End Effector")

    # Legend and grid
    ax.legend(fontsize=styles.LEGEND_FONTSIZE)
    ax.grid(styles.GRID_ENABLED, alpha=styles.GRID_ALPHA, linestyle=styles.GRID_LINESTYLE)

    # Axis formatting
    ax.set_aspect("equal", adjustable='box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax