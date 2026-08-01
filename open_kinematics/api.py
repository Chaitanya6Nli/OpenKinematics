"""
Public API facade for OpenKinematics.

This module provides a unified user-facing interface while delegating all robotics computations
to the underlying robot, solver, and visualization layers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.solvers.geometric_ik import planar_2r_ik, scara_ik

from open_kinematics.solvers.jacobian_solver import geometric_jacobian, pseudoinverse_ik

from open_kinematics.visualization import styles
from open_kinematics.visualization.plot_2d import plot_2d
from open_kinematics.visualization.plot_3d import plot_3d

from open_kinematics.visualization.workspace import plot_workspace as _visualize_workspace
from open_kinematics.visualization.animate import animate as _animate_trajectory

class Robot:
    """
    Public API facade for OpenKinematics.

    This class wraps an existing robot model and provides a unified interface for forward kinematics, inverse kinematics,
    Jacobian computation, and visualization without exposing the underlying implementation details.
    """

    # Maps each supported robot type to the appropriate plotting function.
    _PLOT_DISPATCH: dict[type[BaseRobot], Callable] = {PlanarRobot: plot_2d, ScaraRobot: plot_2d, ArticulatedRobot: plot_3d}

    # Maps (robot type, IK method) pairs to the private adapter method
    # name, resolved dynamically using getattr().
    _IK_DISPATCH = {
        (PlanarRobot, "geometric"): "_ik_planar_geometric",
        (ScaraRobot, "geometric"): "_ik_scara_geometric",
        (ArticulatedRobot, "pseudoinverse"): "_ik_articulated_pseudoinverse",
    }

    def __init__(self, robot_instance):
        """
        Construct a Robot facade around an existing robot model.

        :param robot_instance: Robot model instance to wrap
        """
        self._robot: BaseRobot = robot_instance

    # Factory Methods

    @classmethod
    def planar(cls, link_lengths):
        """
        Create a PlanarRobot facade.

        :param link_lengths: Sequence of link lengths defining the planar manipulator.
        :return: Robot facade wrapping a ``PlanarRobot`` instance.
        :raises InvalidDHParameters: Propagated if the supplied link lengths are invalid.
        """
        return cls(PlanarRobot(link_lengths))

    @classmethod
    def scara(cls, link_lengths, d_range):
        """
        Create a ScaraRobot facade.

        :param link_lengths: Sequence containing the two planar link lengths.
        :param d_range: Permitted range of motion for the prismatic joint.
        :return: Robot facade wrapping a ``ScaraRobot`` instance.
        :raises InvalidDHParameters: Propagated if the supplied robot parameters are invalid.
        """
        return cls(ScaraRobot(link_lengths, d_range))

    @classmethod
    def articulated(cls, dh_table):
        """
        Create an ArticulatedRobot facade.

        :param dh_table: DH parameter table describing the robot.
        :return: Robot facade wrapping an ``ArticulatedRobot`` instance.
        :raises InvalidDHParameters: Propagated if the supplied DH table is malformed.
        """
        return cls(ArticulatedRobot(dh_table))

    # Forward Kinematics

    def fk(self, joint_values, return_all=False):
        """
        Compute forward kinematics for the wrapped robot.

        :param joint_values: Joint values for the robot configuration.
        :param return_all: If ``True``, return every intermediate homogeneous transformation matrix.
        :return: End-effector homogeneous transformation matrix when ``return_all`` is ``False``.
            Otherwise, a list containing every intermediate homogeneous transformation matrix.
        :raises TypeError: Propagated from ``BaseRobot.forward_kinematics()`` if ``joint_values`` has an invalid type.
        :raises ValueError: Propagated from ``BaseRobot.forward_kinematics()`` if the number of supplied joint values does not match the robot.
        :raises JointLimitViolation: Propagated from ``BaseRobot.forward_kinematics()`` if a joint exceeds its configured limits.
        """
        return self._robot.forward_kinematics(joint_values, return_all=return_all)

    # Jacobian

    def jacobian(self, joint_values):
        """
        Compute the geometric Jacobian of the wrapped robot.

        :param joint_values: Joint values describing the robot configuration.
        :return: The 6xn geometric Jacobian matrix.
        :raises TypeError: Propagated from ``forward_kinematics()`` if ``joint_values`` has an invalid type.
        :raises ValueError: Propagated from ``forward_kinematics()`` if the number of supplied joint values is incorrect.
        :raises JointLimitViolation: Propagated from ``forward_kinematics()`` if a joint exceeds its configured limits.
        """
        transforms = self._robot.forward_kinematics(joint_values, return_all=True)
        return  geometric_jacobian(transforms, self._robot.joint_types)

    # Plotting

    def plot(self, joint_values, ax=None, **plot_kwargs):
        """
        Visualize the wrapped robot configuration.

        :param joint_values: Joint values describing the robot configuration.
        :param ax: Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
        :param plot_kwargs: Additional keyword arguments forwarded to the visualization function (for example trajectory or view).
        :return: Tuple containing the created figure and axes.
        :raises TypeError:  If the wrapped robot type is unsupported,
            or propagated from ``forward_kinematics()`` if ``joint_values`` has an invalid type.
        :raises ValueError: Propagated from ``forward_kinematics()`` if the number of supplied joint values is incorrect.
        :raises JointLimitViolation: Propagated from ``forward_kinematics()`` if a joint exceeds its configured limits.
        """
        renderer = self._PLOT_DISPATCH.get(type(self._robot))

        if renderer is None:
            raise TypeError(f"Unsupported robot type: {type(self._robot).__name__}")

        return renderer(self._robot, joint_values, ax=ax, **plot_kwargs)

    # Validation Helper

    def _validate_target_pose(self, target_pose):
        """
        Validate and normalize a target homogeneous transformation.

        :param target_pose: Target pose supplied as either a NumPy array or a nested Python sequence.
        :return: Target pose converted to a NumPy array.
        :raises ValueError: If the supplied pose does not have shape ``(4, 4)``.
        """
        target_pose = np.asarray(target_pose, dtype=float)
        if not target_pose.shape == (4, 4):
            raise ValueError(f"target_pose must be a 4x4 homogeneous transform, got shape {target_pose.shape}.")
        return target_pose

    # Extraction Helpers

    def _extract_xy(self, T):
        x = T[0, 3]
        y = T[1, 3]
        return (x, y)

    def _extract_scara_pose(self, T):
        """
        Extract the pose representation required by the SCARA geometric inverse kinematics solver.

        :param T: Homogeneous transformation matrix.
        :return: Tuple ``(x, y, z, phi)`` containing the end-effector position and orientation.
        """
        x, y = self._extract_xy(T)
        z = T[2, 3]
        phi = np.arctan2(T[1, 0], T[0, 0])
        return (x, y, z, phi)

    def _ik_planar_geometric(self, target_pose, **kwargs):
        """
        Dispatch geometric inverse kinematics for a planar robot.

        :param target_pose: Target homogeneous transformation matrix.
        :return: List containing the two geometric inverse kinematics solutions.
        :raises IKNoSolution: Propagated from ``planar_2r_ik()`` if the target position is outside the robot workspace.
        """
        x, y = self._extract_xy(target_pose)
        l1, l2 = self._robot.link_lengths
        return planar_2r_ik(l1, l2, x, y)

    def _ik_scara_geometric(self, target_pose, **kwargs):
        """
        Dispatch geometric inverse kinematics for a SCARA robot.

        :param target_pose: Target homogeneous transformation matrix.
        :return: List containing the two geometric inverse kinematics solutions.
        :raises IKNoSolution: Propagated from ``scara_ik()`` if no valid inverse kinematics solution exists.
        """
        x, y, z, phi = self._extract_scara_pose(target_pose)
        return scara_ik(self._robot, x, y, z, phi)

    def _ik_articulated_pseudoinverse(self, target_pose, **kwargs):
        """
        Dispatch pseudoinverse inverse kinematics for an articulated robot.

        :param target_pose: Target homogeneous transformation matrix.
        :param kwargs: Optional solver settings including ``q0``, ``tol``, ``max_iter``, and ``step_size``.
        :return: Converged joint values.
        :raises ValueError: If the required initial guess ``q0`` is not supplied.
        :raises IKNoSolution: Propagated from ``pseudoinverse_ik()`` if the solver fails to converge or no valid solution exists.
        """
        q0 = kwargs.get("q0")
        if q0 is None:
            raise ValueError("pseudoinverse IK requires an initial guess 'q0'.")

        tol = kwargs.get("tol", 1e-6)
        max_iter = kwargs.get("max_iter", 1000)
        step_size = kwargs.get("step_size", 0.1)

        return pseudoinverse_ik(self._robot, target_pose, q0, tol, max_iter, step_size)

    # Inverse Kinematics

    def ik(self, target_pose, method='geometric', **kwargs):
        """
        Compute inverse kinematics for the wrapped robot.

        :param target_pose: Desired homogeneous transformation matrix.
        :param method: Inverse kinematics method to use.
        :param kwargs: Additional arguments required by the selected solver.
        :return: For geometric IK, returns a list containing the valid joint solutions.
            For pseudoinverse IK, returns a single converged joint configuration.
        :raises ValueError: If ``target_pose`` is not a valid 4x4 homogeneous transformation matrix,
            or if the requested robot type and inverse kinematics method combination is unsupported.
        """
        target_pose = self._validate_target_pose(target_pose)
        key = (type(self._robot), method)
        method_name = self._IK_DISPATCH.get(key)
        if method_name is None:
            raise ValueError(f"Unsupported combination: {type(self._robot)} with method '{method}'.")

        handler = getattr(self, method_name)
        return handler(target_pose, **kwargs)

    # Workspace Visualization

    def plot_workspace(self, joint_values=None, num_samples=8000, seed=None, ax=None):
        """
        Visualize the reachable workspace of the wrapped robot.

        :param joint_values: Reserved for future workspace-overlay support.
        :param num_samples: Number of random workspace samples.
        :param seed: Optional random seed.
        :param ax: Existing Matplotlib axes.
        :return: Tuple (fig, ax).
        :raises TypeError: Propagated from ``plot_workspace()`` if the robot type is unsupported.
        :raises ValueError: Propagated if ``num_samples`` is not a positive integer.
        """
        return _visualize_workspace(self._robot, joint_values=joint_values, num_samples=num_samples, seed=seed, ax=ax)

    # Animate

    def animate(self, trajectory, ax=None, interval=styles.ANIMATION_INTERVAL_MS, **plot_kwargs):
        """
        Animate the wrapped robot following a sequence of joint configurations.

        :param trajectory: Non-empty sequence of joint configurations.
        :param ax: Existing Matplotlib axes.
        :param interval: Delay between animation frames.
        :param plot_kwargs: Additional keyword arguments forwarded to the visualization function.
        :return: Tuple (fig, ax, anim).
        :raises TypeError: Propagated from ``animate()`` if the robot type is unsupported,
            or if a trajectory configuration contains invalid joint value types.
        :raises ValueError: Propagated from ``animate()`` if ``trajectory`` is empty,
            or if a trajectory configuration contains an invalid number of joint values.
        :raises JointLimitViolation: Propagated from ``animate()`` if any trajectory
            configuration exceeds the configured joint limits.

        .. note::
            Keep a reference to the returned ``anim`` object.
        """
        return _animate_trajectory(self._robot, trajectory, ax=ax, interval=interval, **plot_kwargs)