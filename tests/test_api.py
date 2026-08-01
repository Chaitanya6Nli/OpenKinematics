import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from open_kinematics.api import Robot

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot

from open_kinematics.solvers.jacobian_solver import geometric_jacobian

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.planar = Robot.planar([1.0, 1.0])

        self.scara = Robot.scara([1.0, 1.0], (0.0, 0.5))

        self.dh_table = [
            {
                "theta": 0.0,
                "d": 0.3,
                "r": 0.2,
                "alpha": np.pi / 2,
            },
            {
                "theta": 0.0,
                "d": 0.0,
                "r": 0.4,
                "alpha": 0.0,
            },
            {
                "theta": 0.0,
                "d": 0.0,
                "r": 0.4,
                "alpha": 0.0,
            },
        ]

        self.articulated = Robot.articulated(self.dh_table)

    # Factory Tests

    def test_planar_factory(self):
        self.assertIsInstance(self.planar, Robot)
        self.assertIsInstance(self.planar._robot, PlanarRobot)

    def test_scara_factory(self):
        self.assertIsInstance(self.scara, Robot)
        self.assertIsInstance(self.scara._robot, ScaraRobot)

    def test_articulated_factory(self):
        self.assertIsInstance(self.articulated, Robot)
        self.assertIsInstance(self.articulated._robot, ArticulatedRobot)

    # FK Delegation Tests

    def test_planar_fk_delegation(self):
        joint_values = [0.0, 0.0]

        expected = self.planar._robot.forward_kinematics(joint_values)
        actual = self.planar.fk(joint_values)

        self.assertTrue(np.allclose(actual, expected))

    def test_scara_fk_delegation(self):
        joint_values = [0.0, 0.0, 0.2, 0.0]

        expected = self.scara._robot.forward_kinematics(joint_values)
        actual = self.scara.fk(joint_values)

        self.assertTrue(np.allclose(actual, expected))

    def test_articulated_fk_delegation(self):
        joint_values = [0.0, 0.0, 0.0]

        expected = self.articulated._robot.forward_kinematics(joint_values)
        actual = self.articulated.fk(joint_values)

        self.assertTrue(np.allclose(actual, expected))

    # return_all Delegation

    def test_fk_return_all_delegation(self):
        joint_values = [0.0, 0.0]

        expected = self.planar._robot.forward_kinematics(joint_values, return_all=True)
        actual = self.planar.fk(joint_values, return_all=True,)

        self.assertEqual(len(actual), len(expected))

        for actual_transform, expected_transform in zip(actual, expected):
            self.assertTrue(np.allclose(actual_transform, expected_transform))

    # Jacobian Delegation

    def test_planar_jacobian_delegation(self):
        joint_values = [0.0, 0.0]
        transforms = self.planar._robot.forward_kinematics(joint_values, return_all=True)

        expected = geometric_jacobian(transforms, self.planar._robot.joint_types)
        actual = self.planar.jacobian(joint_values)

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.shape, (6, 2))

    def test_scara_jacobian_delegation(self):
        joint_values = [0.0, 0.0, 0.2, 0.0]
        transforms = self.scara._robot.forward_kinematics(joint_values, return_all=True)

        expected = geometric_jacobian(transforms, self.scara._robot.joint_types)
        actual = self.scara.jacobian(joint_values)

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.shape, (6, 4))

    def test_articulated_jacobian_delegation(self):
        joint_values = [0.0, 0.0, 0.0]
        transforms = self.articulated._robot.forward_kinematics(joint_values, return_all=True)

        expected = geometric_jacobian(transforms, self.articulated._robot.joint_types)
        actual = self.articulated.jacobian(joint_values)

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.shape, (6, 3))

    # Plot Dispatch

    def test_planar_plot_dispatch(self):
        fig, ax = self.planar.plot([0.0, 0.0])
        self.assertEqual(ax.name, "rectilinear")
        plt.close(fig)

    def test_scara_plot_dispatch(self):
        fig, ax = self.scara.plot([0.0, 0.0, 0.2, 0.0])
        self.assertEqual(ax.name, "rectilinear")
        plt.close(fig)

    def test_articulated_plot_dispatch(self):
        fig, ax = self.articulated.plot([0.0, 0.0, 0.0])
        self.assertEqual(ax.name, "3d")
        plt.close(fig)

    def test_planar_plot_reuses_existing_axes(self):
        fig, ax = plt.subplots()
        _, returned_ax = self.planar.plot([0.0, 0.0], ax=ax)
        self.assertIs(returned_ax, ax)
        plt.close(fig)

    def test_articulated_plot_reuses_existing_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        _, returned_ax = self.articulated.plot([0.0, 0.0, 0.0], ax=ax)
        self.assertIs(returned_ax, ax,)
        plt.close(fig)

    # IK Tests

    def test_planar_ik_round_trip(self):
        q_true = [0.3, 0.5]
        target = self.planar.fk(q_true)
        solutions = self.planar.ik(target)

        self.assertEqual(len(solutions), 2)

        for solution in solutions:
            recovered = self.planar.fk(solution)
            self.assertTrue(np.allclose(recovered[:3, 3], target[:3, 3], atol=1e-6))

    def test_scara_ik_round_trip(self):
        q_true = [0.4, 0.6, 0.2, 0.3]
        target = self.scara.fk(q_true)
        solutions = self.scara.ik(target)

        self.assertEqual(len(solutions), 2)

        for solution in solutions:
            recovered = self.scara.fk(solution)
            self.assertTrue(np.allclose(recovered, target, atol=1e-6))

    def test_articulated_pseudoinverse_round_trip(self):
        dh_table = [
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": np.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.3, "alpha": np.pi / 2},
            {"theta": 0.0, "d": 0.2, "r": 0.2, "alpha": -np.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.2, "alpha": np.pi / 2},
            {"theta": 0.0, "d": 0.1, "r": 0.1, "alpha": 0.0},
        ]
        robot6 = Robot.articulated(dh_table)
        q_true = [0.30, -0.40, 0.25, -0.20, 0.50, -0.35]
        target = robot6.fk(q_true)
        q0 = [0.10, -0.15, 0.10, -0.10, 0.15, -0.05]
        solution = robot6.ik(target, method="pseudoinverse", q0=q0)
        recovered = robot6.fk(solution)

        self.assertTrue(np.allclose(recovered, target, atol=1e-5))

    def test_unsupported_ik_method(self):
        target = self.planar.fk([0.3, 0.5])

        with self.assertRaises(ValueError):
            self.planar.ik(target, method="pseudoinverse")

    def test_missing_q0_raises_value_error(self):
        target = self.articulated.fk([0.2, -0.5, 0.4])

        with self.assertRaises(ValueError):
            self.articulated.ik(target, method="pseudoinverse")

    def test_invalid_target_pose_shape(self):
        with self.assertRaises(ValueError):
            self.planar.ik(np.eye(3))

    def test_target_pose_accepts_list_of_lists(self):
        q_true = [0.3, 0.5]
        target = self.planar.fk(q_true)
        target_list = target.tolist()
        ndarray_solution = self.planar.ik(target)
        list_solution = self.planar.ik(target_list)

        self.assertEqual(len(ndarray_solution), len(list_solution))

        for expected, actual in zip(ndarray_solution, list_solution):
            self.assertTrue(np.allclose(expected, actual, atol=1e-6))

    def test_planar_plot_workspace_dispatch(self):
        fig, ax = self.planar.plot_workspace(num_samples=100)
        self.assertEqual(ax.name, "rectilinear")
        plt.close(fig)

    def test_scara_plot_workspace_dispatch(self):
        fig, ax = self.scara.plot_workspace(num_samples=100)
        self.assertEqual(ax.name, "rectilinear")
        plt.close(fig)

    def test_articulated_plot_workspace_dispatch(self):
        fig, ax = self.articulated.plot_workspace(num_samples=100)
        self.assertEqual(ax.name, "3d")
        plt.close(fig)

    def test_animate_returns_expected_types(self):
        fig, ax, anim = self.planar.animate([[0.0, 0.0], [0.3, 0.2]])
        self.assertIsInstance(fig, Figure)
        self.assertEqual(ax.name, "rectilinear")
        self.assertIsInstance(anim, FuncAnimation)
        plt.close(fig)

    def test_animate_empty_trajectory_raises(self):
        with self.assertRaises(ValueError):
            self.planar.animate([])

    def test_plot_workspace_unsupported_input_raises(self):
        with self.assertRaises(TypeError):
            Robot(object()).plot_workspace()

    def test_planar_plot_forwards_trajectory(self):
        trajectory = [
            [0.0, 0.0],
            [0.4, 0.3],
            [0.8, -0.2],
        ]
        fig, ax = self.planar.plot([0.5, 0.3], trajectory=trajectory)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Trajectory", labels)
        plt.close(fig)

    def test_articulated_plot_forwards_trajectory(self):
        trajectory = [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3],
            [0.4, -0.2, 0.5],
        ]
        fig, ax = self.articulated.plot([0.0, 0.0, 0.0], trajectory=trajectory)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Trajectory", labels)

        plt.close(fig)

if __name__ == "__main__":
    unittest.main()