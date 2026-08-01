import unittest
import matplotlib
# Use a non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy

from open_kinematics.visualization import styles
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.visualization.plot_3d import plot_3d


class TestPlot3D(unittest.TestCase):

    def setUp(self):
        self.dh_table = [
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.5, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.4, "r": 0.0, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.0, "alpha": -numpy.pi / 2},
            {"theta": 0.0, "d": 0.2, "r": 0.0, "alpha": 0.0},
        ]

        self.robot = ArticulatedRobot(self.dh_table)
        self.joint_values = [0.4, -0.7, 0.8, -0.5, 0.6, -0.3]

        self.trajectory = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3, 0.0, 0.1, -0.2],
            [0.4, -0.3, 0.5, -0.2, 0.3, 0.0],
        ]

    def tearDown(self):
        plt.close("all")

    def test_returns_figure_and_axes(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes3D)

    def test_reuses_existing_axes(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        returned_fig, returned_ax = plot_3d(self.robot, self.joint_values, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)

    def test_planar_robot_raises_type_error(self):
        robot = PlanarRobot([1.0, 1.0])
        with self.assertRaises(TypeError):
            plot_3d(robot, [0.0, 0.0])

    def test_scara_robot_raises_type_error(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        with self.assertRaises(TypeError):
            plot_3d(robot, [0.0, 0.0, 0.0, 0.0])

    def test_link_data_matches_forward_kinematics(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        transforms = self.robot.forward_kinematics(self.joint_values, return_all=True)
        augmented = [identity_matrix(4)] + transforms

        expected_x = [T[0, 3] for T in augmented]
        expected_y = [T[1, 3] for T in augmented]
        expected_z = [T[2, 3] for T in augmented]
        link_line = ax.lines[0]

        numpy.testing.assert_allclose(link_line.get_xdata(), expected_x)
        numpy.testing.assert_allclose(link_line.get_ydata(), expected_y)
        numpy.testing.assert_allclose(link_line.get_data_3d()[2], expected_z)

    def test_coordinate_frames_drawn(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        transforms = self.robot.forward_kinematics(self.joint_values, return_all=True)
        n_frames = len(transforms) + 1 # +1 for base frame
        expected_quivers = 3 * n_frames # X, Y, Z axes
        self.assertEqual(len(ax.collections), expected_quivers)

    def test_default_figure_size(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        width, height = fig.get_size_inches()
        self.assertEqual((width, height), styles.DEFAULT_FIGSIZE_3D)
        plt.close(fig)

    def test_existing_axes_preserve_figure_size(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        returned_fig, returned_ax = plot_3d(self.robot, self.joint_values, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertEqual(tuple(fig.get_size_inches()), (10, 8))
        plt.close(fig)

    def test_legend_contains_expected_entries(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        expected = {"Base", "Links", "Revolute Joint", "End Effector"}
        self.assertEqual(set(labels), expected)
        plt.close(fig)

    def test_single_joint_robot_no_phantom_joint_label(self):
        robot = ArticulatedRobot([{"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0}])
        fig, ax = plot_3d(robot, [0.0])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("Revolute Joint", labels)
        plt.close(fig)

    def test_equal_aspect_ratio_set(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        box_aspect = ax.get_box_aspect()
        self.assertIsNotNone(box_aspect)
        self.assertEqual(len(box_aspect), 3)
        plt.close(fig)

    def test_planar_configuration_does_not_crash(self):
        dh_table = [
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
        ]
        robot = ArticulatedRobot(dh_table)
        try:
            fig, ax = plot_3d(robot, [0.0, 0.0, 0.0])
        except Exception as e:
            self.fail(f"plot_3d raised unexpectedly on a degenerate planar (z=0) configuration: {e}")
        plt.close(fig)

    def test_camera_preset_applied(self):
        fig, ax = plot_3d(self.robot, self.joint_values, view="top")
        expected = styles.VIEW_PRESETS["top"]
        self.assertAlmostEqual(ax.elev, expected["elev"])
        self.assertAlmostEqual(ax.azim, expected["azim"])
        plt.close(fig)

    def test_default_view_is_isometric(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        expected = styles.VIEW_PRESETS[styles.DEFAULT_VIEW]
        self.assertAlmostEqual(ax.elev, expected["elev"])
        self.assertAlmostEqual(ax.azim, expected["azim"])
        plt.close(fig)

    def test_invalid_view_raises_value_error(self):
        with self.assertRaises(ValueError):
            plot_3d(self.robot, self.joint_values, view="nonexistent")

    def test_end_effector_frame_is_emphasized(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        base_frame_linewidth = ax.collections[0].get_linewidth()[0]
        ee_frame_linewidth = ax.collections[-1].get_linewidth()[0]
        self.assertAlmostEqual(base_frame_linewidth, styles.FRAME_LINEWIDTH)
        self.assertAlmostEqual(ee_frame_linewidth, styles.END_EFFECTOR_FRAME_LINEWIDTH)
        self.assertGreater(ee_frame_linewidth, base_frame_linewidth)
        plt.close(fig)

    def test_grid_is_enabled(self):
        fig, ax = plot_3d(self.robot, self.joint_values)
        self.assertTrue(ax._draw_grid)
        plt.close(fig)

    def test_trajectory_none_unchanged(self):
        fig, ax = plot_3d(self.robot, self.joint_values, trajectory=None)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("Trajectory", labels)
        plt.close(fig)

    def test_trajectory_adds_legend_entry(self):
        fig, ax = plot_3d(self.robot, self.joint_values, trajectory=self.trajectory)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Trajectory", labels)
        plt.close(fig)

    def test_trajectory_path_matches_end_effector_positions(self):
        fig, ax = plot_3d(self.robot, self.joint_values, trajectory=self.trajectory)
        expected_x = []
        expected_y = []
        expected_z = []

        for config in self.trajectory:
            transforms = self.robot.forward_kinematics(config, return_all=True)
            ee = transforms[-1][:3, 3]

            expected_x.append(ee[0])
            expected_y.append(ee[1])
            expected_z.append(ee[2])

        trajectory_line = ax.lines[0]  # precedes links per the ordering comment
        numpy.testing.assert_allclose(trajectory_line.get_xdata(), expected_x)
        numpy.testing.assert_allclose(trajectory_line.get_ydata(), expected_y)
        numpy.testing.assert_allclose(trajectory_line.get_data_3d()[2], expected_z)
        plt.close(fig)

    def test_empty_trajectory_behaves_like_none(self):
        fig, ax = plot_3d(self.robot, self.joint_values, trajectory=[])
        self.assertNotIn("Trajectory", [t.get_text() for t in ax.get_legend().get_texts()])
        plt.close(fig)

if __name__ == "__main__":
    unittest.main()