import unittest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from open_kinematics.visualization.plot_2d import plot_2d
from open_kinematics.visualization import styles
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot

class TestPlot2D(unittest.TestCase):

    def setUp(self):
        self.planar_robot = PlanarRobot([1.0, 1.0])
        self.scara_robot = ScaraRobot(link_lengths=[1.0, 1.0], d_range=[0.0, 0.5])
        self.articulated_robot = ArticulatedRobot([
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.5, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.4, "r": 0.0, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.0, "alpha": -numpy.pi / 2},
            {"theta": 0.0, "d": 0.2, "r": 0.0, "alpha": 0.0},
        ])

    def test_returns_figure_and_axes(self):
        fig, ax = plot_2d(self.planar_robot,[numpy.pi / 4, numpy.pi / 6])

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        plt.close(fig)

    def test_reuses_existing_axes(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_2d(self.planar_robot,[numpy.pi / 4, numpy.pi / 6], ax=ax)

        self.assertIs(returned_ax, ax)
        self.assertIs(returned_fig, fig)

        plt.close(fig)

    def test_plotted_data_matches_planar_joint_positions(self):
        joint_values = [numpy.pi / 4, numpy.pi / 6]

        expected_points = self.planar_robot.get_joint_positions(joint_values)

        expected_x = [point[0] for point in expected_points]
        expected_y = [point[1] for point in expected_points]

        fig, ax = plot_2d(self.planar_robot, joint_values)
        plotted_line = ax.lines[0]

        self.assertTrue(numpy.allclose(plotted_line.get_xdata(), expected_x))
        self.assertTrue(numpy.allclose(plotted_line.get_ydata(), expected_y))

        plt.close(fig)

    def test_scara_projection_to_xy(self):
        joint_values = [numpy.pi / 6, numpy.pi / 4, 0.2, numpy.pi / 3,]
        expected_points = self.scara_robot.get_joint_positions(joint_values)

        expected_x = [point[0] for point in expected_points]
        expected_y = [point[1] for point in expected_points]

        fig, ax = plot_2d(self.scara_robot, joint_values)
        plotted_line = ax.lines[0]
        self.assertTrue(numpy.allclose(plotted_line.get_xdata(), expected_x))
        self.assertTrue(numpy.allclose(plotted_line.get_ydata(), expected_y))

        plt.close(fig)

    def test_articulated_robot_raises_type_error(self):
        with self.assertRaises(TypeError):
            plot_2d(self.articulated_robot, [0.0] * 6)

    def test_default_figure_size(self):
        fig, ax = plot_2d(self.planar_robot, [numpy.pi / 4, numpy.pi / 6])
        width, height = fig.get_size_inches()
        self.assertEqual((width, height), styles.DEFAULT_FIGSIZE_2D)
        plt.close(fig)

    def test_existing_axes_preserve_figure_size(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        returned_fig, returned_ax = plot_2d(self.planar_robot, [numpy.pi / 4, numpy.pi / 6], ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        width, height = fig.get_size_inches()
        self.assertEqual((width, height), (10, 5))
        plt.close(fig)

    def test_legend_contains_expected_entries_planar(self):
        fig, ax = plot_2d(self.planar_robot, [numpy.pi / 4, numpy.pi / 6])
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        expected = {"Base", "Links", "Revolute Joint", "End Effector"}
        self.assertEqual(set(labels), expected)
        plt.close(fig)

    def test_legend_contains_prismatic_joint_scara(self):
        fig, ax = plot_2d(self.scara_robot, [numpy.pi / 6, numpy.pi / 4, 0.2, numpy.pi / 3])
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        expected = {"Base", "Links", "Revolute Joint", "Prismatic Joint", "End Effector"}
        self.assertEqual(set(labels), expected)
        plt.close(fig)

    def test_no_duplicate_joint_labels_planar(self):
        fig, ax = plot_2d(self.planar_robot, [numpy.pi / 4, numpy.pi / 6])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels.count("Revolute Joint"), 1)
        plt.close(fig)

    def test_no_duplicate_joint_labels_scara(self):
        fig, ax = plot_2d(self.scara_robot, [numpy.pi / 6, numpy.pi / 4, 0.2, numpy.pi / 3])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels.count("Revolute Joint"), 1)
        self.assertEqual(labels.count("Prismatic Joint"), 1)
        plt.close(fig)

    def test_grid_is_enabled(self):
        fig, ax = plot_2d(self.planar_robot, [numpy.pi / 4, numpy.pi / 6])
        visible_gridlines = [line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()]
        self.assertTrue(any(visible_gridlines))
        plt.close(fig)

    def test_trajectory_none_unchanged(self):
        fig, ax = plot_2d(self.planar_robot, [0.5, 0.3], trajectory=None)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("Trajectory", labels)
        plt.close(fig)

    def test_trajectory_adds_legend_entry(self):
        trajectory = [[0.0, 0.0], [0.3, 0.2], [0.5, 0.4]]  # adapt per robot type
        fig, ax = plot_2d(self.planar_robot, [0.5, 0.3], trajectory=trajectory)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Trajectory", labels)
        plt.close(fig)

    def test_trajectory_path_matches_end_effector_positions(self):
        trajectory = [[0.0, 0.0], [0.3, 0.2], [0.5, 0.4]]
        fig, ax = plot_2d(self.planar_robot, [0.5, 0.3], trajectory=trajectory)
        expected_x = [self.planar_robot.get_joint_positions(c)[-1][0] for c in trajectory]
        expected_y = [self.planar_robot.get_joint_positions(c)[-1][1] for c in trajectory]
        trajectory_line = ax.lines[0]  # precedes links per the ordering comment
        numpy.testing.assert_allclose(trajectory_line.get_xdata(), expected_x)
        numpy.testing.assert_allclose(trajectory_line.get_ydata(), expected_y)
        plt.close(fig)

    def test_empty_trajectory_behaves_like_none(self):
        fig, ax = plot_2d(self.planar_robot, [0.5, 0.3], trajectory=[])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("Trajectory", labels)
        plt.close(fig)

if __name__ == "__main__":
    unittest.main()