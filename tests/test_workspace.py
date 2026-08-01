import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import unittest
import numpy

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from open_kinematics.visualization import styles
from open_kinematics.visualization.workspace import sample_workspace
from open_kinematics.visualization.workspace import plot_workspace

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot

class TestWorkspace(unittest.TestCase):

    def setUp(self):
        self.planar = PlanarRobot([1.0, 1.0])
        self.scara = ScaraRobot(link_lengths=[1.0, 1.0], d_range=[0.0, 0.5])
        self.articulated = ArticulatedRobot([
            {"theta":0.0,"d":0.3,"r":0.2,"alpha":numpy.pi/2},
            {"theta":0.0,"d":0.0,"r":0.5,"alpha":0.0},
            {"theta":0.0,"d":0.0,"r":0.4,"alpha":0.0},
            {"theta":0.0,"d":0.4,"r":0.0,"alpha":numpy.pi/2},
            {"theta":0.0,"d":0.0,"r":0.0,"alpha":-numpy.pi/2},
            {"theta":0.0,"d":0.2,"r":0.0,"alpha":0.0},
        ])

    def test_returns_expected_shape(self):
        points = sample_workspace(self.planar, num_samples=500)
        self.assertEqual(points.shape, (500, 3))

    def test_returns_numpy_array(self):
        points = sample_workspace(self.planar)
        self.assertIsInstance(points, numpy.ndarray)

    def test_seed_produces_reproducible_samples(self):
        p1 = sample_workspace(self.planar, num_samples=100, seed=42)
        p2 = sample_workspace(self.planar, num_samples=100, seed=42)
        numpy.testing.assert_allclose(p1, p2)

    def test_different_seeds_produce_different_samples(self):
        p1 = sample_workspace(self.planar, num_samples=100, seed=1)
        p2 = sample_workspace(self.planar, num_samples=100, seed=2)
        self.assertFalse(numpy.allclose(p1, p2))

    def test_invalid_num_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_workspace(self.planar, num_samples=0)

    def test_planar_workspace_has_zero_z(self):
        points = sample_workspace(self.planar, num_samples=500)
        self.assertTrue(numpy.allclose(points[:, 2], 0.0))

    def test_scara_workspace_has_variable_z(self):
        points = sample_workspace(self.scara, num_samples=500)
        self.assertGreater(numpy.ptp(points[:, 2]), 0.0)

    def test_workspace_contains_no_nan_values(self):
        points = sample_workspace(self.articulated, num_samples=500)
        self.assertFalse(numpy.isnan(points).any())

    def test_unsupported_robot_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            plot_workspace("not a robot")

    def test_existing_axes_preserve_figure_size(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        returned_fig, returned_ax = plot_workspace(self.planar, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertEqual(tuple(fig.get_size_inches()), (10, 5))
        plt.close(fig)

    def test_returns_figure_and_axes_planar(self):
        fig, ax = plot_workspace(self.planar, num_samples=200)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        plt.close(fig)

    def test_returns_figure_and_axes_articulated(self):
        fig, ax = plot_workspace(self.articulated, num_samples=200)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes3D)
        plt.close(fig)

    def test_reuses_existing_2d_axes(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_workspace(self.planar, num_samples=100, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        plt.close(fig)

    def test_reuses_existing_3d_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned_fig, returned_ax = plot_workspace(self.articulated, num_samples=100, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        plt.close(fig)

    def test_default_figure_size_2d(self):
        fig, ax = plot_workspace(self.planar)
        self.assertEqual(tuple(fig.get_size_inches()), styles.DEFAULT_FIGSIZE_2D)
        plt.close(fig)

    def test_default_figure_size_3d(self):
        fig, ax = plot_workspace(self.articulated)
        self.assertEqual(tuple(fig.get_size_inches()), styles.DEFAULT_FIGSIZE_3D)
        plt.close(fig)

    def test_legend_contains_workspace(self):
        fig, ax = plot_workspace(self.planar)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(set(labels), {"Workspace"})
        plt.close(fig)

    def test_default_camera_is_isometric(self):
        fig, ax = plot_workspace(self.articulated)
        expected = styles.VIEW_PRESETS[styles.DEFAULT_VIEW]
        self.assertAlmostEqual(ax.elev, expected["elev"])
        self.assertAlmostEqual(ax.azim, expected["azim"])
        plt.close(fig)

    def test_workspace_alpha_matches_style(self):
        fig, ax = plot_workspace(self.planar)
        collection = ax.collections[0]
        self.assertAlmostEqual(collection.get_alpha(), styles.WORKSPACE_ALPHA)
        plt.close(fig)
