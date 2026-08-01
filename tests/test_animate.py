import unittest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D

from open_kinematics.visualization import styles
from open_kinematics.visualization.animate import animate

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot


class TestAnimate(unittest.TestCase):

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

        self.planar_trajectory = [
            [0.0, 0.0],
            [0.3, 0.2],
            [0.5, 0.4],
        ]

        self.scara_trajectory = [
            [0.0, 0.0, 0.10, 0.0],
            [0.2, 0.1, 0.20, 0.1],
            [0.4, 0.2, 0.30, 0.2],
        ]

        self.articulated_trajectory = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3, 0.0, 0.1, -0.2],
            [0.4, -0.3, 0.5, -0.2, 0.3, 0.0],
        ]

    def tearDown(self):
        plt.close("all")

    # Return values
    def test_returns_figure_axes_animation_planar(self):
        fig, ax, anim = animate(self.planar_robot,self.planar_trajectory)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIsInstance(anim, FuncAnimation)

    def test_returns_figure_axes_animation_articulated(self):
        fig, ax, anim = animate(self.articulated_robot, self.articulated_trajectory)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes3D)
        self.assertIsInstance(anim, FuncAnimation)

    # Existing axes
    def test_reuses_existing_2d_axes(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax, anim = animate(self.planar_robot, self.planar_trajectory, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)

    def test_reuses_existing_3d_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned_fig, returned_ax, anim = animate(self.articulated_robot, self.articulated_trajectory, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)

    # Validation
    def test_empty_trajectory_raises_value_error(self):
        with self.assertRaises(ValueError):
            animate(self.planar_robot, [])

    def test_unsupported_robot_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            animate("not a robot", self.planar_trajectory)

    # Interval
    def test_default_interval_used(self):
        fig, ax, anim = animate(self.planar_robot, self.planar_trajectory)
        self.assertEqual(anim.event_source.interval, styles.ANIMATION_INTERVAL_MS)

    def test_custom_interval_used(self):
        fig, ax, anim = animate(self.planar_robot, self.planar_trajectory, interval=250)
        self.assertEqual(anim.event_source.interval, 250)

    # Frame count
    def test_animation_frame_count(self):
        fig, ax, anim = animate(self.planar_robot, self.planar_trajectory)
        frames = list(anim.new_frame_seq())
        self.assertEqual(len(frames), len(self.planar_trajectory))

    # plot_kwargs forwarding
    def test_view_argument_forwarded(self):
        fig, ax, anim = animate(self.articulated_robot, self.articulated_trajectory, view="top")
        expected = styles.VIEW_PRESETS["top"]
        self.assertAlmostEqual(ax.elev, expected["elev"])
        self.assertAlmostEqual(ax.azim, expected["azim"])

    # Existing figure preserved
    def test_existing_axes_preserve_figure_size(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        returned_fig, returned_ax, anim = animate(self.planar_robot, self.planar_trajectory, ax=ax)
        self.assertEqual(tuple(fig.get_size_inches()), (10, 5))

if __name__ == "__main__":
    unittest.main()