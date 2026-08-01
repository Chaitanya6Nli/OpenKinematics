"""
Visualization styling constants for OpenKinematics.

This module is the single source of truth for visualization styling used by
plot_2d.py, plot_3d.py, workspace.py, and animate.py.

It intentionally contains only constants and preset dictionaries.
"""

# Color palette
BASE_COLOR = "#2F2F2F"
LINK_COLOR = "#37474F"                # used by plot_2d.py, plot_3d.py — link segments, shared
JOINT_REVOLUTE_COLOR = "#1F77B4"       # blue — revolute joint markers, both files
JOINT_PRISMATIC_COLOR = "#FF7F0E"      # orange — prismatic joint markers (ScaraRobot, plot_2d.py only)
END_EFFECTOR_COLOR = "#D62728"         # red — end-effector marker, both files
FRAME_X_COLOR = "#E74C3C"              # RGB=XYZ convention, plot_3d.py
FRAME_Y_COLOR = "#2ECC71"              # RGB=XYZ convention, plot_3d.py
FRAME_Z_COLOR = "#3498DB"              # RGB=XYZ convention, plot_3d.py

# Typography
LEGEND_FONTSIZE = 10                   # ax.legend() in plot_2d.py, plot_3d.py

# Figure defaults
DEFAULT_FIGSIZE_2D = (6, 6)            # square — matches equal-aspect enforcement
DEFAULT_FIGSIZE_3D = (7.5, 7.5)        # extra room for legends and coordinate frames
DEFAULT_SAVEFIG_DPI = 200              # docs/images/*.png regeneration
DEFAULT_SCREEN_DPI = 100               # interactive plt.show()

# Marker & line styles
BASE_MARKER = "D"      # diamond
BASE_MARKERSIZE = 8
LINK_LINEWIDTH = 6                     # shared 2D/3D value, between the old separate 2/3
LINK_EDGEWIDTH = 1.5
JOINT_MARKERSIZE = 7                   # shared 2D/3D value, between the old separate 6/8
JOINT_REVOLUTE_MARKER = "o"            # circle — shape difference, independent of color
JOINT_PRISMATIC_MARKER = "s"           # square — shape difference, independent of color
END_EFFECTOR_MARKER = "*"              # shared, both files
END_EFFECTOR_MARKERSIZE = 14           # shared, between the old separate 12/15

# Coordinate frame styles
FRAME_LINEWIDTH = 1.5                  # intermediate joint frames, plot_3d.py
FRAME_ALPHA = 0.85                     # intermediate joint frames, plot_3d.py
END_EFFECTOR_FRAME_LINEWIDTH = 2.5     # heavier than FRAME_LINEWIDTH — this is the EE emphasis

# Camera presets
# Angles may be refined after visual validation with real robot models.
VIEW_PRESETS = {
    "isometric": {"elev": 35.264, "azim": 45},
    "front":     {"elev": 0,      "azim": -90},
    "side":      {"elev": 0,      "azim": 0},
    "top":       {"elev": 90,     "azim": -90},
}
DEFAULT_VIEW = "isometric"             # plot_3d.py default camera preset
PROJECTION_TYPE = "ortho"              # plot_3d.py — Decision #3, clarity over realism

# Grid styles
GRID_ENABLED = True                    # plot_2d.py, plot_3d.py
GRID_ALPHA = 0.3                       # plot_2d.py, plot_3d.py
GRID_LINESTYLE = "--"                  # plot_2d.py, plot_3d.py

# Workspace styles
WORKSPACE_MARKER_SIZE = 8
WORKSPACE_FACE_COLOR = "#3498DB"       # visualization/workspace.py
WORKSPACE_EDGE_COLOR = "#2980B9"       # visualization/workspace.py
WORKSPACE_ALPHA = 0.15                 # low alpha — must not obscure the robot

# Trajectory styles
TRAJECTORY_COLOR = "#9B59B6"           # purple, unused elsewhere — stays visually distinct from links/joints/EE/frames
TRAJECTORY_LINEWIDTH = 1.5             # visualization/animate.py, trajectory overlays
TRAJECTORY_LINESTYLE = "--"            # visualization/animate.py, trajectory overlays

# Animation styles
ANIMATION_INTERVAL_MS = 100

__all__ = [
    "BASE_COLOR",
    "LINK_COLOR",
    "JOINT_REVOLUTE_COLOR",
    "JOINT_PRISMATIC_COLOR",
    "END_EFFECTOR_COLOR",
    "FRAME_X_COLOR",
    "FRAME_Y_COLOR",
    "FRAME_Z_COLOR",
    "LEGEND_FONTSIZE",
    "DEFAULT_FIGSIZE_2D",
    "DEFAULT_FIGSIZE_3D",
    "DEFAULT_SAVEFIG_DPI",
    "DEFAULT_SCREEN_DPI",
    "BASE_MARKER",
    "BASE_MARKERSIZE",
    "LINK_LINEWIDTH",
    "LINK_EDGEWIDTH",
    "JOINT_MARKERSIZE",
    "JOINT_REVOLUTE_MARKER",
    "JOINT_PRISMATIC_MARKER",
    "END_EFFECTOR_MARKER",
    "END_EFFECTOR_MARKERSIZE",
    "FRAME_LINEWIDTH",
    "FRAME_ALPHA",
    "END_EFFECTOR_FRAME_LINEWIDTH",
    "VIEW_PRESETS",
    "DEFAULT_VIEW",
    "PROJECTION_TYPE",
    "GRID_ENABLED",
    "GRID_ALPHA",
    "GRID_LINESTYLE",
    "WORKSPACE_MARKER_SIZE",
    "WORKSPACE_FACE_COLOR",
    "WORKSPACE_EDGE_COLOR",
    "WORKSPACE_ALPHA",
    "TRAJECTORY_COLOR",
    "TRAJECTORY_LINEWIDTH",
    "TRAJECTORY_LINESTYLE",
    "ANIMATION_INTERVAL_MS"
]