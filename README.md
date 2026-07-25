# OpenKinematics

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/devwithchai/OpenKinematics)](LICENSE)
[![Release](https://img.shields.io/github/v/release/devwithchai/OpenKinematics)](https://github.com/devwithchai/OpenKinematics/releases)
[![Tests](https://github.com/devwithchai/OpenKinematics/actions/workflows/tests.yml/badge.svg)](https://github.com/devwithchai/OpenKinematics/actions/workflows/tests.yml)

> An educational Python library for robot kinematics built from first principles.

OpenKinematics is an open-source Python library that implements fundamental
robot kinematics algorithms from scratch using only NumPy and Matplotlib.

The project is designed to bridge robotics theory and software engineering by
providing clean, readable implementations of forward kinematics, Jacobians,
inverse kinematics, and robot visualization. Rather than treating robotics
algorithms as black boxes, OpenKinematics focuses on helping students,
educators, and developers understand how these algorithms work while exposing
them through a simple and consistent API.

---

## Why OpenKinematics?

Many robotics libraries prioritize industrial applications and hide the
underlying mathematics behind large software frameworks. OpenKinematics takes a
different approach.

It is designed as an educational library where every major algorithm is
implemented from first principles, thoroughly documented, and organized using a
clean layered architecture. The goal is to make robotics mathematics easier to
understand while demonstrating how those algorithms can be structured as a
maintainable software library.

In addition to being a usable robotics library, OpenKinematics is intended to
serve as a learning resource for students, educators, and developers who want
to understand how robotics algorithms translate into clean, maintainable
software.

---

## Features

- Forward Kinematics using the Denavit–Hartenberg (DH) convention
- Geometric Jacobian computation
- Geometric Inverse Kinematics for Planar and SCARA manipulators
- Numerical Pseudoinverse Inverse Kinematics for articulated manipulators
- 2D visualization for Planar and SCARA robots
- 3D visualization for articulated robots
- Unified `Robot` facade API
- Fully documented public API (Sphinx-style docstrings)
- Pure Python implementation using only NumPy and Matplotlib
- Released under the MIT License

---

## Installation

### Requirements

- Python 3.10 or later
- NumPy 1.24 or later
- Matplotlib 3.7 or later

OpenKinematics is currently installed directly from the source repository.

```bash
git clone https://github.com/devwithchai/OpenKinematics.git

cd OpenKinematics

pip install -e .
```

---

## Quick Start

```python
import matplotlib.pyplot as plt
from open_kinematics.api import Robot

robot = Robot.planar([1.0, 1.0])

joint_values = [0.5, 0.3]

transform = robot.fk(joint_values)

print(transform)

robot.plot(joint_values)

plt.show()
```

For more complete examples covering forward kinematics, inverse kinematics, and
visualization, see the programs in the `examples/` directory.

---

## Examples

OpenKinematics includes complete, runnable example programs demonstrating the
library's public API and supported robot models.

| Example | Description |
|---------|-------------|
| `examples/planar_fk_ik.py` | Forward kinematics, geometric inverse kinematics, round-trip verification, and 2D visualization for a two-link planar manipulator. |
| `examples/scara_fk_ik.py` | Forward kinematics, geometric inverse kinematics, and 2D visualization for a SCARA robot. |
| `examples/articulated_fk_ik.py` | Forward kinematics, numerical pseudoinverse inverse kinematics, full-pose verification, and 3D visualization for an articulated robot. |

Each example is self-contained and demonstrates the recommended usage of the
public `Robot` facade API.

---

## Visualization

### 2R Planar Robot
![2R Planar Robot](docs/images/planar_robot.png)

### SCARA Robot
![SCARA Robot](docs/images/scara_robot.png)

### 6-DOF Articulated Robot
![6-DOF Articulated Robot](docs/images/articulated_robot.png)

---

## Supported Robot Models

OpenKinematics currently supports the following robot manipulators:

| Robot | Forward Kinematics | Jacobian | Inverse Kinematics | Visualization |
|--------|-------------------|----------|--------------------|---------------|
| Planar (2R) | ✅ | ✅ | Geometric | 2D |
| SCARA (RRP-R) | ✅ | ✅ | Geometric | 2D |
| Articulated (6-DOF) | ✅ | ✅ | Numerical Pseudoinverse | 3D |

---

## Project Architecture

The library follows a layered architecture where higher-level modules depend
only on lower-level modules.

```text
                +----------------------+
                |        api.py        |
                +----------+-----------+
                           |
                +----------v-----------+
                |       robots/        |
                +----------+-----------+
                           |
                +----------v-----------+
                |      solvers/        |
                +----------+-----------+
                           |
                +----------v-----------+
                |        math/         |
                +----------+-----------+
                           |
                +----------v-----------+
                |        NumPy         |
                +----------------------+
```

This dependency direction is intentionally enforced throughout the project to
keep the codebase modular, maintainable, and easy to extend.

---

## Roadmap

The library currently provides:

- ✅ Foundation mathematics and transformation utilities
- ✅ Robot models (Planar, SCARA, Articulated)
- ✅ Forward Kinematics
- ✅ Geometric Jacobian
- ✅ Geometric Inverse Kinematics
- ✅ Numerical Pseudoinverse Inverse Kinematics
- ✅ 2D and 3D Visualization
- ✅ Unified public `Robot` facade API
- ✅ Runnable example programs
- ✅ Packaging and installation support
- ✅ Comprehensive API documentation
- ✅ Continuous Integration (GitHub Actions)
- ✅ Expanded developer documentation

Planned future work includes:

- Dynamics module
- Theory documentation
- PyPI release

---

## Documentation

The library's public API is documented directly in the source code using
Sphinx-style docstrings.

Additional theory documentation is planned under:

```text
docs/
└── theory/
    ├── dh_method.md
    ├── geometric_ik.md
    ├── jacobian.md
    └── dynamics.md
```

These documents are planned for a future release and are not yet available.

---

## Contributing

Contributions, bug reports, feature requests, and documentation improvements
are welcome.

See `CONTRIBUTING.md` for development setup, coding standards, testing expectations, and contribution workflow.

Until then, contributors are encouraged to:

- Open an issue before proposing significant changes.
- Keep pull requests focused on a single logical change.
- Ensure existing tests continue to pass.
- Follow the project's layered architecture.

---

## License

OpenKinematics is released under the MIT License.

See the [LICENSE](LICENSE) file for the full license text.

---

## Acknowledgements

OpenKinematics is an independent educational project created to help bridge
robotics theory and software engineering through clean, readable
implementations of fundamental robotics algorithms.

The project is inspired by standard robotics textbooks, university coursework,
and the open-source software community.