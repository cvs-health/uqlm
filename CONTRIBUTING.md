# Contributing to UQLM

Welcome and thank you for considering contributing to UQLM!

It takes a lot of time and effort to use software much less build upon it, so we deeply appreciate your desire to help make this project thrive.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Pull Requests](#pull-requests)
2. [Development Setup](#development-setup)
3. [Style Guides](#style-guides)
    - [Code Style](#code-style)

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by opening an issue on GitHub. Include as much detail as possible:
- Steps to reproduce the bug.
- Expected and actual behavior.
- Screenshots if applicable.
- Any other information that might help us understand the problem.

### Suggesting Enhancements

We welcome suggestions for new features or improvements. To suggest an enhancement, please open an issue on GitHub and include:
- A clear description of the suggested enhancement.
- Why you believe this enhancement would be useful.
- Any relevant examples or mockups.

### Pull Requests

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

Please ensure your pull request adheres to the following guidelines:
- Follow the project's code style.
- Include tests for any new features or bug fixes.

## Development Setup

1. Clone the repository: `git clone https://github.com/cvs-health/uqlm.git`
2. Navigate to the project directory: `cd uqlm`
3. Create and activate a virtual environment (using `venv` or `conda`)
4. Install uv (if you don't already have it): `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. Install uqlm with dev dependencies: `uv sync --group dev`
6. Install our pre-commit hooks to ensure code style compliance: `pre-commit install`
7. Run tests to ensure everything is working: `pre-commit run --all-files`

You're ready to develop!

## Style Guides

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) to lint and format our files.
- Our pre-commit hook will run Ruff linting and formatting when you commit.
- You can manually run Ruff at any time (see [Ruff usage](https://github.com/astral-sh/ruff#usage)).

Please ensure your code is properly formatted and linted before committing.

## License

Before contributing to this CVS Health sponsored project, you will need to sign the associated [Contributor License Agreement (CLA)](https://forms.office.com/r/iFZWwzjt9C)

---

Thanks again for using and supporting uqlm!