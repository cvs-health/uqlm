.. _contribute:

Contributing to UQLM
====================

Welcome and thank you for considering contributing to UQLM!

It takes a lot of time and effort to use software much less build upon it, so we deeply appreciate your desire to help make this project thrive.

Table of Contents
-----------------
1. :ref:`How to Contribute<how-to-contribute>`
    * :ref:`Reporting Bugs<reporting-bugs>`
    * :ref:`Suggesting Enhancements<suggesting-enhancements>`
    * :ref:`Pull Requests<pull-requests>`
2. :ref:`Development Setup<development-setup>`
3. :ref:`Style Guides<style-guides>`
    * :ref:`Code Style<code-style>`
4. :ref:`License<license>`

.. _how-to-contribute:

How to Contribute
-----------------

.. _reporting-bugs:

Reporting Bugs
**************

If you find a bug, please report it by opening an issue on GitHub. Include as much detail as possible:
* Steps to reproduce the bug.
* Expected and actual behavior.
* Screenshots if applicable.
* Any other information that might help us understand the problem.

.. _suggesting-enhancements:

Suggesting Enhancements
***********************

We welcome suggestions for new features or improvements. To suggest an enhancement, please open an issue on GitHub and include:

* A clear description of the suggested enhancement.
* Why you believe this enhancement would be useful.
* Any relevant examples or mockups.

.. _pull-requests:

Pull Requests
*************

1. Fork the repository.
2. Create a new branch (``git checkout -b feature/your-feature-name``).
3. Make your changes.
4. Commit your changes (``git commit -m 'Add some feature'```).
5. Push to the branch (``git push origin feature/your-feature-name``).
6. Open a pull request.

Please ensure your pull request adheres to the following guidelines:

* Follow the project's code style.
* Include tests for any new features or bug fixes.

.. _development-setup:

Development Setup
-----------------

1. Clone the repository: ``git clone https://github.aetna.com/analytics-org/uqlm.git``
2. Navigate to the project directory: ``cd uqlm``
3. Create and activate a virtual environment (using ``venv`` or ``conda``)
4. Install uv (if you don't already have it): ``pip install uv`` or ``curl -LsSf https://astral.sh/uv/install.sh | sh``
5. Install uqlm with dev dependencies: ``uv sync --group dev``
6. Install our pre-commit hooks to ensure code style compliance: ``pre-commit install``
7. Run tests to ensure everything is working: ``pre-commit run --all-files```

You're ready to develop!

**For documentation contributions**

Our documentation lives on the gh-pages branch and is hosted via GitHub Pages.

There are two relevant directories:

* ``docs_src`` - where source documentation files are located
* ``docs`` - where the built documentation is located that is served by GitHub Pages

To build the documentation locally:

#. Create a virtual environment with your favorite tool(ex. conda, virtualenv, uv, and etc.)

#. Checkout the ``gh-pages`` branch and create new branch from it

#. Navigate to the ``docs_src/latest`` directory

  * If this is version upgrade:

    #. Copy ``latest`` contents to ``docs_src/{version_number}`` folder update the version in ``conf.py`` file

    #. Copy ``latest`` contents from ``docs/`` to ``docs/{version_number}`` folder

    #. Update the versions in ``docs_src/latest/index.rst`` file and ``docs_src/versions.json``

#. ``cd uqlm``

#. ``pip install -e .`` # installs current uqlm repo as package to environment

#. ``cd docs_src/latest``

#. ``brew install pandoc`` # to use nbsphinx extension

#. ``make install`` # installs sphinx related python packages

#. ``make github`` # builds docs html

#. ``make local`` # locally test doc site


.. _style-guides:

Style Guides
------------

.. _code-style:

Code Style
**********

- We use `Ruff <https://github.com/astral-sh/ruff>`_ to lint and format our files.
- Our pre-commit hook will run Ruff linting and formatting when you commit.
- You can manually run Ruff at any time `Ruff usage <https://github.com/astral-sh/ruff#usage>`_.

Please ensure your code is properly formatted and linted before committing.

.. _license:

License
-------

Before contributing to this CVS Health sponsored project, you will need to sign the associated `Contributor License Agreement (CLA) <https://TBD>`_.


Thanks again for using and supporting uqlm!