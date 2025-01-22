Installation
============

Development environment
-----------------------

Requirements
~~~~~~~~~~~~

- Python: https://www.python.org
- Poetry: https://python-poetry.org/docs

After installing Poetry and cloning the project from GitHub, execute the following command in the root directory of the cloned project:

.. code:: sh

    poetry install

All of the project's dependencies should be installed and the project should be ready for further development. Note that Poetry creates a separate virtual environment for the project.

Development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

List of NiaAutoARM's dependencies:

+----------------------+----------------------+
| Package              | Version              |
+======================+======================+
| niapy                | ^2.0.5               |
+----------------------+----------------------+
| pandas               | ^2.1.1               |
+----------------------+----------------------+
| niaarm               | ^0.3.12              |
+----------------------+----------------------+
| sphinx               | ^5.0                 |
+----------------------+----------------------+
| sphinx-rtd-theme     | ^1.0.0               |
+----------------------+----------------------+