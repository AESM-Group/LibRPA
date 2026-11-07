.. LibRPA documentation master file, created by
   sphinx-quickstart on Thu Sep  7 16:55:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LibRPA
======

**LibRPA** is a library for efficient and accurate first-principles calculations
based on many-body perturbation theory with numerical atomic orbitals (NAOs).
It currently supports random-phase approximation (RPA) correlation-energy
calculations and GW quasiparticle calculations for periodic systems. By
leveraging the localized resolution of identity (LRI) technique, LibRPA
achieves low-scaling algorithms suitable for large-scale systems. Implemented
in C++ with MPI/OpenMP parallelism and equipped with C and Fortran
interfaces, LibRPA can be integrated with density functional theory (DFT) codes
through both file-based and API-based workflows.

.. note::

   LibRPA is under active development, and its input/output interfaces are
   prone to change.

LibRPA is :doc:`licensed <develop/licensing>` under LGPL-3.0.

**Key Features**

- RPA correlation energy calculations and GW quasiparticle calculations for atomic basis framework.
- Low-scaling algorithms based on the localized resolution of identity (LRI) technique.
- Hybrid MPI/OpenMP parallelism for scalability and performance.
- Flexible integration with external DFT packages through driver-based and API-based interfaces.

**Get Started**

- :doc:`Installation guide <user_guide/install>` together with :doc:`example build scripts <examples/build/index>`
- Tutorials by functionality: :doc:`RPA <tutorial/rpa/index>`, :doc:`one-shot GW <tutorial/g0w0/index>`, and more to come.
- :doc:`Driver examples <examples/driver/index>`

**User Guide**

- :doc:`Compile Options <user_guide/compile_options>`
- :doc:`Runtime Parameters <user_guide/runtime_parameters>`
- :doc:`Driver Usage <user_guide/driver_usage>`
- :doc:`API Usage <user_guide/api_usage>`
- :doc:`Changelog <changelog>`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install
   user_guide/compile_options
   user_guide/runtime_parameters
   user_guide/driver_usage
   user_guide/api_usage
   user_guide/analytic_continuation
   user_guide/cite
   Changelog <changelog>

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Tutorials

   tutorial/rpa/index
   tutorial/g0w0/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   Build <examples/build/index>
   Driver <examples/driver/index>
   API <examples/api/index>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Zone

   develop/documentation
   develop/design
   develop/develop_tips
   develop/dataset_format
   develop/known_issues
   develop/licensing
   API Documentation <doxygen/librpa/html/index>

..
   Indices and tables
   ==================
   
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
