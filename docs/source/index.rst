.. moolib documentation master file, created by
   sphinx-quickstart on Thu Aug 19 14:02:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Moolib Documentation
==================================


.. code-block:: html

   moolib  - a communications library for distributed ml training

   moolib offers general purpose RPC with automatic transport
   selection (shared memory, tcp/ip, infiniband) allowing models 
   to data-parallelise their training and synchronize gradients 
   and model weights across many nodes.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:
   
   What is Moolib? <self>


Getting Started
---------------
   
   Install from GitHub
   
   .. code-block:: bash

      pip install git+https://github.com/facebookresearch/moolib

   Build from source: **Linux**
   
   .. code-block:: bash

      git clone --recursive git@github.com:facebookresearch/moolib
      cd moolib && pip install .

   Build from source: **MacOS**

   .. code-block:: bash

      git clone --recursive git@github.com:facebookresearch/moolib
      cd moolib && USE_CUDA=0 pip install .

   How to host docs:

   .. code-block:: bash

      # after installation
      pip install sphinx==4.1.2
      cd docs && ./run_docs.sh


API
-----------------

Classes
"""""""

.. currentmodule:: moolib
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: moolib_class_template.rst

   Accumulator
   Batcher
   Broker
   EnvPool
   EnvStepper
   Group
   Rpc

Methods
"""""""

.. currentmodule:: moolib
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: 

   create_uid
   set_logging
   set_log_level
   set_max_threads

Futures
"""""""

.. currentmodule:: moolib
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: moolib_result_template.rst

   AllReduce
   Future
   EnvStepperFuture

Examples
-----------------

   Some examples are in the ``./examples`` directory.


.. .. automodule:: moolib
..    :members:


Search
-----------------

* :ref:`search`
