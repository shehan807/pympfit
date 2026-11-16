============
Installation
============

Dependencies
------------

PyMPFIT depends on `OpenFF <https://openforcefield.org/>`_ packages and
`Psi4 <https://psicode.org/>`_/`GDMA <https://github.com/psi4/gdma>`_,
which are distributed via conda-forge. *Works with conda, mamba, or micromamba.*

.. code-block:: bash

   conda create -n pympfit python=3.12 openff-recharge openff-interchange openff-utilities psi4 pygdma -c conda-forge -y
   conda activate pympfit

Install
-------

.. code-block:: bash

   pip install pympfit

MBIS multipoles use Psi4's built-in support, so they need no packages beyond the
``psi4`` installed above. The GDMA path uses ``pygdma``, which is installed above too.

Optional: Bayesian Virtual Site Fitting
---------------------------------------

.. code-block:: bash

   pip install pyro-ppl arviz matplotlib sphericart-torch

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/shehan807/pympfit.git
   cd pympfit
   pip install -e ".[test]"
