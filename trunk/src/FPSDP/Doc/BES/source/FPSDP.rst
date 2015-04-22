Beam Emission Spectroscopy Documentation
========================================

The Beam Emission Spectroscopy (BES) synthetic diagnostic use a configuration file (e.g. :download:`bes.in <../../../../FPSDP/Diagnostics/BES/bes.in>`).
The only object that should be used for the diagnostic is :class:`BES <FPSDP.Diagnostics.BES.bes.BES>`,
it creates everything from the config file and with the function :func:`get_bes <FPSDP.Diagnostics.BES.bes.BES.get_bes>` will compute the number of photons captured by each fiber.
:download:`example.py <../../../../FPSDP/Diagnostics/BES/example.py>` is an example of how to use the code.

In the documentation, I tried to specify the unit of each variable, but it is possible that I forgot a few.
I choose to use the SI system as often as possible (except for the beam energy), therefore it should be the first guess.

FPSDP Directory
---------------

The main directory of the library contains the following directory:

* :ref:`diag_dir` contains the files for the beam and the BES optics.
* :ref:`IO` contains a function for parsing a string.
* :ref:`Maths` contains the quadratures points and weights.
* :ref:`Plasma` contains the files about the collisions and the output of the simulations.
* :ref:`General` contains the definition of a few constants.

Table of content
----------------

.. toctree::
	   
   FPSDP.Diagnostics
   FPSDP.IO
   FPSDP.Maths
   FPSDP.Plasma
   FPSDP.GeneralSettings

