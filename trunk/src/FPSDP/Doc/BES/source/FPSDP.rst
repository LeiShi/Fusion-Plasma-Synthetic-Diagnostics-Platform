Beam Emission Spectroscopy Documentation
========================================

The Beam Emission Spectroscopy (BES) synthetic diagnostic use a configuration file (e.g. :download:`bes.in <../../../../FPSDP/Diagnostics/BES/bes.in>`).
The only object that should be used for the diagnostic is :class:`FPSDP.Diagnostics.BES.bes.BES`,
it creates everything from the config file and with the function :func:`FPSDP.Diagnostics.BES.bes.BES.get_bes` will compute the number of photons captured by each fiber.


FPSDP Directory
---------------

Main directory of the library.

* :mod:`FPSDP.Diagnostics` contains the files for the beam and the BES optics.
* :mod:`FPSDP.IO` contains a function for parsing a string.
* :mod:`FPSDP.Maths` contains the quadratures points and weights.
* :mod:`FPSDP.Plasma` contains the files about the collisions and the output of the simulations.


.. toctree::
	   
   FPSDP.Diagnostics
   FPSDP.IO
   FPSDP.Maths
   FPSDP.Plasma


