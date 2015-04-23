Beam Emission Spectroscopy Documentation
========================================

The Beam Emission Spectroscopy (BES) synthetic diagnostic use a configuration file (e.g. :download:`bes.in <../../../../FPSDP/Diagnostics/BES/bes.in>`).
The only object that should be used for the diagnostic is the class :class:`BES <FPSDP.Diagnostics.BES.bes.BES>`,
it creates everything from the config file and with the method :func:`get_bes() <FPSDP.Diagnostics.BES.bes.BES.get_bes>` will compute the number of photons captured by each fiber.
:download:`example.py <../../../../FPSDP/Diagnostics/BES/example.py>` is an example of how to use the code.

In the documentation, I tried to specify the unit of each variable, but it is possible that I forgot a few.
I choose to use the SI system as often as possible (except for the beam energy), therefore it should be the first guess.   
   

The following graph shows the dependencies between the different classes.
A black arrow shows an attribut dependency and a red one an inheritance.

.. graphviz::
   
   digraph FPSDP{
   size="5"; ratio=fill; node[fontsize=24];

   BES->Beam1D->Collisions->ADAS_file; BES_ideal->XGC_Loader_BES;
   Beam1D->XGC_Loader_BES; ADAS_file->ADAS21[color="red"]; ADAS_file->ADAS22[color="red"];

   subgraph cluster_bes { label="FPSDP/Diagnostics/BES/bes.py"; BES; BES_ideal; }

   subgraph cluster_beam { label="FPSDP/Diagnostics/Beam/beam.py"; Beam1D; }

   subgraph cluster_XGC { label="FPSDP/Plasma/XGC_Profile/XGC_Loader_BES.py"; XGC_Loader_BES; }

   subgraph cluster_ADAS { label="FPSDP/Plasma/Collisions/ADAS_file.py"; ADAS_file; ADAS21; ADAS22; }

   subgraph cluster_collisions { label="FPSDP/Plasma/Collisions/collisions.py"; Collisions; }

   }

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

