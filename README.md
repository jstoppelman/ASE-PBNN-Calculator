# ASE_PBNN_Calculator

**This repo is now an archive. The updated code is being worked on [here](https://github.com/jstoppelman/PBNN)**

This code computes the energy and force for a given structure using the "PB/NN" potential. The output energy units are eV and the output force units are eV/A (default ASE units). The code is based around an ASE calculator, and modeled after the SchNetPack ASE Interface. The ff_files directory contains the OpenMM force fields and neural network models. In order to run a MD simulation of the dimer, switch 
line 46 to "ase_md.run_md(1000)", or however many simulation steps you would like to take. 

Dependencies:
  OpenMM  
  NumPy  
  ASE  
  PyTorch  
  SchNetPack (download custom version from https://github.com/jstoppelman/schnetpack as this contains the code needed to run AP-Net)
