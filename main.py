#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from ase.io import read, write
from SAPTFF_OpenMM import SAPT_ForceField
from schnet_nn import *
import torch
import sys, time

def main():
    #Reference electronic energies, the shift variable is used for the H22 element
    ref_A_acetate = -228.3671507582417632
    ref_B_emim = -344.2359084163039711
    ref_A_acetic = -228.9282132443254056
    ref_B_nhc = -343.8068210638700180
    shift = ref_A_acetic + ref_B_nhc - ref_A_acetate - ref_B_emim
    H_kJmol = 2625.5
    shift *= H_kJmol

    #PDB, forcefield and residue file for constructing SAPT_ForceField classes
    pdbtemplate_d1 = "ff_files/emim.pdb"
    pdbtemplate_d2 = "ff_files/nhc.pdb"
    xml_ff_file = "ff_files/pb.xml"
    xml_res_file = "ff_files/pb_residues.xml"

    diabat1_saptff = SAPT_ForceField(pdbtemplate_d1, xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_intra_res=[0, 1], platformtype='OpenCL')
    diabat1_saptff.set_react_res(0, 3)
    nn_atoms = diabat1_saptff.get_nn_atoms()

    #Constuct residue list in order to partition total dimer positions into separate monomer
    #positions for the intramolecular neural networks included in the Diabat_NN classes
    res_list1 = diabat1_saptff.res_list()
    diabat1_nn = Diabat_NN("ff_files/emim_model", "ff_files/acetate_model", "ff_files/d1_apnet", res_list1, diabat1_saptff.has_periodic_box)

    nn_indices_diabat1 = np.arange(0, len(nn_atoms), 1).tolist()
    diabat1 = Diagonal(diabat1_saptff, diabat1_nn, nn_atoms, nn_indices_diabat1, return_positions)

    diabat2_saptff = SAPT_ForceField(pdbtemplate_d2, xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_intra_res=[0, 1], platformtype='OpenCL')
    diabat2_saptff.set_react_res(1, 7)
    
    res_list2 = diabat2_saptff.res_list()
    diabat2_nn = Diabat_NN("ff_files/nhc_model", "ff_files/acetic_model", "ff_files/d2_apnet", res_list2, diabat1_saptff.has_periodic_box)
    
    nn_indices_diabat2 = np.arange(0, len(nn_atoms), 1).tolist()
    nn_indices_diabat2.insert(25, nn_indices_diabat2.pop(3))
    diabat2 = Diagonal(diabat2_saptff, diabat2_nn, nn_atoms, nn_indices_diabat2, pos_diabat2, shift=shift)
    diabats = [diabat1, diabat2]

    #Model for predicting the H12 energy and force
    h12 = Coupling("ff_files/h12_model", diabat1_saptff.has_periodic_box)

    coupling = [h12]

    #Main class for MD simulation inside of ASE
    ase_md = ASE_MD("input.xyz", './test', diabats, coupling, nn_atoms)
    energy, force = ase_md.calculate_single_point()
    print(energy)

if __name__ == "__main__":
    main()

