import sys
from simtk.openmm.app import *
from SAPTFF_OpenMM import SAPT_ForceField
from pbnn_calculator import *
import random as rd
import numpy as np
from ase.optimize import BFGS
from ase import Atom
from copy import deepcopy
from ase.constraints import FixCom

def read_file(fname):
    """
    Reads input file for PB/NN and gets software options.
    Options are listed in pbnn_input.txt file

    Parameters
    ------------
    fname : str
        str pointing to input file name

    Returns
    ------------
    sim_arg_values : dict
        Dictionary containing arguments for setting up the simulation
    system_arg_values : dict
        Dictionary containing arguments for setting up the Hamiltonian
    plumed_arg_values : dict
        Dictionary containing arguments for setting up Plumed for biased sampling. May be empty if not present in the input file
    adaptive_arg_values : dict
        Dictionary containing arguments for setting up adaptive residues. May be missing if not present in the input file
    """

    #Required argument. Dictionary indicates if they have been set or not
    required_sim_args = {
            'ensemble': False,
            'time_step': False, 
            'run_time': False,
            'number_of_diabats': False,
            'OpenMM_ForceField': False,
            'OpenMM_Residue': False,
            'Reacting_Atom': False,
            'Accepting_Atoms': False,
            'NN_Intra': False,
            'NN_Inter': False,
            'Couplings': False
            }

    #Optional arguments
    optional_sim_args = [
            'temperature',
            'friction_coefficient',
            'sim_dir',
            'store_frq'
            ]

    sim_arg_values = {}
    system_arg_values = {}
    plumed_arg_values = {}
    adaptive_arg_values = {}

    #Identify different parts of the input file and grab the arguments. A bit messy currently...
    sim_settings = False
    system_settings = False
    plumed_settings = False
    adaptive_settings = False
    diabat_shift = []
    nn_intra_models = []
    nn_inter_models = []
    nn_inter_resids = []
    coupling_models = []
    coupling_diabatids = []
    for line in open(fname):
        ls = line.split()
        if sim_settings:
            if ls[0] in required_sim_args:
                required_sim_args[ls[0]] = True
                sim_arg_values[ls[0]] = ls[2]
            elif ls[0] in optional_sim_args:
                sim_arg_values[ls[0]] = ls[2]
            elif "#" in line:
                sim_settings = False
        if system_settings: 
            if "number_of_diabats" in line:
                required_sim_args[ls[0]] = True
                num_diabats = int(ls[2])
            elif "OpenMM_pdbfile" in line:
                pdbfiles = ls[2:2+num_diabats]
                system_arg_values["OpenMM_pdbfile"] = pdbfiles
            elif "Reacting_Atom" in line:
                required_sim_args["Reacting_Atom"] = True
                system_arg_values["Reacting_Atom"] = ls[2:2+num_diabats]
            elif "Accepting_Atoms" in line:
                required_sim_args["Accepting_Atoms"] = True
                accepting_atoms = []
                for val in ls[2:]:
                    if "!" in val:
                        break
                    accepting_atoms.append(val)
                system_arg_values["Accepting_Atoms"] = accepting_atoms
            elif "Diabat_shift" in line:
                for val in ls[2:2+num_diabats-1]: diabat_shift.append(float(val))
                system_arg_values["Diabat_shift"] = diabat_shift
            elif "NN_Intra" in line:
                required_sim_args["NN_Intra"] = True
                nn_intra_models.append(ls[2:2+num_diabats])
                system_arg_values["NN_Intra"] = nn_intra_models
            elif "NN_Inter" in line:
                required_sim_args["NN_Inter"] = True
                read_input = False
                nn_inter_resid = []
                nn_inter_model = []
                for i in range(len(ls)):
                    if read_input:
                        if "," in ls[i]:
                            resids = []
                            for val in ls[i].split(","):
                                resids.append(int(val))
                            nn_inter_resid.append(resids)
                        elif '!' not in ls[i]:
                            nn_inter_model.append(ls[i])
                        if (ls[i] == ls[-1]) or ('!' in ls[i]):
                            read_input = False
                            nn_inter_resids.append(nn_inter_resid)
                            nn_inter_models.append(nn_inter_model)
                    if "=" in ls[i]: read_input = True
                system_arg_values["NN_Inter_Models"] = nn_inter_models
                system_arg_values["NN_Inter_Resids"] = nn_inter_resids
            elif "Couplings" in line:
                required_sim_args["Couplings"] = True
                read_input = False
                for i in range(len(ls)):
                    if read_input: 
                        if ('!' in ls[i]):
                            read_input = False
                        elif "," in ls[i]:
                            diabatids = []
                            for val in ls[i].split(","):
                                diabatids.append(int(val))
                            coupling_diabatids.append(diabatids)
                        else:
                            coupling_models.append(ls[i])
                        if (ls[i] == ls[-1]):
                            read_input = False
                    if "=" in ls[i]: read_input = True
                system_arg_values["Coupling_Models"] = coupling_models
                system_arg_values["Coupling_Diabatids"] = coupling_diabatids
            elif ls[0] in required_sim_args:
                required_sim_args[ls[0]] = True
                system_arg_values[ls[0]] = ls[2]
        if plumed_settings:
            if "Umbrella_residues" in line:
                umbrella_residue = {}
                residue_num = 0
                for res in range(2, 5, 2):
                    if ',' in ls[res+1]:
                        umbrella_residue[residue_num] = {int(ls[res]):[int(atom) for atom in ls[res+1].split(',')]}
                    else:
                        umbrella_residue[residue_num] = {int(ls[res]):int(ls[res+1])}
                    residue_num += 1
                plumed_arg_values["Umbrella_residues"].append(umbrella_residue)
            if "Umbrella_force_constants" in line:
                umbrella_force_constant = []
                for f in range(2, 2+len(plumed_arg_values["Umbrella_residues"])):
                    umbrella_force_constant.append(float(ls[f]))
                plumed_arg_values["Umbrella_force_constants"] = umbrella_force_constant
            if "Umbrella_center_positions" in line:
                umbrella_ref_positions = []
                for pos in range(2, 2+len(plumed_arg_values["Umbrella_residues"])):
                    umbrella_ref_positions.append(float(ls[pos]))
                plumed_arg_values["Umbrella_center_positions"] = umbrella_ref_positions
        if adaptive_settings:
            adaptive_arg_values["adaptive_region"] = True
            if "Molecule_adapt" in line:
                atom_adapt_name = {}
                atom_adapt_name[ls[2]] = []
                for name in ls[3:]:
                    if "!" in name:
                        break
                    atom_adapt_name[ls[2]].append(name)
                adaptive_arg_values["Molecules_adapt"].append(atom_adapt_name)
            if "Core_mol_id" in line:
                adaptive_arg_values["Core_mol_id"] = ls[2]
            if "Core_mol_atoms" in line:
                read_input = False
                atoms = []
                for i in range(len(ls)):
                    if read_input:
                        if ('!' in ls[i]):
                            read_input = False
                        else:
                            atoms.append(int(ls[i]))
                        if (ls[i] == ls[-1]):
                            read_input = False
                    if "=" in ls[i]: read_input = True
                adaptive_arg_values["Core_mol_atoms"] = atoms
            if "R_inner" in line:
                adaptive_arg_values["R_inner"] = float(ls[2])
            if "R_outer" in line:
                adaptive_arg_values["R_outer"] = float(ls[2])
            if "FD_beta" in line:
                adaptive_arg_values["FD_beta"] = float(ls[2])
            if "FD_mu" in line:
                adaptive_arg_values["FD_mu"] = float(ls[2])
            if "adapt_nn_model" in line:
                read_input = False
                nn_res_dict = {}
                for arg in range(len(ls)):
                    if read_input:
                        if ('!' in ls[arg]):
                            read_input = False
                        else:
                            if arg%2 == 0:
                                nn_res_dict[ls[arg]] = ls[arg+1]
                        if (ls[arg] == ls[-1]):
                            read_input = False
                    if "=" in ls[arg]: read_input = True
                adaptive_arg_values["adapt_nn_model"] = nn_res_dict
            if "adapt_pdb" in line:
                adaptive_arg_values["adapt_pdb"] = ls[2]
            if "ff_shift" in line:
                adaptive_arg_values["ff_shift"] = float(ls[2])
        if "Simulation Settings" in line:
            sim_settings = True
        if "System Settings" in line:
            system_settings = True
        if "Plumed Settings" in line:
            plumed_settings = True
            plumed_arg_values["plumed_input"] = True
            plumed_arg_values["Umbrella_residues"] = []
        if "Adaptive PBNN Molecules" in line:
            adaptive_settings = True
            adaptive_arg_values["Molecules_adapt"] = []

    if not adaptive_arg_values:
        adaptive_arg_values["adaptive_region"] = False
    if not plumed_arg_values:
        plumed_arg_values["plumed_input"] = False

    #Consistency checks for various arguments

    if len(pdbfiles) != num_diabats:
        print("Number of pdb files needs to be the same as the number of diabats\n")
        sys.exit()

    if not diabat_shift:
        diabat_shift = [0 for i in range(num_diabats)]
        print("Assuming diabat shifts should all be zero, as there were none entered in the PDBFile\n")
    elif len(diabat_shift) == num_diabats - 1:
        diabat_shift.insert(0, 0.0)
    elif len(diabat_shift) != num_diabats - 1:
        print("The number of diabatic states entered in the input file should be one less than the number of diabatic states\n")
        sys.exit()
    
    if len(nn_intra_models) != num_diabats:
        print("Enter intramolecular neural network models for all diabats\n")
        sys.exit()
    else:
        nn_intra_len = False
        for diabat, nn_intra_model in enumerate(nn_intra_models):
            if len(nn_intra_model) != num_diabats:
                print(f"Diabat {diabat} has an incorrect number of intramolecular neural network models listed\n")
                sys.exit()

    if len(nn_inter_models) != num_diabats:
        print("Enter intermolecular neural network models for all diabats\n")
        sys.exit()
    else:
        nn_inter_len = False
        for diabat, nn_inter_model in enumerate(nn_inter_models):
            if len(nn_inter_model) != num_diabats-1:
                print(f"Diabat {diabat} has an incorrect number of intermolecular neural network models listed \n")
                sys.exit()
            elif len(nn_inter_model) != len(nn_inter_resid):
                print(f"Diabat {diabat} has an incorrect number of intermolecular neural network models compared to the number of intermolecular residues ids\n")
                sys.exit()

    if False in required_sim_args.values():
        missing_args = [k for k, v in required_sim_args.items() if not v]
        for arg in missing_args:
            print(f"Argument {arg} missing in input file\n")
        sys.exit()

    return sim_arg_values, system_arg_values, plumed_arg_values, adaptive_arg_values

def construct_intra_nn_list(nn_intra_model, res_list, exclude_intra_res, periodic):
    """
    Parameters
    --------------
    nn_intra_model : list
        List of strings pointing to saved NN models
    res_list : list
        List containing atoms that are part of each residue
    exclude_intra_res : list
        List containing residue index
    periodic : bool
        Bool denoting whether the system has a periodic box
    
    Returns
    --------------
    diabat_intra_nn : list
        List of containing NN_Intra_Diabat classes for each residue
    """
    diabat_intra_nn = []
    for j in range(len(res_list)):
        intrann_diabat = NN_Intra_Diabat("ff_files/"+nn_intra_model[j], res_list[j], exclude_intra_res[j], periodic)
        diabat_intra_nn.append(intrann_diabat)
    return diabat_intra_nn

def construct_inter_nn_list(nn_inter_model, nn_inter_resid, res_list, periodic):
    """
    Parameters
    --------------
    nn_inter_model : list
        List of strings pointing to saved NN models
    nn_inter_resid : list
        List containing residue indices for which the neural network will be applied to
    res_list : list
        List containing atom indices of each residue
    periodic : bool
        Bool denoting whether the system has a periodic box
    
    Returns
    --------------
    diabat_inter_nn : list
        List of containing NN_Inter_Diabat classes for each residue
    """
    diabat_inter_nn = []
    for j in range(len(nn_inter_resid)):
        residues = nn_inter_resid[j]
        internn_res_list = [res_list[residues[0]], res_list[residues[1]]]
        internn_diabat = NN_Inter_Diabat("ff_files/"+nn_inter_model[j], internn_res_list, residues, periodic)
        diabat_inter_nn.append(internn_diabat)
    return diabat_inter_nn

def initialize_system(atoms, fname):
    """
    Sets up the ASE_MD object and passes all arguments it needs. Potentially could be simplified...

    Parameters 
    -------------
    atoms : ASE atoms object
        The atoms object
    fname : str
        String pointing to input file
 
    Returns
    -------------
    ase_md : ASE_MD object
        object for running the simulation
    run_time : int
        number of time steps the simulation will be run for
    """

    #Read file input and get back info needed to build the system
    sim_arg_values, system_arg_values, plumed_arg_values, adaptive_arg_values = read_file(fname)
    
    #PDB files, one for each diabat
    pdbfiles = system_arg_values["OpenMM_pdbfile"]
    number_of_diabats = len(pdbfiles)
    sim_dir = sim_arg_values["sim_dir"]
    store_frq = int(sim_arg_values["store_frq"])
    if not os.path.isdir(sim_dir):
        os.mkdir(sim_dir)
    #Name of the reacting atom (atom that is donated or accepted) in each diabat
    reacting_atom = system_arg_values["Reacting_Atom"]
    #Name of the atom(s) that can accept the reacting proton in each diabat
    accepting_atoms = system_arg_values["Accepting_Atoms"]
    #OpenMM residue file
    xml_res_file = "ff_files/"+system_arg_values["OpenMM_Residue"]
    #OpenMM force field file
    xml_ff_file = "ff_files/"+system_arg_values["OpenMM_ForceField"]
    #List of intramolecular neural network models for each diabat 
    nn_intra_models = system_arg_values["NN_Intra"]
    #List of intermolecular neural network models for each diabat
    nn_inter_models = system_arg_values["NN_Inter_Models"]
    #List of residue ids that the intermolecular neural network is applied to
    nn_inter_resids = system_arg_values["NN_Inter_Resids"]
    #List of models used for coupling between diabats
    coupling_models = system_arg_values["Coupling_Models"]
    #List of residue ids that the coupling elements are applied to
    coupling_diabatids = system_arg_values["Coupling_Diabatids"]
    #List of shifts with respect to diabat 1 for each diabat
    shifts = system_arg_values["Diabat_shift"]
    #Whether plumed will be used for biasing simulations
    plumed_input = plumed_arg_values["plumed_input"]
    #Bool: whether the molecules the PB/NN potential can be applied to are allowed to vary
    adapt = adaptive_arg_values["adaptive_region"]
    #Gets options for adapted NN residues
    if adapt:
        #Residue name of molecules for which the intramolecular neural network can be applied to
        molecules_adapt = adaptive_arg_values["Molecules_adapt"]
        #Molecule which represents center of adaptive site
        core_mol_id = int(adaptive_arg_values["Core_mol_id"])
        #List of atoms within the core molecule. A dummy particle will be placed at their center of mass and the interpolating function will use this as the reference position
        core_mol_atoms = adaptive_arg_values["Core_mol_atoms"]
        #All adaptive molecules will be treated with the neural network within R_inner.
        R_inner = adaptive_arg_values["R_inner"]
        #All adaptive molecules will be trated with the force field beyond R_outer
        R_outer = adaptive_arg_values["R_outer"]
        #Neural network molecule used for adaptive molecules
        adapt_nn_model = adaptive_arg_values["adapt_nn_model"]
        #PDB used for force field between R_inner and R_outer
        adapt_pdb = adaptive_arg_values["adapt_pdb"]
        #Beta for exponent of Fermi-Dirac function
        FD_beta = adaptive_arg_values["FD_beta"]
        #Center for Fermi-Dirac function
        FD_mu = adaptive_arg_values["FD_mu"]
        ff_shift = adaptive_arg_values["ff_shift"]

    diabats = []
    #Number of atoms that can accept the reacting group
    can_accept = len(accepting_atoms)
    plumed_commands = []
    #Loop through the number of states and set up a Diabat class for each one
    for i in range(number_of_diabats):
        if i == 0:
            pdb = PDBFile(f"ff_files/{pdbfiles[i]}")
            #Find list of all molecules that can undergo reactions
            #Residue IDs that can donate a group
            react_resids = []
            #Atom IDs making up the donating group
            react_atomids = []
            #Residue IDs that can accept a group
            accept_resids = []
            #Atom IDs that can accept a group
            accept_atomids = []
            #Residue names within Diabat 1
            diabat1_resnames = []
            #Name of the atom that is leaving/being accepted
            reacting_atom_name = reacting_atom[i] 
            for res in pdb.topology.residues():
                diabat1_resnames.append(res.name)
                accept_atomids_mol = []
                for atom in range(len(res._atoms)):
                    #Donating group info
                    if res._atoms[atom].name in reacting_atom:
                        react_resids.append(res.index)
                        react_atomids.append(res._atoms[atom].index)
                        #ID of the leaving/accepted atom within a residue
                        reacting_atom_mol_id = atom
                    if res._atoms[atom].name in accepting_atoms:
                        accept_resids.append(res.index)
                        accept_atomids.append(res._atoms[atom].index)
                        accept_resname = res.name
                        accept_atomids_mol.append(atom)

            #Randomly select one of the possible reacting molecules to be "reactive"
            #select_reacting_mol = rd.choice(react_resids)
            select_reacting_mol = 36
            #Get atom index of the reacting atom within the reacting molecule
            select_reacting_atom = react_atomids[react_resids.index(select_reacting_mol)]

            #Distances from the reacting atom to all potential accepting atoms
            distances = atoms.get_distances(select_reacting_atom, accept_atomids, mic=True)
            #Reshape distances array so that it is a (num_accept_molecules, num_accept_atoms_within_molecule) array
            distances = distances.reshape(int(distances.shape[0]/can_accept), can_accept)
            #For each accepting atom within a molecule, get the minimum distance of all atoms within that molecule with the reacting atom
            dist_min = distances.min(axis=1)
            
            #Sort array to identify the residues with the shortest distance to the reacting atom
            dist_order = np.argsort(dist_min)
            #Reshape the array that contains the accepting atomids
            accept_atomids = np.asarray(accept_atomids)
            accepting_atomids = accept_atomids.reshape(int(accept_atomids.shape[0]/can_accept), can_accept)
            #Get accepting atom ids that correspond to the residues with the shortest distance to the reacting group. Determined by number of molecules that can accept.
            accepting_atomids = accepting_atomids[dist_order[0:can_accept]]
            #dist_order_top is rereferenced so that the accepting residue ids are within the global topology, not just within the potential accepting residue ids
            dist_order_top = dist_order + accept_resids[0]
            dist_order_top = dist_order_top.tolist()
            #Residue ids within the topology that can accept the protons
            accepting_mol = dist_order_top[0:number_of_diabats-1]

            #Residues that will be treated with the neural network. Exclusions will be created within the SAPT_ForceField class for the intramolecular components
            exclude_res = [select_reacting_mol, *accepting_mol]
            diabat_saptff = SAPT_ForceField("ff_files/"+pdbfiles[i], xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_intra_res=deepcopy(exclude_res), platformtype='CUDA')
            #Set reacting residue and reacting atom within residue for saptff
            diabat_saptff.set_react_res(select_reacting_mol, reacting_atom_mol_id)
            nn_atoms = diabat_saptff.get_nn_atoms()
            
            nn_res_list = diabat_saptff.get_nn_res_list()
            print("Optimizing initial neural network molecules with BFGS\n")
            for idx, mol in enumerate(nn_res_list):
                atoms_nn = atoms[mol]
                res = exclude_res[idx]
                #Hamiltonian for singular neural network
                nn_hamiltonian = NeuralNetworkHamiltonian(atoms_nn, "ff_files/"+nn_intra_models[i][idx], res, diabat_saptff.has_periodic_box)
                atoms_nn.set_calculator(nn_hamiltonian)
                atoms_nn.set_constraint(FixCom())
                opt = BFGS(atoms=atoms_nn)
                opt.run(fmax=0.05)
                atoms.positions[mol] = atoms_nn.positions
            
            #Build list of molecules for which the intramolecular neural networks will be applied to
            res_list = diabat_saptff.get_nn_res_list()
            exclude_intra_res = diabat_saptff.exclude_intra_res
            nn_intra_model = nn_intra_models[i]
            diabat_intra_nn = construct_intra_nn_list(nn_intra_model, res_list, exclude_intra_res, diabat_saptff.has_periodic_box)

            #Build list of molecules for which the intermolecular neural networks will be applied to
            nn_inter_resid = nn_inter_resids[i]
            nn_inter_model = nn_inter_models[i]
            diabat_inter_nn = construct_inter_nn_list(nn_inter_model, nn_inter_resid, res_list, diabat_saptff.has_periodic_box)

            #Reorders positions to be in the order for that diabat. Since this is diabat 1, the positions will be returned as they are
            return_positions = Reorder_Indices()
            kwargs = {}
            #Performs some routines for adaptive molecules for which the PBNN intramolecular neural networks can be turned on and off for
            if adapt:
                real_atom_res_list = diabat_saptff.get_res_list()
                #Reacting residue. An umbrella potential will be placed on the center of mass of selected atoms for this residue
                core_res = real_atom_res_list[exclude_intra_res[core_mol_id]]
                com_umbrella_atoms = []
                for atom in core_mol_atoms:
                    com_umbrella_atoms.append(core_res[atom])
                atoms_center_mol = atoms[com_umbrella_atoms]

                #Put dummy atom on center of mass of selected core atoms. This is so spurious forces for adaptive smoothing potential are only applied to the dummy atom for reacting group
                com_pos = atoms_center_mol.get_center_of_mass()
                
                #Get distances from center of active site to atoms that can adapt their Hamiltonian
                disp = com_pos - atoms.positions[accept_atomids]
                cell = np.asarray(atoms.get_cell())
                cell = np.expand_dims(cell, axis=0)
                cell = np.repeat(cell, accept_atomids.shape[0], axis=0)
                diff = np.zeros((accept_atomids.shape[0], 3))
                shift_z = -np.sign(disp[:,2]) * np.floor(abs(disp[:,2]/cell[:,2,2])+0.5)
                shift_z = np.repeat(shift_z[...,None], 3, axis=1)
                shift_y = -np.sign(disp[:,1]) * np.floor(abs(disp[:,1]/cell[:,1,1])+0.5)
                shift_y = np.repeat(shift_y[...,None], 3, axis=1)
                shift_x = -np.sign(disp[:,0]) * np.floor(abs(disp[:,0]/cell[:,0,0])+0.5)
                shift_x = np.repeat(shift_x[...,None], 3, axis=1)
                diff += cell[:, 2] * shift_z
                diff += cell[:, 1] * shift_y
                diff += cell[:, 0] * shift_x
                
                disp += diff
                distances = np.sqrt((disp*disp).sum(axis=1))
                distances_adapt = distances.reshape(int(distances.shape[0]/can_accept), can_accept)
                dist_min_adapt = distances_adapt.min(axis=1)
                dist_order_adapt = np.argsort(dist_min_adapt)
                dist_order_adapt_top = dist_order_adapt + accept_resids[0]

                #Determine which residues are currently excluded in SAPT_ForceField class and which ones need to be excluded based on their location from center of active site
                exclude_dist_adapt = dist_min_adapt[dist_order_adapt]
                #[1:] removes the already excluded reacting molecule from the exclude_intra_res list
                current_exclude_residues = diabat_saptff.exclude_intra_res[1:]
                exclude = np.where(exclude_dist_adapt < R_outer)[0]
                correct_exclude_residues = [dist_order_adapt_top[res] for res in exclude]

                #Determine if any residues listed in current_exclude_residues are not in correct_exclude_residues
                remove_residue = [res for res in current_exclude_residues if res not in correct_exclude_residues]
                #Determines if any residues listed in correct_exclude_residues are not listed in current_exclude_residues
                add_residue = [res for res in correct_exclude_residues if res not in current_exclude_residues]
                #Determines which residues are solely in the adaptive region
                adaptive_region = np.where(np.logical_and(exclude_dist_adapt < R_outer, exclude_dist_adapt > R_inner))[0]
                #Determines which residues are in the environment region
                environment_region = np.where(exclude_dist_adapt > R_outer)[0]
                adaptive_residues = [dist_order_adapt_top[res] for res in adaptive_region]
                environment_residues = [dist_order_adapt_top[res] for res in environment_region]
                
                #Removes and adds exclusions from residues 
                if remove_residue:
                    diabat_saptff.create_exclusions_intra_remove(remove_residue)
                    current_exclude = diabat_saptff.exclude_intra_res
                if add_residue:
                    diabat_saptff.create_exclusions_intra_add(add_residue)
                    current_exclude = diabat_saptff.exclude_intra_res

                #Idenfity whether the residues currently treated solely with intramolecular neural networks are not in the adaptive/environment region
                adjust_intra_nn = []
                for nn_intra in diabat_intra_nn:
                    if (nn_intra.residue not in adaptive_residues) and (nn_intra.residue not in environment_residues):
                        adjust_intra_nn.append(nn_intra)
                diabat_intra_nn = adjust_intra_nn
                
                #For residues in adaptive region, create Adaptive_Intramolecular class
                res_names = diabat_saptff.get_res_names()
                #Residue name for adaptive residues
                adaptive_res = [res_names[idx] for idx in adaptive_residues][0]
                #Number of atoms for which the adaptive energy/force will be computed for
                adapt_atom_num = len(real_atom_res_list[adaptive_residues[0]])
                adapt_saptff = SAPT_ForceField("ff_files/"+adapt_pdb, xml_res_file, xml_ff_file, Drude_hyper_force=True, platformtype='CUDA') 
                adaptive_intramolecular = Adaptive_Intramolecular(adapt_nn_model, adapt_saptff, adaptive_res, adapt_atom_num, real_atom_res_list, accept_atomids_mol, R_inner, R_outer, com_umbrella_atoms, ff_shift)
                adaptive_intramolecular.set_adaptive_residues(adaptive_residues)

                res_list = diabat_saptff.get_res_list()
                kwargs['res_list'] = res_list
                kwargs['ff_shift'] = ff_shift
                kwargs['num_accept'] = len(accept_atomids)/can_accept
                diabat = Diabat(diabat_saptff, diabat_intra_nn, diabat_inter_nn, nn_atoms, return_positions, shift=shifts[i], adaptive=True, **kwargs)
            
            else:
                diabat = Diabat(diabat_saptff, diabat_intra_nn, diabat_inter_nn, nn_atoms, return_positions, shift=shifts[i])

            diabats.append(diabat)

        else:
            #Do the same for the other diabats
            pdb = PDBFile(f"ff_files/{pdbfiles[i]}")
            #Determine which residues are different compared to diabat 1
            #Dictionary of initial resnames (names of the changed residue in diabat 1)
            resname_initial = {}
            #Dictionary of final resnames (names of the changed residues in the current diabat)
            resname_final = {}
            reacting_atom_name = reacting_atom[i]
            new_resname = []
            new_name = {}
            changed_residue = exclude_res[i]
            switch = True
            #Determine initial resnames
            for index, res in enumerate(pdb.topology.residues()):
                if res.name != diabat1_resnames[index]:
                    new_name[diabat1_resnames[index]] = res.name
                if res.name != diabat1_resnames[index] and select_reacting_mol != index and index != changed_residue:
                    resname_initial[index] = diabat1_resnames[index]
                if res.name != diabat1_resnames[index] and select_reacting_mol != index and index == changed_residue:
                    switch = False

            #Determine final resnames
            for index, res in enumerate(pdb.topology.residues()):
                if index == select_reacting_mol and diabat1_resnames[index] in list(resname_initial.values()):
                    resname_final[select_reacting_mol] = new_name[diabat1_resnames[index]]
                if index == changed_residue:
                    resname_final[index] = new_name[diabat1_resnames[index]]
                for atom in range(len(res._atoms)):
                    if res._atoms[atom].name in reacting_atom:
                        reacting_atom_mol_id = atom
            
            diabat_saptff = SAPT_ForceField(f"ff_files/{pdbfiles[i]}", xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_intra_res=deepcopy(exclude_res), platformtype='CUDA')
            #The residues currently selected to accept the reacting atoms may not be the same residues which are labeled as the accepting residues in the PDB File. Change the topology so this is the case
            if switch: diabat_saptff.change_topology(resname_initial, resname_final)
            diabat_saptff.set_react_res(changed_residue, reacting_atom_mol_id)
            name_switch_accept = diabat1_resnames[changed_residue]
            diabat_saptff.set_accept_res(changed_residue, name_switch_accept)
            nn_atoms = diabat_saptff.get_nn_atoms()

            res_list = diabat_saptff.get_nn_res_list()
            exclude_intra_res = diabat_saptff.exclude_intra_res
            diabat_intra_nn = []
            nn_intra_model = nn_intra_models[i]
            diabat_intra_nn = construct_intra_nn_list(nn_intra_model, res_list, exclude_intra_res, diabat_saptff.has_periodic_box)

            nn_inter_resid = nn_inter_resids[i]
            nn_inter_model = nn_inter_models[i]
            diabat_inter_nn = []
            diabat_inter_nn = construct_inter_nn_list(nn_inter_model, nn_inter_resid, res_list, diabat_saptff.has_periodic_box)

            #insert_loc: location to insert the reacting atom in the Reorder_Indices class 
            insert_loc = diabat_saptff.nn_res_list[i][-1]
            pos_reorder = Reorder_Indices(diabats[0].saptff.react_atom, accepting_atomids[i-1], insert_loc)

            kwargs = {}
            if adapt:
                #Same function as above to determine which residues need to be treated with the Adaptive_Intramolecular class and which do not
                correct_exclude_residues = [dist_order_adapt_top[res] for res in exclude]
                remove_residue = [res for res in current_exclude_residues if res not in correct_exclude_residues]
                add_residue = [res for res in correct_exclude_residues if res not in current_exclude_residues]
                adaptive_region = np.where(np.logical_and(exclude_dist_adapt < R_outer, exclude_dist_adapt > R_inner))[0]
                environment_region = np.where(exclude_dist_adapt > R_outer)[0]
                adaptive_residues = [dist_order_adapt_top[res] for res in adaptive_region]
                environment_residues = [dist_order_adapt_top[res] for res in environment_region]

                if remove_residue:
                    diabat_saptff.create_exclusions_intra_remove(remove_residue)
                    current_exclude = diabat_saptff.exclude_intra_res
                if add_residue:
                    diabat_saptff.create_exclusions_intra_add(add_residue)
                    current_exclude = diabat_saptff.exclude_intra_res

                adjust_intra_nn = []
                for nn_intra in diabat_intra_nn:
                    if (nn_intra.residue not in adaptive_residues) and (nn_intra.residue not in environment_residues):
                        adjust_intra_nn.append(nn_intra)
                diabat_intra_nn = adjust_intra_nn

                #Form a list of atoms in each residue, with the atom indices being the same indices in diabat 1
                diabat_res_list = diabat_saptff.get_res_list()
                #Remove dummy atom
                positions = atoms.get_positions()
                inds = [index for index in range(len(positions))]
                inds.insert(insert_loc, inds.pop(diabats[0].saptff.react_atom))
                diabat_res_list_d1 = []
                begin = 0
                for res in diabat_res_list:
                    end = begin + len(res)
                    diabat_res_list_d1.append(inds[begin:end])
                    begin = end
                kwargs['res_list'] = diabat_res_list_d1
                kwargs['ff_shift'] = ff_shift
                kwargs['num_accept'] = len(accept_atomids)/can_accept
                diabat = Diabat(diabat_saptff, diabat_intra_nn, diabat_inter_nn, nn_atoms, pos_reorder, shift=shifts[i], name=f'diabat{i}', adaptive=True, **kwargs)

            else:
                diabat = Diabat(diabat_saptff, diabat_intra_nn, diabat_inter_nn, nn_atoms, pos_reorder, shift=shifts[i], name=f'diabat{i}', **kwargs)

            diabats.append(diabat)
    
    couplings = []
    #Create the coupling elements
    for i in range(len(coupling_models)):
        res_list = diabats[0].saptff.get_res_list()
        res_list = [res_list[res] for res in exclude_res]
        residues = coupling_diabatids[i]
        nn_indices = [i for i in res_list[residues[0]]]
        if residues[0] != 0:
            nn_indices.append(diabats[0].saptff.react_atom)
        [nn_indices.append(i) for i in res_list[residues[1]]]
        coupling = Coupling(f"ff_files/{coupling_models[i]}", nn_indices, residues, diabats[0].saptff.has_periodic_box)
        couplings.append(coupling)

    #Plumed from ASE takes in a list of the Plumed commmands. Only support umbrella sampling here for now, could expand for others
    if plumed_input:
        #Residues for plumed
        umbrella_residues = plumed_arg_values["Umbrella_residues"]
        umbrella_force_constants = plumed_arg_values["Umbrella_force_constants"]
        #Positions to place force constants
        umbrella_center_positions = plumed_arg_values["Umbrella_center_positions"]
        #Start the line for the umbrella restraints and the print property line
        restraint = f"restraint: RESTRAINT ARG="
        print_line = f"PRINT ARG="
        #Loop through individual umbrella potentials
        for umbrella in umbrella_residues:
            #Loop through residues that the umbrella potential will be used for
            for residues in umbrella:
                resid = umbrella[residues]
                mul_atoms = []
                umbrella_atoms = []
                for key, val in resid.items():
                    if isinstance(val, list):
                        mul_atoms.append(residues)
            #if have multiple atoms that the proton could be accepted on, then we need Plumed groups and DISTANCES LOWEST
            if not mul_atoms:
                umb_number = 1
                #Create distance line and restrain/print that distance
                distance = f"d{umb_number}: DISTANCE ATOMS="
                restraint += f"d{umb_number},"
                print_line += f"d{umb_number},"
                umb_number += 1
                for residues in umbrella:
                    resid = umbrella[residues]
                    for key, val in resid.items():
                        umbrella_res = exclude_res[key]
                        res_list = diabats[0].saptff.get_res_list()
                        atom_index = res_list[umbrella_res][val]
                        distance += str(atom_index+1)+","
            else:
                group_num = 1
                for residues in umbrella:
                    if residues not in mul_atoms:
                        resid = umbrella[residues]
                        for key, val in resid.items():
                            umbrella_res = exclude_res[key]
                            res_list = diabats[0].saptff.get_res_list()
                            atom_index = res_list[umbrella_res][val]+1
                        group = f"group{group_num}: GROUP ATOMS={atom_index},"
                        group_num += 1
                        plumed_commands.append(group[:-1])
                    else:
                        for mul_atom in mul_atoms:
                            group = f"group{group_num}: GROUP ATOMS="
                            group_num += 1
                            for key, val in umbrella[mul_atom].items():
                                res_list = diabats[0].saptff.get_res_list()[exclude_res[key]]
                                atom_indices = [res_list[atom] for atom in val]
                            for atom in atom_indices:
                                group += str(atom+1)+","
                        plumed_commands.append(group[:-1])
                #Create DISTANCES LOWEST line and restrain/print out lowest distance
                distance = f"d{umb_number}: DISTANCES GROUPA=group1 GROUPB=group2 LOWEST\n" 
                restraint += f"d{umb_number}.lowest,"
                print_line += f"d{umb_number}.lowest,"
                umb_number += 1
            plumed_commands.append(distance[:-1])
        
        #Finish restraint line with the centers and force constants of the umbrella potentials
        restraint = restraint[:-1]
        restraint += " AT="
        for at in umbrella_center_positions:
            restraint += str(at/10)+","
        restraint = restraint[:-1]
        restraint += " KAPPA="
        for kappa in umbrella_force_constants:
            restraint += str(kappa)+","
        restraint = restraint[:-1]
        plumed_commands.append(restraint)
        #Finish print line
        print_line = print_line[:-1]
        print_line += f" FILE={sim_dir}/colvar.dat STRIDE=1"
        plumed_commands.append(print_line)
        for fname in os.listdir(sim_dir):
            if "colvar" in fname:
                os.remove(f"{sim_dir}/{fname}")

    #Add additional restraint on the specifiied COM
    if adapt:
        define_com = "RING_COM: COM MASS ATOMS="
        for atom in com_umbrella_atoms:
            define_com += f"{atom+1},"
        define_com = define_com[:-1] + "\n"
        position = "POSITION ATOM=RING_COM LABEL=p\n"
        plumed_commands.append(define_com)
        plumed_commands.append(position)
        define_fixed_atom = f"atom: FIXEDATOM AT={com_pos[0]/10},{com_pos[1]/10},{com_pos[2]/10}\n"
        plumed_commands.append(define_fixed_atom)
        plumed_commands.append("d: DISTANCE ATOMS=RING_COM,atom\n")
        plumed_commands.append("RESTRAINT ARG=d AT=0 KAPPA=20000.0 LABEL=restraint_com\n")

    #kwargs for ASE_MD: multiple accept residues and multiple accepting atoms, list them here. Also add the reacting atom and the molecule the reacting atom is on
    kwargs = {'mul_accept_res': accept_resname, 'mul_accept_atom': accepting_atoms, 'react_atom': diabats[0].saptff.react_atom, 'react_residue': select_reacting_mol}
    
    #Main class for MD simulation inside of ASE
    res_list = diabats[0].saptff.get_res_list()

    if diabats[0].adaptive:
        kwargs['R_inner'] = R_inner
        kwargs['R_outer'] = R_outer
        kwargs['adaptive_intramolecular'] = adaptive_intramolecular
        kwargs['center_mol_indices'] = com_umbrella_atoms

    if plumed_input or diabats[0].adaptive:
        ase_md = ASE_MD(atoms, sim_dir, diabats, couplings, coupling_diabatids, plumed_input=plumed_commands, **kwargs)
    else:
        ase_md = ASE_MD(atoms, sim_dir, diabats, couplings, coupling_diabatids, **kwargs)

    #Arguments for MD
    time_step = float(sim_arg_values["time_step"])
    ensemble = sim_arg_values["ensemble"]
    run_time = float(sim_arg_values["run_time"])
    if ensemble == "NVT":
        friction_coefficient = float(sim_arg_values["friction_coefficient"])
        temperature = float(sim_arg_values["temperature"])
        ase_md.create_system("traj", time_step=time_step, temp=temperature, store=store_frq, nvt=True, friction=friction_coefficient)
    elif ensemble == "NVE":
        ase_md.create_system("traj", time_step=time_step, store=store_frq)
    return ase_md, run_time

