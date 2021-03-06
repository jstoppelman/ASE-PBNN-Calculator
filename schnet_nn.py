import schnetpack as sch
import time 
from schnetpack import AtomsData, Properties
from schnetpack.environment import (
    APNetEnvironmentProvider,
    APModEnvironmentProvider,
    APNetPBCEnvironmentProvider, 
    APModPBCEnvironmentProvider,
    TorchEnvironmentProvider
)
from schnetpack.data import AtomsConverter
import schnetpack.train as trn
from schnetpack.md import MDUnits
import numpy as np
import torch
from torch.optim import Adam
import sys, os, shutil
from copy import deepcopy
from ase import units, Atoms
from ase.io import read, write
from ase.io import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.geometry import find_mic

def write_forces(fnm, xyz, forces, frame):
    out = open(fnm, 'a')
    sym = xyz.get_chemical_symbols()
    out.write("Frame {}\n".format(frame))
    for i in range(0, 26):
        out.write(" {0}  {1:13.10f}  {2:13.10f}  {3:13.10f}\n".format(sym[i], forces[i][0], forces[i][1], forces[i][2]))
    out.write("\n")
    out.close()

def pos_diabat2(atoms, positions):
    """
    Reorder diabat 1 positions to be in the same order as diabat 2.
    Hard-coded in for now to work for EMIM+/acetate 

    Parameters
    -----------
    positions : np.ndarray
        array containing 3*N positions for the simulated dimer

    Returns
    -----------
    positions : np.ndarray
        array containing the reordered 3*N positions
    inds : list
        list containing the indices at which the new positions
        have been modified in comparison to the old positions.
    """
    inds = [i for i in range(len(positions))]
    dists = atoms.get_distances(3, [19, 20], mic=True)
    dist1, dist2 = dists[0], dists[1]
    if dist1 < dist2:
        inds.insert(25, inds.pop(3))
        h_atom = positions[3]
        positions = np.delete(positions, 3, axis=0)
        positions = np.insert(positions, 25, [h_atom], axis=0)
    else:
        inds.insert(19, inds.pop(20))
        inds.insert(25, inds.pop(3))
        new_pos = np.empty_like(positions)
        new_pos[:] = positions
        o1_atom = positions[19]
        o2_atom = positions[20]
        new_pos[19] = o2_atom
        new_pos[20] = o1_atom
        h_atom = new_pos[3]
        new_pos = np.delete(new_pos, 3, axis=0)
        new_pos = np.insert(new_pos, 25, [h_atom], axis=0)
        positions = new_pos
    return positions, inds

def reorder(coords, inds):
    """
    Reorder the input (positions or forces from diabat 2)
    to align with the positions and forces of diabat 1

    Parameters
    -----------
    coords : np.ndarray
        array containing 3*N positions or forces for the simulated dimer
    inds : list
        list containing the indices that the diabat 1 positions
        were reordered to in pos_diabat2

    Returns
    -----------
    coords : np.ndarray
        array containing the reorderd posiitons or forces
    """
    reord_list = [inds.index(i) for i in range(len(inds))]
    return coords[reord_list]

def shift_atoms(xyz, box, res_list):
    """Function used in OpenMM to shift
    center of mass of each molecule to 
    the principal box; not currently used
    """
    for mol in res_list:
        center = np.sum(xyz[mol], axis=0)
        center *= 1.0/len(mol)
        diff = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        diff += box[2] * np.floor(abs(center[2]/box[2][2]))
        diff += box[1] * np.floor(abs((center[1] - diff[1])/box[1][1]))
        diff += box[0] * np.floor(abs((center[0] - diff[0])/box[0][0]))
        xyz[mol] -= diff
    return xyz

def shift_reacting_atom(xyz, box, res_list, react_atom):
    """
    OpenMM Drude oscillators don't work with PBCs, so 
    molecules cannot be broken up by a periodic boundary.
    This function ensures the reacting atom is shifted to be on the same 
    side of the box as the residue its bonded to in a diabatic state.

    Parameters
    -----------
    xyz : np.ndarray
        array containing 3*N positions 
    box : ASE cell object
        Cell object from ASE, which contains the box vectors
    res_list : list
        list containing the indices of the atoms of the monomer
        which the reacting atom is contained in 
    react_atom : int
        index of the reacting atom in the complex

    Returns
    -----------
    xyz : np.ndarray
        array containing the shifted position of the reacting atom 

    """
    disp = xyz[res_list[0]] - xyz[react_atom]
    diff = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    diff += box[2] * -np.sign(disp[2]) * np.floor(abs(disp[2]/box[2][2])+0.5)
    diff += box[1] * -np.sign(disp[1]) * np.floor(abs(disp[1]/box[1][1])+0.5)
    diff += box[0] * -np.sign(disp[0]) * np.floor(abs(disp[0]/box[0][0])+0.5)
    xyz[react_atom] -= diff
    return xyz

def make_molecule_whole(xyz, box, res_list):
    """
    Atoms are wrapped to stay inside of the periodic box. 
    This function ensures molecules are not broken up by 
    a periodic boundary, as OpenMM electrostatics will be 
    incorrect if all atoms in a molecule are not on the 
    same side of the periodic box.

    Parameters
    -----------
    xyz : np.ndarray
        array containing 3*N positions
    box : ASE cell object
        Cell object from ASE, which contains the box vectors
    res_list : list
        list containing the indices of the atoms of the monomer
        which the reacting atom is contained in

    Returns
    -----------
    xyz : np.ndarray
        array containing the shifted positions

    """
    shifts = np.zeros_like(xyz)
    for mol in res_list:
        mol_coords = xyz[mol]
        disp_0 = np.subtract(mol_coords[0], mol_coords[1:])
        diff = box[0][0] * -np.sign(disp_0) * np.floor(abs(disp_0/box[0][0])+0.5)
        shifts[mol[1:]] += diff
    return shifts

class Diabat_NN:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks. This computes energies and forces from the
    intra- and intermolecular neural networks for the diabatic states.
    """
    def __init__(self, nn_monA, nn_monB, nn_dimer, res_list, periodic_boundary, device='cuda'):
        """
        Parameters
        -----------
        nn_monA : str
            location of the neural network for monomer A in the diabat
        nn_monB : str
            location of the neural network for monomer B in the diabat
        nn_dimer : str
            location of the neural network for the dimer
        res_list : dictionary
            dictionary containing the indices of the monomers in the diabat
        periodic_boundary : bool
            bool indicating whether periodic boundaries are being used. 
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.nn_monA = torch.load(nn_monA)
        self.nn_monB = torch.load(nn_monB)
        self.nn_dimer = torch.load(nn_dimer)
        self.res_list = res_list
        if periodic_boundary:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=TorchEnvironmentProvider(8., device=torch.device(device)))
            self.inter_converter = AtomsConverter(device=torch.device(device), environment_provider=APNetPBCEnvironmentProvider(), res_list=res_list)
        else:
            self.converter = AtomsConverter(device=torch.device(device))
            self.inter_converter = AtomsConverter(device=torch.device(device), environment_provider=APNetEnvironmentProvider(), res_list=res_list)

    def compute_energy_intra(self, xyz):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        xyz : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        
        Returns
        -----------
        energy : np.ndarray
            Intramoleculer energy in kJ/mol
        forces : np.ndarray
            Intramolecular forces in kJ/mol/A
        """ 
        monA = self.res_list[0]
        monB = self.res_list[1]
        
        xyzA = xyz[monA]
        xyzB = xyz[monB]

        inputA = self.converter(xyzA)
        inputB = self.converter(xyzB)
        resultA = self.nn_monA(inputA)
        resultB = self.nn_monB(inputB)

        energyA = resultA['energy'].detach().cpu().numpy()[0][0]
        forcesA = resultA['forces'].detach().cpu().numpy()[0]
        energyB = resultB['energy'].detach().cpu().numpy()[0][0]
        forcesB = resultB['forces'].detach().cpu().numpy()[0]
        intra_eng = energyA + energyB
        intra_forces = np.append(forcesA, forcesB, axis=0)
        return np.asarray(intra_eng), np.asarray(intra_forces)

    def compute_energy_inter(self, xyz):
        """
        Compute the energy for the intermolecular components of the dimer.

        Parameters
        -----------
        xyz : ASE Atoms Object
            ASE Atoms Object used as the input for the neural network.

        Returns
        -----------
        energy : np.ndarray
            Intermoleculer energy 
        forces : np.ndarray
            Intermolecular forces
        """
        inp = self.inter_converter(xyz)
        result = self.nn_dimer(inp)

        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        forces[forces != forces] = 0.0
        return np.asarray(energy), np.asarray(forces)

class Harmonic_Bias:
    """
    Adds harmonic bias to the Hamiltonian, based on the ASE Harmonic calculator
    """
    def __init__(self, spring_constant, r0, index1, index2):
        self.k = spring_constant
        self.r0 = r0
        self.index1 = index1
        self.index2 = index2

    def compute_energy_force(self, atoms, output_file):
        if not isinstance(self.index1, list) and not isinstance(self.index2, list):
            dist = atoms.get_distance(self.index1, self.index2, mic=True)
            index1 = self.index1
            index2 = self.index2
        elif isinstance(self.index1, list):
            dists = atoms.get_distances(self.index2, self.index1, mic=True)
            dist = np.min(dists)
            index1 = self.index1[np.argmin(dists)]
            index2 = self.index2
        elif isinstance(self.index2, list):
            dists = atoms.get_distances(self.index1, self.index2, mic=True)
            dist = np.min(dists)
            index1 = self.index1
            index2 = self.index2[np.argmin(dists)]

        forces = np.zeros_like(atoms.positions)
        disps = atoms.positions[index1, :] - atoms.positions[index2, :]
        disps = disps.reshape(1, -1)
        D, D_len = find_mic(disps, atoms.get_cell(), pbc=True)
        disps = D[0]

        energy = 0.5 * self.k * (dist - self.r0)**2
        forces[index1, :] += -self.k * (dist - self.r0) * disps/dist
        forces[index2, :] -= -self.k * (dist - self.r0) * disps/dist
        output_file.write("{}   ".format(dist))
        return energy, forces

class EVB_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running EVB simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, saptff_d1, saptff_d2, nn_d1, nn_d2, off_diag, tmpdir, nn_atoms, res_list, periodic_box, shift=0, bias_dicts=[], device='cuda', **kwargs):
        """
        Parameters
        -----------
        saptff_d1 : Object
            Contains OpenMM force field for diabat 1.
        saptff_d2 : Object
            Contains OpenMM force field for diabat 2.
        nn_d1 : Object
            Diabat_NN instance for diabat 1.
        nn_d2 : Object
            Diabat NN instance for diabat 2.
        off_diag : PyTorch model
            Model for predicting H12 energy and forces.
        shift : float
            Diabat 2 energy shift to diabat 1 reference.
        bias_dicts : list
            list of dictionaries containing information for bias potentials
        device : str
            device where the neural networks will be run. Default is cuda.
        **kwargs : dict
            additional args for ASE base calculator.
        """
        Calculator.__init__(self, **kwargs)
        self.saptff_d1 = saptff_d1
        self.saptff_d2 = saptff_d2
        self.nn_d1 = nn_d1
        self.nn_d2 = nn_d2
        self.off_diag = off_diag
        self.shift = shift
        self.nn_atoms = nn_atoms
        self.has_periodic_box = periodic_box

        self.ff_time = 0
        if self.has_periodic_box:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModPBCEnvironmentProvider(), collect_triples=True)
        else:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModEnvironmentProvider(), collect_triples=True)
        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom
        self.frame = 0
        self.debug_forces = False
        self.res_list = res_list

        self.previous_positions = None 
        self.react_res_d1 = self.res_list[0]
        self.react_atom_d1 = self.react_res_d1[3]
        
        res_list_d2 = self.saptff_d2.res_list()
        self.react_res_d2 = res_list_d2[1]
        self.react_atom_d2 = self.react_res_d2[7]
        
        self.bias_potentials = []
        for bias in bias_dicts:
            self.bias_potentials.append(Harmonic_Bias(bias['k'], bias['r0'], bias['index1'], bias['index2']))

        if bias_dicts:
            self.bias_file = open("{}/umbrella_log.txt".format(tmpdir), 'w')
            self.bias_file.close()

        if os.path.isfile("diabat1_forces_ff.txt"): os.remove("diabat1_forces_ff.txt")
        if os.path.isfile("diabat1_forces_intrann.txt"): os.remove("diabat1_forces_intrann.txt")
        if os.path.isfile("diabat1_forces_internn.txt"): os.remove("diabat1_forces_internn.txt")
        if os.path.isfile("diabat1_forces.txt"): os.remove("diabat1_forces.txt")

        if os.path.isfile("diabat2_forces_ff.txt"): os.remove("diabat2_forces_ff.txt")
        if os.path.isfile("diabat2_forces_intrann.txt"): os.remove("diabat2_forces_intrann.txt")
        if os.path.isfile("diabat2_forces_internn.txt"): os.remove("diabat2_forces_internn.txt")
        if os.path.isfile("diabat2_forces.txt"): os.remove("diabat2_forces.txt")

        if os.path.isfile("h12_forces.txt"): os.remove("h12_forces.txt")
        if os.path.isfile("total_forces.txt"): os.remove("total_forces.txt")
        if os.path.isfile("diabat1_total_forces.txt"): os.remove("diabat1_total_forces.txt")
        if os.path.isfile("diabat2_total_forces.txt"): os.remove("diabat2_total_forces.txt")

    def saptff_energy_force_d1(self, xyz):
        """
        Compute OpenMM energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the SAPTFF_Forcefield class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """
        pos_d1 = xyz.get_positions()
        if self.has_periodic_box:
            pos_d1 = shift_reacting_atom(pos_d1, xyz.get_cell(), self.react_res_d1, self.react_atom_d1)
        self.saptff_d1.set_xyz(pos_d1)
        energy, forces = self.saptff_d1.compute_energy()
        if self.debug_forces:
            print("FF D1", energy)
            write_forces("diabat1_forces_ff.txt", xyz, forces, self.frame)

        return energy, forces

    def saptff_energy_force_d2(self, xyz):
        """
        Compute OpenMM energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the SAPTFF_Forcefield class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """
        
        pos, reord_inds = pos_diabat2(xyz, xyz.get_positions())
        if self.has_periodic_box:
            pos = shift_reacting_atom(pos, xyz.get_cell(), self.react_res_d2, self.react_atom_d2)
        self.saptff_d2.set_xyz(pos)
        energy, forces = self.saptff_d2.compute_energy()
        forces = reorder(forces, reord_inds)
        
        if self.debug_forces:
            print("FF D2", energy)
            write_forces("diabat2_forces_ff.txt", xyz, forces, self.frame)

        return energy, forces

    def nn_energy_force_d1(self, xyz):
        """
        Compute Diabat neural network energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the Diabat_NN class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """

        intra_eng, intra_forces = self.nn_d1.compute_energy_intra(xyz)
        inter_eng, inter_forces = self.nn_d1.compute_energy_inter(xyz)
        total_eng = intra_eng + inter_eng
        total_forces = intra_forces + inter_forces
        if self.debug_forces:
            print("Intra D1", intra_eng)
            print("Inter D1", inter_eng)
            write_forces("diabat1_forces_internn.txt", xyz, inter_forces, self.frame)
            write_forces("diabat1_forces_intrann.txt", xyz, intra_forces, self.frame)

        return total_eng, total_forces

    def nn_energy_force_d2(self, xyz):
        """
        Compute Diabat neural network energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the Diabat_NN class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """

        symb = xyz.get_chemical_symbols()
        symb.insert(25, symb.pop(3))
        pos, reord_inds = pos_diabat2(xyz, xyz.get_positions())
        tmp_Atoms = Atoms(symb, positions=pos, cell=xyz.get_cell(), pbc=xyz.pbc)
        intra_eng, intra_forces = self.nn_d2.compute_energy_intra(tmp_Atoms)
        inter_eng, inter_forces = self.nn_d2.compute_energy_inter(tmp_Atoms)
        total_eng = intra_eng + inter_eng
        total_forces = intra_forces + inter_forces
        total_forces = reorder(total_forces, reord_inds)
        if self.debug_forces:
            print("Intra Eng D2", intra_eng)
            print("Inter Eng D2", inter_eng)
            write_forces("diabat2_forces_internn.txt", tmp_Atoms, inter_forces, self.frame)
            write_forces("diabat2_forces_intrann.txt", tmp_Atoms, intra_forces, self.frame)

        return total_eng, total_forces

    def diagonalize(self, d1_energy, d2_energy, h12_energy):
        """
        Forms matrix and diagonalizes using np to obtain ground-state
        eigenvalue and eigenvector.

        Parameters
        -----------
        d1_energy : np.ndarray
            Total diabat 1 energy
        d2_energy : np.ndarray
            Total diabat 2 energy
        h12_energy : np.ndarray
            Off-diagonal energy

        Returns
        -----------
        eig[l_eig] : np.ndarray
            Ground-state eigenvalue
        eigv[:, l_eig] : np.ndarray
            Ground-state eigenvector
        """
        hamiltonian = [[d1_energy, h12_energy], [h12_energy, d2_energy]]
        eig, eigv = np.linalg.eig(hamiltonian)
        l_eig = np.argmin(eig)
        return eig[l_eig], eigv[:, l_eig]

    def calculate_forces(self, d1_forces, d2_forces, h12_forces, ci):
        """
        Uses Hellmann-Feynman theorem to calculate forces on each atom.

        Parameters
        -----------
        d1_forces : np.ndarray
            forces for diabat 1
        d2_forces : np.ndarray
            forces for diabat 2
        h12_forces : np.ndarray
            forces from off-diagonal elements
        ci : np.ndarray
            ground-state eigenvector

        Returns
        -----------
        np.ndarray
            Forces calculated from Hellman-Feynman theorem
        """
        return ci[0]**2 * d1_forces + 2 * ci[0] * ci[1] * h12_forces + ci[1]**2 * d2_forces

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Obtains the total energy and force using the above methods.

        Parameters
        -----------
        atoms : ASE Atoms object
            atoms object containing coordinates
        properties : list
            Not used, follows SchNetPack format
        system_changes : list
            List of changes for ASE
        """
        
        result = {}
        
        if self.has_periodic_box:
            atoms.wrap()
            shifts = make_molecule_whole(atoms.get_positions(), atoms.get_cell(), self.res_list)
            atoms.positions -= shifts
        
        Calculator.calculate(self, atoms)

        symbs = atoms.get_chemical_symbols()
        symbs = [symbs[i] for i in self.nn_atoms]
        nn_atoms = Atoms(symbols=symbs, positions=atoms.positions[self.nn_atoms], cell=atoms.get_cell(), pbc=atoms.pbc)

        ff_energy_d1, ff_forces_d1 = self.saptff_energy_force_d1(atoms)
        ff_energy_d2, ff_forces_d2 = self.saptff_energy_force_d2(atoms)
        nn_energy_d1, nn_forces_d1 = self.nn_energy_force_d1(nn_atoms)
        nn_energy_d2, nn_forces_d2 = self.nn_energy_force_d2(nn_atoms)

        diabat1_energy = ff_energy_d1 + nn_energy_d1
        diabat2_energy = ff_energy_d2 + nn_energy_d2 + self.shift

        ff_forces_d1[self.nn_atoms] += nn_forces_d1
        ff_forces_d2[self.nn_atoms] += nn_forces_d2

        diabat1_forces = ff_forces_d1
        diabat2_forces = ff_forces_d2

        inp = self.converter(nn_atoms)
        off_diag = self.off_diag(inp)
       
        h12_energy = off_diag['energy'].detach().cpu().numpy()[0][0]
        h12_forces = off_diag['forces'].detach().cpu().numpy()[0]
        h12_forces[h12_forces != h12_forces] = 0.0

        zero_forces = np.zeros_like(ff_forces_d1)
        zero_forces[self.nn_atoms] += h12_forces
        h12_forces = zero_forces
  
        energy, ci = self.diagonalize(diabat1_energy, diabat2_energy, h12_energy)
        forces = self.calculate_forces(diabat1_forces, diabat2_forces, h12_forces, ci)

        if self.debug_forces:
            print(ci[0]**2, ci[1]**2)
            print(diabat1_energy)
            print(diabat2_energy)
            print(atoms.get_distance(2,3))
            print(min(atoms.get_distances(3, [19, 20])))
            write_forces("h12_forces.txt", atoms, h12_forces, self.frame)
            write_forces("total_forces.txt", atoms, forces, self.frame)
            write_forces("diabat1_total_forces.txt", atoms, diabat1_forces, self.frame)
            write_forces("diabat2_total_forces.txt", atoms, diabat2_forces, self.frame)

        if self.bias_potentials:
            self.bias_file = open(self.bias_file.name, 'a+')
            self.bias_file.write("{}   ".format(self.frame))

        for bias in self.bias_potentials:
            bias_energy, bias_force = bias.compute_energy_force(atoms, self.bias_file)
            energy += bias_energy
            forces += bias_force

        if self.bias_potentials:
            self.bias_file.write("\n")
            self.bias_file.close()
        
        self.frame += 1
        
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class ASE_MD:
    """
    Setups and runs the MD simulation. Serves as an interface to the EVB Hamiltonian class and ASE.
    """
    def __init__(self, ase_atoms, tmp, calc_omm_d1, calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, nn_atoms, res_list, shift=0, frame=-1, bias_dicts=[], device='cuda'):
        """
        Parameters
        -----------
        ase_atoms : str
            Location of input structure, gets created to ASE Atoms object.
        tmp : str
            Location for tmp directory.
        calc_omm_d1 : Object
            Contains OpenMM force field for diabat 1.
        calc_omm_d2 : Object
            Contains OpenMM force field for diabat 2.
        calc_nn_d1 : Object
            Diabat_NN instance for diabat 1.
        calc_nn_d2 : Object
            Diabat NN instance for diabat 2.
        off_diag : PyTorch model
            Model for predicting H12 energy and forces.
        shift : float
            Diabat 2 energy shift to diabat 1 reference.
        """

        self.working_dir = tmp
        self.tmp = tmp
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)
        
        self.mol = read(ase_atoms, index="{}".format(frame))
        self.mol.set_masses(calc_omm_d1.get_masses())
        self.calc_omm_d1 = calc_omm_d1
        self.calc_omm_d1.set_initial_positions(self.mol.get_positions())
        d2_pos, reord_inds = pos_diabat2(self.mol, self.mol.get_positions())
        self.calc_omm_d2 = calc_omm_d2
        self.calc_omm_d2.set_initial_positions(d2_pos)
        periodic_box = self.calc_omm_d1.has_periodic_box
        calculator = EVB_Hamiltonian(self.calc_omm_d1, self.calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, self.tmp, nn_atoms, res_list, periodic_box, shift, bias_dicts, device)
        self.mol.set_calculator(calculator)
        self.md = False

    def create_system(self, name, time_step=1.0, temp=300, temp_init=None, restart=False, store=1, nvt=False, friction=0.001):
        """
        Parameters
        -----------
        name : str
            Name for output files.
        time_step : float, optional
            Time step in fs for simulation.
        temp : float, optional
            Temperature in K for NVT simulation.
        temp_init : float, optional
            Optional different temperature for initialization than thermostate set at.
        restart : bool, optional
            Determines whether simulation is restarted or not, 
            determines whether new velocities are initialized.
        store : int, optional
            Frequency at which output is written to log files.
        nvt : bool, optional
            Determines whether to run NVT simulation, default is False.
        friction : float, optional
            friction coefficient in fs^-1 for Langevin integrator
        """
        if temp_init is None: temp_init = temp
        if not self.md or restart:
            MaxwellBoltzmannDistribution(self.mol, temp_init * units.kB)
            Stationary(self.mol)
            ZeroRotation(self.mol)
        
        if not nvt:
            self.md = VelocityVerlet(self.mol, time_step * units.fs)
        else:
            self.md = Langevin(self.mol, time_step * units.fs, temp * units.kB, friction/units.fs)

        logfile = os.path.join(self.tmp, "{}.log".format(name))
        trajfile = os.path.join(self.tmp, "{}.traj".format(name))

        logger = MDLogger(self.md, self.mol, logfile, stress=False, peratom=False, header=True, mode="w")
        trajectory = Trajectory(trajfile, "w", self.mol)
        self.md.attach(logger, interval=store)
        self.md.attach(trajectory.write, interval=store)

    def write_mol(self, name, ftype="xyz", append=False):
        """
        Write out current molecule structure.
        Parameters
        -----------
        name : str
            Name of the output file.
        ftype : str, optional
            Determines output file format, default xyz.
        append : bool, optional
            Append to existing output file or not.
        """
        path = os.path.join(self.tmp, "{}.{}".format(name, ftype))
        write(path, self.mol, format=ftype, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        self.calc_omm_d1.set_initial_positions(self.mol.get_positions())
        d2_pos, reord_inds = pos_diabat2(self.mol, self.mol.get_positions())
        self.calc_omm_d2.set_initial_positions(d2_pos)

        energy = self.mol.get_potential_energy()
        forces = self.mol.get_forces()
        self.mol.energy = energy
        self.mol.forces = forces
        return energy, forces

    def run_md(self, steps):
        """
        Run MD simulation.
        Parameters
        -----------
        steps : int
            Number of MD steps
        """
        self.md.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
        fmax (float): Maximum residual force change (default 1.e-2)
        steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
          self.mol,
          trajectory="%s.traj" % optimize_file,
          restart="%s.pkl" % optimize_file,
          )
        optimizer.run(fmax, steps)
     
        # Save final geometry in xyz format
        self.save_molecule(name)


