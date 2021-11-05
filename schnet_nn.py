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
from ase.calculators.plumed import Plumed
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

def pos_diabat1_acetic(atoms, positions):
    inds = [i for i in range(len(positions))]
    dists = atoms.get_distances(7, [0, 1], mic=True)
    dist1, dist2 = dists[0], dists[1]
    if dist1 < dist2:
        return positions, inds
    else:
        new_pos = np.empty_like(positions)
        new_pos[:] = positions
        o1 = positions[0]
        o2 = positions[1]
        new_pos[0] = o2
        new_pos[1] = o1
        positions = new_pos
        return positions, inds

def pos_diabat2_acetic(atoms, positions):
    inds = [i for i in range(len(positions))]
    mol1 = np.arange(8, 15, 1).tolist()
    mol2 = np.arange(0, 7, 1).tolist()
    mol1.append(7)
    for m in mol2: mol1.append(m)
    index = mol1
    positions = positions[index]
    dists = atoms.get_distances(7, [8, 9])
    dist1, dist2 = dists[0], dists[1]
    if dist1 < dist2:
        return positions, index
    else:
        new_pos = np.empty_like(positions)
        new_pos[:] = positions
        o1, o2 = positions[0], positions[1]
        new_pos[0] = o2
        new_pos[1] = o1
        positions = new_pos
        index[0] = 9
        index[1] = 8
        return positions, index 

def return_positions(atoms, positions):
    inds = [i for i in range(len(positions))]
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

class Diabat:
    """
    Contains the SAPT-FF and Diabat_NN classes for a diabatic state and uses
    them to compute the diabatic energy for a system.
    """
    def __init__(self, saptff, nn, nn_atoms, nn_indices, reorder_func, shift=0):
        """
        Parameters
        -----------
        saptff : Object
            Instance of SAPT_ForceField class
        nn : Object
            Instance of Diabat_NN class
        nn_atoms : list
            List containing the indices of the atoms the neural network is applied to
        nn_indices : list
            List containing reordered indices for the neural network atoms, specific to each diabat
        reorder : function
            Function to reorder positions, see return_positions and pos_diabat2 above
        shift : float
            Shift in the diabatic energy
        """
        self.saptff = saptff
        self.nn = nn
        self.nn_atoms = nn_atoms
        self.nn_indices = nn_indices
        self.reorder_func = reorder_func
        self.shift = shift

    def setup_saptff(self, atoms, positions):
        positions, inds = self.reorder_func(atoms, positions)
        self.saptff.set_initial_positions(positions)

    def compute_energy_force(self, atoms, nn_atoms):
        saptff_positions = atoms.get_positions()
        saptff_positions, reorder_inds = self.reorder_func(atoms, saptff_positions)
        if self.saptff.has_periodic_box:
            saptff_positions = shift_reacting_atom(saptff_positions, atoms.get_cell(), self.saptff.react_res, self.saptff.react_atom)
        self.saptff.set_xyz(saptff_positions)
        saptff_energy, saptff_forces = self.saptff.compute_energy()

        saptff_forces = reorder(saptff_forces, reorder_inds)

        symbols = nn_atoms.get_chemical_symbols()
        new_symbols = [symbols[i] for i in self.nn_indices]
        nn_positions, reorder_nn_inds = self.reorder_func(nn_atoms, nn_atoms.get_positions())
        reorder_Atoms = Atoms(new_symbols, positions=nn_positions, cell=nn_atoms.get_cell(), pbc=nn_atoms.pbc)
        
        nn_intra_energy, nn_intra_forces = self.nn.compute_energy_intra(reorder_Atoms)
        nn_inter_energy, nn_inter_forces = self.nn.compute_energy_inter(reorder_Atoms)

        energy = saptff_energy + nn_intra_energy + nn_inter_energy + self.shift

        nn_forces = nn_intra_forces + nn_inter_forces
        nn_forces = reorder(nn_forces, reorder_nn_inds)
        saptff_forces[self.nn_atoms] += nn_forces
        
        forces = saptff_forces
        
        return energy, forces

class Coupling: 
    """
    Class used to compute the coupling between two diabatic states
    """
    def __init__(self, nn, periodic, device='cuda'):
        """
        Parameters
        -----------
        nn : str
            location of the neural network used to compute the coupling term
        periodic : bool
            bool indicating whether periodic bundaries are being used.
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.nn = torch.load(nn)
        if periodic:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModPBCEnvironmentProvider(), collect_triples=True)
        else:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModEnvironmentProvider(), collect_triples=True)

    def compute_energy_force(self, nn_atoms):
        inputs = self.converter(nn_atoms)
        result = self.nn(inputs)

        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        forces[forces != forces] = 0.0
        return np.asarray(energy), np.asarray(forces)

class NN_Diagonal:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks. This computes energies and forces from the
    intra- and intermolecular neural networks for the diabatic states.
    """
    def __init__(self, nn_monA, nn_monB, nn_dimer, res_list, periodic, device='cuda'):
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
        periodic : bool
            bool indicating whether periodic boundaries are being used. 
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.nn_monA = torch.load(nn_monA)
        self.nn_monB = torch.load(nn_monB)
        self.nn_dimer = torch.load(nn_dimer)
        self.res_list = res_list
        if periodic:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=TorchEnvironmentProvider(8., device=torch.device(device)))
            self.inter_converter = AtomsConverter(device=torch.device(device), environment_provider=APNetPBCEnvironmentProvider(), res_list=res_list)
        else:
            self.converter = AtomsConverter(device=torch.device(device))
            self.inter_converter = AtomsConverter(device=torch.device(device), environment_provider=APNetEnvironmentProvider(), res_list=res_list)

    def compute_energy_intra(self, atoms):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        atoms : ASE Atoms Object
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
        
        atomsA = atoms[monA]
        atomsB = atoms[monB]

        inputsA = self.converter(atomsA)
        inputsB = self.converter(atomsB)
        resultA = self.nn_monA(inputsA)
        resultB = self.nn_monB(inputsB)

        energyA = resultA['energy'].detach().cpu().numpy()[0][0]
        forcesA = resultA['forces'].detach().cpu().numpy()[0]
        energyB = resultB['energy'].detach().cpu().numpy()[0][0]
        forcesB = resultB['forces'].detach().cpu().numpy()[0]
        intra_eng = energyA + energyB
        intra_forces = np.append(forcesA, forcesB, axis=0)
        intra_forces[intra_forces!=intra_forces] = 0
        return np.asarray(intra_eng), np.asarray(intra_forces)

    def compute_energy_inter(self, atoms):
        """
        Compute the energy for the intermolecular components of the dimer.

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural network.

        Returns
        -----------
        energy : np.ndarray
            Intermoleculer energy 
        forces : np.ndarray
            Intermolecular forces
        """
        inputs = self.inter_converter(atoms)
        result = self.nn_dimer(inputs)

        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        forces[forces!=forces] = 0
        return np.asarray(energy), np.asarray(forces)

class PBNN_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running PBNN simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, diabats, couplings, tmpdir, nn_atoms, res_list, device='cuda', **kwargs):
        """
        Parameters
        -----------
        diabats : list
            List containing Diagonal diabatic state classes
        coupling : list
            List containing Coupling classes 
        tmpdir : str
            directory where the simulation logs are stored
        nn_atoms : list
            List containing the set of atoms that the neural network computes
        res_list : list
            List containing the atoms in each residue from OpenMM
        device : str
            device where the neural networks will be run. Default is cuda.
        **kwargs : dict
            additional args for ASE base calculator.
        """
        Calculator.__init__(self, **kwargs)
        self.diabats = diabats
        self.couplings = couplings
        self.nn_atoms = nn_atoms
        self.has_periodic_box = self.diabats[0].saptff.has_periodic_box

        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom
        self.frame = 0
        self.debug_forces = False
        self.res_list = res_list

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

    def diagonalize(self, diabat_energies, coupling_energies):
        """
        Forms matrix and diagonalizes using np to obtain ground-state
        eigenvalue and eigenvector.

        Parameters
        -----------
        diabat_energies : list
            List containing the energies of the diabatic states
        coupling_energies : list
            List containing the coupling energies between diabatic states

        Returns
        -----------
        eig[l_eig] : np.ndarray
            Ground-state eigenvalue
        eigv[:, l_eig] : np.ndarray
            Ground-state eigenvector
        """
        num_states = len(diabat_energies)
        hamiltonian = np.zeros((num_states, num_states))
        for i, energy in enumerate(diabat_energies):
            hamiltonian[i, i] = energy
        for i, energy in enumerate(coupling_energies):
            j = i + 1
            hamiltonian[i, j] = energy
            hamiltonian[j, i] = energy

        eig, eigv = np.linalg.eig(hamiltonian)
        l_eig = np.argmin(eig)
        return eig[l_eig], eigv[:, l_eig]

    def calculate_forces(self, diabat_forces, coupling_forces, ci):
        """
        Uses Hellmann-Feynman theorem to calculate forces on each atom.

        Parameters
        -----------
        diabat_forces : list
            List containing the forces for each diabat
        coupling_forces : list
            List containing the forces from the coupling elements
        ci : np.ndarray
            ground-state eigenvector

        Returns
        -----------
        np.ndarray
            Forces calculated from Hellman-Feynman theorem
        """
        num_states = len(diabat_forces)
        hamiltonian_force = np.zeros((num_states, num_states), dtype=np.ndarray)
        for i, force in enumerate(diabat_forces):
            hamiltonian_force[i, i] = force

        for i, force in enumerate(coupling_forces):
            j = i + 1
            hamiltonian_force[i, j] = force
            hamiltonian_force[j, i] = force
        
        total_forces = 0
        for i in range(num_states):
            for j in range(num_states):
                total_forces += ci[i] * ci[j] * hamiltonian_force[i, j]
        
        return total_forces

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

        diabat_energies = []
        diabat_forces = []
        for diabat in self.diabats: 
            energy, forces = diabat.compute_energy_force(atoms, nn_atoms)
            diabat_energies.append(energy)
            diabat_forces.append(forces)

        coupling_energies = []
        coupling_forces = []
        for coupling in self.couplings:
            energy, forces = coupling.compute_energy_force(nn_atoms)
            zero_forces = np.zeros_like(diabat_forces[-1])
            zero_forces[self.nn_atoms] += forces
            coupling_energies.append(energy)
            coupling_forces.append(zero_forces)
        
        energy, ci = self.diagonalize(diabat_energies, coupling_energies)
        forces = self.calculate_forces(diabat_forces, coupling_forces, ci)
        
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

        self.frame += 1
        
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class ASE_MD:
    """
    Setups and runs the MD simulation. Serves as an interface to the PBNN Hamiltonian class and ASE.
    """
    def __init__(self, ase_atoms, tmp, diabats, coupling, nn_atoms, frame=-1, plumed_input=[], device='cuda'):
        """
        Parameters
        -----------
        ase_atoms : str
            Location of input structure, gets created to ASE Atoms object.
        tmp : str
            Location for tmp directory.
        diabats : list
            List containing Diagonal objects representing the diabatic states of the system
        coupling : list
            List containing the coupling terms for the matrix. Order so that they correspond with the diabats list
        plumed_input : list
            List containing the commands for Plumed
        nn_atoms : list
            List containing the atoms the neural network is applied to
        """

        self.working_dir = tmp
        self.tmp = tmp
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)
        
        self.mol = read(ase_atoms, index="{}".format(frame))
        self.diabats = diabats
        self.coupling = coupling
        self.mol.set_masses(self.diabats[0].saptff.get_masses())

        for diabat in self.diabats:
            diabat.setup_saptff(self.mol, self.mol.get_positions())

        res_list = self.diabats[0].saptff.res_list()
        pbnn_calculator = PBNN_Hamiltonian(diabats, coupling, self.tmp, nn_atoms, res_list, device)
        if not plumed_input:
            self.mol.set_calculator(pbnn_calculator)
        else:
            plumed_calculator = Plumed(pbnn_calculator, plumed_input, 1.0, atoms=self.mol, kT=300.0*units.kB, log=f'{self.tmp}/colvar.dat')
            self.mol.set_calculator(plumed_calculator)
        
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
        for diabat in self.diabats:
            diabat.setup_saptff(self.mol, self.mol.get_positions())

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

