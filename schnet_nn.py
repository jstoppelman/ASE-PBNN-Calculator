import schnetpack as sch
import time 
from schnetpack import AtomsData, Properties
from schnetpack.environment import APNetEnvironmentProvider, APModEnvironmentProvider
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

def write_forces(fnm, xyz, forces, frame):
    out = open(fnm, 'a')
    sym = xyz.get_chemical_symbols()
    out.write("Frame {}\n".format(frame))
    for i in range(len(forces)):
        out.write(" {0}  {1:13.10f}  {2:13.10f}  {3:13.10f}\n".format(sym[i], forces[i][0], forces[i][1], forces[i][2]))
    out.write("\n")
    out.close()

def pos_diabat2(positions):
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
    p_o1 = positions[19]
    p_o2 = positions[20]
    h = positions[3]
    dist1 = p_o1 - h
    dist2 = p_o2 - h
    dist1 = np.power(np.dot(dist1, dist1), 1/2)
    dist2 = np.power(np.dot(dist2, dist2), 1/2)
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

class Diabat_NN:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks. This computes energies and forces from the
    intra- and intermolecular neural networks for the diabatic states.
    """
    def __init__(self, nn_monA, nn_monB, nn_dimer, res_list):
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
        """
        self.nn_monA = torch.load(nn_monA)
        self.nn_monB = torch.load(nn_monB)
        self.nn_dimer = torch.load(nn_dimer)
        self.res_list = res_list
        self.converter = AtomsConverter(device='cuda')
        self.inter_converter = AtomsConverter(device='cuda', environment_provider=APNetEnvironmentProvider(), res_list=res_list)

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
        #Derivative of Fermi-Dirac functions is infinite from PyTorch autograd
        #once the bond has stretched too far. This line recognizes whether inf is present
        #in the forces array, and sets each element to 0 if this occurs.
        forces[forces != forces] = 0.0
        return np.asarray(energy), np.asarray(forces)

class PBNN_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running PBNN simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, saptff_d1, saptff_d2, nn_d1, nn_d2, off_diag, tmpdir, nn_atoms, shift=0, **kwargs):
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

        self.converter = AtomsConverter(device='cuda', environment_provider=APModEnvironmentProvider(), collect_triples=True)
        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom
        self.frame = 0
        self.debug_forces = False

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

        pos, reord_inds = pos_diabat2(xyz.get_positions())
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
        pos, reord_inds = pos_diabat2(xyz.get_positions())
        tmp_Atoms = Atoms(symb, positions=pos)
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
        Calculator.calculate(self, atoms)
        result = {}

        symbs = atoms.get_chemical_symbols()
        symbs = [symbs[i] for i in self.nn_atoms]
        nn_atoms = Atoms(symbols=symbs, positions=atoms.positions[self.nn_atoms])

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
        #Derivative of Fermi-Dirac functions is infinite from PyTorch autograd
        #once the bond has stretched too far. This line recognizes whether inf is present
        #in the forces array, and sets each element to 0 if this occurs.
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

        self.frame += 1
       
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class ASE_MD:
    """
    Setups and runs the MD simulation. Serves as an interface to the PBNN Hamiltonian class and ASE.
    """
    def __init__(self, ase_atoms, tmp, calc_omm_d1, calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, nn_atoms, shift=0, frame=-1):
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
        self.calc_omm_d2 = calc_omm_d2

        self.calc_omm_d1.set_initial_positions(self.mol.get_positions())
        
        d2_pos, reord_inds = pos_diabat2(self.mol.get_positions())
        self.calc_omm_d2.set_initial_positions(d2_pos)

        calculator = PBNN_Hamiltonian(self.calc_omm_d1, self.calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, self.tmp, nn_atoms, shift)
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

    def run_md(self, steps):
        """
        Run MD simulation.
        Parameters
        -----------
        steps : int
            Number of MD steps
        """
        self.md.run(steps)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        self.calc_omm_d1.set_initial_positions(self.mol.get_positions())
        d2_pos, reord_inds = pos_diabat2(self.mol.get_positions())
        self.calc_omm_d2.set_initial_positions(d2_pos)

        energy = self.mol.get_potential_energy()
        forces = self.mol.get_forces()
        self.mol.energy = energy
        self.mol.forces = forces
        return energy, forces

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


