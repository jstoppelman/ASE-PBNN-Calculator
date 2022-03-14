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
from plumed_calculator import Plumed
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.geometry import find_mic

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
        diff_x = box[0][0] * -np.sign(disp_0[:,0]) * np.floor(abs(disp_0[:,0]/box[0][0])+0.5)
        diff_y = box[1][1] * -np.sign(disp_0[:,1]) * np.floor(abs(disp_0[:,1]/box[1][1])+0.5)
        diff_z = box[2][2] * -np.sign(disp_0[:,2]) * np.floor(abs(disp_0[:,2]/box[2][2])+0.5)
        shifts[mol[1:],0] += diff_x
        shifts[mol[1:],1] += diff_y
        shifts[mol[1:],2] += diff_z
    return shifts

class Reorder_Indices:
    """
    As the atom ordering changes within each diabat, this class shifts the atom ordering so that the positions are correctly changed for each diabat
    This isn't completely general for now, as it is essentially hardwired to work with EMIM/acetate
    """
    def __init__(self, react_atom=None, accept_atom=None, insert_react=None):
        """
        Parameters 
        -------------
        react_atom : int, optional
            Atoms which gets donated/accepted
        accept_atom : list, optional
            List of atoms that can accept the proton
        insert_react : int, optional
            Location in the list of atom indices at which to place the react atom in the new diabat
        """
        self.react_atom = react_atom
        self.accept_atom = accept_atom
        self.insert_react = insert_react

    def pos_diabat(self, atoms, positions):
        """
        Parameters
        ------------
        atoms : Atoms object
            System atoms object
        positions : np.ndarray
            Current positions
        """
        if self.react_atom is None and self.accept_atom is None:
            inds = [i for i in range(len(positions))]
        else:
            inds = [i for i in range(len(positions))]
            dists = atoms.get_distances(self.react_atom, self.accept_atom, mic=True)
            dist1, dist2 = dists[0], dists[1]
            if dist1 < dist2:
                inds.insert(self.insert_react, inds.pop(self.react_atom))
                h_atom = positions[self.react_atom]
                positions = np.delete(positions, self.react_atom, axis=0)
                positions = np.insert(positions, self.insert_react, [h_atom], axis=0)
            else:
                inds.insert(self.accept_atom[1], inds.pop(self.accept_atom[0]))
                inds.insert(self.insert_react, inds.pop(self.react_atom))
                new_pos = np.empty_like(positions)
                new_pos[:] = positions
                o1_atom = positions[self.accept_atom[0]]
                o2_atom = positions[self.accept_atom[1]]
                new_pos[self.accept_atom[0]] = o2_atom
                new_pos[self.accept_atom[1]] = o1_atom
                h_atom = new_pos[self.react_atom]
                new_pos = np.delete(new_pos, self.react_atom, axis=0)
                new_pos = np.insert(new_pos, self.insert_react, [h_atom], axis=0)
                positions = new_pos
        return positions, inds

class Diabat:
    """
    Contains the SAPT-FF and Diabat_NN classes for a diabatic state and uses
    them to compute the diabatic energy for a system.
    """
    def __init__(self, saptff, nn_intra, nn_inter, nn_indices, reorder_func, shift=0, name=None, adaptive=False, **kwargs):
        """
        Parameters
        ----------
        saptff : Object
            Instance of SAPT_ForceField class
        nn_intra : list
            List of NN_Intra_Diagonal classes
        nn_inter : list
            List of NN_Inter_Diagonal classes
        nn_indices : list
            List containing reordered indices for the neural network atoms with respect to 
            diabat 1, specific to each diabat
        reorder_func : function
            Function to reorder positions, see return_positions and pos_diabat2 above
        shift : float
            Shift in the diabatic energy
        """
        self.saptff = saptff
        self.nn_intra = nn_intra
        self.nn_inter = nn_inter
        self.nn_atoms = np.sort(np.asarray(nn_indices)).tolist()
        self.reorder_func = reorder_func
        self.shift = shift
        self.name = name
        self.adaptive = adaptive
        if self.adaptive:
            self.res_list = kwargs.get('res_list')
            self.num_accept = kwargs.get('num_accept')
            self.ff_shift = kwargs.get('ff_shift')
       
    def setup_saptff(self, atoms, positions):
        """
        Sets initial posiitons for SAPT_ForceField class

        Parameters
        ------------
        atoms : ASE atoms object
            atoms object
        positions : np.ndarray
            Current positions
        """
        positions, inds = self.reorder_func.pos_diabat(atoms, positions)
        self.saptff.set_initial_positions(positions)

    def compute_energy_force(self, atoms, adapt_dict=None):
        """
        Gets energy and force from different parts of the Hamiltonian

        Parameters 
        --------------
        atoms : ASE atoms object
            atoms object
        adapt_dict : dict, optional
            If a residue is in the adaptive zone, adapt_dict containg the energy and force from the adaptive class

        Returns
        --------------
        energy : float
            energy for the diabat
        forces : np.ndarray
            forces from the diabat ordered in Diabat 1 order
        """
        saptff_positions = atoms.get_positions()
        saptff_positions, reorder_inds = self.reorder_func.pos_diabat(atoms, saptff_positions)
        nn_atoms = atoms[reorder_inds]

        if self.saptff.has_periodic_box:
            saptff_positions = shift_reacting_atom(saptff_positions, atoms.get_cell(), self.saptff.react_res, self.saptff.react_atom)
        
        self.saptff_energy, self.saptff_forces = self.compute_saptff_energy_force(saptff_positions) 

        self.nn_intra_energy, self.nn_intra_forces = self.compute_nn_intra_energy_force(nn_atoms)
       
        self.nn_inter_energy, self.nn_inter_forces = self.compute_nn_inter_energy_force(nn_atoms)
        
        energy = self.saptff_energy + self.nn_intra_energy + self.nn_inter_energy + self.shift
        forces = self.saptff_forces + self.nn_intra_forces + self.nn_inter_forces
        forces = reorder(forces, reorder_inds)
        if self.adaptive:
            num_exclude = len(self.saptff.exclude_intra_res)
            n_shift = self.num_accept - num_exclude
            energy += n_shift * self.ff_shift
            for key, val in adapt_dict.items():
                index = self.res_list[key]
                adapt_energy = adapt_dict[key]["energy"]
                energy += adapt_energy
                adapt_forces = adapt_dict[key]["forces"]
                if len(index) != adapt_forces.shape[0]:
                    index = index[:-1]
                forces[index] += adapt_forces
        return energy, forces

    def compute_saptff_energy_force(self, positions):
        """
        Parameters
        -------------
        positions : np.ndarray
            current positions
        """
        self.saptff.set_xyz(positions)
        saptff_energy, saptff_forces = self.saptff.compute_energy()
        return saptff_energy, saptff_forces

    def compute_nn_intra_energy_force(self, nn_atoms):
        """
        Parameters 
        -------------
        nn_atoms : ASE atoms object
            atoms object ordered for the current diabat
        """
        nn_intra_energy = 0
        nn_intra_forces = np.zeros_like(self.saptff_forces)
        for nn_intra in self.nn_intra:
            energy, nn_intra_forces = nn_intra.compute_energy_force(nn_atoms, nn_intra_forces)
            nn_intra_energy += energy
        return nn_intra_energy, nn_intra_forces

    def compute_nn_inter_energy_force(self, nn_atoms):
        """
        Parameters 
        -------------
        nn_atoms : ASE atoms object
            atoms object ordered for the current diabat
        """
        nn_inter_energy = 0
        nn_inter_forces = np.zeros_like(self.saptff_forces)
        for nn_inter in self.nn_inter:
            energy, nn_inter_forces = nn_inter.compute_energy_force(nn_atoms, nn_inter_forces)
            nn_inter_energy += energy
        return nn_inter_energy, nn_inter_forces

class Coupling: 
    """
    Class used to compute the coupling between two diabatic states
    """
    def __init__(self, nn, force_indices, nn_list_loc, periodic, device='cuda'):
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
        self.force_indices = force_indices
        self.nn_list_loc = nn_list_loc
        if periodic:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModPBCEnvironmentProvider(), collect_triples=True)
        else:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APModEnvironmentProvider(), collect_triples=True)

    def compute_energy_force(self, atoms, total_forces):
        """
        atoms : ASE atoms object
            current atoms object ordered in Diabat 1 order
        total_forces : np.ndarray
            Shape of the total atoms in the system, filled in with forces that the coupling residue is applied to
        """
        inputs = self.converter(atoms[self.force_indices])
        result = self.nn(inputs)

        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        forces[forces != forces] = 0.0
        total_forces[self.force_indices] += forces
        return np.asarray(energy), total_forces

class NN_Intra_Diabat:
    """ 
    Class for obtaining the energies and forces from SchNetPack
    neural networks, designed for single molecules intramolecular interactions
    """
    def __init__(self, model, force_atoms, residue, periodic, device='cuda'):
        """
        Parameters
        -----------
        model : str
            location of the neural network model for the monomer
        monomer_indices : list
            List of atom indices that correspond to the atoms the neural network is applied to
        periodic : bool
            bool indicating whether periodic boundaries are being used.
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.model_name = model
        self.model = torch.load(self.model_name)
        self.nn_force_atoms = force_atoms
        self.residue = residue
        if periodic:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=TorchEnvironmentProvider(8., device=torch.device(device)))
        else:
            self.converter = AtomsConverter(device=torch.device(device))

    def compute_energy_force(self, atoms, total_forces):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        total_forces : np.ndarray
            numpy array containing the total intramolecular forces for a diabat

        Returns
        -----------
        energy : np.ndarray
            Intramoleculer energy in kJ/mol
        total_forces : np.ndarray
            Intramolecular forces in kJ/mol/A
        """
        inputs = self.converter(atoms[self.nn_force_atoms])
        result = self.model(inputs)
        energy = result["energy"].detach().cpu().numpy()[0]
        forces = result["forces"].detach().cpu().numpy()[0]
        forces[forces!=forces] = 0
        total_forces[self.nn_force_atoms] += forces
        return np.asarray(energy), total_forces

class NN_Inter_Diabat:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks, designed for intermolecular dimer interactions.
    """
    def __init__(self, model, res_list, nn_res_loc, periodic, device='cuda'):
        """
        Parameters
        -----------
        model : str
            location of the neural network model for the dimer
        res_list : list
            List of atom indices that correspond to the atoms the neural network is applied to
        periodic : bool
            bool indicating whether periodic boundaries are being used.
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.model = torch.load(model)
       
        self.nn_force_atoms = [atom_id for res in res_list for atom_id in res]
        self.nn_res_loc = nn_res_loc
        nn_res_list = [[i for i in range(len(res_list[0]))]]
        monomer_2 = [i for i in range(res_list[0][-1]+1, res_list[0][-1]+1+len(res_list[1]))]
        nn_res_list.append(monomer_2)
        if periodic:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APNetPBCEnvironmentProvider(), res_list=nn_res_list)
        else:
            self.converter = AtomsConverter(device=torch.device(device), environment_provider=APNetEnvironmentProvider(), res_list=nn_res_list)

    def compute_energy_force(self, atoms, total_forces):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        total_forces : np.ndarray
            numpy array containing the total intramolecular forces for a diabat

        Returns
        -----------
        energy : np.ndarray
            Intramoleculer energy in kJ/mol
        forces : np.ndarray
            Intramolecular forces in kJ/mol/A
        """
        inputs = self.converter(atoms[self.nn_force_atoms])
        result = self.model(inputs)
        energy = result["energy"].detach().cpu().numpy()[0]
        forces = result["forces"].detach().cpu().numpy()[0]
        forces[forces!=forces] = 0
        total_forces[self.nn_force_atoms] += forces
        return np.asarray(energy), total_forces

class Adaptive_Intramolecular:
    """
    Adaptive Intramolecular class. Interpolates energy for all residues in the adaptive region.
    Interpolation function from https://pubs.acs.org/doi/10.1021/jp0673617
    """
    def __init__(self, models, saptff, res_name, num_atoms, res_list, accept_atomids, R_inner, R_outer, core_mol_atoms, ff_shift):
        """
        Parameters
        ---------------
        models : dict
            Dictionary containing NN models for the adaptive residue
        saptff : SAPT_ForceField object
            SAPT_ForceField object containing only force field definitions for the adaptive monomer
        res_name : str
            Residue name of the adaptive molecule
        res_list : list
            List of atoms in each residue
        accept_atomids : list
            adaptive site identified in terms of the accepting atoms, so we compute the distance from the active site to the accepting atoms
        R_inner : float
            distance from the center of active site that describes the inner region
        R_outer : float
            distance from the center of active site that describes the outer region
        core_mol_atoms : list
            list of atom indices for which the core region is defined
        ff_shift : float
            shift for making the mean force field energy equal to the mean neural network energy
        """
        self.models_dict = models
        self.saptff = saptff
        self.res_name = res_name
        self.num_atoms = num_atoms
        self.res_list = res_list
        self.accept_atomids = accept_atomids
        self.intra_model = self.models_dict[self.res_name]
        self.nn_intra = NN_Intra_Diabat("ff_files/"+self.intra_model, np.arange(num_atoms), 0, saptff.has_periodic_box)
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.core_mol_atoms = core_mol_atoms
        self.ff_shift = ff_shift

    def set_adaptive_residues(self, adapt_list):
        """
        Parameters
        --------------
        adapt_list : list
            List of residues in the adaptive region
        """
        self.adaptive_residues = adapt_list

    def compute_energy_force(self, atoms):
        """
        Parameters
        --------------
        atoms : ASE atoms object
            atoms object
        """
        adapt_dict = {}
        for res in self.adaptive_residues:
            atoms_tmp = atoms[self.res_list[res]]
            self.saptff.set_initial_positions(atoms_tmp.get_positions())
            self.saptff.set_xyz(atoms_tmp.get_positions())
            saptff_energy, saptff_forces = self.saptff.compute_energy()
            saptff_energy += self.ff_shift
            nn_forces = np.zeros_like(saptff_forces)
            nn_energy, nn_forces = self.nn_intra.compute_energy_force(atoms_tmp, nn_forces)
            core_atoms = atoms[self.core_mol_atoms]
            ref_pos = core_atoms.get_center_of_mass()
            energy, forces = self.interpolate_energy_force(atoms_tmp, ref_pos, saptff_energy, nn_energy, saptff_forces, nn_forces)
            energy_force_dict = {'energy': energy, 'forces': forces}
            adapt_dict[res] = energy_force_dict
        return adapt_dict

    def interpolate_energy_force(self, atoms, ref_pos, saptff_energy, nn_energy, saptff_forces, nn_forces):
        """
        Interpolation function. Computes minimum image for interpolation

        Parameters 
        ---------------
        atoms : ASE atoms object
            atoms object
        ref_pos : np.ndarray
            coordinates for center of active site
        saptff_energy : float
            SAPTFF output energy
        nn_energy : float
            NN output energy
        saptff_forces : np.ndarray
            SAPT-FF forces
        nn_forces : np.ndarray
            NN forces
        """
        disp = ref_pos - atoms.positions[self.accept_atomids]
        cell = np.asarray(atoms.get_cell())
        cell = np.expand_dims(cell, axis=0)
        cell = np.repeat(cell, len(self.accept_atomids), axis=0)
        diff = np.zeros((len(self.accept_atomids), 3))
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
        dists = np.sqrt((disp*disp).sum(axis=1))
        r = dists.min()
        index1 = np.argmin(dists)
        disps = atoms.positions[index1, :] - ref_pos
        disps = disps.reshape(1, -1)
        D, D_len = find_mic(disps, atoms.get_cell(), pbc=True)
        disps = D[0]

        piecewise = np.piecewise(r, [np.logical_and(r > self.R_inner, r < self.R_outer), np.logical_or(r <= self.R_inner, r >=self.R_outer)], [1.0, 0.0])
        alpha = (r - self.R_inner) / (self.R_outer - self.R_inner) 
        P = -6.0 * alpha**5 + 15.0 * alpha**4 - 10.0 * alpha**3 + 1.0
        P *= piecewise
        fac = -30.0 * alpha**4 + 60.0 * alpha**3 - 30.0 * alpha**2 
        force = -fac * disps / r / (self.R_outer - self.R_inner)
        P_force = np.zeros_like(atoms.positions)
        P_force[index1, :] += force
        P += np.piecewise(r, [r <= self.R_inner, r > self.R_inner], [1.0, 0.0])
        
        energy = P * (nn_energy) + (1 - P) * saptff_energy
        forces = P_force * nn_energy + P * nn_forces + saptff_forces - (P_force * saptff_energy + P * saptff_forces)
        return energy, forces

class PBNN_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running PBNN simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, diabats, couplings, couplings_loc, tmpdir, res_list, device='cuda', atom_accept_index=None, res_accept_index=None, react_index=None, can_accept=None, plumed_call=False, **kwargs):
        """
        Parameters
        -----------
        diabats : list
            List containing Diagonal diabatic state classes
        couplings : list
            List containing Coupling classes 
        couplings_loc : list
            List containing the residues in the diabatic states that the coupling NN is applied to
        tmpdir : str
            directory where the simulation logs are stored
        res_list : list
            List containing the atoms in each residue from OpenMM
        device : str
            device where the neural networks will be run. Default is cuda.
        atom_accept_index : list, optional
            if more than one atom can accept the reacting group, then these atoms should be contained in a list of lists with this accepting atom in each list
        res_accept_index : list, optional
            list of residues that can accept the reacting atom
        react_index : index
            index for the reacting atom
        can_accept : int
            Number of atoms in each accepting molecule that can accept the reacting atom
        plumed_call : list
            list of plumed commands needed for setting up the Plumed calculator
        **kwargs : dict
            additional args for ASE base calculator.
        """
        Calculator.__init__(self, **kwargs)
        self.diabats = diabats
        self.couplings = couplings
        self.couplings_loc = couplings_loc
        self.has_periodic_box = self.diabats[0].saptff.has_periodic_box
        self.plumed_call = plumed_call

        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom
        self.frame = 0
        self.debug_forces = False
        self.res_list = res_list
        self.tmp = tmpdir

        if atom_accept_index:
            self.atom_accept_index = atom_accept_index
            self.res_accept_index = res_accept_index
            self.react_index = react_index
            self.can_accept = can_accept

        if self.diabats[0].adaptive:
            self.R_inner = kwargs.get('R_inner')
            self.R_outer = kwargs.get('R_outer')
            self.adapt_intra = kwargs.get('adaptive_intramolecular')
            self.center_mol_indices = kwargs.get('center_mol_indices')

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
            index = self.couplings_loc[i]
            hamiltonian[index[0], index[1]] = energy
            hamiltonian[index[1], index[0]] = energy

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
            index = self.couplings_loc[i]
            hamiltonian_force[index[0], index[1]] = force
            hamiltonian_force[index[1], index[0]] = force
        
        total_forces = 0
        for i in range(num_states):
            for j in range(num_states):
                total_forces += ci[i] * ci[j] * hamiltonian_force[i, j]
        
        return total_forces

    def check_topology(self, atoms, atom_accept_index, res_accept_index, react_atom, can_react):
        """
        Parameters 
        --------------
        atoms : ASE atoms object
            atoms object
        atom_accept_index : list
            list of lists of atoms that are in each residue that can accept the reacting atom
        res_accept_index : list
            residue that can accept the reacting atom
        react_atom : int
            reacting atom index
        can_react : int
            number of atoms in each molecule that can accept a reacting atom
        """
        
        react_pos = atoms.positions[react_atom]
        distances = atoms.get_distances(react_atom, atom_accept_index, mic=True)
        distances = distances.reshape(int(distances.shape[0]/can_react), can_react)
        dist_min = distances.min(axis=1)        
        dist_order = np.argsort(dist_min)
        dist_order_top = dist_order + res_accept_index[0]
        dist_order_top = dist_order_top.tolist()
        current_accept_ids = []
        top_change = False

        for i in range(1, len(self.diabats)):
            current_accept_ids.append(self.diabats[i].saptff.accepting_resid)
       
        #print(current_accept_ids)
        #print(dist_order_top)
        for i in range(1, len(self.diabats)):
            if self.diabats[i].saptff.accepting_resid in dist_order_top[0:can_react]:
                pass
            else:
                top_change = True
                resname_initial = {self.diabats[i].saptff.accepting_resid: self.diabats[i].saptff.accept_name_initial}
                old_accept = self.diabats[i].saptff.accepting_resid
                for k in range(can_react):
                    if dist_order_top[k] not in current_accept_ids:
                        new_accept = dist_order_top[k]
                        resname_final = {dist_order_top[k]: self.diabats[i].saptff.accept_name_final}
                        break
              
                real_atom_res_list = self.diabats[0].saptff.real_atom_res_list[self.diabats[i].saptff.accepting_resid]
                accept_ids = [real_atom_res_list.index(atom) for atom in self.diabats[i].reorder_func.accept_atom]

                for diabat in range(len(self.diabats)):
                    self.diabats[diabat].saptff.diabat_resids[i] = new_accept
                    
                    if diabat == i:
                        self.diabats[diabat].saptff.change_topology(resname_initial, resname_final)
                    else:
                        self.diabats[diabat].saptff.change_topology()
                self.diabats[i].saptff.accepting_resid = new_accept

                for diabat in range(len(self.diabats)):
                    #for nn_intra in self.diabats[diabat].nn_intra:
                    #    if nn_intra.residue == old_accept:
                    #        nn_intra.nn_force_atoms = self.diabats[diabat].saptff.get_diabatid_res_list()[i]
                    #        nn_intra.residue = new_accept

                    nn_res_list = self.diabats[diabat].saptff.get_diabatid_res_list()
                    for nn_inter in self.diabats[diabat].nn_inter:
                        loc = nn_inter.nn_res_loc
                        internn_res_list = [atom for atom in nn_res_list[loc[0]]]
                        for atom in nn_res_list[loc[1]]: internn_res_list.append(atom)
                        nn_inter.nn_force_atoms = internn_res_list
                
                real_atom_res_list = self.diabats[0].saptff.real_atom_res_list[new_accept]
                accept_atom = [real_atom_res_list[index] for index in accept_ids]
                self.diabats[i].reorder_func.accept_atom = accept_atom
                self.diabats[i].reorder_func.insert_react = real_atom_res_list[-1]

        if top_change:
            for diabat in range(len(self.diabats)):
                self.diabats[diabat].setup_saptff(atoms, atoms.get_positions())
            nn_list = self.diabats[0].saptff.get_diabatid_res_list()
            for coupling in self.couplings:
                residues = coupling.nn_list_loc
                nn_indices = [i for i in nn_list[residues[0]]]
                if residues[0] != 0:
                    nn_indices.append(self.diabats[0].saptff.react_atom)
                [nn_indices.append(i) for i in nn_list[residues[1]]]
                coupling.force_indices = nn_indices

        if self.diabats[0].adaptive:
            self.check_adaptive(atoms, atom_accept_index, can_react, res_accept_index)

    def check_adaptive(self, atoms, atom_accept_index, can_react, res_accept_index):
        """
        atoms : ASE atoms object
            atoms object
        atom_accept_index : list
            list of lists of atoms that are in each residue that can accept the reacting atom
        can_react : int
            number of atoms in each molecule that can accept a reacting atom
        res_accept_index : list
            residue that can accept the reacting atom
        """        
        com_pos = atoms[self.center_mol_indices].get_center_of_mass()
        disp = com_pos - atoms.positions[atom_accept_index]
        cell = np.asarray(atoms.get_cell())
        cell = np.expand_dims(cell, axis=0)
        cell = np.repeat(cell, len(atom_accept_index), axis=0)
        diff = np.zeros((len(atom_accept_index), 3))
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
        distances_adapt = distances.reshape(int(distances.shape[0]/can_react), can_react)
        dist_min_adapt = distances_adapt.min(axis=1)
        dist_order_adapt = np.argsort(dist_min_adapt)
        dist_order_adapt_top = dist_order_adapt + res_accept_index[0]

        exclude_dist_adapt = dist_min_adapt[dist_order_adapt]
        #print(exclude_dist_adapt)
        #print(dist_order_adapt_top)
        current_exclude_residues = self.diabats[0].saptff.exclude_intra_res[1:]
        exclude = np.where(exclude_dist_adapt < self.R_outer)[0]
        correct_exclude_residues = [dist_order_adapt_top[res] for res in exclude]
        remove_residue = [res for res in current_exclude_residues if res not in correct_exclude_residues]
        add_residue = [res for res in correct_exclude_residues if res not in current_exclude_residues]
        adaptive_region = np.where(np.logical_and(exclude_dist_adapt < self.R_outer, exclude_dist_adapt > self.R_inner))[0]
        environment_region = np.where(exclude_dist_adapt > self.R_outer)[0]
        adaptive_residues = [dist_order_adapt_top[res] for res in adaptive_region]
        environment_residues = [dist_order_adapt_top[res] for res in environment_region]
        intrann_residues = np.where(exclude_dist_adapt < self.R_inner)[0]
        intrann_residues = [dist_order_adapt_top[res] for res in intrann_residues]

        #print(self.diabats[0].saptff.exclude_intra_res)
        #print(add_residue)
        #print(remove_residue)
        self.adapt_intra.set_adaptive_residues(adaptive_residues)
        for diabat in self.diabats:
            if remove_residue:
                diabat.saptff.create_exclusions_intra_remove(remove_residue)
                for diabat in self.diabats:
                    diabat.setup_saptff(atoms, atoms.get_positions())
            if add_residue:
                diabat.saptff.create_exclusions_intra_add(add_residue)
                for diabat in self.diabats:
                    diabat.setup_saptff(atoms, atoms.get_positions())
       
        for diabat in self.diabats:
            adjust_intra_nn = []
            for nn_intra in diabat.nn_intra:
                if (nn_intra.residue not in adaptive_residues) and (nn_intra.residue not in environment_residues):
                    res_list = diabat.saptff.get_res_list()
                    res_name = diabat.saptff.get_res_names()[nn_intra.residue]
                    if nn_intra.nn_force_atoms != res_list[nn_intra.residue]:
                        model = self.adapt_intra.models_dict[res_name]
                        nn_intra_new = NN_Intra_Diabat("ff_files/"+model, res_list[nn_intra.residue], nn_intra.residue, diabat.saptff.has_periodic_box)
                        adjust_intra_nn.append(nn_intra_new)
                    else:
                        adjust_intra_nn.append(nn_intra)
            diabat.nn_intra = adjust_intra_nn

        for residue in intrann_residues:
            for diabat in self.diabats:
                residue_present = False
                for nn_intra in diabat.nn_intra:
                    if nn_intra.residue == residue:
                        residue_present = True
                if not residue_present:
                    res_names = diabat.saptff.get_res_names()
                    res_name = res_names[residue]
                    model = self.adapt_intra.models_dict[res_name]
                    res_list = diabat.saptff.get_res_list()
                    nn_intra = NN_Intra_Diabat("ff_files/"+model, res_list[residue], residue, diabat.saptff.has_periodic_box)
                    diabat.nn_intra.append(nn_intra)

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
        if self.has_periodic_box and not self.plumed_call:
            atoms.wrap()
            shifts = make_molecule_whole(atoms.get_positions(), atoms.get_cell(), self.res_list)
            atoms.positions -= shifts
        
        print(self.frame) 
        if hasattr(self, 'atom_accept_index') and self.frame % 50 == 0:
        #if hasattr(self, 'atom_accept_index'):
            self.check_topology(atoms, self.atom_accept_index, self.res_accept_index, self.react_index, self.can_accept)
       
        Calculator.calculate(self, atoms)

        #atoms.write("current_frame.xyz")
        adapt_dict = None
        if self.diabats[0].adaptive:
            adapt_dict = self.adapt_intra.compute_energy_force(atoms)

        diabat_energies = []
        diabat_forces = []
        for diabat in self.diabats: 
            energy, forces = diabat.compute_energy_force(atoms, adapt_dict=adapt_dict)
            diabat_energies.append(energy)
            diabat_forces.append(forces)
        
        coupling_energies = []
        coupling_forces = []
        for coupling in self.couplings:
            coupling_force = np.zeros_like(diabat_forces[-1])
            energy, forces = coupling.compute_energy_force(atoms, coupling_force)
            coupling_energies.append(energy)
            coupling_forces.append(forces)
      
        energy, ci = self.diagonalize(diabat_energies, coupling_energies)
        forces = self.calculate_forces(diabat_forces, coupling_forces, ci)
        
        ci2_coefficients = open(f"{self.tmp}/ci2_coefficients.txt", "a+")
        ci2_coefficients.write(f"{self.frame} {ci[0]**2} {ci[1]**2}\n")
        ci2_coefficients.close()
        
        self.frame += 1
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class NeuralNetworkHamiltonian(Calculator):
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]
    def __init__(self, ase_atoms, model, res, periodic, **kwargs):
        """
        Parameters
        -----------
        ase_atoms : str
            Location of input structure, gets created to ASE Atoms object.
        model : str
            NN model
        res : int
            arbitrary residue index
        periodic : bool
            bool for whether the system is periodic
        """
        Calculator.__init__(self, **kwargs)

        force_atoms = np.arange(len(ase_atoms))
        self.nn_intra = NN_Intra_Diabat(model, force_atoms, res, periodic)
        
        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom

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
        total_forces = np.zeros_like(atoms.positions)
        energy, forces = self.nn_intra.compute_energy_force(atoms, total_forces)
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms),3)) * self.forces_units

        self.results = result

class ASE_MD:
    """
    Setups and runs the MD simulation. Serves as an interface to the PBNN Hamiltonian class and ASE.
    """
    def __init__(self, ase_atoms, tmp, diabats, coupling, coupling_loc, frame=-1, plumed_input=[], device='cuda', **kwargs):
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

        if isinstance(ase_atoms, Atoms):
            self.mol = ase_atoms
        else:
            self.mol = read(ase_atoms, index="{}".format(frame))
        self.diabats = diabats
        self.coupling = coupling
        
        masses = diabats[0].saptff.get_masses()
        self.mol.set_masses(self.diabats[0].saptff.get_masses())

        for diabat in self.diabats:
            diabat.setup_saptff(self.mol, self.mol.get_positions())
        
        res_list = self.diabats[0].saptff.get_res_list()

        mul_accept_res = kwargs.get('mul_accept_res', None)
        mul_accept_atom = kwargs.get('mul_accept_atom', None)
        react_atom = kwargs.get('react_atom', None)
        react_residue = kwargs.get('react_residue', None)
        self.rewrite = kwargs.get('rewrite_log', True)
        
        if mul_accept_res:
            can_accept = len(mul_accept_atom)
            atom_accept_index, res_accept_index = self.diabats[0].saptff.get_spec_res_index(mul_accept_res, mul_accept_atom)
        else:
            can_accept = None

        if not plumed_input:
            self.calculator = PBNN_Hamiltonian(diabats, coupling, coupling_loc, self.tmp, res_list, device=device, atom_accept_index=atom_accept_index, res_accept_index=res_accept_index, react_index=react_atom, can_accept=can_accept)
        else:
            pbnn_calculator = PBNN_Hamiltonian(diabats, coupling, coupling_loc, self.tmp, res_list, device=device, atom_accept_index=atom_accept_index, res_accept_index=res_accept_index, react_index=react_atom, can_accept=can_accept, plumed_call=True, **kwargs)
            self.calculator = Plumed(pbnn_calculator, plumed_input, 1.0, atoms=self.mol, kT=300.0*units.kB, log=f'{self.tmp}/colvar.dat', res_list=res_list)
        
        self.mol.set_calculator(self.calculator) 
        self.md = False

    def set_mol_positions(self, positions):
        """
        Parameters 
        ------------
        positions : np.ndarray
            positions to set for the molecule object
        """
        self.mol.positions = positions

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
            self.md = Langevin(self.mol, time_step * units.fs, temperature_K=temp, friction=friction/units.fs)

        logfile = os.path.join(self.tmp, "{}.log".format(name))
        trajfile = os.path.join(self.tmp, "{}.traj".format(name))
        if self.rewrite and os.path.isfile(logfile) and os.path.isfile(trajfile):
            os.remove(logfile)
            os.remove(trajfile)
        
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

