from __future__ import print_function
from openmm.app import *
from openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np
import time 
from copy import deepcopy
import gc

#**********************************************
# this routine uses OpenMM to evaluate energies of SAPT-FF force field
# we use the DrudeSCFIntegrator to solve for equilibrium Drude positions.
# positions of atoms are frozen by setting mass = 0 in .xml file
#**********************************************

class SAPT_ForceField:
    """
    Setup OpenMM simulation object with template pdb
    and xml force field files
    """
    def __init__(self, pdbtemplate, residuexml, saptxml , platformtype='CPU', Drude_hyper_force=True, exclude_intra_res=[]):
        """
        Parameters
        -------------
        pdbtemplate : str
            path to pdb file containing the topology of the system
        residuexml : str
            path to residue file containing the bond definitions
        saptxml : str
            path to xml file containing force field definitions
        platformtype : optional, str
            Platform for running the OpenMM calculations on. Default is CPU, choices are Reference, CPU, OpenCL, CUDA
        Drude_hyper_force : optional, bool
            whether to add a restraining force for Drude particles
        exclude_intra_res : optional, list
            list residue indices for which the force field intramolecular DOF will be excluded/zerod out
        """

        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        self.pdbtemplate = pdbtemplate
        self.residuexml = residuexml
        self.Drude_hyper_force = Drude_hyper_force
        self.platformtype = platformtype
        Topology().loadBondDefinitions(residuexml)
        self.pdb = PDBFile(pdbtemplate)  # this is used for topology, coordinates are not used...
        #Real_atom_topology contains the top info for only real atoms, no virutal sites or Drude particles
        self.real_atom_topology, positions = self.pdb.topology, self.pdb.positions

        self.modeller = Modeller(self.real_atom_topology, positions)
        self.forcefield = ForceField(saptxml)
        self.modeller.addExtraParticles(self.forcefield)  # add Drude oscillators
        self.exclude_intra_res = exclude_intra_res 
        self.diabat_resids = deepcopy(exclude_intra_res)

        #Gets periodic box vectors
        cell_vectors = self.real_atom_topology.getPeriodicBoxVectors()
        if cell_vectors is not None: self.has_periodic_box = True
        else: self.has_periodic_box = False

        #Sets up everything related to a simulation object
        self.create_simulation()

        #Set current positions, although these are somewhat useless as they will be overwritten
        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        self.positions = state.getPositions()

    def create_simulation(self):
        """
        Sets up OpenMM simulation object
        """
        #Integrator used to minimizer Drude particle positions
        self.integrator = DrudeSCFIntegrator(0.0005*picoseconds) # Use the SCF integrator to optimize Drude positions
        #self.integrator = VerletIntegrator(0.0005*picoseconds)
        #Use tolerance of 0.01 kj/mol/nm
        self.integrator.setMinimizationErrorTolerance(0.01)

        #If periodic box present, create system object with cutoff
        if self.has_periodic_box:
            cutoff = self.modeller.topology.getPeriodicBoxVectors()[0][0]/nanometers
            self.cutoff = cutoff / 2 - 0.01
            self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff*nanometer, constraints=None, rigidWater=False)
        else:
            # by default, no cutoff is used, so all interactions are computed.  This is what we want for gas phase PES...no Ewald!!
            self.system = self.forcefield.createSystem(self.modeller.topology, constraints=None, rigidWater=False)
        
        #Smaller Ewald error tolerance should help energy conservation with Drudes
        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        nbondedForce.setEwaldErrorTolerance(5.5e-5)
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        customBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        #Determine whether to use PBCs for customBond and nbondedForce
        if self.has_periodic_box:
            customBondForce.setUsesPeriodicBoundaryConditions(True)
            nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        else:
            nbondedForce.setNonbondedMethod(NonbondedForce.NoCutoff)
        customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
        customNonbondedForce.setUseLongRangeCorrection(True)

        # Obtain a list of real atoms excluding Drude particles for obtaining forces later
        # Also set particle mass to 0 in order to optimize Drude positions without affecting atom positions
        self.realAtoms = []
        self.masses = []
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i)/dalton > 1.0:
                self.realAtoms.append(i)
                self.masses.append(self.system.getParticleMass(i)/dalton)
            self.system.setParticleMass(i,0)

        # add "hard wall" hyper force to Drude/parent atoms to prevent divergence with SCF integrator...
        if self.Drude_hyper_force == True:
            self.add_Drude_hyper_force()

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)

        #Platform used for OpenMM
        self.platform = Platform.getPlatformByName(self.platformtype)
        if self.platformtype == 'CPU':
            self.properties = {}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'Reference':
            self.properties = {}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'CUDA':
            self.properties = {'CudaPrecision': 'double'}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'OpenCL':
            self.properties = {'OpenCLPrecision': 'double'}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)

        #Residue list of all the atoms in the simulation
        all_atom_res_list = []
        res_names = []
        for res in self.simulation.topology.residues():
            c_res = []
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            all_atom_res_list.append(c_res)
            res_names.append(res.name)
        self.res_names = res_names
        self.all_atom_res_list = all_atom_res_list

        #Residue list for only real atoms
        real_atom_res_list = []
        residue_topology_types = {}
        for res in self.real_atom_topology.residues():
            c_res = []
            if res.name not in residue_topology_types.keys():
                residue_topology_types[res.name] = res
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            real_atom_res_list.append(c_res)
        self.real_atom_res_list = real_atom_res_list
        self.residue_topology_types = residue_topology_types
        
        #Exclude residues present in exclude_intra_res
        if self.exclude_intra_res is not None:
            self.exclude_atom_list = [z for i in self.exclude_intra_res for z in all_atom_res_list[i]]
            self.diabat_exclude = [z for i in self.diabat_resids for z in self.all_atom_res_list[i]]
            self.add_exclusions_monomer_intra()
            self.nnRealAtoms = [z for i in self.exclude_intra_res for z in self.real_atom_res_list[i]]
            self.nn_res_list = []
            for i in self.exclude_intra_res:
                self.nn_res_list.append(self.real_atom_res_list[i])

        #Collection of Drude pairs
        self.get_drude_pairs()

    def create_exclusions_intra_add(self, res_index):
        """
        Adds residue for exclusions

        Parameters
        ------------------
        res_index : list
            list of residues that will be added to self.exclude_intra_res
        """
        for res in res_index:
            self.exclude_intra_res.append(res)
        self.exclude_atom_list = [z for i in self.exclude_intra_res for z in self.all_atom_res_list[i]]
        self.diabat_exclude = [z for i in self.diabat_resids for z in self.all_atom_res_list[i]]
        self.add_exclusions_monomer_intra()
        self.nnRealAtoms = [z for i in self.exclude_intra_res for z in self.real_atom_res_list[i]]
        self.nn_res_list = []
        for i in self.exclude_intra_res:
            self.nn_res_list.append(self.real_atom_res_list[i])
    
    def create_exclusions_intra_remove(self, res_index):
        """
        Removes exclusions from residues

        Parameters
        ------------------
        res_index : list
            list of residues that will be added to self.exclude_intra_res
        """
        for res in res_index:
            self.exclude_intra_res.remove(res)
        self.create_simulation()

    def change_topology(self, resname_initial={}, resname_final={}):
        """
        Changes the topology if a residue is a new part of the diabatic states

        Parameters 
        -------------------------
        resname_initial : optional, dict
            dictionary of the initial residue index and names which will be changed 
        resname_final : optional, dict
            dictionary of residue names and indexes which will be changed in the simulation
        """
        new_topology = Topology()
        new_topology.loadBondDefinitions(self.residuexml)
        new_topology.addChain()
        for chain in new_topology.chains(): pass
        residue_definitions = []
        for index, res in enumerate(self.real_atom_topology.residues()):
            if index in resname_initial.keys():
                res_type = self.residue_topology_types[resname_initial[index]]
                new_residue = new_topology.addResidue(res_type.name, chain)
                for atom in res_type.atoms(): new_topology.addAtom(atom.name, atom.element, new_residue)
            elif index in resname_final.keys():
                res_type = self.residue_topology_types[resname_final[index]]
                new_residue = new_topology.addResidue(res_type.name, chain)
                for atom in res_type.atoms(): new_topology.addAtom(atom.name, atom.element, new_residue)
            else:
                new_residue = new_topology.addResidue(res.name, chain)
                for atom in res.atoms(): new_topology.addAtom(atom.name, atom.element, new_residue)

        new_topology.createStandardBonds()
        new_topology.setPeriodicBoxVectors(self.real_atom_topology.getPeriodicBoxVectors())
        self.real_atom_topology = new_topology
        self.modeller = Modeller(new_topology, self.pdb.positions)
        self.modeller.addExtraParticles(self.forcefield)
        
        #Only deltes the reference, but python garbage collection should clean it up afterward
        delattr(self, 'simulation')
        delattr(self, 'integrator')
        self.create_simulation() 

    #***********************************
    # this method excludes all intra-molecular non-bonded interactions in the system
    # for Parent/Drude interactions, exclusion is replaced with a damped Thole interaction...
    #***********************************
    def add_exclusions_monomer_intra(self):    
        """
        Zero out/exclude intramolecular interactions for all atoms in exclude_monomer_intra
        """
        harmonicBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicBondForce][0]
        harmonicAngleForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicAngleForce][0]
        periodicTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce][0]
        rbTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == RBTorsionForce][0]
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        customBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        #Zero energies from intramolecular forces for residues with neural networks
        for i in range(harmonicBondForce.getNumBonds()):
            p1, p2, r0, k = harmonicBondForce.getBondParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list:
                k = Quantity(0, unit=k.unit)
                harmonicBondForce.setBondParameters(i, p1, p2, r0, k)

        for i in range(harmonicAngleForce.getNumAngles()):
            p1, p2, p3, r0, k = harmonicAngleForce.getAngleParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list:
                k = Quantity(0, unit=k.unit)
                harmonicAngleForce.setAngleParameters(i, p1, p2, p3, r0, k)

        for i in range(periodicTorsionForce.getNumTorsions()):
            p1, p2, p3, p4, period, r0, k = periodicTorsionForce.getTorsionParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list or p4 in self.exclude_atom_list:
                k = Quantity(0, unit=k.unit)
                periodicTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, period, r0, k)

        for i in range(rbTorsionForce.getNumTorsions()):
            p1, p2, p3, p4, c1, c2, c3, c4, c5, c6 = rbTorsionForce.getTorsionParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list or p4 in self.exclude_atom_list:
                c1 = Quantity(0, unit=c1.unit)
                c2 = Quantity(0, unit=c2.unit)
                c3 = Quantity(0, unit=c3.unit)
                c4 = Quantity(0, unit=c4.unit)
                c5 = Quantity(0, unit=c5.unit)
                c6 = Quantity(0, unit=c6.unit)
                rbTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, c1, c2, c3, c4, c5, c6)
        
        for i in range(customBondForce.getNumBonds()):
            p1, p2, parms = customBondForce.getBondParameters(i) 
            if (p1 not in self.exclude_atom_list and p1 not in self.diabat_exclude) or (p2 not in self.exclude_atom_list and p2 not in self.diabat_exclude):
                customBondForce.setBondParameters(i, p1, p2, (0.0, 0.1, 10.0)) 
            p1, p2, parms = customBondForce.getBondParameters(i)
        
        # map from global particle index to drudeforce object index
        particleMap = {}
        for i in range(drudeForce.getNumParticles()):
            particleMap[drudeForce.getParticleParameters(i)[0]] = i

        # can't add duplicate ScreenedPairs, so store what we already have
        flagexceptions = {}
        for i in range(nbondedForce.getNumExceptions()):
            (particle1, particle2, charge, sigma, epsilon) = nbondedForce.getExceptionParameters(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexceptions[string1]=1
            flagexceptions[string2]=1

        # can't add duplicate customNonbonded exclusions, so store what we already have
        flagexclusions = {}
        for i in range(customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexclusions[string1]=1
            flagexclusions[string2]=1

        print(' adding exclusions ...')

        # add all intra-molecular exclusions, and when a drude pair is
        # excluded add a corresponding screened thole interaction in its place
        current_res = 0
        for res in self.simulation.topology.residues():
            if current_res in self.exclude_intra_res:
                for i in range(len(res._atoms)-1):
                    for j in range(i+1,len(res._atoms)):
                        (indi,indj) = (res._atoms[i].index, res._atoms[j].index)
                        # here it doesn't matter if we already have this, since we pass the "True" flag
                        nbondedForce.addException(indi,indj,0,1,0,True)
                        # make sure we don't already exclude this customnonbond
                        string1=str(indi)+"_"+str(indj)
                        string2=str(indj)+"_"+str(indi)
                        if string1 in flagexclusions or string2 in flagexclusions:
                            continue
                        else:
                            customNonbondedForce.addExclusion(indi,indj)
                        # add thole if we're excluding two drudes
                        if indi in particleMap and indj in particleMap:
                            # make sure we don't already have this screened pair
                            if string1 in flagexceptions or string2 in flagexceptions:
                                continue
                            else:
                                drudei = particleMap[indi]
                                drudej = particleMap[indj]
                                drudeForce.addScreenedPair(drudei, drudej, 2.0)
            current_res += 1
        
        # now reinitialize to make sure changes are stored in context
        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simulation.context.reinitialize()
        self.simulation.context.setPositions(positions)

    # this method adds a "hard wall" hyper bond force to
    # parent/drude atoms to prevent divergence using the 
    # Drude SCFIntegrator ...
    def add_Drude_hyper_force(self):
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]

        hyper = CustomBondForce('step(r-rhyper)*((r-rhyper)*khyper)^powh')
        hyper.addGlobalParameter('khyper', 100.0)
        hyper.addGlobalParameter('rhyper', 0.02)
        hyper.addGlobalParameter('powh', 6)
        hyper.setUsesPeriodicBoundaryConditions(True)
        self.system.addForce(hyper)

        for i in range(drudeForce.getNumParticles()):
            param = drudeForce.getParticleParameters(i)
            drude = param[0]
            parent = param[1]
            hyper.addBond(drude, parent)

    def get_res_list(self):
        """
        Return dictionary of indices of real atoms for each monomer in the dimer. 
        Used for diababt intramolecular neural networks
        """
        res_dict = []
        for res in self.real_atom_res_list:
            res_list = []
            for i in range(len(res)):
                res_list.append(res[i])
            res_dict.append(res_list)
        return res_dict

    def get_all_atom_res_list(self):
        """
        Return dictionary of indices for all atoms in each dimer, including virtual atoms
        """
        res_dict = []
        for res in self.simulation.topology.residues():
            res_list = []
            for i in range(len(res._atoms)):
                res_list.append(res._atoms[i].index)
            res_dict.append(res_list)
        return res_dict

    def get_spec_res_index(self, res_name, atom_name):
        """
        Get index of specific residue
        """
        react_atom_index = []
        react_res_index = []
        self.reorder_mol = True
        for i, res in enumerate(self.real_atom_topology.residues()):
            if res.name == res_name:
                react_res_index.append(i)
                for j, atom in enumerate(res._atoms):
                    if atom.name in atom_name:
                        react_atom_index.append(atom.index)
        return react_atom_index, react_res_index
    
    def get_res_names(self):
        """
        Get dictionary of residue names
        """
        res_names = {}
        for i, res in enumerate(self.real_atom_topology.residues()):
            res_names[i] = res.name
        return res_names

    def get_diabatid_res_list(self):
        """
        Get residue indices of molecules that are currently classified as diabats
        """
        res_lists = self.get_res_list()
        diabatid_res_list = []
        for diabatid in self.diabat_resids:
            diabatid_res_list.append(res_lists[diabatid])
        return diabatid_res_list

    def get_nn_res_list(self):
        """
        indices of all atoms for which neural networks are being applied to
        """
        return self.nn_res_list

    def get_masses(self):
        """
        Return all masses of real atoms
        """
        return self.masses

    def get_nn_atoms(self):
        """
        Return list of real atoms
        """
        return self.nnRealAtoms

    def set_react_res(self, react_res, react_atom):
        """
        Set reacting residue info

        Parameters
        ------------
        react_res : int
            residue which will donate the reacting group
        react_atom : int
            atom index which will donate group
        """
        res_list = self.get_res_list()
        all_atom_res_list = self.get_all_atom_res_list()
        self.react_resid = react_res
        self.react_res = res_list[react_res]
        self.react_atom = self.react_res[react_atom]
        self.react_atom_molindex = react_atom
        self.react_atom_global = all_atom_res_list[react_res][react_atom]

    def set_accept_res(self, index, initial_name):
        """
        Set accepting residue

        Parameters
        ------------
        index : int
            residue which will accept the reacting group
        index : str
            Accepting residue name in Diabat 1 
        """
        self.accepting_resid = index
        self.accept_name_final = self.res_names[index]
        self.accept_name_initial = initial_name
    
    def get_drude_pairs(self):
        """
        Get drude atom pairs
        """
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.drudePairs = []
        for i in range(drudeForce.getNumParticles()):
            parms = drudeForce.getParticleParameters(i)
            self.drudePairs.append((parms[0], parms[1]))

    def set_initial_positions(self, xyz):
        """
        Parameters
        ------------------
        Inputs
        xyz : np.ndarray
            xyz contains positions of all the real atoms
        """
        self.xyz_pos = xyz
        for i in range(len(self.xyz_pos)):
            # update pdb positions
            self.positions[i] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        # now update positions in modeller object
        self.modeller = Modeller(self.real_atom_topology, self.positions)
        # add dummy site and shell initial positions
        self.modeller.addExtraParticles(self.forcefield)
        self.simulation.context.setPositions(self.modeller.positions)
        self.positions = self.modeller.positions

    def set_xyz(self, xyz):
        """
        Parameters
        ------------------
        Inputs
        xyz : np.ndarray
            xyz contains positions of all the real atoms
        """
        self.xyz_pos = xyz/10
        self.initial_positions = self.simulation.context.getState(getPositions=True).getPositions()
        for i in range(len(self.realAtoms)):
            self.positions[self.realAtoms[i]] = Vec3(self.xyz_pos[i][0], self.xyz_pos[i][1], self.xyz_pos[i][2])*nanometer
        res_list = self.get_all_atom_res_list()
        self.drudeDisplacement()
        self.simulation.context.setPositions(self.positions)

    def drudeDisplacement(self):
        """
        Since the real atoms have moved some amount in set_xyz, displace drude positions by the same amount
        """
        for i in range(len(self.drudePairs)):
            disp = self.positions[self.drudePairs[i][1]] - self.initial_positions[self.drudePairs[i][1]]
            self.positions[self.drudePairs[i][0]] += disp
    
    # Compute the energy for a particular configuration
    # the input xyz array should follow the same structure
    # as the template pdb file
    def compute_energy(self): 

        # integrate one step to optimize Drude positions. Note that parent atoms won't move if masses are set to zero
        self.simulation.step(1)
        
        # get energy
        state = self.simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        self.positions = state.getPositions()
        #PDBFile.writeFile(self.simulation.topology, self.positions, open(f'diabat_{self.react_resid}.pdb', 'w'))
        eSAPTFF = state.getPotentialEnergy()/kilojoule_per_mole
        SAPTFF_forces = state.getForces(asNumpy=True)[self.realAtoms]/kilojoule_per_mole*nanometers
        
        # if you want energy decomposition, uncomment these lines...
        #for j in range(self.system.getNumForces()):
        #    f = self.system.getForce(j)
        #    print(type(f), str(self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
        return np.asarray(eSAPTFF), SAPTFF_forces/10.0
