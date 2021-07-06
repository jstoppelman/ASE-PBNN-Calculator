from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np

#**********************************************
# this routine uses OpenMM to evaluate energies of SAPT-FF force field
# we use the DrudeSCFIntegrator to solve for equilibrium Drude positions.
# positions of atoms are frozen by setting mass = 0 in .xml file
#**********************************************

class SAPT_ForceField:
    # Setup OpenMM simulation object with template pdb
    # and xml force field files
    def __init__(self, pdbtemplate, residuexml, saptxml , platformtype='CPU', Drude_hyper_force=True, exclude_intra_res=None):

        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        Topology().loadBondDefinitions(residuexml)
        self.pdb = PDBFile(pdbtemplate)  # this is used for topology, coordinates are not used...
        self.integrator = DrudeSCFIntegrator(0.0005*picoseconds) # Use the SCF integrator to optimize Drude positions    
        self.integrator.setMinimizationErrorTolerance(0.01)
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = ForceField(saptxml)
        self.modeller.addExtraParticles(self.forcefield)  # add Drude oscillators
        self.exclude_intra_res = exclude_intra_res 

        # by default, no cutoff is used, so all interactions are computed.  This is what we want for gas phase PES...no Ewald!!
        self.system = self.forcefield.createSystem(self.modeller.topology, constraints=None, rigidWater=True)

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
        if Drude_hyper_force == True:
            self.add_Drude_hyper_force()

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)

        self.platform = Platform.getPlatformByName(platformtype)
        if platformtype == 'CPU':
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
        elif platformtype == 'Reference': 
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
        elif platformtype == 'OpenCL':
            self.properties = {'OpenCLPrecision': 'double'}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
       
        all_atom_res_list = []
        for res in self.simulation.topology.residues():
            c_res = []
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            all_atom_res_list.append(c_res)

        real_atom_res_list = []
        for res in self.pdb.topology.residues():
            c_res = []
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            real_atom_res_list.append(c_res)
        
        if self.exclude_intra_res is not None:
            self.exclude_atom_list = [z for i in self.exclude_intra_res for z in all_atom_res_list[i]]
            self.add_exclusions_monomer_intra()
            self.nnRealAtoms = [z for i in self.exclude_intra_res for z in real_atom_res_list[i]]

        self.get_drude_pairs()

        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        self.positions = state.getPositions()

    #***********************************
    # this method excludes all intra-molecular non-bonded interactions in the system
    # for Parent/Drude interactions, exclusion is replaced with a damped Thole interaction...
    #***********************************
    def add_exclusions_monomer_intra(self):    
        harmonicBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicBondForce][0]
        harmonicAngleForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicAngleForce][0]
        periodicTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce][0]
        rbTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == RBTorsionForce][0]
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]

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
        self.system.addForce(hyper)

        for i in range(drudeForce.getNumParticles()):
            param = drudeForce.getParticleParameters(i)
            drude = param[0]
            parent = param[1]
            hyper.addBond(drude, parent)

    def res_list(self):
        #Return dictionary of indices for each monomer in the dimer. 
        #Used for diababt intramolecular neural networks
        res_dict = []
        k = 0
        for res in self.pdb.topology.residues():
            res_list = []
            for i in range(len(res._atoms)):
                res_list.append(res._atoms[i].index)
            res_dict.append(res_list)
        return res_dict

    def get_masses(self):
        return self.masses

    def get_nn_atoms(self):
        return self.nnRealAtoms

    def get_drude_pairs(self):
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.drudePairs = []
        for i in range(drudeForce.getNumParticles()):
            parms = drudeForce.getParticleParameters(i)
            self.drudePairs.append((parms[0], parms[1]))

    def set_initial_positions(self, xyz):
        self.xyz_pos = xyz
        for i in range(len(self.xyz_pos)):
            # update pdb positions
            self.positions[i] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        # now update positions in modeller object
        self.modeller = Modeller(self.pdb.topology, self.positions)
        # add dummy site and shell initial positions
        self.modeller.addExtraParticles(self.forcefield)
        self.simulation.context.setPositions(self.modeller.positions)
        self.positions = self.modeller.positions

    def set_xyz(self, xyz):
        self.xyz_pos = xyz
        self.initial_positions = self.simulation.context.getState(getPositions=True).getPositions()
        for i in range(len(self.realAtoms)):
            self.positions[self.realAtoms[i]] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        self.drudeDisplacement()
        self.simulation.context.setPositions(self.positions)

    def drudeDisplacement(self):
        for i in range(len(self.drudePairs)):
            disp = self.positions[self.drudePairs[i][1]] - self.initial_positions[self.drudePairs[i][1]]
            self.positions[self.drudePairs[i][0]] += disp
    
    # Compute the energy for a particular configuration
    # the input xyz array should follow the same structure
    # as the template pdb file
    def compute_energy(self): 

        # integrate one step to optimize Drude positions.  Note that atoms won't move if masses are set to zero
        self.simulation.step(1)
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()
        
        # get energy
        state = self.simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        eSAPTFF = state.getPotentialEnergy()/kilojoule_per_mole
        SAPTFF_forces = state.getForces(asNumpy=True)[self.realAtoms]/kilojoule_per_mole*nanometers

        return np.asarray(eSAPTFF), SAPTFF_forces/10.0

