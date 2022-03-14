#!/usr/bin/env python
import sys, time
from initialize_system import initialize_system
from ase.io import read, write

def main():
    #Just read in the atoms object and pass the input file to initialize_system
    atoms = read(sys.argv[1])
    ase_md, run_time = initialize_system(atoms, sys.argv[2])
   
    #Run the simulation
    ase_md.run_md(run_time)

if __name__ == "__main__":
    main()

