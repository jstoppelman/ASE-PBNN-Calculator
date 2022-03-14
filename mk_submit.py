#!/usr/bin/env python
import numpy as np 

ch = np.arange(1.4, 2.0, 0.15)
oh = np.arange(1.05, 1.41, 0.15)
ch2 = np.arange(1.05, 1.4, 0.15)
oh2 = np.arange(1.4, 2.55, 0.15)

input_tmp = []
for line in open("pbnn_input.txt"):
    input_tmp.append(line)

submit_tmp = []
for line in open("submit2.pbs"):
    submit_tmp.append(line)

sim = 0
for i in ch:
    for j in oh:
        out = open(f"pbnn_input_{sim}.txt", "w")
        for line in input_tmp:
            if "sim_dir" in line:
                out.write(f"sim_dir = window{sim} !Directory to place simulation output\n")
            elif "Umbrella_center_positions" in line:
                out.write(f"Umbrella_center_positions = {i} {j} !Distances (in angstrom) for which the umbrella potentials should be centered\n")
            else:
                out.write(line)
        out.close()
        out = open(f"submit_window_{sim}.pbs", "w")
        for line in submit_tmp:
            if "python" in line:
                out.write(f"python main.py input_40_equil.xyz pbnn_input_{sim}.txt\n\n")
            elif "PBS -N" in line:
                out.write(f"#PBS -N window_{sim}\n")
            else:
                out.write(line)
        sim += 1

for i in ch2:
    for j in oh2:
        out = open(f"pbnn_input_{sim}.txt", "w")
        for line in input_tmp:
            if "sim_dir" in line:
                out.write(f"sim_dir = window{sim} !Directory to place simulation output\n")
            elif "Umbrella_center_positions" in line:
                out.write(f"Umbrella_center_positions = {i} {j} !Distances (in angstrom) for which the umbrella potentials should be centered\n")
            else:
                out.write(line)
        out.close()
        out = open(f"submit_window_{sim}.pbs", "w")
        for line in submit_tmp:
            if "python" in line:
                out.write(f"python main.py input_40_equil.xyz pbnn_input_{sim}.txt\n\n")
            elif "PBS -N" in line:
                out.write(f"#PBS -N window_{sim}\n")
            else:
                out.write(line)
        sim += 1


