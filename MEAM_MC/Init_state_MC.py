#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
import time

from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]

import os
import subprocess
import shutil
import glob
import argparse

RunPath = os.getcwd() + "/"

# First, we write a lammps input script for this run
def write_lammps_input(potPath):

    lines = ["units \t metal\n",
             "atom_style \t atomic\n",
             "atom_modify \t map array\n",
             "boundary \t p p p\n",
             "atom_modify \t sort 0 0.0\n",
             "read_data \t inp_MC.data\n",
             "pair_style \t meam\n",
             "pair_coeff \t * * {0}/library.meam Co Ni Cr Fe Mn {0}/params.meam Co Ni Cr Fe Mn\n".format(potPath),
             "minimize	\t 1e-5 0.0 1000 10000\n",
             "variable x equal pe\n",
             "print \"$x\" file Eng.txt"]

    with open("in.minim", "w") as fl:
        fl.writelines(lines)


# Next, we write the MC loop
def MC_Run(T, SwapRun, ASE_Super, elems,
           N_therm=2000, N_save=200, lastChkPt=0):

    if not os.path.isdir(RunPath + "chkpt"):
        os.mkdir(RunPath + "chkpt")

    if not os.path.isdir(RunPath + "test"):
        os.mkdir(RunPath + "test")

    if not os.path.isdir(RunPath + "History_backup"):
        os.mkdir(RunPath + "History_backup")

    cmdString = "$LMPPATH/lmp -in in.minim > out.txt"

    Natoms = len(ASE_Super)
    N_accept = 0
    N_total = 0
    Eng_steps_all = []
    accepts = []
    rand_steps = []
    swap_steps = []
    
    # Store the starting supercell in the test directory if doing from scratch
    if lastChkPt == 0:
        write_lammps_data("test/inp_MC_test_step_{0}.data".format(N_total), ASE_Super, specorder=elems)
        with open("test/supercell_{}_test.pkl".format(N_total), "wb") as fl_sup:
            pickle.dump(ASE_Super, fl_sup)

    else:
        Eng_steps_all = list(np.load("Eng_all_steps.npy")[:lastChkPt])
        accepts = list(np.load("accepts_all_steps.npy")[:lastChkPt])
        rand_steps = list(np.load("rands_all_steps.npy")[:lastChkPt])
        swap_steps = list(np.load("swap_atoms_all_steps.npy")[:lastChkPt])


    # write the supercell as a lammps file
    write_lammps_data("inp_MC.data", ASE_Super, specorder=elems)

    # evaluate the energy
    cmd = subprocess.Popen(cmdString, shell=True)
    rt = cmd.wait()
    assert rt == 0
    start_time = time.time()
    # read the energy
    with open("Eng.txt", "r") as fl_en:
        e1 = fl_en.readline().split()[0]
        e1 = float(e1)
    
    beta = 1.0/(kB * T)
    Eng_steps_all.append(e1)
    while N_total < SwapRun - lastChkPt: # run the remaining steps only
        
        # Now randomize the atomic occupancies
        site1 = np.random.randint(0, Natoms)
        site2 = np.random.randint(0, Natoms)
        while ASE_Super[site1].symbol == ASE_Super[site2].symbol:
            site1 = np.random.randint(0, Natoms)
            site2 = np.random.randint(0, Natoms)

        # change the occupancies
        tmp = ASE_Super[site1].symbol
        ASE_Super[site1].symbol = ASE_Super[site2].symbol
        ASE_Super[site2].symbol = tmp

        # write the supercell again as a lammps file
        write_lammps_data("inp_MC.data", ASE_Super, specorder=elems)

        # evaluate the energy
        cmd = subprocess.Popen(cmdString, shell=True)
        rt = cmd.wait()
        assert rt == 0
        # read the energy
        with open("Eng.txt", "r") as fl_en:
            e2 = fl_en.readline().split()[0]
            e2 = float(e2)

        # make decision
        de = e2 - e1
        rn = np.random.rand()

        if rn <= np.exp(-beta * de):
            # Then accept the move
            N_accept += 1
            e1 = e2  # set the next initial state energy to the current final energy
            accepts.append(1)

        else:
            # reject the move by reverting the occupancies to initial state values
            tmp = ASE_Super[site1].symbol
            ASE_Super[site1].symbol = ASE_Super[site2].symbol
            ASE_Super[site2].symbol = tmp
            accepts.append(0)

        rand_steps.append(rn)
        swap_steps.append([site1, site2])
        
        # save the history at all steps
        Eng_steps_all.append(e1)
        np.save("Eng_all_steps.npy", np.array(Eng_steps_all))
        np.save("accepts_all_steps.npy", np.array(accepts))
        np.save("rands_all_steps.npy", np.array(rand_steps))
        np.save("swap_atoms_all_steps.npy", np.array(swap_steps))

        N_total += 1

        if N_total%N_save == 0:
            with open("timing.txt", "w") as fl_timer:
                t_now = time.time()
                fl_timer.write("Time Per step ({0} steps): {1}\n".format(N_total, (t_now-start_time)/N_total))

            if N_total + lastChkPt >= N_therm:
                with open("chkpt/supercell_{}.pkl".format(N_total + lastChkPt), "wb") as fl_sup:
                    pickle.dump(ASE_Super, fl_sup)

                with open("chkpt/counter.txt", "w") as fl_counter:
                    fl_counter.write("last step saved\n{}".format(N_total))

            # Back up the trajectory history
            shutil.copy("Eng_all_steps.npy", "History_backup/")
            shutil.copy("accepts_all_steps.npy", "History_backup/")
            shutil.copy("rands_all_steps.npy", "History_backup/")
            shutil.copy("swap_atoms_all_steps.npy", "History_backup/")


        # For the first 20 steps, store all the supercells as well to a test directory if we want to check later
        if N_total <= 20 and lastChkPt == 0:
            np.save("test/Eng_steps_test.npy", np.array(Eng_steps_all))
            np.save("test/rand_steps_test.npy", np.array(rand_steps))
            np.save("test/swap_atoms_test.npy", np.array(swap_steps))
            np.save("test/acceptances_test.npy", np.array(accepts))
            # store the supercells and lammps files too
            write_lammps_data("test/inp_MC_test_step_{0}.data".format(N_total), ASE_Super, specorder=elems)
            with open("test/supercell_{}_test.pkl".format(N_total), "wb") as fl_sup:
                pickle.dump(ASE_Super, fl_sup)

    return N_total, N_accept


def main(args):
    print("Using CheckPoint : {}".format(args.UseLastChkPt))

    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]

    elemsToNum = {}
    for elemInd, el in enumerate(elems):
        elemsToNum[el] = elemInd + 1

    if args.UseLastChkPt:
        ChkPtFiles = os.getcwd() + "/chkpt/*.pkl"
        files = glob.glob(ChkPtFiles)
        
        if len(files) == 0:
            raise FileNotFoundError("No checkpoint found.")

        else:
            files_sorted = sorted(files, key=lambda fl: int(fl.split("/")[-1][10:-4]))
            max_file = files_sorted[-3] # Get the file created 400 steps before last
            with open(max_file, "rb") as fl:
                superFCC = pickle.load(fl)
        
            lastFlName=max_file.split("/")[-1]
            lastSave=int(lastFlName[10:-4])
            print("Loading checkpointed step : {} from : {}".format(lastSave, os.getcwd() + "/chkpt/"))

    else:
        lastSave=0

        # Create an FCC primitive unit cell
        a = args.LatPar
        fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

        # Form a supercell with a vacancy at the centre
        superlatt = np.identity(3) * args.Nunits
        superFCC = make_supercell(fcc, superlatt)

        if not args.NoVac:
            print("Putting vacancy at site 0")
            assert np.allclose(superFCC[0].position, 0)
            del(superFCC[0])
        
        # randomize occupancies of the sites
        Nsites = len(superFCC.get_positions())
        assert sum(args.Natoms, Nsites), "Total number of atoms does not match supercell size."

        Indices = np.random.permutation(Nsites) # store the sites in random order to be occupied randomly.
        NSpec = len(elems)

        for i in range(NSpec):

            lower = sum(args.Natoms[j] for j in range(i))
            upper = lower + args.Natoms[i]
            print("{} : {} atoms".format(elems[i], upper - lower))
            for at_Ind in range(lower, upper):
                permInd = Indices[at_Ind]
                superFCC[permInd].symbol = elems[i]

        print("Supercell formula: {}". format(superFCC.get_chemical_formula()))
        # save the initial supercell
        print("Occupied Sites: \n{}".format(Indices))
        np.save("SiteOccs_initial.npy", Indices)
        with open("superInitial.pkl", "wb") as fl:
            pickle.dump(superFCC, fl)
        

    # Run MC
    if not args.UseLastChkPt:
        # Lammps input script need be written only once. We're also starting from on-lattice positions for
        # reproducibility.
        write_lammps_input(args.potPath)

    start = time.time()
    N_total, N_accept = MC_Run(args.Temp, args.Nsteps, superFCC, elems, N_therm=args.NEqb, N_save=args.Nsave, lastChkPt=lastSave)
    end = time.time()
    print("Thermalization Run acceptance ratio : {}".format(N_accept/N_total))
    print("Thermalization Run accepted moves : {}".format(N_accept))
    print("Thermalization Run total moves : {}".format(N_total))
    print("Thermalization Time Per iteration : {}".format((end-start)/N_total))
    with open("superFCC_End.pkl", "wb") as fl:
        pickle.dump(superFCC, fl)

if __name__ == "__main__":
    # N_units, a0, NoVac, T, N_swap, N_eqb, N_save

    parser = argparse.ArgumentParser(description="Input parameters for using GCnets",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-pp", "--potPath", metavar="/path/to/potential/file", type=str,
                        help="Path to the LAMMPS MEAM potential.")

    parser.add_argument("-na", "--Natoms", metavar="102 104...", nargs="+", type=int, default=[102, 103, 102, 102, 102],
                        help="Number of atoms of each kind of Co, Ni, Cr, Fe, Mn in that order.")


    parser.add_argument("-ckp", "--UseLastChkPt", action="store_true",
                        help="Whether to store final style NEB files for all jumps separately.")

    parser.add_argument("-u", "--Nunits", metavar="0", type=int, default=8,
                        help="Number of unit cells in the supercell.")

    parser.add_argument("-a0", "--LatPar", metavar="1.0", type=float, default=3.59,
                        help="Lattice parameter - multiplied to displacements and used"
                             "to construct LAMMPS coordinates.")

    parser.add_argument("-nv", "--NoVac", action="store_true",
                        help="Whether to disable vacancy creation.")

    parser.add_argument("-T", "--Temp", metavar="1073", type=float, default=1073.0,
                        help="Temperature in Kelvin.")

    parser.add_argument("-nt", "--Nsteps", metavar="60000", type=int, default=60000,
                        help="Total number of Metropolis trials to run. A Metropolis trial consists of swapping two "
                             "sites and accepting with a probability.")

    parser.add_argument("-ne", "--NEqb", metavar="2000", type=int, default=2000,
                        help="Number of equilibrating/thermalizing steps.")

    parser.add_argument("-ns", "--Nsave", metavar="200", type=int, default=200,
                        help="Interval of steps after equilibration after which to collect a state as a sample.")

    parser.add_argument("-dmp", "--DumpArguments", action="store_true",
                        help="Whether to dump all the parsed arguments into a text file.")

    parser.add_argument("-dpf", "--DumpFile", type=str, default="ArgFiles",
                        help="The file in the run directory where all the args will be dumped.")


    args = parser.parse_args()

    if args.DumpArguments:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(os.getcwd()+ '/' + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}:\t{}\n".format(key, val))

    main(args)
