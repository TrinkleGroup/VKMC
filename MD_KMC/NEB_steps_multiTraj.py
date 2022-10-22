#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import pickle
import subprocess
import sys
import time
import pickle
import collections
import h5py
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from KMC_funcs import *
import os
from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]


args = list(sys.argv)
T = int(args[1])
startStep = int(args[2])
Nsteps = int(args[3])
SampleStart = int(args[4])
batchSize = int(args[5]) # Total samples in this batch of calculations.
chunk = int(args[6]) # How many samples to do NEB on at a time.
permuteStates = bool(int(args[7])) # permute the states (applicable if startStep is 0)? 0 if False.
MainPath = args[8] # The path where the potential file is found
#MainPath = "/home/sohamc2/HEA_FCC/MDMC/"
WriteAllJumps = bool(int(args[9]))

SourcePath, _ = os.path.split(os.path.realpath(__file__))[0] # the directory where the main script is
SourcePath += "/"

# Need to get rid of these argument
NImage = 3
RunPath = os.getcwd()+'/'
print("Running from : " + RunPath)

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load(SourcePath + "SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load(MainPath + "CrysDat_FCC/NNsites_sitewise.npy")[1:, :].T  # Nsites x z array of site neighbors

dxList = np.load(MainPath + "CrysDat_FCC/dxList.npy")

assert SiteIndToNgb.shape[0] == SiteIndToPos.shape[0]
assert SiteIndToNgb.shape[1] == dxList.shape[0]


# Do a small check of the displacements and vacancy neighbors
with open(MainPath + "CrysDat_FCC/supercellFCC.pkl", "rb") as fl:
    supFCC = pickle.load(fl)
    latt = supFCC.crys.latt

for siteInd in range(SiteIndToNgb.shape[0]):
    ciSite, Rsite = supFCC.ciR(siteInd)
    assert ciSite == (0,0)
    for jmp in range(dxList.shape[0]):
        dxR, ci = supFCC.crys.cart2pos(dxList[jmp])
        assert ci == (0,0)
        ngb = supFCC.index(dxR + Rsite, ci)
        assert ngb = SiteIndToNgb[jmp, siteInd]

dxList = dxList * 3.59

# load the data
if startStep > 0:
    SiteIndToSpecAll = np.zeros(batchSize, dtype=np.int8)
    for batch in range(0, batchSize, chunk):
        end = min((batch + 1) * chunk, batchSize)
        with h5py.File(RunPath + "data_{0}_{1}_{2}.h5".format(T, startStep, SampleStart), "r") as fl:
            batcStates = np.array(fl["FinalStates"])

        SiteIndToSpecAll[batch : end, :] = batcStates[:, :]

    vacSiteIndAll = np.zeros(batchSize, dtype=np.int8)
    for stateInd in SiteIndToSpecAll.shape[0]:
        state = SiteIndToSpecAll[stateInd]
        vacSite = np.where(state == 0)[0][0]
        vacSiteIndAll[stateInd] = vacSite

    print("Starting from checkpointed step {}".format(startStep))

else:
    print("Starting from step zero.")
    try:
        allStates = np.load(SourcePath + "states_{}.npy".format(T))
    except:
        raise FileNotFoundError("Initial states not found.")
    if permuteStates:
        perm = np.load(SourcePath + "perm_{}.npy".format(T))
        SiteIndToSpecAll = allStates[perm][SampleStart: SampleStart + batchSize]
        np.save("states_perm_step0_{}.npy".format(T), SiteIndToSpecAll)
    else:
        SiteIndToSpecAll = allStates[SampleStart: SampleStart + batchSize]

    assert np.all(SiteIndToSpecAll[:, 0] == 0), "All vacancies must be at the 0th site initially."
    vacSiteIndAll = np.zeros(SiteIndToSpecAll.shape[0], dtype = int)


try:
    with open(SourcePath + "lammpsBox.txt", "r") as fl:
        Initlines = fl.readlines()
except:
    raise FileNotFoundError("Template lammps data file not found.")

assert SiteIndToSpecAll.shape[1] == len(Initlines[12:])

specs, counts = np.unique(SiteIndToSpecAll[0], return_counts=True)
Nspec = len(specs)  # including the vacancy
Ntraj = SiteIndToSpecAll.shape[0]
assert Ntraj == batchSize

Nsites = SiteIndToSpecAll.shape[1]
Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
Initlines[3] = "{} atom types\n".format(Nspec-1)

# Begin KMC loop below
FinalStates = SiteIndToSpecAll
FinalVacSites = vacSiteIndAll
SpecDisps = np.zeros((Ntraj, Nspec, 3))
AllJumpRates = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
AllJumpBarriers = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
tarr = np.zeros(Ntraj)
JumpSelects = np.zeros(Ntraj, dtype=np.int8) # which jump is chosen for each trajectory

# rates will be stored for the first batch for testing
TestRates = np.zeros((batchSize, 12)) # store all the rates to be tested
TestBarriers = np.zeros((batchSize, 12)) # store all the barriers to be tested
TestRandomNums = np.zeros(batchSize) # store the random numbers used in the test trajectories

# Before starting, write the lammps input files
write_input_files(Ntraj, potPath=MainPath)

start = time.time()
NEB_count = 0

for step in range(Nsteps):
    for batch in range(0, Ntraj, chunk):
        # Write the initial states from last accepted state
        sampleStart = batch * chunk
        sampleEnd = min((batch + 1) * chunk, SiteIndToSpecAll.shape[0])

        SiteIndToSpec = FinalStates[sampleStart: sampleEnd].copy()
        vacSiteInd = FinalVacSites[sampleStart: sampleEnd].copy()

        write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines)

        rates = np.zeros((batchSize, SiteIndToNgb.shape[1]))
        barriers = np.zeros((batchSize, SiteIndToNgb.shape[1]))
        for jumpInd in range(SiteIndToNgb.shape[1]):
            # Write the final states in NEB format for lammps
            write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=WriteAllJumps)

            # store the final lammps files for the first batch of states
            if batch == 0:
                # Store the final data for each traj, at each step and for each jump
                for traj in range(batchSize):
                    cmd = subprocess.Popen("cp final_{0}.data final_{0}_{1}.data".format(traj, jumpInd), shell=True)
                    rt = cmd.wait()
                    assert rt == 0

            # Then run lammps
            commands = [
                "mpirun -np {0} $LMPPATH/lmp -p {0}x1 -in in.neb_{1} > out_{1}.txt".format(NImage, traj)
                for traj in range(batchSize)
            ]
            cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

            # wait for the lammps commands to complete
            for c in cmdList:
                rt_code = c.wait()
                assert rt_code == 0  # check for system errors

            # Then read the forward barrier -> ebf
            for traj in range(batchSize):
                with open("out_{0}.txt".format(traj), "r") as fl:
                    for line in fl:
                        continue
                ebfLine = line.split()
                ebf = float(ebfLine[6])
                rates[traj, jumpInd] = np.exp(-ebf / (kB * T))
                barriers[traj, jumpInd] = ebf

                # get the jumping species and store the barrier for later use
                vInd = vacSiteInd[traj]
                vacNgb = SiteIndToNgb[vInd, jumpInd]
                jAtom = SiteIndToSpec[traj, vacNgb]

        # store all the rates
        AllJumpRates[sampleStart:sampleEnd] = rates[:, :]
        AllJumpBarriers[sampleStart:sampleEnd] = barriers[:, :]

        # Then do selection
        jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
        # store the selected jumps
        JumpSelects[sampleStart: sampleEnd] = jumpID[:]

        # store the random numbers for the first set of jumps
        if batch == 0:
            TestRates[:, :] = rates[:, :]
            TestBarriers[:, :] = barriers[:, :]
            TestRandomNums[:] = rndNums[:]

        # Then do the final exchange
        jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
        # def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):

        # save final states, displacements and times
        FinalStates[sampleStart: sampleEnd, :] = SiteIndToSpec[:, :]
        FinalVacSites[sampleStart: sampleEnd, :] = vacSiteInd[:, :]
        SpecDisps[sampleStart:sampleEnd, :, :] = X_traj[:, :, :]
        tarr[sampleStart:sampleEnd] = time_step[:]
        if step == 0:
            with open("BatchTiming.txt", "a") as fl:
                fl.write(
                    "batch {0} of {1} completed in : {2} seconds\n".format(batch + 1, int(np.ceil(Ntraj/chunk)), time.time() - start))

    with open("StepTiming.txt", "a") as fl:
        fl.write("Time per step up to {0} of {1} steps : {2} seconds\n".format(step + 1, Nsteps, (time.time() - start)/(step + 1)))

    # Next, save all the arrays in an hdf5 file for the current step.
    # For the first 10 steps, store test random numbers.
    if startStep + step <= 10:
        with h5py.File("data_{0}_{1}_{2}.h5".format(T, startStep + step + 1, SampleStart), "w") as fl:
            fl.create_dataset("FinalStates", data=FinalStates)
            fl.create_dataset("SpecDisps", data=SpecDisps)
            fl.create_dataset("times", data=tarr)
            fl.create_dataset("AllJumpRates", data=AllJumpRates)
            fl.create_dataset("AllJumpBarriers", data=AllJumpBarriers)
            fl.create_dataset("JumpSelects", data=JumpSelects)
            fl.create_dataset("TestRandNums", data=TestRandomNums)
            fl.create_dataset("TestRates", data=TestRates)
            fl.create_dataset("TestBarriers", data=TestBarriers)
    else:
        with h5py.File("data_{0}_{1}_{2}.h5".format(T, startStep + step + 1, SampleStart), "w") as fl:
            fl.create_dataset("FinalStates", data=FinalStates)
            fl.create_dataset("SpecDisps", data=SpecDisps)
            fl.create_dataset("times", data=tarr)
            fl.create_dataset("AllJumpRates", data=AllJumpRates)
            fl.create_dataset("AllJumpBarriers", data=AllJumpBarriers)
            fl.create_dataset("JumpSelects", data=JumpSelects)
