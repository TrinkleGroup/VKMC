import unittest
import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from tqdm import tqdm
from KMC_funcs import write_init_states, write_final_states, updateStates, getJumpSelects
from NEB_steps_multiTraj import CreateLammpsData
import os
import subprocess
import h5py

if not os.path.isdir(os.getcwd() + "/test_KMC_funcs_files"):
    os.mkdir(os.getcwd() + "/test_KMC_funcs_files")

class test_KMC_funcs(unittest.TestCase):
    def setUp(self):
        self.N_units = 8
        # Create an FCC primitive unit cell
        self.a = 3.59
        a = self.a
        self.prim = True
        self.top = 12 # No. of lines in the lammps coords file until the coordiantes begin

        self.fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=self.prim)

        # Form a supercell with a vacancy at the centre
        self.superlatt = np.identity(3) * self.N_units
        self.superFCCASE = make_supercell(self.fcc, self.superlatt)
        Nsites = len(self.superFCCASE.get_positions())
        self.Nsites = Nsites

        # Write the lammps box
        write_lammps_data("lammpsBox.txt", self.superFCCASE)
        self.Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")
        self.assertEqual(len(self.Sup_lammps_unrelax_coords), Nsites)

        # Store the lammps-basis coordinate of each site
        self.SiteIndToCartPos = np.zeros((Nsites, 3))
        for i in range(Nsites):
            self.SiteIndToCartPos[i, :] = self.Sup_lammps_unrelax_coords[i].position[:]

        sup_coords_script = CreateLammpsData(self.N_units, self.a, prim=self.prim)
        assert np.array_equal(self.SiteIndToCartPos, sup_coords_script)

        # give some random occupancies to 2 random trajectories
        self.NtrajTest = 5
        self.SiteIndToSpec = np.random.randint(1, 6, (self.NtrajTest, self.SiteIndToCartPos.shape[0]))
        self.vacSiteInd = np.random.randint(0, Nsites, self.NtrajTest)
        # self.vacSiteInd = np.zeros(self.NtrajTest, dtype=int)

        for traj in range(self.NtrajTest):
            self.SiteIndToSpec[traj, self.vacSiteInd[traj]] = 0

        np.save("TestOccs.npy", self.SiteIndToSpec)

        with h5py.File("../../CrysDat_FCC/CrystData.h5", "r") as fl:
            self.dxList = np.array(fl["dxList_1nn"])
            self.NNsites = np.array(fl["NNsiteList_sitewise"])[1:, :].T


    def test_write_init_states(self):

        # First, write the output
        with open("lammpsBox.txt", "r") as fl:
            Initlines = fl.readlines()

        siteCount = int(Initlines[2].split()[0])
        InitialSpecCount = int(Initlines[3].split()[0])

        self.assertEqual(siteCount, self.Nsites)
        self.assertEqual(InitialSpecCount, 1)  # should have only the Nickel case

        specs = np.unique(self.SiteIndToSpec[0])
        Nspecs = len(specs)

        Initlines[2] = "{} \t atoms\n".format(self.Nsites - 1)
        Initlines[3] = "{} atom types\n".format(Nspecs - 1)

        # Write out the initial state
        write_init_states(self.SiteIndToSpec, self.SiteIndToCartPos, self.vacSiteInd, Initlines[:self.top])#Initlines[:12])

        # Move the files to the test folder (to prevent cluttering of source directory)
        cmd = "mv initial_* test_KMC_funcs_files"
        c = subprocess.Popen(cmd, shell=True)
        rt_code = c.wait()
        assert rt_code == 0

        # Now read in the initial states and spot the coordinates
        for traj in range(self.NtrajTest):
            print("testing traj: {}".format(traj), flush=True)
            vacInd = self.vacSiteInd[traj]
            # read the input file
            with open("test_KMC_funcs_files/initial_{}.data".format(traj), "r") as fl:
                fileLines = fl.readlines()

            self.assertEqual(fileLines[:self.top], Initlines[:self.top])
            atomLines = fileLines[self.top:]
            self.assertEqual(len(atomLines), self.Nsites - 1)
            # except the vacancy site, sites should not have been skipped
            for lineInd, writtenSite in enumerate(tqdm(atomLines, position=0, leave=True, ncols=65)):
                splitInfo = writtenSite.split()
                lammpsSiteInd = int(splitInfo[0])
                WrittenSpec = int(splitInfo[1])
                x = float(splitInfo[2])
                y = float(splitInfo[3])
                z = float(splitInfo[4])

                if lineInd >= vacInd:
                    mainSiteInd = lammpsSiteInd
                else:
                    mainSiteInd = lammpsSiteInd - 1

                self.assertEqual(WrittenSpec, self.SiteIndToSpec[traj, mainSiteInd],
                                 msg="{} {}\n{}".format(traj, lineInd + 1, writtenSite))
                self.assertEqual(x, self.SiteIndToCartPos[mainSiteInd, 0])
                self.assertEqual(y, self.SiteIndToCartPos[mainSiteInd, 1])
                self.assertEqual(z, self.SiteIndToCartPos[mainSiteInd, 2])

            with open("test_KMC_funcs_files/initial_{}.data".format(traj), "a") as fl:
                fl.write("{}".format(vacInd))

        print("Initial state writing passed.", flush=True)

    def test_write_final_states(self):

        # write the initial lines to compare later
        with open("lammpsBox.txt", "r") as fl:
            Initlines = fl.readlines()

        siteCount = int(Initlines[2].split()[0])
        InitialSpecCount = int(Initlines[3].split()[0])

        self.assertEqual(siteCount, self.Nsites)
        self.assertEqual(InitialSpecCount, 1)  # should have only the Nickel case

        specs = np.unique(self.SiteIndToSpec[0])
        Nspecs = len(specs)
        print("Nspec: ", Nspecs)
        Initlines[2] = "{} \t atoms\n".format(self.Nsites - 1)
        Initlines[3] = "{} atom types\n".format(Nspecs - 1)

        # Write out the initial state
        write_init_states(self.SiteIndToSpec, self.SiteIndToCartPos, self.vacSiteInd, Initlines[:self.top])

        # Move the files to the test folder (to prevent cluttering of source directory)
        cmd = "mv initial_* test_KMC_funcs_files"
        c = subprocess.Popen(cmd, shell=True)
        rt_code = c.wait()
        assert rt_code == 0

        with open("test_KMC_funcs_files/vacneighbors.txt", "w") as fl:
            for traj in range(self.NtrajTest):
                vacSite = self.vacSiteInd[traj]
                vacNgb = self.NNsites[vacSite, :]
                st = ["{} ".format(i) for i in vacNgb]
                st = "{}    ".format(vacSite) + "".join(st) + "\n"
                fl.write(st)

        for jInd in range(self.NNsites.shape[1]):
            print(jInd, flush=True)
            write_final_states(self.SiteIndToCartPos, self.vacSiteInd, self.NNsites, jInd)
            # Now go through each trajectory and check that the proper site has been translated
            for traj in range(self.NtrajTest):
                cmd = "cp final_{0}.data final_{0}_{1}.data".format(traj, jInd)
                c = subprocess.Popen(cmd, shell=True)
                rt_code = c.wait()
                self.assertEqual(rt_code, 0)

            cmd = "mv final_* test_KMC_funcs_files"
            c = subprocess.Popen(cmd, shell=True)
            rt_code = c.wait()
            self.assertEqual(rt_code, 0)

            for traj in tqdm(range(self.NtrajTest)):

                vacSite = self.vacSiteInd[traj]
                vacCoords = self.SiteIndToCartPos[vacSite]

                # Now read the file
                with open("test_KMC_funcs_files/final_{}.data".format(traj), "r") as fl:
                    finalLines = fl.readlines()
                line0 = finalLines[0].split()
                self.assertEqual(len(line0), 1)
                self.assertEqual(int(line0[0]), 1)

                # check that only the jumping atom was considered
                atomLines = finalLines[1:]
                self.assertEqual(len(atomLines), 1)

                # check that the correct lammps site index was recorded for it
                coords = atomLines[0]
                splitInfo = coords.split()
                lammpsSiteInd = int(splitInfo[0])

                # Get the vacancy neighbor at this site
                vacNgb = self.NNsites[vacSite, jInd]
                pos = self.SiteIndToCartPos[vacSite]
                if vacNgb < vacSite:
                    self.assertEqual(lammpsSiteInd, vacNgb + 1)
                else:
                    self.assertEqual(lammpsSiteInd, vacNgb)


                # Now check that the correct positions are recorded
                x = float(splitInfo[1])
                y = float(splitInfo[2])
                z = float(splitInfo[3])

                self.assertEqual(x, pos[0])
                self.assertEqual(y, pos[1])
                self.assertEqual(z, pos[2])


                # Check that initially, this atom was indeed the vacancy neighbor

                # Read in the initial coordinates
                with open("test_KMC_funcs_files/initial_{}.data".format(traj), "r") as fl:
                    AllLines = fl.readlines()

                InitCoords = AllLines[self.top:]
                initCoord = self.SiteIndToCartPos[vacNgb]
                spliInitCoords = InitCoords[lammpsSiteInd - 1].split() # the sites start from 1
                self.assertEqual(initCoord[0], float(spliInitCoords[2]), msg="{} {} {}".format(vacSite, vacNgb, lammpsSiteInd))
                self.assertEqual(initCoord[1], float(spliInitCoords[3]), msg="{} {} {}".format(vacSite, vacNgb, lammpsSiteInd))
                self.assertEqual(initCoord[2], float(spliInitCoords[4]), msg="{} {} {}".format(vacSite, vacNgb, lammpsSiteInd))

    def test_getJumpSelects_updateStates(self):

        # For each trajectory, we'll give some random rates
        rates = np.random.rand(self.NtrajTest, self.NNsites.shape[1])
        jumpID, ratesProb, ratesProbSum, rn, timeStep = getJumpSelects(rates)

        # Now for each trajectory, test the selection
        for traj in range(self.NtrajTest):
            self.assertTrue(np.allclose(ratesProb[traj], rates[traj] / np.sum(rates[traj])))
            self.assertTrue(np.allclose(ratesProbSum[traj], np.cumsum(rates[traj] / np.sum(rates[traj]))))
            self.assertTrue(np.math.isclose(ratesProbSum[traj, -1], 1.))

            self.assertTrue(np.math.isclose(timeStep[traj], 1./np.sum(rates[traj])))

            idx = np.searchsorted(ratesProbSum[traj], rn[traj])
            self.assertEqual(idx, jumpID[traj])

            for id in range(idx):
                self.assertTrue(ratesProbSum[traj, id] <= rn[traj])
            for id in range(idx, self.NNsites.shape[1]):
                self.assertTrue(ratesProbSum[traj, id] >= rn[traj])

        Nspec = len(np.unique(self.SiteIndToSpec[0]))
        vacSiteInd_init = self.vacSiteInd.copy()
        SiteIndToSpec_init = self.SiteIndToSpec.copy()
        jumpAtomSelectArray, X =\
            updateStates(self.NNsites, Nspec, self.SiteIndToSpec, self.vacSiteInd, jumpID, self.dxList)

        for traj in range(self.NtrajTest):
            jSelect = jumpID[traj]
            jSite = self.NNsites[vacSiteInd_init[traj], jSelect]
            vacsiteInit = vacSiteInd_init[traj]
            specJump = SiteIndToSpec_init[traj, jSite]
            self.assertEqual(jumpAtomSelectArray[traj], specJump)
            self.assertEqual(self.SiteIndToSpec[traj, jSite], 0)
            self.assertEqual(self.SiteIndToSpec[traj, vacsiteInit], specJump)
            self.assertEqual(self.vacSiteInd[traj], jSite)

            self.assertTrue(np.array_equal(X[traj, 0, :], self.dxList[jSelect]))
            self.assertTrue(np.array_equal(X[traj, specJump, :], -self.dxList[jSelect]))

            self.assertTrue(np.allclose(np.sum(X[traj, :, :], axis=0), np.zeros(3)))


class test_KMC_funcs_orthogonal(test_KMC_funcs):
    def setUp(self):
        self.N_units = 5
        with h5py.File("../../CrysDat_FCC/CrystData_ortho_{}_cube.h5".format(self.N_units), "r") as fl:
            self.dxList = np.array(fl["dxList_1nn"])
            self.NNsites = np.array(fl["NNsiteList_sitewise"])[1:, :].T

        # Create an FCC primitive unit cell
        self.a = 3.59
        a = self.a
        self.prim=False
        self.top = 11
        self.fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=self.prim)

        # Form a supercell with a vacancy at the centre
        self.superlatt = np.identity(3) * self.N_units
        self.superFCCASE = make_supercell(self.fcc, self.superlatt)
        Nsites = len(self.superFCCASE.get_positions())
        self.Nsites = Nsites

        # Write the lammps box
        write_lammps_data("lammpsBox.txt", self.superFCCASE)
        self.Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")
        self.assertEqual(len(self.Sup_lammps_unrelax_coords), Nsites)

        # Store the lammps-basis coordinate of each site
        self.SiteIndToCartPos = np.zeros((Nsites, 3))
        for i in range(Nsites):
            self.SiteIndToCartPos[i, :] = self.Sup_lammps_unrelax_coords[i].position[:]

        sup_coords_script = CreateLammpsData(self.N_units, self.a, prim=self.prim)
        assert np.array_equal(self.SiteIndToCartPos, sup_coords_script)

        # give some random occupancies to 2 random trajectories
        self.NtrajTest = 6
        self.SiteIndToSpec = np.random.randint(1, 6, (self.NtrajTest, self.SiteIndToCartPos.shape[0]))
        self.vacSiteInd = np.random.randint(0, Nsites, self.NtrajTest)
        # self.vacSiteInd = np.zeros(self.NtrajTest, dtype=int)

        for traj in range(self.NtrajTest):
            self.SiteIndToSpec[traj, self.vacSiteInd[traj]] = 0

        np.save("TestOccs.npy", self.SiteIndToSpec)

