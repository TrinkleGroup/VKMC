-This file describes all the options that can be used in running the Init_state_MC module
with the LAMMPS package to compute energies of FCC CoNiCrFeMn supercells to perform Metropolis
Monte Carlo runs.

NOTE: the location of the LAMMPS executable must be saved in the $PATH environment
variable with the name LMPPATH.

-Minimization is done with the conjugate gradient algorithm as implemented in LAMMPS with a
force tolerance of 1e-3 by default.

-The Monte Carlo simulations consist of successive Metropolis trial steps, which correspond to
swapping the positions of two different atoms and accepting the move with a probability.

-At first, a few thermalization steps are performed (specified with option --Neqb) and
then the supercells are saved after every specified number of steps (specified by --Nsave).

- To get started:
	1. An example slurm job script ("job_script_MC.sb") to run the simulations is provided to
	give an example of how to run the simulations.
	- Typically, we run multiple Monte Carlo simulations (we call them Job1, Job2,...)
	- In each simulation, we run multiple trajectories (trajectories 1, 2, ...)
	- In each trajectory, we start from a random state and gather states from --Neqb + --Nsave to --Nsteps number of steps
	(see option descriptions below).

	2. An example notebook named "Extract_states.ipynb" is provided, to show how to extract these states saved as
	ASE supercells (in pickle files) and store them into numpy arrays for use as initial states for Kinetic Monte
	Carlo simulations.

-For more details, please see the test notebook in the same directory as the Init_state_MC module.


usage: Init_state_MC.py [-h] [-pp /path/to/potential/file]
                        [-na NATOMS [NATOMS ...]] [-ckp] [-u NUNITS]
                        [-a0 LATPAR] [-nv] [-T TEMP] [-nt NSTEPS] [-ne NEQB]
                        [-ns NSAVE] [-dmp] [-dpf DUMPFILE]

Input parameters for Metropolis Monte Carlo simulations with ASE.

optional arguments:
  -h, --help            show this help message and exit
  -pp /path/to/potential/file, --potPath /path/to/potential/file
                        Path to the LAMMPS MEAM potential. (default: None)
  -na int [int ...], --Natoms int [int ...]
                        Number of atoms of each kind of Co, Ni, Cr, Fe, Mn in
                        that order. (default: None)
  -ckp, --UseLastChkPt  Whether to restart simulations from saved checkpoints.
                        If used, this flag will restart the Monte Carlo runs
                        from the second last saved ASE supercell in the
                        checkpoint directory. (default: False)
  -u int [int ...], --Nunits int [int ...]
                        Number of unit cells in the supercell. (default: [5,
                        5, 5])
  -pr, --Prim           Whether to use primitive cell (default: False)
  -nosr, --NoSrun       Whether to use srun on not to launch lammps jobs.
                        (default: False)
  -a0 float, --LatPar float
                        Lattice parameter (default: 3.595)
  -nv, --NoVac          Whether to disable vacancy creation. (default: False)
  -T float, --Temp float
                        Temperature in Kelvin. (default: None)
  -nt int, --Nsteps int
                        Total number of Metropolis trials to run. A Metropolis
                        trial consists of swapping two sites and accepting
                        with a probability. (default: 60000)
  -ne int, --NEqb int   Number of equilibrating/thermalizing steps. (default:
                        2000)
  -ns int, --Nsave int  Interval of steps after equilibration after which to
                        collect a state as a sample. (default: 200)
  -ftol float, --ForceTol float
                        Force tolerance to stop CG minimization of energies.
                        (default: 0.001)
  -etol float, --EnTol float
                        Relative energy change tolerance to stop CG
                        minimization of energies. (default: 0.0)
  -dmp, --DumpArguments
                        Whether to dump all the parsed arguments into a text
                        file. (default: False)
  -dpf string, --DumpFile string
                        The file in the run directory where all the args will
                        be dumped. (default: ArgFiles)

