import os
import sys
import argparse
RunPath = os.getcwd() + "/"
#CrysPath = "/home/sohamc2/HEA_FCC/MDMC/CrysDat_FCC/"
#DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
ModulePath = "/home/sohamc2/HEA_FCC/MDMC/Symm_Network/"

sys.path.append(ModulePath)

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import h5py
import pickle
from tqdm import tqdm
from SymmLayers import GCNet
from GCNetRun import Load_crysDats
import copy

# Function to set up parallel process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# The data partitioning function
def splitData():
    pass

# The training function
def train():
    pass

# The evaluation function
def Eval():
    pass

# The function to get the Y vectors
def getY():
    pass

# Next, the main function - this main function is the one that will be
# run on parallel instances of the code
def main(rank, world_size, args):

    # Extract parsed arguments

    # Initiate process group
    setup(rank, world_size)
    
    # Load the crystal data
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats(filter_nn, CrysDatPath) 
    
    # Load the KMC Trajectory data - we'll need rates from both state 1 and state 2
    state1List, state2List, allRates_st1, allRates_st2, dispList, escRateList = Load_Data(DataPath, f1, f2)

    # Convert to necessary tensors - portions extracted based on rank
    state1NgbTens, state2NgbTens, avDispSpecTrain, rateProbTens, escTest, dispTens =\
            splitData(rank, state1List, state2List, allRates_st1, allRates_st2, dispList, dxJumps, a0, escRateList)
   
    net_dir = "epochs_T_{0}_n{1}c{2}_NgbAvg".format(T_net, nch, nl)
    # if from scratch, create new network
    gNet = GCNet() # pass in arguments to make the GCNet
    if not from_scratch:
        # load unwrapped state dict
        state_dict = torch.load(RunPath + net_dir + "/ep_{}.pt".format(sep))
        gNet.load_state_dict(state_dict)

    # send to ranked gpu
    gNet.to(rank)

    # Wrap with DDP
    gNet = DDP(gNet, device_ids=[rank], output_device=rank)

    # Pass the partitioned data to the training function
    if mode == "train":
        # Call training function
    elif mode == "eval":
        # Call evaluation function
    elif mode == "getY":
        # Call getY function


    # Lastly, clean things up by destroying the process group
    dist.destroy_process_group()


# Add argument parser
    # arguments needed:
    #   DataPath, f1, f2 - directory of data files, file for step 1 data, file for step 2 data
    #   a0 (float), from_scratch_bool, T_data, T_net, filter_nn
    #   CrysDatPath, sep (Start epoch - int), nch (int), nl(int)
parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str, help="Path to Data files.")
parser.add_argument("-f1", "--FileStep1", metavar="data_stp1.h5", type=str, help="Data file for step 1.")
parser.add_argument("-f2", "--FileStep2", metavar="data_stp2.h5", type=str, help="Data file for step 2.")
parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")

parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")

parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, help="No. of layers of the neural network.")
parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, help="No. of representation channels in non-input layers.")
parser.add_argument("-cngb", "--ConvNgbRange", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filters.")


parser.add_argument("-scr", "--Scratch", action="store_true", help="Whether to create new network and start from scratch")

parser.add_argument("-td", "--Tdata", metavar="T", type=int, help="Temperature to read data from")
parser.add_argument("-tn", "--TNet", metavar="T", type=int, help="Temperature to use networks from\n For example one can evaluate a network trained on 1073 K data, on the 1173 K data, to see what it does.")
parser.add_argument("-sep", "--Start_epoch", metavar="Ep", type=int, help="Starting epoch (for training, this network will be read in.)")
parser.add_argument("-eep", "--End_epoch", metavar="Ep", type=int, help="Ending epoch (for training, this will be the last epoch.)")

parser.add_argument("-sp", "--SpecTrain", metavar="s1s2s3", type=str, help="species to consider, order independent (Eg, 123 or 213 etc for species 1, 2 and 3")
parser.add_argument("-vSp", "--VacSpec", metavar="SpV", type=int, default=0, help="species index of vacancy, must match dataset, default 0")

parser.add_argument("-nt", "--N_train", type=int, default=10000, help="No. of training samples.")
parser.add_argument("-i", "--Interval", type=int, default=1, help="Epoch intervals in which to save or load networks.")
parser.add_argument("-lr", "--Learning_rate", type=float, default=0.001, help="Learning rate for Adam algorithm.")
parser.add_argument("-bs", "--Batch_size", type=int, default=128, help="size of a single batch of samples.")
parser.add_argument("-wm", "--Mean_wt", type=float, default=0.02, help="Initialization mean value of weights.")
parser.add_argument("-ws", "--Std_wt", type=float, default=0.2, help="Initialization standard dev of weights.")

parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")

# Then, we need to spawm multiple processes to run the main function

if __name__ == "__main__":
    
    args = parser.parse_args()    
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    if pt.cuda.is_available():
        DeviceIDList = list(range(pt.cuda.device_count()))
    if len(DeviceIDList == 0):
        raise ValueError("No Gpu found for distributed training.")
        device = pt.device("cpu")

    # Then spawn processes - we'll do one GPU per process
    world_size = len(DeviceIDList)
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
