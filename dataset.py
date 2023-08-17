# This routine creates a dataset loader
# mode ---------> 'train', 'valid', 'test', 'all'. How to create the dataset
# f_SFRD -------> file containing the data for the SFRD
# f_params -----> file containing the value of the cosmological and astrophysical parameters
# batch_size ---> number of elements in the batch
# seed ---------> the data is randomly shuffled before being split into training, validation and testing. This set the random seed for that.
import numpy as np
import torch
from torch.utils.data import DataLoader


def create_dataset(mode, f_SFRD, f_params, batch_size, seed):

    # create the class with the dataset
    data_set = make_dataset(mode, f_SFRD, f_params, seed)

    # create the data loader
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)


# This class creates the dataset
class make_dataset():

    def __init__(self, mode, f_SFRD, f_params, seed):

        # read the data
        SFRD   = np.load(f_SFRD)      #read SFRD data
        params = np.loadtxt(f_params) #read value of the parameters

        # normalize the value of the SFRD
        SFRD[np.where(SFRD==0.0)] = 1e-12 #avoid points with SFRD=0
        SFRD = np.log10(SFRD)
        mean = np.mean(SFRD, axis=0, dtype=np.float64)
        std  = np.std(SFRD,  axis=0, dtype=np.float64)
        SFRD = (SFRD - mean)/std

        s8 = params[:, 1] * np.sqrt(params[:, 0])
        params[:, 1] = s8

        # Normalize the value of the parameters
        min_params = np.min(params, axis=0)
        max_params = np.max(params, axis=0)
        params     = (params - min_params)/(max_params - min_params)

        # get the number of simulations and number of bins in the SFRD
        simulations = SFRD.shape[0]
        bins        = SFRD.shape[1]

        # get the size and offset depending on the type of dataset
        if   mode=='train':
            size, offset = int(simulations*0.70), int(simulations*0.00)
        elif mode=='valid':
            size, offset = int(simulations*0.15), int(simulations*0.70)
        elif mode=='test':
            size, offset = int(simulations*0.15), int(simulations*0.85)
        elif mode=='all':
            size, offset = int(simulations*1.00), int(simulations*0.00)
        else:    raise Exception('Wrong name!')

        # define size, input and output arrays containing the data
        self.size   = size
        #self.input  = torch.zeros((size,bins), dtype=torch.float) #array with the SFRD
        #self.output = torch.zeros((size,6),    dtype=torch.float) #array with the parameters

        # randomly shuffle the data. Instead of 0 1 2 3...999 have a
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(simulations)
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of the mode

        # get the corresponding parameters and SFRH
        self.input  = torch.tensor(SFRD[indexes],   dtype=torch.float32)
        self.output = torch.tensor(params[indexes], dtype=torch.float32)
        self.weights = torch.pow(
            torch.subtract(
                torch.tensor([0.8]),
                torch.tensor(params[indexes, 1])
            ),
            2
        )

    # This protocol returns the size of the dataset
    def __len__(self):
        return self.size

    # This protocol returns
    def __getitem__(self, idx):
        return self.input[idx], self.output[idx], self.weights[idx]
