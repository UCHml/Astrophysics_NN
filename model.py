from torch import nn

# model parameters
seed           = 1                         #random seed to shuffle data before splitting into training, validation and testing
f_SFRD         = 'SFRH_IllustrisTNG.npy'   #file with the SFRD data
f_params       = 'params_IllustrisTNG.txt' #file with the value of the parameters
min_valid_loss = 1e7                       #set this to a large number. Used to compute

batch_size     = 32                        #number of elements each batch contains. Hyper-parameter
lr             = 1e-4                      #value of the learning rate. Hyper-parameter
wd             = 0.0                       #value of the weight decay. Hyper-parameter
dr             = 0.3                       #dropout rate. Hyper-parameter
epochs         = 450                       #number of epochs to train the network. Hyper-parameter

f_model = 'best_model.pt'


class three_hidden_layers(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_rate):
        super(three_hidden_layers, self).__init__()

        # define the fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        # define the other layers
        self.dropout   = nn.Dropout(p=dropout_rate)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)

    # forward pass
    def forward(self, x):
        out = self.fc1(x)
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out

# get the model and move it to the GPU
model = three_hidden_layers(100, 128, 256, 128, 12, dr) #architecture (we output 12 parameters, 6 posterior means, and 6 posterior standard deviations)
model.to(device=device) #move the architecture to the GPU, if available

# compute the number of parameters in the model
network_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model = %d'%network_total_params)
    