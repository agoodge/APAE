import torch 

#set device
device ='cuda:0' if torch.cuda.is_available() else 'cpu'

#hyperparameters for AE model
n_epochs = 200
batch_size = 256
lr = 1e-3     

#sliding window length for time-series
seq_len = 10

def get_model_size(dataset):
    #redacted
    pass

alpha = 1e-2 # learning rate for attacks/defenses