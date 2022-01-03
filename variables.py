import torch 

#set device
device ='cuda:0' if torch.cuda.is_available() else 'cpu'

#hyperparameters for AE model
n_epochs = 10
batch_size = 256
lr = 1e-3                                                                                                                                                                                                           
seq_len = 10

#AE model size for datasets
def get_model_size(dataset):
    assert dataset in ['SWaT', 'WADI']
    if dataset == 'SWaT':
        enc_hidden = [500, 200, 100, 50]
    if dataset == 'WADI':
        enc_hidden = [1220,500, 200, 100]
    dec_hidden = enc_hidden[::-1]

    return enc_hidden, dec_hidden

alpha = 1e-2 # learning rate for attacks/defenses
epsilon = 10e-4 # epsilon for feature weighting