import time
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import os

import variables as var

class Autoencoder(nn.Module):
    def __init__(self,enc_hidden,dec_hidden):
        
        super(Autoencoder,self).__init__()
    
        # encoder
        self.enc_list = []
        for i in range(1,len(enc_hidden)):
            self.enc_list.append(nn.Linear(enc_hidden[i-1],enc_hidden[i]))
            self.enc_list.append(nn.ReLU(True))
        self.enc_list.pop()
        self.enc_list = nn.ModuleList(self.enc_list)

        #decoder
        self.dec_list = []
        for i in range(1,len(dec_hidden)):
            self.dec_list.append(nn.Linear(dec_hidden[i-1],dec_hidden[i]))
            self.dec_list.append(nn.ReLU(True))
        self.dec_list.pop()
        self.dec_list = nn.ModuleList(self.dec_list)

    def forward(self,x):
        
        for f in self.enc_list:
            x = f(x)
           
        encoding = x

        for f in self.dec_list:
            x = f(x)
            
        reconstruction = x
    
        return encoding, reconstruction


class Decoder(nn.Module):
    def __init__(self,dec_hidden):
        
        super(Decoder,self).__init__()
        
        #decoder
        self.dec_list = []
        for i in range(1,len(dec_hidden)):
            self.dec_list.append(nn.Linear(dec_hidden[i-1],dec_hidden[i]))
            self.dec_list.append(nn.ReLU(True))
        self.dec_list.pop()
        self.dec_list = nn.ModuleList(self.dec_list)
        
    def forward(self,x):
                
        for f in self.dec_list:
            x = f(x)
            
        reconstruction = x
        
        return reconstruction

def train_model(dataset,net,train_loader,val_loader,save_model=True):

    optimizer = optim.Adam(net.parameters(), lr = var.lr, betas=(0.5, 0.999))
    loss_fn = nn.SmoothL1Loss(reduction = "none")

    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(1,var.n_epochs+1):
        #training
        net.train()
        train_batch_loss = []    
        for x_batch in train_loader:
            x_batch = x_batch.to(var.device)
            # Makes predictions
            _, x_rec = net(x_batch)
            # Computes loss
            loss = loss_fn(x_batch,x_rec).mean()
            # Computes gradients
            loss.backward()
            # Updates parameters
            optimizer.step()
            #zero gradient 
            optimizer.zero_grad()
            # Returns the loss
            train_losses.append(loss.item())
            train_batch_loss.append(loss.item())
    
        #validation
        with torch.no_grad():
            val_batch_loss = []
            net.eval()
            for x_batch in val_loader:   
                x_batch = x_batch.to(var.device)         
                _, x_rec = net(x_batch.to(var.device))
                val_loss = loss_fn(x_batch, x_rec).mean()
                val_losses.append(val_loss.item())
                val_batch_loss.append(val_loss.item())

        #print progress
        print("Epoch: %d, Loss %.8f, Validation Loss %.8f" % (epoch, np.mean(train_batch_loss), np.mean(val_batch_loss)))
        
        #early stopping
        if val_loss < 0.003:
            break

    end = time.time()
    print("Training time: %.8f minutes" %((end-start)/60))

    model_save_file = "saved_models/%s/" %dataset
    if not os.path.exists(os.path.dirname(model_save_file)):
        os.makedirs(os.path.dirname(model_save_file))

    torch.save(
        {'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss' : loss,
        'val_loss': val_loss
        }, "%snet.pth" %model_save_file)

    return net
    