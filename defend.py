import numpy as np
import torch.nn as nn
import torch
import time
from tqdm import tqdm

import variables as var

# approximate projection defense
def approximate_projection(decoder_model, encodings, attack_x, num_iter=1000):

    second_losses = []
    loss_fn = nn.SmoothL1Loss(reduction ='sum')
    decoder_model.eval()

    for epoch in tqdm(range(1,num_iter+1)):
        # Makes predictions
        x_rec = decoder_model(encodings.to(var.device))
        # Computes loss
        second_loss = loss_fn(x_rec.cpu(),attack_x)
        # Computes gradients
        second_loss.backward()
        #gradient step
        encodings.data -= var.alpha*encodings.grad
        second_losses.append(second_loss.item())

    # returns rec. error and encodings of the defended points
    return encodings, second_loss

def feature_weighting(train_errors, test_errors, epsilon = 10e-4, train = True):

    if train == True:    
        weights = 1/(epsilon+np.median(train_errors,axis = 0))
    elif train == False:
        weights  = 1/(epsilon+np.median(test_errors,axis = 0))

    test_errors = np.matmul(test_errors,weights)

    return test_errors