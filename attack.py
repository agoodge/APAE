import numpy as np
import torch.nn as nn
import torch
import time
from tqdm import tqdm
import variables as var

# random attack: add random vectors to anomaly and keep those that reduce loss fn
def random_attack(model, x, y, num_vectors = 1000):


    loss_fn = nn.SmoothL1Loss(reduction ='none')
    #only perturb anomalies
    anom_idx = np.argwhere(y==1)

    adv_attack_x = x[anom_idx].clone().detach().requires_grad_(False).squeeze(1)

    _, output = model(adv_attack_x.to(var.device))
    original_losses = loss_fn(output.cpu(), adv_attack_x).mean(axis=1)
        
    current_losses = original_losses.clone().detach()
    for _ in tqdm(np.arange(num_vectors)):
        #generate random vector
        random_vectors = (torch.randn_like(adv_attack_x)*0.01)
        # make perturbation
        perturbed_attack_x = adv_attack_x + random_vectors

        with torch.no_grad():
            # into model
            _, yhat = model(perturbed_attack_x.to(var.device))

            perturbed_losses = loss_fn(yhat.cpu(), perturbed_attack_x).mean(axis=1)
            idx = perturbed_losses < current_losses
            current_losses[idx] = perturbed_losses[idx]
            adv_attack_x[idx] = perturbed_attack_x[idx]  
    return adv_attack_x

# L2 attack with perturbation budget
def L2_attack(model,attack_x,attack_y, budget=50):

    losses = []
    #huber
    loss_fn = nn.SmoothL1Loss(reduction = "sum")
    # L2
    loss_fn = nn.MSELoss(reduction = "sum")
    adv_attack_x = attack_x.clone().detach().requires_grad_(True)
    anom_idx = np.argwhere(attack_y==1)
    change = torch.sum(torch.abs(adv_attack_x[anom_idx] - attack_x[anom_idx]))*100/torch.sum(torch.abs(attack_x[anom_idx])).item()
    
    while change <= budget:
        #into model
        _, output = model(adv_attack_x.to(var.device))
        #find loss
        loss = loss_fn(output[anom_idx].cpu(), adv_attack_x[anom_idx])
        #find gradients
        loss.backward()
        losses.append(loss.item())
        # gradient descent on data to reduce loss fn
        adv_attack_x.data -= var.alpha*adv_attack_x.grad
        #log perturbation size made
        change = torch.sum(torch.abs(adv_attack_x[anom_idx] - attack_x[anom_idx]))*100/torch.sum(torch.abs(attack_x[anom_idx])).item()
    return adv_attack_x

# L0 attack with 
def L0_attack(model,attack_x,attack_y, k=50, num_iter=300):
    
    losses = []
    loss_fn = nn.SmoothL1Loss(reduction = "sum")
    adv_attack_x = attack_x.clone().detach().requires_grad_(True)
    anom_idx = np.argwhere(attack_y==1)
    _, output = model(adv_attack_x.to(var.device))
    loss = loss_fn(output[anom_idx].cpu(),adv_attack_x[anom_idx])
    # get gradient
    loss.backward()
    losses.append(loss.item())
    # gradients of the samples  
    featurewise_grads = torch.sum(adv_attack_x.grad,dim=0)
    # the top k features with the highest grad
    top_k_features = featurewise_grads.argsort()[-k:]
    for _ in tqdm(range(num_iter)):
        _, output = model(adv_attack_x.to(var.device))
        loss = loss_fn(output[anom_idx,top_k_features].cpu(),adv_attack_x[anom_idx,top_k_features])
        loss.backward()
        losses.append(loss.item())
        adv_attack_x.data[:,top_k_features] -= var.alpha*adv_attack_x.grad[:,top_k_features]
    return adv_attack_x


def adv_attack(attack_type, model, attack_x, attack_y):
    
    if attack_type == 'Random':
        adv_attack_x = random_attack(model, attack_x, attack_y)
    elif attack_type == 'L0':
        adv_attack_x = L0_attack(model, attack_x, attack_y)
    elif attack_type == 'L2':
        adv_attack_x = L2_attack(model, attack_x, attack_y)

    return adv_attack_x 
