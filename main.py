#imports
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time
import os
import torch.optim as optim
import torch.nn as nn
import argparse
from sklearn.metrics import roc_auc_score
import utils
import variables as var
import model
import attack
import defend

def main(args):

    dataset = args.dataset
    attack_type = args.attack_type
    train_model = args.train_model

    train_loader, val_loader, test_x, test_y = utils.import_data(dataset)

    # MODEL TRAINING
    enc_hidden, dec_hidden = var.get_model_size(dataset)
    net = model.Autoencoder(enc_hidden, dec_hidden).to(var.device)
    if train_model == True:
        net = model.train_model(dataset,net,train_loader,val_loader)
    else:
        model_save_file = "saved_models/%s/net.pth" %dataset
        checkpoints = torch.load(model_save_file)
        net.load_state_dict(checkpoints['model_state_dict'])

    loss_fn = nn.SmoothL1Loss(reduction='none')
    
    #original score
    with torch.no_grad():
        _, test_rec = net(test_x.to(var.device))
        test_losses = loss_fn(test_x, test_rec.cpu()).mean(1)
        original_auc = roc_auc_score(test_y, test_losses.cpu())
        print('Original AUC Score: %.4f' %original_auc)

    # ATTACKS
    print('Attacking model...')

    att_x = attack.adv_attack(attack_type, net, test_x, test_y)

    # attacked score
    with torch.no_grad():
        _, att_rec = net(att_x.to(var.device))
        test_featurewise_losses = loss_fn(att_x, att_rec.cpu())
        test_losses = test_featurewise_losses.mean(1)
        attacked_auc = roc_auc_score(test_y, test_losses.cpu())
        print('Attacked AUC Score: %.4f' %attacked_auc)

    # DEFENSES
    print('Defending model...')

    decoder_net = model.Decoder(dec_hidden).to(var.device)
    checkpoints = torch.load("saved_models/%s/net.pth" %dataset)
    decoder_net.load_state_dict(checkpoints['model_state_dict'], strict = False)
    
    with torch.no_grad():
        att_enc, att_rec = net(att_x.to(var.device))
        att_enc.requires_grad = True

    # approximate projection
    att_enc, test_losses = defend.approximate_projection(decoder_net, att_enc, att_x)

    # feature weighting 
    train_losses = []
    with torch.no_grad():
        for x_batch in train_loader:
            _, x_rec = net(x_batch.to(var.device))
            loss = loss_fn(x_batch,x_rec.cpu())
            train_losses.append(loss)
    train_losses = torch.cat(train_losses)
    test_losses = defend.feature_weighting(train_losses, test_featurewise_losses, epsilon = var.epsilon, train = True)

    # defended score
    with torch.no_grad():
        defended_auc = roc_auc_score(test_y, test_losses.cpu())
        print('Defended AUC Score: %.4f' %defended_auc)

if __name__ == "__main__":
    
    torch.manual_seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'WADI')
    parser.add_argument("--attack_type", type = str, default = 'L2', help = 'Type of adversarial attack')
    parser.add_argument("--train_model", action="store_true", help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    assert args.dataset in ['WADI', 'SWaT']
    assert args.attack_type in ['L0','L2','Random']

    main(args)