import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train_model(
        epochs: int,
        model: nn.Module,
        f_model: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: str,
        optimizer: Optimizer
):
    start = time.time()

    train_value = np.zeros(epochs)
    valid_value = np.zeros(epochs)
    min_valid_loss = np.inf

    # do a loop over all epochs
    for epoch in range(epochs):

        # do training
        train_loss, points = 0.0, 0
        model.train()  # set this when training. Some architecture pieces, e.g. dropout behave differently
        for SFRH_train, params_train, w_train in train_loader:  # do a loop over all elements in the training set

            # get the size of the batch
            bs = SFRH_train.shape[0]

            # move data to GPU
            params_train = params_train.to(device)
            SFRH_train = SFRH_train.to(device)

            # compute the value predicted by the network
            p = model(SFRH_train)

            # compute loss
            params_NN = p[:, :6]  # posterior mean
            errors_NN = p[:, 6:]  # posterior std
            loss1 = torch.mean((params_NN - params_train) ** 2, axis=0)
            loss2 = torch.mean(((params_NN - params_train) ** 2 - errors_NN ** 2) ** 2, axis=0)
            loss = torch.mean(torch.log(loss1) + torch.log(loss2))
            #loss = criterion(params_pred, params_train)
            loss = torch.mean(loss * w_train)

            # compute cumulative loss and number of examples used
            train_loss += loss * bs
            points += bs

            # zero gradients and do backpropagation. This is the magic!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # get the average training loss
        train_loss /= points
        train_value[epoch] = train_loss

        # do validation
        valid_loss, points = 0.0, 0
        model.eval()  # set this for validation and testing. Some architecture pieces, e.g. dropout behave differently
        for SFRH_val, params_val, w_train in valid_loader:  # do a loop over all elements in the validation set
            with torch.no_grad():  # put this for validation/testing, so save memory and be more efficient. It tells to dont save gradients.

                # get the size of the batch
                bs = SFRH_val.shape[0]

                # move data to the GPU
                params_val = params_val.to(device)
                SFRH_val = SFRH_val.to(device)

                # compute prediction by the network
                p = model(SFRH_val)

                params_NN = p[:, :6]  # posterior mean
                errors_NN = p[:, 6:]  # posterior std
                loss1 = torch.mean((params_NN - params_val) ** 2, axis=0)
                loss2 = torch.mean(((params_NN - params_val) ** 2 - errors_NN ** 2) ** 2, axis=0)
                loss = torch.mean(torch.log(loss1) + torch.log(loss2))
                loss = torch.mean(loss * w_train)

                # compute cumulative loss and number of examples used
                valid_loss += loss * bs
                points += bs

        # get the average validation loss
        valid_loss /= points
        valid_value[epoch] = valid_loss

        # save model if it has a lower validation loss
        print('%03d %.3e %.3e' % (epoch, train_loss, valid_loss), end='')
        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), f_model)
            min_valid_loss = valid_loss
            print(' (best-model)')
        else:
            print('')

    stop = time.time()
    print('Time taken (seconds):', "{:.4f}".format(stop - start))

    return model
