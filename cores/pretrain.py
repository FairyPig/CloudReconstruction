import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import params
from utils import make_variable, save_model


class BalanceMSE(nn.Module):
    def __init__(self):
        super(BalanceMSE, self).__init__()

    def forward(self, cloud_pre, cloud_res, cloud_gt):
        one_blance = 0.15
        cloud_gt = cloud_gt.cpu()
        cloud_pre = cloud_pre.cpu()
        cloud_res = cloud_res.cpu()
        weight = torch.where(torch.eq(cloud_gt, 1.), torch.tensor(1. - one_blance), torch.tensor(one_blance))
        loss = (cloud_res - cloud_pre) * (cloud_res - cloud_pre)
        return torch.mean(weight * loss)


def train_cloud3D(encoder, decoder, data_loader, val_data_loader):
    """Train cloud3d auto-encoder."""
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    encoder.train(True)
    decoder.train(True)

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=params.encoder3D_learning_rate)

    # criterion = nn.MSELoss()
    criterion = BalanceMSE()
    criterion_mask = nn.MSELoss()

    ####################
    # 2. train network #
    ####################
    loss_total = 0
    loss_mask_total = 0
    loss_count = 0
    for epoch in range(params.num_epochs_3d):
        # the cloud3D is a res model and cloud_gt is a 0.0/1.0 array
        for step, (cloud3D, cloud_gt, cloud_mask, cloud_mean) in enumerate(data_loader):
            # make images and labels variable
            cloud_mean = cloud_mean.float().cuda()
            cloud_input = make_variable(cloud3D)
            cloud_gt = make_variable(cloud_gt).view(-1, 64*64*64)
            labels = cloud_input.float().view(-1, 64*64*64)
            cloud_mask = cloud_mask.view(-1, 64*64).float().cpu()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds, pre_mask = decoder(encoder(cloud_input), cloud_mean)
            preds = preds.view(-1, 64*64*64)
            pre_mask = pre_mask.view(-1, 64*64).cpu()
            # loss = criterion(preds, labels)
            loss = criterion(preds, labels, cloud_gt)
            loss_mask = criterion_mask(pre_mask, cloud_mask)

            loss += loss_mask
            loss_total += loss.data
            loss_mask_total += loss_mask.data.cpu()
            loss_count += 1

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_3d == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}  loss_mask={}"
                      .format(epoch + 1,
                              params.num_epochs_3d,
                              step + 1,
                              len(data_loader),
                              loss_total / float(loss_count),
                              loss_mask_total / float(loss_count)))
                loss_total = 0
                loss_mask_total = 0
                loss_count = 0

        # eval model on test set
        if ((epoch + 1) % params.eval_step_3d == 0):
            eval_src(encoder, decoder, val_data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_3d == 0):
            save_model(encoder, "Cloud3D-auto-encoder-{}.pt".format(epoch + 1))
            save_model(
                decoder, "Cloud3D-auto-decoder-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "Cloud3D-auto-encoder-final_mask.pt")
    save_model(decoder, "Cloud3D-auto-decoder-final_mask.pt")

    return encoder, decoder

def eval_src(encoder, decoder, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    decoder.eval()

    # init loss and accuracy
    loss = 0

    # set loss function
    criterion = nn.MSELoss()

    # evaluate network
    for step, (cloud3D, cloud_gt, cloud_mask, cloud_mean) in enumerate(data_loader):
        cloud3D = make_variable(cloud3D, volatile=True)
        labels = make_variable(cloud3D, volatile=True)
        cloud_mean = cloud_mean.float().cuda()
        labels = labels.view(-1, 64*64*64)
        preds, pre_mask = decoder(encoder(cloud3D), cloud_mean)
        preds = preds.view(-1, 64*64*64)
        loss += criterion(preds, labels.float())

    loss /= len(data_loader)
    print("Avg Loss = {}".format(loss))

    encoder.train()
    decoder.train()