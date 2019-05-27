"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable
from core.pretrain import *


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, src_classifier, val_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate)
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        count = 0
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((id_src, images_src, labels_src), (id_tgt, images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            images_src_split = torch.stack((images_src[:, 19, :, :], images_src[:, 20, :, :], images_src[:, 21, :, :]),
                                           1)
            images_tgt_split = torch.stack((images_tgt[:, 19, :, :], images_tgt[:, 20, :, :], images_tgt[:, 21, :, :]),
                                           1)
            images_src = make_variable(images_src_split)
            images_tgt = make_variable(images_tgt_split)

            # make images variable
            if count % 100 == 0:

                # zero gradients for optimizer
                optimizer_critic.zero_grad()

                # extract and concat features
                feat_src = src_encoder(images_src)
                feat_tgt = tgt_encoder(images_tgt)
                feat_concat = torch.cat((feat_src, feat_tgt), 0)

                # predict on discriminator
                pred_concat = critic(feat_concat.detach())

                # prepare real and fake label
                label_src = make_variable(torch.ones(feat_src.size(0)).long())
                label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat)
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()

                pred_cls = torch.squeeze(pred_concat.max(1)[1])
                acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################
            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())
            labels_src = make_variable(labels_src.squeeze_()).long()

            preds = src_classifier(tgt_encoder(images_src))

            # compute loss for target encoder
            loss_tgt = 1e-5 * criterion(pred_tgt, label_tgt) + criterion_2(preds, labels_src)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data,
                              loss_tgt.data,
                              acc.data))

            count += 1
        eval_src(tgt_encoder, src_classifier, val_data_loader)
        # eval_test_csv(str(epoch + 1) + "end", tgt_encoder, src_classifier, tgt_data_loader)


        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder
