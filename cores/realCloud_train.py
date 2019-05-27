import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import params
from utils import make_variable, save_model

def train_realImage(encoder_real, encoder_sy, decoder, discri_encoder, discri_3d, data_loader):
    """Train cloud3d auto-encoder."""
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    encoder_real.train()
    discri_encoder.train()
    discri_3d.train()
    encoder_sy.eval()
    decoder.eval()


    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder_real.parameters()), lr=params.encoder_real_learning_rate)
    optimizer_encoder = optim.Adam(
        list(discri_encoder.parameters()), lr=params.discrimator_encoder_learning_rate)
    optimizer_3d = optim.Adam(
        list(discri_3d.parameters()), lr=params.discrimator_3d_learning_rate)

    criterion_encoder = nn.BCELoss()
    criterion_3d = nn.BCELoss()
    criterion_mask = nn.MSELoss()
    ####################
    # 2. train network #
    ####################
    loss_count = 0
    for epoch in range(params.num_epochs_sy):
        count = 0
        # image is the real cloud, mask is corrsponding real cloud mask, image_sy and cloud_3d is random selected.
        for step, (image, image_mask, image_sy, syimage_seg, cloud_3d, cloud_mean) in enumerate(data_loader):

            cloud_mean = cloud_mean.float().cuda()

            ###########################
            # 2.1 train discriminator #
            ###########################
            # make images and labels variable
            image = make_variable(image)
            image_sy = make_variable(image_sy)
            image_mask = make_variable(image_mask)

            # 这个暂时没用上？
            cloud_3d = make_variable(cloud_3d)

            if count % 5 == 0:
                ###########################
                # 2.1 train encoder-dis   #
                ###########################
                # zero gradients for optimizer
                optimizer_encoder.zero_grad()

                # compute loss for critic
                sy_encoder = encoder_sy(image_sy)
                real_encoder = encoder_real(image)
                feat_concat = torch.cat((sy_encoder, real_encoder), 0)
                feat_concat = torch.squeeze(feat_concat)

                # predict on discriminator
                pred_concat = discri_encoder(feat_concat)

                # prepare real and fake label
                label_sy = make_variable(torch.ones(sy_encoder.size(0)).float())
                label_real = make_variable(torch.zeros(real_encoder.size(0)).float())
                label_concat = torch.cat((label_sy, label_real), 0)

                # compute loss for critic
                loss_encoder_critic = criterion_encoder(pred_concat, label_concat)
                loss_encoder_critic.backward()

                # optimize critic
                optimizer_encoder.step()

                acc_encoder = (torch.squeeze(torch.round(pred_concat)) == label_concat).float().mean()
                print("trainEncoder: acc_encoder = {}".format(acc_encoder))

            if count % 50 == 0:
                ###########################
                # 2.1 train cloud3d-dis   #
                ###########################
                # zero gradients for optimizer
                optimizer_3d.zero_grad()

                # compute loss for critic
                sy_cloud3d, sy_premask = decoder(encoder_sy(image_sy), cloud_mean)
                real_cloud3d, real_premask = decoder(encoder_real(image), cloud_mean)
                cloud3d_concat = torch.cat((sy_cloud3d, real_cloud3d), 0)

                # predict on discriminator
                pred3d_concat = discri_3d(cloud3d_concat)

                # prepare real and fake label
                label_sy = make_variable(torch.ones(sy_cloud3d.size(0)).float())
                label_real = make_variable(torch.zeros(real_cloud3d.size(0)).float())
                label_concat = torch.cat((label_sy, label_real), 0)

                # compute loss for critic
                loss_3d_critic = criterion_3d(pred3d_concat, label_concat)
                loss_3d_critic.backward()

                # optimize critic
                optimizer_3d.step()

                pred3d_concat = torch.squeeze(torch.round(pred3d_concat))
                acc_3d = (pred3d_concat == label_concat).float().mean()
                print("train3D: acc_3d = {}".format(acc_3d))

            # optimize step
            count += 1

            ############################
            # 2.2 train target encoder #
            ############################
            # zero gradients for optimizer
            optimizer_encoder.zero_grad()
            optimizer_3d.zero_grad()
            optimizer.zero_grad()

            # extract and target features
            real_encoder = encoder_real(image)
            preds_3d, preds_mask = decoder(real_encoder, cloud_mean)

            # predict on discriminator
            pred_tgt = discri_encoder(torch.squeeze(real_encoder))
            pred_tgt_3d = torch.squeeze(discri_3d(preds_3d))

            # prepare fake labels
            label_tgt = make_variable(torch.ones(real_encoder.size(0)).float())
            label_tgt_3d = make_variable(torch.ones(preds_3d.size(0)).float())

            # compute loss for mask
            loss_mask = criterion_mask(preds_mask, image_mask)
            # compute loss for target encoder
            loss_encoder = criterion_encoder(pred_tgt, label_tgt)
            loss_3d = criterion_3d(pred_tgt_3d, label_tgt_3d)

            # rate for three part of loss
            rate_mask = 1
            rate_encoder = 0
            rate_3d = 0
            loss_total = rate_mask * loss_mask + rate_encoder * loss_encoder + rate_3d * loss_3d
            loss_total.backward()

            # optimize target encoder
            optimizer_encoder.step()

            # print step info
            if ((step + 1) % params.log_step_real == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss_mask={} loss_encoder={} loss_3d={} loss_total={}"
                      .format(epoch + 1,
                              params.num_epochs_3d,
                              step + 1,
                              len(data_loader),
                              loss_mask.data,
                              loss_encoder.data,
                              loss_3d.data,
                              loss_total.data))

        # test model
        if ((epoch + 1) % params.eval_step_real == 0):
            evalReal(encoder_real, decoder, data_loader)
            encoder_real.train()
            decoder.eval()

        # save model parameters
        if ((epoch + 1) % params.save_step_real == 0):
            save_model(encoder_real, "RealCloudImage-encoder-{}.pt".format(epoch + 1))
            save_model(discri_encoder, "EncoderDiscriminator-encoder-{}.pt".format(epoch + 1))
            save_model(discri_3d, "Cloud3dDiscriminator-encoder-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder_real, "RealCloudImage-encoder-final.pt")
    save_model(discri_encoder, "EncoderDiscriminator-encoder-final.pt")
    save_model(discri_3d, "Cloud3dDiscriminator-encoder-final.pt")
    return encoder_real


def evalReal(encoder_real, decoder, data_loader):
    encoder_real.eval()
    decoder.eval()

    # init loss and accuracy
    mask_loss = 0
    count = 0

    # set loss function
    criterion = nn.MSELoss()
    for step, (image, image_mask, image_sy, syimage_seg, cloud_3d, cloud_mean) in enumerate(data_loader):

        # make images and labels variable
        image = make_variable(image)
        cloud_mean = cloud_mean.float().cuda()
        image_mask = make_variable(image_mask)

        real_encoder = encoder_real(image)
        pred, pred_mask = decoder(real_encoder, cloud_mean)
        mask_loss += criterion(pred_mask, image_mask).data
        count += 1
    print('Eval : mask_loss = {}'.format(mask_loss/float(count)))