import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import params
from utils import make_variable, save_model

def maskLoss(cloud_pre, cloud_mask):
    pass

def train_syImage(encoder_3D, encoder_sy, decoder, data_loader, val_data_loader):
    """Train cloud3d auto-encoder."""
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    encoder_sy.train()
    decoder.eval()
    encoder_3D.eval()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder_sy.parameters()), lr=params.encoder_sy_learning_rate)

    criterion = nn.MSELoss()
    criterion_3D = nn.MSELoss()
    criterion_mask = nn.MSELoss()

    ####################
    # 2. train network #
    ####################
    loss_total = 0
    loss_count = 0
    for epoch in range(params.num_epochs_sy):
        for step, (image, cloud3D, img_seg, cloud_mean) in enumerate(data_loader):
            # set the train stage
            encoder_sy.train()

            # make images and labels variable
            cloud_3d = make_variable(cloud3D)
            image = make_variable(image)
            cloud_mean = cloud_mean.float().cuda()
            img_seg = img_seg.float().cuda()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            image_encoder = encoder_sy(image)
            cloud_3d_encoder = encoder_3D(cloud_3d)
            preds_3d, preds_mask = decoder(image_encoder, cloud_mean)
            preds_3d = preds_3d.view(-1, 64 * 64 * 64)
            labels = cloud_3d.float().view(-1, 64 * 64 * 64)
            loss_encoder = criterion(image_encoder, cloud_3d_encoder)
            loss_mask = criterion_mask(preds_mask, img_seg)
            loss_recon = criterion_3D(preds_3d, labels)

            rate_encoder = 0.5
            rate_mask = 0.3
            rate_recon = 0.2
            loss = rate_recon * loss_recon + rate_encoder * loss_encoder + rate_mask * loss_mask
            loss_total += loss.data
            loss_count += 1

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_sy == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss_encoder={} recon_loss={} mask_loss={}"
                      .format(epoch + 1,
                              params.num_epochs_sy,
                              step + 1,
                              len(data_loader),
                              loss_total / float(loss_count),
                              loss_recon.data,
                              loss_mask.data))
                loss_total = 0
                loss_count = 0

        # eval model on test set
        if ((epoch + 1) % params.eval_step_sy == 0):
            eval_src(encoder_sy, decoder, val_data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_sy == 0):
            save_model(encoder_sy, "syCloudImage-encoder-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder_sy, "syCloudImage-encoder-final_mask.pt")
    return encoder_sy, decoder


def eval_src(encoder_sy, decoder, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder_sy.eval()
    decoder.eval()

    # setup mean cloud_3d
    cloud_mean = np.load(params.src_3D_cloudMean).astype(np.float)
    cloud_mean = cloud_mean[np.newaxis, :]
    cloud_mean = torch.from_numpy(np.repeat(cloud_mean, params.batch_size, axis=0))

    # init loss and accuracy
    loss = 0

    # set loss function
    criterion = nn.MSELoss()

    # evaluate network
    for step, (image, cloud3D, img_seg, cloud_mean) in enumerate(data_loader):
        image = make_variable(image, volatile=True)
        cloud_mean = cloud_mean.float().cuda()

        cloud3D = cloud3D.view(-1, 64*64*64).cpu()
        preds, preds_mask = decoder(encoder_sy(image), cloud_mean)
        preds = preds.view(-1, 64*64*64).cpu().float()

        loss += criterion(preds, cloud3D.float()).data

    loss /= len(data_loader)
    print("Avg Loss = {}".format(loss))
