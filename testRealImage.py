from cores.pretrain import *
from utils import *
from cores.cloud3D import *
from utils import get_data_loader, init_model, init_random_seed
import torch
import numpy as np
import cv2
import os

if __name__ == '__main__':
    torch.cuda.set_device(2)
    realImage_train_data_loader = get_data_loader(name='', stage='realImage')
    cloud3D_mean = np.load(params.src_3D_cloudMean)
    save_dir = './realTest_plot/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load models
    print("begin load models")
    cloud_image_encoder = init_model(net=CloudImageEncoderResNet(),
                                      restore=params.real_Imageencoder_restore)
    cloud3D_auto_decoder = init_model(net=Cloud3DDecoder(),
                                      restore=params.src_3Ddecoder_restore)

    img_count = 0
    show_num = 1
    for step, (image, image_mask, image_sy, syimage_seg, cloud_3d, cloud_mean) in enumerate(realImage_train_data_loader):
        image = make_variable(image, volatile=True)
        cloud_mean = cloud_mean.float().cuda()
        preds, preds_mask = cloud3D_auto_decoder(cloud_image_encoder(image), cloud_mean)
        preds = preds.cpu().data.numpy()
        preds_mask = preds_mask.cpu().data.numpy()
        preds_mask[preds_mask > 1] = 1
        preds_mask[preds_mask < 0] = 0

        for i in range(preds.shape[0]):
            cloud_shape = preds[i, 0, :] + cloud3D_mean
            img_path = os.path.join(save_dir, 'image_' + str(img_count) + '_orj.png')
            mesh_path = os.path.join(save_dir, 'image_' + str(img_count) + '.off')
            mask_path = os.path.join(save_dir, 'image_' + str(img_count) + '_mask.png')
            gt_path = os.path.join(save_dir, 'image_' + str(img_count) + '_gt.png')
            img_count += 1
            plot_3d(cloud_shape, threshold=0.5, fig_save_path=img_path, mesh_save_path=mesh_path)
            cv2.imwrite(mask_path, np.array(preds_mask[i, 0, :] * 255).astype(np.uint8))
            cv2.imwrite(gt_path, np.array(image_mask[i, :] * 255).astype(np.uint8))
            if img_count >= show_num:
                break

        if img_count >= show_num:
            break