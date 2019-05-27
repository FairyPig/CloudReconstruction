from cores.pretrain import *
from utils import *
from cores.cloud3D import *
from utils import get_data_loader, init_model, init_random_seed
import torch
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    add_mean = True
    torch.cuda.set_device(1)
    cloud3D_test_data_loader = get_data_loader(params.src_cloud_test, stage='3D')
    save_dir = './test_plot/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load models
    print("begin load models")
    cloud3D_auto_encoder = init_model(net=Cloud3DEncoder(),
                                      restore=params.src_3Dencoder_restore)
    cloud3D_auto_decoder = init_model(net=Cloud3DDecoder(),
                                      restore=params.src_3Ddecoder_restore)

    cloud3D_auto_encoder.eval()
    cloud3D_auto_decoder.eval()

    cloud_mean_np = np.load(params.src_3D_cloudMean).astype(np.float)
    img_count = 0
    orj_count = 0
    for step, (cloud3D, cloud_gt, cloud_mask, cloud_mean) in enumerate(cloud3D_test_data_loader):
        for i in range(cloud3D.size()[0]):
            if add_mean:
                cloud_shape = cloud3D.numpy()[i, 0, :] + cloud_mean_np
            else:
                cloud_shape = cloud3D.numpy()[i, 0, :]
            img_path = os.path.join(save_dir, 'image_orj_' + str(orj_count) + '.png')
            orj_count += 1
            plot_3d(cloud_shape, threshold=0.5, fig_save_path=img_path)

        cloud3D = make_variable(cloud3D, volatile=True)
        cloud_mean = cloud_mean.float().cuda()
        preds, pred_mask = cloud3D_auto_decoder(cloud3D_auto_encoder(cloud3D), cloud_mean)
        preds = preds.cpu().data.numpy()
        pred_mask = pred_mask.cpu().data.numpy()
        pred_mask[pred_mask > 1] = 1
        pred_mask[pred_mask < 0] = 0

        for i in range(preds.shape[0]):
            if add_mean:
                cloud_shape = preds[i, 0, :] + cloud_mean_np
            else:
                cloud_shape = preds[i, 0, :]
            img_path = os.path.join(save_dir, 'image_' + str(img_count) + '.png')
            mask_path = os.path.join(save_dir, 'image_' + str(img_count) + '_mask.png')
            mask_gt_path = os.path.join(save_dir, 'image_' + str(img_count) + '_mask_gt.png')
            img_count += 1
            plot_3d(cloud_shape, threshold=np.mean(cloud_shape), fig_save_path=img_path)
            cv2.imwrite(mask_path, np.array(pred_mask[i, 0, :] * 255).astype(np.uint8))
            cv2.imwrite(mask_gt_path, np.array(cloud_mask[i, 0, :] * 255).astype(np.uint8))