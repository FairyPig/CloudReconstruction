import torch
from datasets.cloud3DDataset import Cloud3D_Dataset
from datasets.cloudsyImageDataset import CloudsyImage_Dataset
from datasets.cloudNatureImageDataset import CloudNatureImage_Dataset
import params
import numpy as np
import os
from datasets.utils import *
def get_dataset(name, stage):
    if stage == '3D':
        if name == "train":
            # dataset and data loader
            train_dataset = Cloud3D_Dataset(params.src_3Dcloud_npy_dataset, '/home/afan/Reconstruction/Cloud_Recon/datasets/train_3D.csv')

            train_data_loader = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_size = params.batch_size,
                num_workers = 8,
                shuffle = True)

            return train_data_loader

        if name == "test":
            test_dataset = Cloud3D_Dataset(params.src_3Dcloud_npy_dataset, '/home/afan/Reconstruction/Cloud_Recon/datasets/test_3D.csv')
            test_dataset = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=params.batch_size,
                num_workers=8,
                shuffle=True)
            return test_dataset

    if stage == 'syImage':
        if name == "train":
            # dataset and data loader cloud3Dpath='', cloudrenderPath='', labelPath=''
            train_dataset = CloudsyImage_Dataset(cloud3Dpath=params.src_3Dcloud_npy_dataset,
                                                 cloudrenderPath=params.sy_cloud_dataset,
                                            labelPath='/home/afan/Reconstruction/Cloud_Recon/datasets/train_3D.csv')
            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=params.batch_size,
                num_workers=8,
                shuffle=True)
            return train_data_loader

        if name == "test":
            test_dataset = CloudsyImage_Dataset(cloud3Dpath=params.src_3Dcloud_npy_dataset,
                                                 cloudrenderPath=params.sy_cloud_dataset,
                                            labelPath='/home/afan/Reconstruction/Cloud_Recon/datasets/test_3D.csv')

            test_dataset = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=params.batch_size,
                num_workers=8,
                shuffle=True)
            return test_dataset

    if stage == 'realImage':
        train_dataset = CloudNatureImage_Dataset(cloud3Dpath=params.src_3Dcloud_npy_dataset,
                                             cloudNatureImgPath=params.real_cloud_dataset,
                                             cloudImgMaskPath=params.real_cloud_mask_dataset,
                                             cloudrenderPath=params.sy_cloud_dataset,
                                             labelPath='/home/afan/Reconstruction/Cloud_Recon/datasets/train_3D.csv')
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=params.batch_size,
            num_workers=8,
            shuffle=True)
        return train_data_loader

def make_csv():
    train_file = open('./train_3D.csv', 'w')
    for i in range(190):
        train_file.write(str(i + 1)+'.npy\n')
    train_file.close()
    test_file = open('./test_3D.csv', 'w')
    for i in range(10):
        test_file.write(str(i + 1 + 190)+'.npy\n')
    test_file.close()

def calculateMean3D():
    cloud_mean = np.zeros([64, 64, 64]).astype(np.float)
    mean = 0.0
    std = 0.0
    npy_dir = params.src_3Dcloud_npy_dataset
    npy_names = os.listdir(npy_dir)
    repeat_n = 100
    for i in range(repeat_n):
        for name in npy_names:
            npy_path = os.path.join(npy_dir, name)
            cloud_3d = np.load(npy_path).astype(np.float)
            cloud_3d = rotateData(cloud_3d, 360)
            cloud_mean += cloud_3d
    cloud_mean /= (len(npy_names) * repeat_n)
    np.save("./cloud3d_mean.npy", cloud_mean)

if __name__ == '__main__':
    cloud_mean = np.load(params.src_3D_cloudMean).astype(np.float)
    cloud_mean = cloud_mean[np.newaxis, :]
    cloud_mean = torch.from_numpy(np.repeat(cloud_mean, params.batch_size, axis=0))
    print(cloud_mean.size())