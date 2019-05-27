import torch
from PIL import Image
import cv2
import torch.utils.data
import datasets.binvox_rw as binvox_rw
import csv
import numpy as np
import os
import params
from datasets.utils import *
import random

class Cloud3D_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, labelPath):
        self.aug = True
        with open(labelPath, 'r', encoding='utf-8-sig') as f:
            csvData = list(csv.reader(f))
        self.csvData = csvData
        self.dataroot = dataroot
        self.mean_cloud = np.load(params.src_3D_cloudMean).astype(np.float)

    def __len__(self):
        return len(self.csvData)

    def __getitem__(self, id):
        line = self.csvData[id]
        npy_path = os.path.join(self.dataroot, line[0])
        # with open(binvox_path, 'rb') as f:
        #     m = binvox_rw.read_as_3d_array(f)
        m = np.load(npy_path).astype(np.float)
        if self.aug:
            m = rotateData(m, 360)
            m = flipData_3D(m)
            m = scaleCenter(m)
        cloud_mask = np.expand_dims(projectPre(m), axis=0)
        m_shape = m[np.newaxis, :]
        m_res = m_shape - self.mean_cloud
        return np.array(m_res, dtype=np.float32), np.array(m_shape, dtype=np.float32), np.array(cloud_mask, dtype=np.float), np.array(np.expand_dims(self.mean_cloud.copy(), 0), dtype=np.float)