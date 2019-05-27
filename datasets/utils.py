import numpy as np
from skimage import transform, data
import torch
import random
import cv2
import os
import csv
import pydicom

def img_preprocess(array):
    array = np.array(array,dtype=np.float32)
    array /= 255
    array = np.transpose(array, (2, 0, 1))  # c*h*w
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        array[i] -= means[i]
        array[i] /= stds[i]
    return array

def getSeg_syImg(img_path, resize=64):
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    raw_img[raw_img == 0.6313726] = 0
    raw_img[raw_img > 0] = 1
    raw_img = cv2.dilate(raw_img, np.ones((3, 3)), iterations=2)
    raw_img = cv2.erode(raw_img, np.ones((3, 3)), iterations=1)
    raw_img = cv2.resize(raw_img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    raw_img[raw_img > 0] = 1
    return np.array(raw_img).astype(np.uint8)

def getSeg(img_path):
    raw_img = cv2.imread(img_path).astype(np.float32) / 255
    temp = raw_img.copy()
    temp = temp[:, :, -1] - temp[:, :, 0]
    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

    mask = cv2.threshold(temp, 0.4, 1, cv2.THRESH_BINARY)[1]
    mask_refine = cv2.erode(mask, np.ones((3, 3)), iterations=4)
    mask_refine = cv2.dilate(mask_refine, np.ones((3, 3)), iterations=3)
    return mask_refine

def projectPre(model):
    h, w = model.shape[0:2]
    pro_Map = np.zeros([h, w])
    for i in range(model.shape[1]):
        # pro_Map += np.fliplr(model[:, i, :])
        pro_Map += model[:, i, :]
    pro_Map[pro_Map > 1] = 1
    return pro_Map

def scaleCenter(cloud_3d):
    if random.random() > 0.3:
        scale = random.uniform(0.85, 1.0)
        orj_size = cloud_3d.shape[0]
        new_size = int(orj_size * scale)
        for i in range(orj_size):
            trans = transform.resize(cloud_3d[i, :, :], (new_size, new_size))
            cloud_3d[i, :, :] = 0
            cloud_3d[i, int((orj_size - new_size) / 2):(int((orj_size - new_size) / 2) + new_size),
            int((orj_size - new_size) / 2):(int((orj_size - new_size) / 2) + new_size)] = trans
    return cloud_3d

def flipData_3D(cloud_3d):
    # 左右翻转
    if random.random() > 0.5:
        for i in range(cloud_3d.shape[1]):
            cloud_3d[:, i, :] = np.fliplr(cloud_3d[:, i, :])
    # 前后翻转
    if random.random() > 0.5:
        for i in range(cloud_3d.shape[2]):
            cloud_3d[:, :, i] = np.fliplr(cloud_3d[:, :, i])
    return cloud_3d

def resnet_preprocess_batch(array):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    for i in range(array.shape[0]):
        for j in range(3):
            array[i][j] -= means[j]
            array[i][j] /= stds[j]
    return array

def squeezeChannel(d):
    assert d.shape[2] == 1 and d.shape[3] == 1
    d = torch.squeeze(d, 1)
    d = torch.squeeze(d,1)
    return d

def scaleData(matrix,targetChannel,targetSize):
    matrix = np.transpose(matrix, (2, 0, 1))#wch
    m_ = []
    for i in range(matrix.shape[0]):#scale w c
        m_.append(cv2.resize(matrix[i], (targetSize, targetChannel), interpolation=cv2.INTER_LINEAR))
    matrix = np.array(m_)
    matrix = np.transpose(matrix, (1, 2, 0))#chw

    m_=[]
    for i in range(targetChannel):#schle wh
        m_.append(cv2.resize(matrix[i], (targetSize, targetSize), interpolation=cv2.INTER_LINEAR))
    matrix = np.array(m_)

    return matrix

def randomCrop(data,channelShift = 5,imgShift=20):
    srcShape = data.shape

    channelshift_ = random.randint(1,5)
    if random.random()>0.5:
        data = scaleData(data[channelshift_:],srcShape[0],srcShape[1])
    else:
        data = scaleData(data[:-channelshift_],srcShape[0],srcShape[1])
    if imgShift <=1:return data
    shifty = random.randint(1,imgShift)
    if random.random() > 0.5:
        data = scaleData(data[:,shifty:], srcShape[0], srcShape[1])
    else:
        data = scaleData(data[:,:-shifty], srcShape[0], srcShape[1])

    shiftx = random.randint(1,imgShift)
    if random.random() > 0.5:
        data = scaleData(data[:,:,shiftx:], srcShape[0], srcShape[1])
    else:
        data = scaleData(data[:,:,:-shiftx], srcShape[0], srcShape[1])
    return data

class LambdaModel(torch.nn.Module):
    def __init__(self,func):
        super(LambdaModel,self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def rotateData(data, angle):
    m = []
    for i in range(data.shape[0]):
        m.append(rotate(data[i], angle))
    return np.array(m)


def toCubeData(data):
    data = [scaleData(d[:,:,:].cpu().data.numpy(),40,40) for d in data]
    data = torch.unsqueeze(torch.from_numpy(np.array(data)).cuda().float(),dim=1)
    return data

def readCsv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]
        return csvData

def writeCsv(path,csvData):
    f= open(path,'w')

    f.write("id,ret\r\n")
    for k,v in csvData:
        f.write("%s,%s\r\n"%(k,str(v)))
    f.close()

