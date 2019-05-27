import cv2
import numpy as np
import os
from skimage import transform
import random

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

# 合成图片分割（直接根据背景颜色分割）
def getSeg_syImg(img_path):
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    raw_img[raw_img == 0.6313726] = 0
    raw_img[raw_img > 0] = 1
    raw_img = cv2.dilate(raw_img, np.ones((3, 3)), iterations=2)
    raw_img = cv2.erode(raw_img, np.ones((3, 3)), iterations=1)
    return raw_img

# 将三维模型向前投影，返回一个0/1值的图
def projectPre(model):
    h, w = model.shape[0:2]
    pro_Map = np.zeros([h, w])
    for i in range(model.shape[1]):
        # pro_Map += np.fliplr(model[:, i, :])
        pro_Map += model[:, i, :]
    pro_Map[pro_Map > 1] = 1
    return pro_Map

# 第一个合成图片数据集，比较三维模型的位置和图片中云的位置是否一致
def compareImgModel(img_path = '/home/afan/CloudData/Cloud_render/2/cloud2_1.png', model_path = '/home/afan/CloudData/Cloud3D_npy/2.npy'):
    img_dir = '/home/afan/CloudData/Cloud_render/'
    model_dir = '/home/afan/CloudData/Cloud3D_npy/'
    model_names = os.listdir(img_dir)
    for model_name in model_names:
        img_path = os.path.join(img_dir, model_name, 'cloud'+model_name+"_1.png")
        model_path = os.path.join(model_dir, model_name+'.npy')
        seg_Img = getSeg_syImg(img_path)
        model = np.load(model_path)
        pro_Map = projectPre(model)
        seg_Img = np.array(cv2.resize(seg_Img, (64, 64), interpolation=cv2.INTER_CUBIC))
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        seg_Img = seg_Img * raw_img
        diff = seg_Img - pro_Map
        diff = (diff - np.min(diff))/(np.max(diff) - np.min(diff))
        print(model_name)
        # cv2.imshow("seg", (seg_Img * 255).astype(np.uint8))
        cv2.imshow("diff", (np.concatenate((seg_Img*255, pro_Map*255, diff*255), axis=1)).astype(np.uint8))
        cv2.waitKey(0)

# 将模型旋转一定角度后再投影，注意：Cloud_render_aug中的角度用本脚本中的rotate旋转360-angle才能投影一致
def projectRot(model, angle):
    model = rotateData(model, angle)
    h, w = model.shape[0:2]
    pro_Map = np.zeros([h, w])
    for i in range(model.shape[1]):
        pro_Map += model[:, i, :]
    pro_Map[pro_Map > 1] = 1
    return pro_Map

# 第二个合成图片数据集render_aug，比较三维模型旋转后的位置和图片中云的位置是否一致
def compareImgRot():
    img_dir = '/home/afan/CloudData/Cloud_render_aug/'
    model_dir = '/home/afan/CloudData/Cloud3D_npy/'
    model_names = os.listdir(img_dir)
    for model_name in model_names:
        img_names = os.listdir(os.path.join(img_dir, model_name))
        img_path = os.path.join(img_dir, model_name, img_names[0])
        model_path = os.path.join(model_dir, model_name + '.npy')
        seg_Img = getSeg_syImg(img_path)
        model = np.load(model_path)
        angle = 360 - int(img_names[0].split('_')[-1].split('.')[0])
        print("model " + str(model_name) + " rotate angle: " + str(angle))

        pro_Map = projectRot(model, angle)
        seg_Img = np.array(cv2.resize(seg_Img, (64, 64), interpolation=cv2.INTER_CUBIC))
        # raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        # seg_Img = seg_Img * raw_img
        diff = seg_Img - pro_Map
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
        cv2.imshow("diff", (np.concatenate((seg_Img * 255, pro_Map * 255, diff * 255), axis=1)).astype(np.uint8))
        cv2.waitKey(0)

# 自然图像的分割
def getSeg(img_path):
    raw_img = cv2.imread(img_path).astype(np.float32) / 255
    temp = raw_img.copy()
    temp = temp[:, :, -1] - temp[:, :, 0]
    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

    mask = cv2.threshold(temp, 0.5, 1, cv2.THRESH_BINARY)[1]
    mask_refine = cv2.erode(mask, np.ones((3, 3)), iterations=4)
    mask_refine = cv2.dilate(mask_refine, np.ones((3, 3)), iterations=5)
    mask_refine = mask_refine[:, :, np.newaxis]
    mask_refine = np.repeat(mask_refine, 3, axis=2)
    # mask_refine *= raw_img
    # print(mask_refine.shape)
    return mask_refine


def cloudNature_Segall(img_dir='/home/afan/CloudData/cloud_nature/', save_dir='/home/afan/CloudData/cloud_nature_mask/'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img_seg = getSeg(img_path)
        img_orj = cv2.imread(img_path)
        save_path = os.path.join(save_dir, img_name.split('.')[0] + '.png')
        cv2.imwrite(save_path, (img_seg*255).astype(np.uint8))
        # cv2.imshow("img_seg", (np.concatenate((img_seg * 255, img_orj), axis=1)).astype(np.uint8))
        # cv2.waitKey(0)


def showImg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def showSeg():
    img_root = '/home/afan/CloudData/Cloud_render/'
    cloud_item = os.listdir(img_root)
    for cloud_id in cloud_item:
        cloud_names = os.listdir(os.path.join(img_root, cloud_id))
        for cloud_name in cloud_names:
            img_path = os.path.join(img_root, cloud_id, cloud_name)
            img_seg = getSeg_syImg(img_path)
            showImg(img_seg)
            print(img_seg.shape)
            print(np.max(img_seg))
            break


def scale_model(cloud_3d):
    if random.random() > 0.3:
        scale = random.uniform(0.85, 1.0)
        orj_size = cloud_3d.shape[0]
        new_size = int(orj_size * scale)
        for i in range(orj_size):
            trans = transform.resize(cloud_3d[i, :, :], (new_size, new_size))
            cloud_3d[i, :, :] = 0
            cloud_3d[i, int((orj_size - new_size)/2):(int((orj_size - new_size)/2) + new_size), int((orj_size - new_size)/2):(int((orj_size - new_size)/2) + new_size)] = trans
    return cloud_3d

# 计算三维模型数据集的样本不平衡问题
def getUnbalance(cloud_npy_dir='/home/afan/CloudData/Cloud3D_npy/'):
    cloud_names = os.listdir(cloud_npy_dir)
    one_num = 0
    zero_num = 0
    for name in cloud_names:
        npy_path = os.path.join(cloud_npy_dir, name)
        cloud_3d = np.load(npy_path).astype(np.uint8)
        one_num += np.sum(cloud_3d == 1)
        zero_num += np.sum(cloud_3d == 0)

    print("1的个数：" + str(one_num))
    print("0的个数：" + str(zero_num))
    print("1的百分比： " + str(float(one_num)/float(one_num + zero_num)))
    print("0的百分比： " + str(float(zero_num) / float(one_num + zero_num)))

if __name__ == '__main__':
    cloudNature_Segall()