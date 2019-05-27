import cv2
import numpy as np
import os
from PIL import Image
import json

def cutImg(img, img_mask, rec, save_path):
    orj_width = img.shape[1]
    orj_height = img.shape[0]
    width = 250
    height = 250
    cloud_max = max(rec[2]-rec[0], rec[3]-rec[1]) + 5
    mid_x = int((rec[2]+rec[0])/2)
    mid_y = int((rec[3]+rec[1])/2)
    max_x = min(int(mid_x + cloud_max/2), orj_width)
    max_y = min(int(mid_y + cloud_max/2), orj_height)
    min_x = max(int(mid_x - cloud_max/2), 0)
    min_y = max(int(mid_y - cloud_max/2), 0)
    cropped = img[min_y:max_y, min_x:max_x]
    cropped_mask = img_mask[min_y:max_y, min_x:max_x]
    cropped_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_CUBIC)
    cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
    cropped_mask[cropped_mask > 200] = 255
    cropped_mask[cropped_mask <= 200] = 0
    cv2.imwrite(save_path, cropped)
    save_path_mask = save_path.split('.')[0] + '_mask.png'
    cv2.imwrite(save_path_mask, cropped_mask)
    # cv2.imshow("mask", cropped_mask)
    # cv2.waitKey(0)
    cropped2 = img[rec[1]:rec[3], rec[0]:rec[2]]
    cropped2 = cv2.resize(cropped2, (width, height), interpolation=cv2.INTER_CUBIC)
    save_path2 = save_path.split('.')[0] + '_two.jpg'
    cv2.imwrite(save_path2, cropped2)


def testjson(json_path, save_dir):
    img_path = json_path.split('.')[0] + '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    size = img.shape
    with open(json_path, 'r') as load_json:
        load_dict = json.load(load_json)['objects']
        cloud_num = len(load_dict)
        for i in range(cloud_num):
            cloud_info = load_dict[i]["polygon"][0]
            pts = []
            assert len(cloud_info) % 2 == 0
            for j in range(int(len(cloud_info) / 2)):
                pts.append([cloud_info[2 * j], cloud_info[2 * j + 1]])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 2))
            right = 0
            left = size[1]
            down = 0
            up = size[0]
            for point in pts:
                x = point[0]
                y = point[1]
                right = max(right, x)
                left = min(left, x)
                down = max(down, y)
                up = min(up, y)
            # img = cv2.rectangle(img, (left, up), (right, down), (255, 0, 0), 2)
            # cv2.imshow('test', img)
            # cv2.waitKey(0)
            img_mask = np.zeros([size[0], size[1]], dtype=np.uint8)
            img_mask = cv2.polylines(img_mask, [pts], True, 255)
            img_mask = cv2.fillPoly(img_mask, [pts], 255, cv2.LINE_AA)
            # cv2.imshow("mask", img_mask)
            # cv2.waitKey(0)
            save_path = os.path.join(save_dir, img_path.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg')
            cutImg(img, img_mask, [left, up, right, down], save_path)


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

def saveImgSeg():
    cloud_pkg_dir = '/home/afan/CloudData/CloudDataset_GX/annotation-converted/'
    save_dir = '/home/afan/CloudData/CloudDataset_GX/seg_result/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pkg_names = os.listdir(cloud_pkg_dir)
    for pkg_name in pkg_names:
        file_path = os.path.join(cloud_pkg_dir, pkg_name)
        if pkg_name.split('.')[-1] == 'json' and pkg_name != 'imagelist.json':
            print(file_path)
            testjson(file_path, save_dir)


def analyseMask():
    mask_dir = '/home/afan/CloudData/cloud_nature_mask/'
    imglist = os.listdir(mask_dir)
    for img_name in imglist:
        print(img_name)
        img_path = os.path.join(mask_dir, img_name)
        img_mask = np.array(Image.open(img_path).convert('L'))
        img_mask[img_mask > 0.1] = 1
        cv2.imshow("mask", (img_mask*255).astype(np.uint8))
        cv2.waitKey(0)

def showIMG():
    img_dir = '/home/afan/CloudData/GX_cloud/'
    mask_dir = '/home/afan/CloudData/GX_cloud_mask/'
    imglist = os.listdir(img_dir)
    for img_name in imglist:
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img_seg = getSeg(img_path)
        mask_path = os.path.join(mask_dir, img_name.split('.')[0] + '_mask.png')
        mask_save_path = os.path.join(mask_dir, img_name.split('.')[0] + '.png')
        img = np.array(cv2.imread(img_path))
        mask = np.array(cv2.imread(mask_path)) / 255
        mask[mask <= 0.1] = 0
        mask[mask > 0.1] = 1
        print(img.shape)
        print(mask.shape)
        print(img)
        img = img * mask #  * img_seg
        # cv2.imshow('img', img.astype(np.uint8))
        # cv2.waitKey(0)
        cv2.imwrite(mask_save_path, (mask*255).astype(np.uint8))


if __name__ == '__main__':
    showIMG()