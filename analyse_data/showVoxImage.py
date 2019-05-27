import sys
sys.path.append("..")
import binvox_rw as binvox_rw
import numpy as np
import cv2
import os
import params

def test_binvox():
    cloud_3D_dir = params.src_3Dcloud_dataset
    show_num = 10
    cloud_names = os.listdir(cloud_3D_dir)
    save_dir = './cloud3D_Img'
    for i in range(show_num):
        cloud_path = os.path.join(cloud_3D_dir, cloud_names[i])
        with open(cloud_path, 'rb') as f:
            m = binvox_rw.read_as_3d_array(f)
        m_shape = np.reshape(m.data, (64, 64, 64))
        m_shape = np.transpose(m_shape, (1, 2, 0))
        # m_shape = m_shape[::-1, :, :]
        cloud_pixs = np.where(m_shape == 1)
        shift_0 = int((64 - cloud_pixs[0].max())/2)
        shift_1 = int((64 - cloud_pixs[1].max())/2)
        new_shape = np.zeros((64, 64, 64))
        new_shape[shift_0:, shift_1:, :] = m_shape[0:(64-shift_0), 0:(64-shift_1), :]
        new_shape = new_shape[::-1, :, :]
        save_cloud_dir = os.path.join(save_dir, cloud_names[i].split('.')[0])
        if not os.path.exists(save_cloud_dir):
            os.mkdir(save_cloud_dir)
        for j in range(new_shape.shape[0]):
            img_path = os.path.join(save_cloud_dir, 'cloud_'+ str(j) + '.png')
            cv2.imwrite(img_path, (new_shape[:, j, :] * 255).astype(np.uint8))

def Binvox2Npy(binvox_path, npy_path):
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    cloud_names = os.listdir(binvox_path)
    for i in range(len(cloud_names)):
        cloud_path = os.path.join(binvox_path, cloud_names[i])
        with open(cloud_path, 'rb') as f:
            m = binvox_rw.read_as_3d_array(f)
        m_shape = np.reshape(m.data, (64, 64, 64))
        m_shape = np.transpose(m_shape, (1, 2, 0))
        cloud_pixs = np.where(m_shape == 1)
        shift_0 = int((64 - cloud_pixs[0].max()) / 2)
        shift_1 = int((64 - cloud_pixs[1].max()) / 2)
        shift_2 = int((64 - cloud_pixs[2].max()) / 2)
        new_shape = np.zeros((64, 64, 64))
        if shift_2 == 0 or shift_2 == 1:
            new_shape[shift_0:, shift_1:, :] = m_shape[0:(64 - shift_0), 0:(64 - shift_1), :]
        elif shift_1 == 0:
            new_shape[shift_0:, :, shift_2:] = m_shape[0:(64 - shift_0), :, 0:(64 - shift_2)]
        else:
            new_shape[:, shift_1:, shift_2:] = m_shape[:, 0:(64 - shift_1), 0:(64 - shift_2)]
        new_shape = new_shape[::-1, :, :]

        for j in range(new_shape.shape[1]):
            new_shape[:, j, :] = np.fliplr(new_shape[:, j, :])
        save_path = os.path.join(npy_path, cloud_names[i].split('.')[0] + '.npy')
        np.save(save_path, new_shape)

def showNpy(npy_path, img_save_path):
    show_num = 1
    cloud_names = os.listdir(npy_path)
    for i in range(show_num):
        print(cloud_names[i])
        cloud_path = os.path.join(npy_path, cloud_names[i])
        cloud_shape = np.load(cloud_path)
        save_cloud_dir = os.path.join(img_save_path, cloud_names[i].split('.')[0])
        if not os.path.exists(save_cloud_dir):
            os.mkdir(save_cloud_dir)
        for j in range(cloud_shape.shape[0]):
            img_path = os.path.join(save_cloud_dir, 'cloud_' + str(j) + '.png')
            cv2.imwrite(img_path, (cloud_shape[:, j, :] * 255).astype(np.uint8))

if __name__ == '__main__':
    binvox_path = params.src_3Dcloud_dataset
    npy_path = params.src_3Dcloud_npy_dataset
    Binvox2Npy(binvox_path, npy_path)
    showNpy(npy_path, './cloud3D_Img')