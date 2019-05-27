from PIL import Image
import torch.utils.data
import datasets.binvox_rw as binvox_rw
from datasets.utils import *
import params


class CloudsyImage_Dataset(torch.utils.data.Dataset):
    def __init__(self, cloud3Dpath='', cloudrenderPath='', labelPath='', renderParaPath=''):
        with open(labelPath, 'r', encoding='utf-8-sig') as f:
            csvData = list(csv.reader(f))
        newcsvData = []
        for i in range(len(csvData)):
            cloud_dir_num = csvData[i][0].split('.')[0]
            cloud_names = os.listdir(os.path.join(cloudrenderPath, cloud_dir_num))
            for cloud_name in cloud_names:
                cloud_path = cloud_dir_num + '/' + cloud_name
                line = csvData[i]
                line = line[:]
                line.append(cloud_path)
                newcsvData.append(line)
        self.aug = True
        self.csvData = newcsvData
        self.cloud3Dpath = cloud3Dpath
        self.cloudrenderPath = cloudrenderPath
        self.mean_cloud = np.load(params.src_3D_cloudMean).astype(np.float)

    def __len__(self):
        return len(self.csvData)

    def __getitem__(self, id):
        line = self.csvData[id]
        npy_path = os.path.join(self.cloud3Dpath, line[0])
        img_path = os.path.join(self.cloudrenderPath, line[1])
        img_seg = np.array(getSeg_syImg(img_path)).astype(np.float)
        img_help = np.array(getSeg_syImg(img_path, resize=224)).astype(np.uint8)
        model_rotate_angle = 360 - int(line[1].split('_')[-1].split('.')[0])

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float)[:, :, :3]
        img = img_preprocess(img)
        for i in range(img.shape[0]):
            img[i, :] = img[i, :] * img_help
        m = np.load(npy_path).astype(np.float)
        m = rotateData(m, model_rotate_angle)

        m_shape = m[np.newaxis, :]
        m_res = m_shape - self.mean_cloud
        # return img, np.array(m_res, dtype=np.float32)
        return img, np.array(m_res, dtype=np.float32), img_seg, np.array(np.expand_dims(self.mean_cloud.copy(), 0), dtype=np.float)

# if __name__ == '__main__':
#     img_path = '/home/afan/CloudData/Cloud_render/100/cloud100_1.png'
#     image = scipy.misc.imread(img_path, mode='RGB')
#     image = img_preprocess(image)
#     img = np.swapaxes(img,axis1,axis2)
#     print(np.max(img))
#     print(img.shape)