from PIL import Image
import torch.utils.data
import datasets.binvox_rw as binvox_rw
from datasets.utils import *
import params


class CloudNatureImage_Dataset(torch.utils.data.Dataset):
    def __init__(self, cloud3Dpath='', cloudNatureImgPath='', cloudImgMaskPath='', labelPath='', cloudrenderPath=''):
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
        self.cloudrenderPath = cloudrenderPath
        self.csvData = newcsvData
        self.cloud3Dpath = cloud3Dpath
        self.cloudNatureImgPath = cloudNatureImgPath
        self.imglist = os.listdir(cloudNatureImgPath)
        self.cloud3dlist = os.listdir(cloud3Dpath)
        self.cloudImgMaskPath = cloudImgMaskPath
        self.mean_cloud = np.load(params.src_3D_cloudMean).astype(np.float)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, id):
        random_id = random.randint(0, len(self.csvData))
        line = self.csvData[random_id]
        npy_path = os.path.join(self.cloud3Dpath, line[0])
        syimg_path = os.path.join(self.cloudrenderPath, line[1])
        syimg_seg = np.array(getSeg_syImg(syimg_path)).astype(np.uint8)
        model_rotate_angle = 360 - int(line[1].split('_')[-1].split('.')[0])

        syimg = Image.open(syimg_path)
        syimg = syimg.resize((224, 224))
        syimg = np.array(syimg).astype(np.float)[:, :, :3]
        syimg = img_preprocess(syimg)
        cloud_model = np.load(npy_path).astype(np.float)
        cloud_model = rotateData(cloud_model, model_rotate_angle)
        cloud_model = cloud_model[np.newaxis, :]
        m_res = cloud_model - self.mean_cloud

        img_path = os.path.join(self.cloudNatureImgPath, self.imglist[id])
        img_mask_path = os.path.join(self.cloudImgMaskPath, self.imglist[id].split('.')[0] + '.png')

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float)[:, :, :3]
        img = img_preprocess(img)
        img_mask = Image.open(img_mask_path).convert('L')
        img_mask = img_mask.resize((224, 224))
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0.1] = 1
        img = img * img_mask
        img_mask = Image.open(img_mask_path).convert('L')
        img_mask = img_mask.resize((64, 64))
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0.1] = 1
        return img, np.array(img_mask, dtype=np.float32), syimg, syimg_seg, np.array(m_res, dtype=np.float32), \
               np.array(np.expand_dims(self.mean_cloud.copy(), 0), dtype=np.float)