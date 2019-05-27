import numpy as np  # 插入numpy库
import PIL.Image as image  # 加载pil的包
from sklearn.cluster import KMeans


def loadData(filePath):
    f = open(filePath, 'rb')  # 以二进制读取文件
    data = []
    img = image.open(f)  # 返回图片的像素值
    m, n = img.size  # 返回图片的大小
    for i in range(m):
        for j in range(n):
            x, y, z, _ = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


imgData, row, col = loadData('/home/afan/CloudData/Cloud_render/100/cloud100_21.png')
label = KMeans(n_clusters=2).fit_predict(imgData)  # 图片聚成4类

label = label.reshape([row, col])
pic_new = image.new("L", (row, col))
for i in range(row):  # 根据所属类别给图片添加灰度
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save("result-bull-4.jpg", "JPEG")