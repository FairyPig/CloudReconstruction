import os
import math
import numpy as np
import random
import math

cloud_obj_dir = '/home/afan/CloudData/Cloud3D_obj'
render_output_dir = '/home/afan/CloudData/Cloud_render_aug'
blank_blender_path = './blank.blend'
render_code = './single_cloud_render.py'
if not os.path.exists(render_output_dir):
    os.mkdir(render_output_dir)

render_para = []
render_txt = './render_para.txt'
render_txt_file = open(render_txt)
lines = render_txt_file.readlines()
for line in lines:
    line = line.strip()
    paras = line.split(' ')
    render_para.append(paras)
repeat_time = 5
cloud_objs = os.listdir(cloud_obj_dir)
cloud_objs = cloud_objs[100:]
for model_name in cloud_objs:
    if model_name.split('.')[-1] == 'obj':
        print(model_name)
        model_path = os.path.join(cloud_obj_dir, model_name)
        render_model_dir = os.path.join(render_output_dir, model_name.split('.')[0])
        if not os.path.exists(render_model_dir):
            os.mkdir(render_model_dir)
        num = 1
        for repeat in range(repeat_time):
            for i in range(len(render_para)):
                para = render_para[i]
                random_angel = random.randint(0, 360)
                toRad = float(random_angel) / 360.0 * 2 * math.pi
                #out_name = model_name.split('.')[0] + '_' + str(int(para[0])) + '_' + str(para[1]) + '_' + str(para[2]) + '_' + str(para[3])+ '.png'
                out_name = 'cloud' + model_name.split('.')[0] + '_' + str(num)  + '_' + str(random_angel) + '.png'
                num += 1
                out_path = os.path.join(render_model_dir, out_name)
                render_cmd = '%s %s --background --python %s -- %s %s %s %s %s %s %s' % ( \
                'blender', blank_blender_path, render_code, toRad, model_path, str(para[3]), str(para[0]), str(para[1]), str(para[2]), out_path)
                print(render_cmd)
                os.system(render_cmd)