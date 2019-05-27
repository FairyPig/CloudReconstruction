import os
import math

cloud_obj_dir = '/home/afan/Data/Cloud3D_obj_scale'
rotate_output_dir = '/home/afan/Data/rotate'
rotate_code = '/home/afan/Data/CloudRender/obj_rotate.py'
blank_blender_path = '/home/afan/Data/CloudRender/blank.blend'
if not os.path.exists(rotate_output_dir):
    os.mkdir(rotate_output_dir)
rotate_out_dir1 = '/home/afan/Data/rotate/rotate_90'
rotate_out_dir2 = '/home/afan/Data/rotate/rotate_180'
rotate_out_dir3 = '/home/afan/Data/rotate/rotate_270'

rotate_off_dir1 = '/home/afan/Data/rotate/rotate_off_90'
rotate_off_dir2 = '/home/afan/Data/rotate/rotate_off_180'
rotate_off_dir3 = '/home/afan/Data/rotate/rotate_off_270'

rotate_off_scale_dir1 = '/home/afan/Data/rotate/rotate_scale_90'
rotate_off_scale_dir2 = '/home/afan/Data/rotate/rotate_scale_180'
rotate_off_scale_dir3 = '/home/afan/Data/rotate/rotate_scale_270'

rotate_result_dir1 = '/home/afan/Data/rotate/rotate_result_90'
rotate_result_dir2 = '/home/afan/Data/rotate/rotate_result_180'
rotate_result_dir3 = '/home/afan/Data/rotate/rotate_result_270'

if not os.path.exists(rotate_out_dir1):
    os.mkdir(rotate_out_dir1)
if not os.path.exists(rotate_out_dir2):
    os.mkdir(rotate_out_dir2)
if not os.path.exists(rotate_out_dir3):
    os.mkdir(rotate_out_dir3)
    
if not os.path.exists(rotate_off_dir1):
    os.mkdir(rotate_off_dir1)
if not os.path.exists(rotate_off_dir2):
    os.mkdir(rotate_off_dir2)
if not os.path.exists(rotate_off_dir3):
    os.mkdir(rotate_off_dir3)
    
if not os.path.exists(rotate_off_scale_dir1):
    os.mkdir(rotate_off_scale_dir1)
if not os.path.exists(rotate_off_scale_dir2):
    os.mkdir(rotate_off_scale_dir2)
if not os.path.exists(rotate_off_scale_dir3):
    os.mkdir(rotate_off_scale_dir3)
    
if not os.path.exists(rotate_result_dir1):
    os.mkdir(rotate_result_dir1)
if not os.path.exists(rotate_result_dir2):
    os.mkdir(rotate_result_dir2)
if not os.path.exists(rotate_result_dir3):
    os.mkdir(rotate_result_dir3)

cloud_objs = os.listdir(cloud_obj_dir)
for model_name in cloud_objs:
    if model_name.split('.')[-1] == 'obj':
        model_path = os.path.join(cloud_obj_dir, model_name)
        out_path1 = os.path.join(rotate_out_dir1, model_name.split('.')[0] + '_90' + '.obj')
        rotate_1 = math.pi / 2
        render_cmd = '%s %s --background --python %s -- %s %s %s' % ( \
            'blender', blank_blender_path, rotate_code, model_path, out_path1, rotate_1)
        os.system(render_cmd)
        
        out_path2 = os.path.join(rotate_out_dir2, model_name.split('.')[0] + '_180' + '.obj')
        rotate_2 = math.pi
        render_cmd = '%s %s --background --python %s -- %s %s %s' % ( \
            'blender', blank_blender_path, rotate_code, model_path, out_path2, rotate_2)
        os.system(render_cmd)
        
        out_path3 = os.path.join(rotate_out_dir3, model_name.split('.')[0] + '_270' + '.obj')
        rotate_3 = math.pi * 3 / 2
        render_cmd = '%s %s --background --python %s -- %s %s %s' % ( \
            'blender', blank_blender_path, rotate_code, model_path, out_path3, rotate_3)
        os.system(render_cmd)

classList = os.listdir(rotate_out_dir1)
for model_name in classList:
    if model_name.split('.')[-1] == 'obj':
        objmodel_path = os.path.join(rotate_out_dir1, model_name)
        off_model_path = os.path.join(rotate_off_dir1, model_name[:-3] + 'off')
        python_cmd = 'meshlabserver -i ' + objmodel_path + ' -o ' + off_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))
        
classList = os.listdir(rotate_out_dir2)
for model_name in classList:
    if model_name.split('.')[-1] == 'obj':
        objmodel_path = os.path.join(rotate_out_dir2, model_name)
        off_model_path = os.path.join(rotate_off_dir2, model_name[:-3] + 'off')
        python_cmd = 'meshlabserver -i ' + objmodel_path + ' -o ' + off_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))

classList = os.listdir(rotate_out_dir3)
for model_name in classList:
    if model_name.split('.')[-1] == 'obj':
        objmodel_path = os.path.join(rotate_out_dir3, model_name)
        off_model_path = os.path.join(rotate_off_dir3, model_name[:-3] + 'off')
        python_cmd = 'meshlabserver -i ' + objmodel_path + ' -o ' + off_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))

tool_dir = "/home/afan/Data/scale_off_right.py"
scale_str = "python " + tool_dir + " " + rotate_off_dir1 + " " + rotate_off_scale_dir1 + " --height "+str(2)+" --width "+str(2)+" --depth "+str(2) 
val_s = os.system(scale_str)

scale_str = "python " + tool_dir + " " + rotate_off_dir2 + " " + rotate_off_scale_dir2 + " --height "+str(2)+" --width "+str(2)+" --depth "+str(2) 
val_s = os.system(scale_str)

scale_str = "python " + tool_dir + " " + rotate_off_dir3 + " " + rotate_off_scale_dir3 + " --height "+str(2)+" --width "+str(2)+" --depth "+str(2) 
val_s = os.system(scale_str)

classList = os.listdir(rotate_off_scale_dir1)
for model_name in classList:
    if model_name.split('.')[-1] == 'off':
        off_model_path = os.path.join(rotate_off_scale_dir1, model_name)
        obj_model_path = os.path.join(rotate_result_dir1, model_name[:-3] + 'obj')
        python_cmd = 'meshlabserver -i ' + off_model_path + ' -o ' + obj_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))
        
classList = os.listdir(rotate_off_scale_dir2)
for model_name in classList:
    if model_name.split('.')[-1] == 'off':
        off_model_path = os.path.join(rotate_off_scale_dir2, model_name)
        obj_model_path = os.path.join(rotate_result_dir2, model_name[:-3] + 'obj')
        python_cmd = 'meshlabserver -i ' + off_model_path + ' -o ' + obj_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))
        
classList = os.listdir(rotate_off_scale_dir3)
for model_name in classList:
    if model_name.split('.')[-1] == 'off':
        off_model_path = os.path.join(rotate_off_scale_dir3, model_name)
        obj_model_path = os.path.join(rotate_result_dir3, model_name[:-3] + 'obj')
        python_cmd = 'meshlabserver -i ' + off_model_path + ' -o ' + obj_model_path
        print(">> Running rendering command: \n \t %s" % (python_cmd))
        os.system('%s' % (python_cmd))