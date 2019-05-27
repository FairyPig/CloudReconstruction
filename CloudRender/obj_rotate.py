import bpy
import os
import math
import sys

obj_model_path = sys.argv[-3]
out_path = sys.argv[-2]
rotate_angle = float(sys.argv[-1])
m_obj_name = obj_model_path.split('/')[-1].split('.')[0]
bpy.ops.import_scene.obj(filepath = obj_model_path,split_mode='OFF')

bpy.data.objects[m_obj_name].hide = False
bpy.data.objects[m_obj_name].hide_render = False
bpy.data.objects[m_obj_name].location = (-1, 1, -1)
bpy.data.objects[m_obj_name].rotation_euler[2] = rotate_angle
bpy.ops.export_scene.obj(filepath=out_path, use_selection=True)