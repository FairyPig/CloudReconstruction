import bpy
import os
import math

obj_model_path = '/home/afan/0.obj'
out_path = '/home/afan/image.jpg'
m_obj_name = '0'
# y axis is grean
# x axis is red
# z axis is blue
bpy.ops.import_scene.obj(filepath = obj_model_path,split_mode='OFF')
bpy.data.objects['Cube'].hide = True
bpy.data.objects['Cube'].hide_render = True
bpy.data.objects[m_obj_name].hide = False
bpy.data.objects[m_obj_name].hide_render = False
bpy.context.scene.objects.active = bpy.data.objects[m_obj_name]  #get object
# bpy.data.objects[m_obj_name].scale = (0.02, 0.02, 0.02)
bpy.data.objects[m_obj_name].location = (-1, 1, -1)
bpy.data.objects['Camera'].location = (0, 6.0, 0)
bpy.data.objects['Camera'].rotation_euler = (math.pi/2, 0, math.pi)
#set the render engine to CYCLES
bpy.context.scene.render.engine = 'CYCLES'

#set the perlin noise
bpy.ops.object.modifier_add(type = 'DISPLACE')
bpy.ops.texture.new()
bpy.data.textures['Texture'].name = m_obj_name
bpy.data.objects[m_obj_name].modifiers['Displace'].texture = bpy.data.textures[m_obj_name]
bpy.data.textures[m_obj_name].type = 'CLOUDS'
bpy.data.textures[m_obj_name].noise_basis = 'ORIGINAL_PERLIN'
bpy.data.textures[m_obj_name].noise_scale = 0.45
#set the material and scatter info
bpy.ops.material.new()
mat = bpy.data.materials['Material']
mat.name = m_obj_name
mat.use_nodes = True
bpy.data.objects[m_obj_name].data.materials.append(mat)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.remove(nodes[1])
nodes.new('ShaderNodeVolumeScatter')
links.clear()
links.new(nodes[1].outputs[0], nodes[0].inputs[1])
nodes[1].inputs[1].default_value = 2.5
nodes[1].inputs[2].default_value = 0.95
#set the sun
bpy.context.scene.objects.active = bpy.data.objects['Lamp']
#set the type of lamp to SUN
bpy.context.object.data.type = 'SUN'
bpy.data.lamps['Lamp'].use_nodes = True
emission = bpy.data.lamps['Lamp'].node_tree.nodes[1]
#set the sun intention
emission.inputs[1].default_value = 500

#set the BG
bpy.data.worlds['World'].use_nodes = True
m_world = bpy.data.worlds['World'].node_tree
#set the BG color
m_world.nodes[1].inputs[0].default_value = (0.16, 0.31, 0.5, 1)
#set the BG intensity
m_world.nodes[1].inputs[1].default_value = 1.3

#set the Camera
bpy.data.scenes['Scene'].render.resolution_x = 500
bpy.data.scenes['Scene'].render.resolution_y = 500
bpy.context.scene.objects.active = bpy.data.objects['Camera']
bpy.context.object.data.type = 'ORTHO'
bpy.context.object.data.ortho_scale = 3

#set the sampling 
cycles = bpy.context.scene.cycles
cycles.use_square_samples = True

# Path Trace
cycles.samples = 24
cycles.preview_samples = 12

# Branched Path Trace
cycles.aa_samples = 8
cycles.preview_aa_samples = 4
cycles.diffuse_samples = 3
cycles.glossy_samples = 2
cycles.transmission_samples = 2
cycles.ao_samples = 1
cycles.mesh_light_samples = 2
cycles.subsurface_samples = 2
cycles.volume_samples = 2
bpy.data.scenes['Scene'].render.filepath = out_path
bpy.ops.render.render(write_still=True)
