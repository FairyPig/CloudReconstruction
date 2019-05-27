import bpy
import os
import math

# y axis is grean
# x axis is red
# z axis is blue
def LoadandSetCloud(cloud_path, cloud_name):
    bpy.ops.import_scene.obj(filepath = cloud_path, split_mode='OFF')
    bpy.data.objects['Cube'].hide = True
    bpy.data.objects['Cube'].hide_render = True
    bpy.data.objects[cloud_name].hide = False
    bpy.data.objects[cloud_name].hide_render = False
    bpy.context.scene.objects.active = bpy.data.objects[cloud_name]  #get object
    # bpy.data.objects[cloud_name].scale = (0.02, 0.02, 0.02)
    bpy.data.objects[cloud_name].location = (-1, 1, -1)
    bpy.ops.object.editmode_toggle()
    bpy.ops.transform.rotate(value=3.14, axis=(0,0,1),constraint_axis=(False,False,True),constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',proportional_edit_falloff='SMOOTH',proportional_size=1,release_confirm=True)


def preSetBeforeRender(image_size_x, image_size_y):
    #set the render engine to CYCLES
    bpy.context.scene.render.engine = 'CYCLES'
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
    
    #set the Camera
    bpy.data.objects['Camera'].location = (0, 6.0, 0)
    bpy.data.objects['Camera'].rotation_euler = (math.pi/2, 0, math.pi)
    #set the Render Option
    bpy.data.scenes['Scene'].render.resolution_x = image_size_x
    bpy.data.scenes['Scene'].render.resolution_y = image_size_y
    bpy.context.scene.objects.active = bpy.data.objects['Camera']
    bpy.context.object.data.type = 'ORTHO'
    #ORTHO render scale
    bpy.context.object.data.ortho_scale = 3
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

def bindTextureNode(cloud_name):
    #set the perlin noise
    bpy.ops.object.modifier_add(type = 'DISPLACE')
    bpy.ops.texture.new()
    bpy.data.textures['Texture'].name = cloud_name
    bpy.data.objects[cloud_name].modifiers['Displace'].texture = bpy.data.textures[cloud_name]
    bpy.data.textures[cloud_name].type = 'CLOUDS'
    bpy.data.textures[cloud_name].noise_basis = 'ORIGINAL_PERLIN'
    bpy.data.textures[cloud_name].noise_scale = 0.45
    #set the material and scatter info
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = cloud_name
    mat.use_nodes = True
    bpy.data.objects[cloud_name].data.materials.append(mat)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.remove(nodes[1])
    nodes.new('ShaderNodeVolumeScatter')
    links.clear()
    links.new(nodes[1].outputs[0], nodes[0].inputs[1])
    nodes[1].inputs[1].default_value = 2.5
    nodes[1].inputs[2].default_value = 0.95

def setCloudDensity(density, beta, cloud_name):
    mat = bpy.data.materials[cloud_name]
    nodes = mat.node_tree.nodes
    nodes[1].inputs[1].default_value = density
    nodes[1].inputs[2].default_value = beta

def setSunIntensity(sun_intensity):
    emission = bpy.data.lamps['Lamp'].node_tree.nodes[1]
    #set the sun intention
    emission.inputs[1].default_value = 500
    
def setBackGroudColor(bg_color):
    m_world = bpy.data.worlds['World'].node_tree
    #set the BG color
    m_world.nodes[1].inputs[0].default_value = bg_color#(0.16, 0.31, 0.5, 1)

def setBackGroudIntensity(bg_intensity):
    m_world = bpy.data.worlds['World'].node_tree
    #set the BG intensity
    m_world.nodes[1].inputs[1].default_value = bg_intensity
    
def setSunPosition(sun_position):
    bpy.data.objects['Lamp'].location = sun_position
    x_pos = sun_position[0]
    y_pos = sun_position[1]
    z_pos = sun_position[2]
    rotate_x = -atan(float(y_pos)/float(z_pos))
    rotate_y = -atan(float(x_pos)/float(z_pos))
    bpy.data.objects['Lamp'].rotation_euler = (rotate_x, rotate_y, 0)
    
def render(img_out_path):
    bpy.data.scenes['Scene'].render.filepath = img_out_path
    bpy.ops.render.render(write_still=True)

def renderSingle():
    obj_model_path = '/home/afan/1.obj'
    out_path = '/home/afan/image_1.png'
    m_obj_name = '1'
    preSetBeforeRender(500, 500)
    LoadandSetCloud(obj_model_path, m_obj_name)
    bindTextureNode(m_obj_name)
    render(out_path)

if __name__ == "__main__":
    renderSingle()
