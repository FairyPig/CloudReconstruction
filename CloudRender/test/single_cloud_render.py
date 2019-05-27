import bpy
import os
import math
import sys
import numpy as np

# y axis is grean
# x axis is red
# z axis is blue
def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def LoadandSetCloud(cloud_path, cloud_name):
    bpy.ops.import_scene.obj(filepath = cloud_path, split_mode='OFF')
    bpy.data.objects['Cube'].hide = True
    bpy.data.objects['Cube'].hide_render = True
    bpy.data.objects[cloud_name].hide = False
    bpy.data.objects[cloud_name].hide_render = False
    bpy.context.scene.objects.active = bpy.data.objects[cloud_name]  #get object
    #bpy.data.objects[cloud_name].scale = (0.02, 0.02, 0.02)
    bpy.data.objects[cloud_name].location = (-1, 1, -1)
    bpy.ops.object.editmode_toggle()
    bpy.ops.transform.rotate(value=0.5, axis=(0,0,1),constraint_axis=(False,False,True),constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',proportional_edit_falloff='SMOOTH',proportional_size=1,release_confirm=True)
    bpy.ops.object.editmode_toggle()
    #AFAN 2018.12.30 背面render的时候scale
    #bpy.data.objects[cloud_name].location = (-2, 2, -2)
    #bpy.data.objects[cloud_name].scale = (1.0, 1.0, 1.0)

def preSetBeforeRender(image_size_x, image_size_y):
    #set the render engine to CYCLES
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.volume_bounces = 0
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
    bpy.data.objects['Camera'].location = (0.0, 6.0, 0)
    bpy.data.objects['Camera'].rotation_euler = (math.pi/2, 0, math.pi)
    #set the Render Option
    bpy.data.scenes['Scene'].render.resolution_x = image_size_x
    bpy.data.scenes['Scene'].render.resolution_y = image_size_y
    bpy.context.scene.objects.active = bpy.data.objects['Camera']
    bpy.context.object.data.type = 'ORTHO'
    #ORTHO render scale
    bpy.context.object.data.ortho_scale = 1.7
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
    bpy.data.objects[cloud_name].modifiers['Displace'].mid_level = 0.0
    bpy.data.objects[cloud_name].modifiers['Displace'].strength = 0.0

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
    bpy.data.objects['Lamp'].location = sun_position#(0.16, 0.31, 0.5)
    x_pos = sun_position[0]
    y_pos = sun_position[1]
    z_pos = sun_position[2]
    
def setSunRotation(sun_rotation):
    bpy.data.objects['Lamp'].rotation_mode = 'QUATERNION'
    bpy.data.objects['Lamp'].rotation_quaternion[0] = sun_rotation[0]
    bpy.data.objects['Lamp'].rotation_quaternion[1] = sun_rotation[1]
    bpy.data.objects['Lamp'].rotation_quaternion[2] = sun_rotation[2]
    bpy.data.objects['Lamp'].rotation_quaternion[3] = sun_rotation[3]

def render(img_out_path):
    bpy.data.scenes['Scene'].render.filepath = img_out_path
    bpy.ops.render.render(write_still=True)


# shape_file = sys.argv[-6]
# cloud_density = float(sys.argv[-5])
# sun_azimuth_deg = int(sys.argv[-4])
# sun_elevation_deg = int(sys.argv[-3])
# sun_Intensity = int(sys.argv[-2])
# out_path = sys.argv[-1]
# obj_name = shape_file.split('/')[-1].split('.')[0]
# sun_dist = 5
# theta_deg  = 0
# lx, ly, lz = obj_centened_camera_pos(sun_dist, sun_azimuth_deg, sun_elevation_deg)
# q1 = camPosToQuaternion(lx, ly, lz)
# q2 = camRotQuaternion(lx, ly, lz, theta_deg)
# q = quaternionProduct(q2, q1)
# print("======================")
# print("light position: " + str(lx) + " " + str(ly) + " " + str(lz))
# print("light rotate: " + str(q))
# print("======================")

# preSetBeforeRender(512, 512)
# LoadandSetCloud(shape_file, obj_name)
# bindTextureNode(obj_name)
# setSunPosition((lx,ly,lz))
# setSunRotation(q)
# setCloudDensity(cloud_density, 0.95, obj_name)
# setSunIntensity(sun_Intensity)
# bpy.data.scenes["Scene"].cycles.device='GPU'
# bpy.context.scene.cycles.device = 'GPU'
# render(out_path)
# for scene in bpy.data.scenes:
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.data.scenes["Scene"].cycles.device='GPU'
#     bpy.context.scene.cycles.device = 'GPU'
#
def renderSingle():
    obj_model_path = '/home/afan/1.obj'
    out_path = '/home/afan/image_2.jpg'
    m_obj_name = '1'
    preSetBeforeRender(500, 500)
    LoadandSetCloud(obj_model_path, m_obj_name)
    bindTextureNode(m_obj_name)
    render(out_path)

if __name__ == "__main__":
    renderSingle()
