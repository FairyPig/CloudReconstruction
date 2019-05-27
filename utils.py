import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import scipy.ndimage
import matplotlib.pyplot as plt
import params
from datasets.dataset import get_dataset
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
import pymesh
import numpy as np

def plot_3d(image, threshold=-300, fig_save_path='', mesh_save_path=''):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes_classic(p, threshold)
    if mesh_save_path != '':
        save_mesh_obj(verts, faces, mesh_save_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    fig = plt.gcf()
    # plt.show()
    fig.savefig(fig_save_path, dpi=100)


def save_mesh_obj(verts, faces, file_name):
    verts = np.array(verts, dtype=float)
    faces = np.array(faces, dtype=float)
    mesh = pymesh.form_mesh(verts, faces)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    pymesh.save_mesh(file_name, mesh)
    cmd = 'meshlabserver -i ' + file_name + ' -o ' + file_name + ' -s ' + params.post_mlx_file_path
    os.system(cmd)


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        torch.nn.init.xavier_normal_(layer.weight.data)
    elif layer_name.find('Linear') != -1:
        torch.nn.init.xavier_normal_(layer.weight.data)
        torch.nn.init.constant_(layer.bias.data, 0.0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, stage):
    """Get data loader by name."""
    return get_dataset(name, stage)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))