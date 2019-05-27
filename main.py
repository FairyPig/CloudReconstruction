import params
from cores.pretrain import *
from cores.cloud3D import *
from cores.syImage_train import *
from cores.realCloud_train import *
from utils import get_data_loader, init_model, init_random_seed
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # init random seed
    torch.cuda.set_device(0)
    init_random_seed(params.manual_seed)

    # load dataset
    cloud3D_train_data_loader = get_data_loader(params.src_cloud_train, stage='3D')
    cloud3D_test_data_loader = get_data_loader(params.src_cloud_test, stage='3D')

    # load models
    print("begin load models")
    cloud3D_auto_encoder = init_model(net=Cloud3DEncoder(),
                             restore=params.src_3Dencoder_restore)
    print("src_encoder")
    cloud3D_auto_decoder = init_model(net=Cloud3DDecoder(),
                                restore=params.src_3Ddecoder_restore)

    # train Cloud3D auto-encoder model
    print("=== Training Cloud3D auto-encoder ===")

    if not (cloud3D_auto_encoder.restored and cloud3D_auto_decoder.restored and
            params.src_model_trained):
        cloud3D_auto_encoder, cloud3D_auto_decoder = train_cloud3D(
            cloud3D_auto_encoder, cloud3D_auto_decoder, cloud3D_train_data_loader, cloud3D_test_data_loader)


    # load image-encoder models
    cloud_syImage_encoder = init_model(net=CloudImageEncoderResNet(), restore=params.src_syImageencoder_restore)

    # load dataset
    syImage_train_data_loader = get_data_loader(params.src_cloud_train, stage='syImage')
    syImage_test_data_loader = get_data_loader(params.src_cloud_test, stage='syImage')

    # train syImage auto-encoder model
    print("=== Training syImage encoder ===")
    if not (cloud_syImage_encoder.restored and cloud3D_auto_decoder.restored and cloud3D_auto_encoder.restored and
            params.sy_Image_model_trained):
        cloud_syImage_encoder = train_syImage(
            cloud3D_auto_encoder, cloud_syImage_encoder, cloud3D_auto_decoder, syImage_train_data_loader,
            syImage_test_data_loader)


    # load real-image-encoder models
    cloud_realImage_encoder = init_model(net=CloudImageEncoderResNet(), restore=params.src_syImageencoder_restore)
    cloud_encoder_discrimination = init_model(net=CloudEncoderDiscriminator(input_dims=200, hidden_dims=100, output_dims=1),
                                              restore=params.cloud_encoder_discrimination_restore)
    cloud_3d_discrimination = init_model(net=CloudDiscriminator(), restore=params.cloud3d_discrimination_restore)
    # load dataset
    realImage_train_data_loader = get_data_loader(name='', stage='realImage')

    # domain adaption for real cloud images
    print("=== Training realImage encoder ===")
    realImage_train_data_loader = train_realImage(cloud_realImage_encoder, cloud_syImage_encoder, cloud3D_auto_decoder,
                                                  cloud_encoder_discrimination, cloud_3d_discrimination,
                                                  realImage_train_data_loader)

    print("Train End!!!")