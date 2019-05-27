# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.27
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
syImageNum = 96
batch_size = 8
image_size = 128
image_depth = 3
manual_seed = 2019

# mlx file for fix hole, reverse normal and remesh by meshlabserver
post_mlx_file_path = '/home/afan/Reconstruction/Cloud_Recon/post_refinemesh.mlx'

# params for 3D cloud encoder
src_cloud_train = 'train'
src_cloud_test = 'test'
model_root = '/home/afan/Reconstruction/Cloud_Recon/snapshots_resnet1'

src_3D_cloudMean = '/home/afan/Reconstruction/Cloud_Recon/datasets/cloud3d_mean.npy'
src_3Dcloud_dataset = '/home/afan/CloudData/Cloud3D_vox/'
src_3Dcloud_npy_dataset = '/home/afan/CloudData/Cloud3D_npy/'
src_3Dencoder_restore = './snapshots_mask/Cloud3D-auto-encoder-final_mask.pt'
src_3Ddecoder_restore = './snapshots_mask/Cloud3D-auto-decoder-final_mask.pt'
src_model_trained = True

# params for training 3D cloud
num_epochs_3d = 150
log_step_3d = 10
eval_step_3d = 5
save_step_3d = 10
encoder3D_learning_rate = 1e-3

# params for training syImage encoder
sy_cloud_train = 'train_syImage'
sy_cloud_test = 'test_syImage'
src_syImageencoder_restore = './snapshots_resnet1/syCloudImage-encoder-100.pt'
sy_cloud_dataset = '/home/afan/CloudData/Cloud_render_aug/'
sy_Image_model_trained = True

num_epochs_sy = 500
log_step_sy = 500
eval_step_sy = 1
save_step_sy = 10
encoder_sy_learning_rate = 1e-3

# params for adapt encoder
real_cloud_dataset = '/home/afan/CloudData/cloud_real/'
real_cloud_mask_dataset = '/home/afan/CloudData/cloud_real_mask/'
real_Imageencoder_restore = './snapshots_resnet1/RealCloudImage-encoder-15.pt'
cloud_encoder_discrimination_restore = './snapshots_mask/cloud-encoder-discrimination-final.pt'
cloud3d_discrimination_restore = './snapshots_mask/cloud3d-discrimination-final.pt'
save_step_real = 5
eval_step_real = 1
log_step_real = 10
encoder_real_learning_rate = 1e-3
discrimator_encoder_learning_rate = 1e-3
discrimator_3d_learning_rate = 1e-5