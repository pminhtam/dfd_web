import os
import argparse
from make_df.stargan.solver import Solver
from torch.backends import cudnn
from PIL import Image
import numpy as np


config = lambda: None
config.attr_path = 'attr.txt'
config.batch_size = 1
config.beta1 = 0.5
config.beta2 = 0.999
config.c2_dim = 8
config.c_dim = 5
config.celeba_crop_size = 178
config.celeba_image_dir = 'data/celeba/images'
config.d_conv_dim = 64
config.d_lr = 0.0001
config.d_repeat_num = 6
config.dataset = 'CelebA'
config.g_conv_dim = 64
config.g_lr = 0.0001
config.g_repeat_num = 6
config.image_size = 128
config.lambda_cls = 1
config.lambda_gp = 10
config.lambda_rec = 10
config.log_dir = 'stargan/logs'
config.log_step = 10
config.lr_update_step = 1000
config.mode = 'test'
config.model_save_dir = 'stargan_celeba_128/models'
config.model_save_step = 10000
config.n_critic = 5
config.num_iters = 200000
config.num_iters_decay = 100000
config.num_workers = 1
config.rafd_crop_size = 256
config.rafd_image_dir = 'data/RaFD/train'
config.result_dir = 'stargan/results'
config.resume_iters = None
config.sample_dir = 'stargan/samples'
config.sample_step = 1000
config.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
config.test_iters = 200000
config.use_tensorboard = True

cudnn.benchmark = True


solver = Solver(None, None, config)

def str2bool(v):
    return v.lower() in ('true')

def adj_image(image,image_size):
    # For fast training.
    h_ori, w_ori,_ = image.shape
    image = Image.fromarray(np.uint8(image))
    # image = image.resize((image_size, image_size), Image.ANTIALIAS)
    image = image.resize((128, 128), Image.ANTIALIAS)
    result = solver.test_one(image,image_size,w_ori,h_ori)
    image_df =np.array(result)
    # print(image_df.shape)
    return image_df
    # print(result)



if __name__ == '__main__':


    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    # Solver for training and testing StarGAN.



    # attr_path='attr.txt', batch_size=1, beta1=0.5, beta2=0.999, c2_dim=8, c_dim=5, celeba_crop_size=178,
    #           celeba_image_dir='data/celeba/images', d_conv_dim=64, d_lr=0.0001, d_repeat_num=6, dataset='CelebA',
    #           g_conv_dim=64, g_lr=0.0001, g_repeat_num=6, image_size=128, lambda_cls=1, lambda_gp=10, lambda_rec=10,
    #           log_dir='stargan/logs', log_step=10, lr_update_step=1000, mode='test',
    #           model_save_dir='stargan_celeba_128/models', model_save_step=10000, n_critic=5, num_iters=200000,
    #           num_iters_decay=100000, num_workers=1, rafd_crop_size=256, rafd_image_dir='data/RaFD/train',
    #           result_dir='stargan/results', resume_iters=None, sample_dir='stargan/samples', sample_step=1000,
    #           selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'], test_iters=200000,
    #           use_tensorboard=True
