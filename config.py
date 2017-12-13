#-*- coding: utf-8 -*-
import argparse
import datetime

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#flags
flags = add_argument_group('Flags')
flags.add_argument("--dataset", type=str, default= "20171205_128_gray")
flags.add_argument("--data_dir", type=str, default="C:/samples/img_download/wheels/data2/img/")
flags.add_argument("--log_dir", type=str, default="C:/samples/img_download/wheels/data2/output/began2_07_data21_17-12-05-10-13/")
flags.add_argument("--load_dir", type=str, default="C:/samples/img_download/wheels/data2/output/began2_07_data21_17-12-05-10-13/")
flags.add_argument("--curr_time", type=str, default=datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
flags.add_argument("--checkpoint_dir", type=str, default= "checkpoint")
#flags.add_argument("--npz_itr", type=str, default= "111_")
flags.add_argument("--ckpt_nm", type=str, default= "222111_began2_model.ckpt")
flags.add_argument("--load_target", type=str, default= "G")

flags.add_argument('--gamma', type=float, default=0.7)
flags.add_argument('--lambda_k', type=float, default=0.001)
flags.add_argument('--d_lr', type=float, default=0.00008)
flags.add_argument('--g_lr', type=float, default=0.00008)

flags.add_argument('--n_conv_hidden', type=int, default=128,choices=[64, 128],help='n in the paper')
flags.add_argument("--n_epoch", type=int, default=25)
flags.add_argument("--n_z", type=int, default=64)

flags.add_argument("--n_batch", type=int, default=64)
flags.add_argument("--n_img_pix", type=int, default=128)
flags.add_argument("--n_img_out_pix", type=int, default=128)
flags.add_argument("--n_save_log_step", type=int, default=1)
flags.add_argument("--n_save_img_step", type=int, default=2)
flags.add_argument("--n_save_ckpt_epoch", type=int, default=2)
flags.add_argument("--n_buffer", type=int, default=1)

flags.add_argument("--is_gray", type=str2bool, default=True)
flags.add_argument("--is_train", type=str2bool, default=True)
flags.add_argument("--is_crop", type=str2bool, default=False)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
