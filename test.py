import os
import os.path as osp
import pathlib
import argparse
import yaml
import glob
import pickle
import torch

from viper_rl.videogpt.models import VQGAN, extract_iteration
from viper_rl.videogpt.models import AE, VideoGPT

def main(config):
    # vqgan0 = VQGAN(image_size=config.image_size, **config.ae)
    # vqgan0_params = list(vqgan0.named_parameters())

    # vqgan1 = VQGAN(image_size=config.image_size, **config.ae)
    # vqgan1_params = list(vqgan1.named_parameters())

    # m = len(list(model0.named_parameters()))


    # vqgan_files = glob.glob(f"viper_rl_data/checkpoints/dmc_vqgan/checkpoints/*.pth")
    # vqgan_checkpoint_paths = sorted(vqgan_files, key=extract_iteration)

    # vqgan0.load_state_dict(torch.load(vqgan_checkpoint_paths[1])["model_state_dict"])
    # print("load vqgan weights from {}".format(vqgan_checkpoint_paths[0]))
    # vqgan1.load_state_dict(torch.load(vqgan_checkpoint_paths[-1])["model_state_dict"])
    # print("load vqgan weights from {}".format(vqgan_checkpoint_paths[-1]))
    
    
    # for i in range(m):
    #     print(vqgan0_params[i][0])
    #     print(torch.sum(torch.abs(vqgan0_params[i][1]-vqgan1_params[i][1])))
 
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)


    # # print("----------------------")
    # # print("load vqgan weights from {}".format(checkpoint_paths[-1]))
    # model.load_state_dict(torch.load(checkpoint_paths[-1]), strict=False)
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    config.ae["ddp"] = config.ddp = False

    ae = AE(config.ae_ckpt, config.ae)

    gpt0 = VideoGPT(config, ae)
    gpt1 = VideoGPT(config, ae)

    m = len(list(gpt0.named_parameters()))

    gpt_files = glob.glob(f"viper_rl_data/checkpoints/dmc_videogpt_l16_s1/checkpoints/*.pth")
    gpt_checkpoint_paths = sorted(gpt_files, key=extract_iteration)

    
    gpt0.model.load_state_dict(torch.load(gpt_checkpoint_paths[0])['model_state_dict'])
    print("load videogpt weights from {}".format(gpt_checkpoint_paths[0]))

    gpt1.model.load_state_dict(torch.load(gpt_checkpoint_paths[-1])['model_state_dict'])
    print("load videogpt weights from {}".format(gpt_checkpoint_paths[-1]))

    gpt0_params = list(gpt0.model.named_parameters())
    gpt1_params = list(gpt1.model.named_parameters())
    for i in range(m):
        print(gpt0_params[i][0])
        print(torch.sum(torch.abs(gpt0_params[i][1]-gpt1_params[i][1])))







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    args_d = vars(args)
    args_d.update(config)
    config = args
    main(config)



# import pickle

# class_map = {'acrobot_swingup': 0, 'cartpole_balance': 1, 'cartpole_swingup': 2, 'cheetah_run': 3, 
#                 'cup_catch': 4, 'finger_spin': 5, 'finger_turn_hard': 6, 'hopper_stand': 7, 
#                 'manipulator_bring_ball': 8, 'pendulum_swingup': 9, 'pointmass_easy': 10, 
#                 'pointmass_hard': 11, 'quadruped_run': 12, 'quadruped_walk': 13, 'reacher_easy': 14, 
#                 'reacher_hard': 15, 'walker_walk': 16}

# with open('class_map.pkl', 'wb') as f:
#     pickle.dump(class_map, f)


