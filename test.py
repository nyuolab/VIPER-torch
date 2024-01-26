# import os
# import os.path as osp
# import pathlib
# import argparse
# import yaml
# import glob
# import pickle
# import torch

# from viper_rl.videogpt.models import VQGAN, extract_iteration

# def main():
#     model0 = VQGAN(image_size=config.image_size, **config.ae)
#     model0_params = list(model0.named_parameters())

#     model1 = VQGAN(image_size=config.image_size, **config.ae)
#     model1_params = list(model1.named_parameters())

#     m = len(list(model0.named_parameters()))


#     model_files = glob.glob(f"viper_rl_data/checkpoints/dmc_vqgan/checkpoints/*.pth")
#     checkpoint_paths = sorted(model_files, key=extract_iteration)

#     model0.load_state_dict(torch.load(checkpoint_paths[1])["model_state_dict"])
#     print("load vqgan weights from {}".format(checkpoint_paths[1]))
#     model1.load_state_dict(torch.load(checkpoint_paths[-1])["model_state_dict"])
#     print("load vqgan weights from {}".format(checkpoint_paths[-1]))
    
    
#     for i in range(m):
#         print(model0_params[i][0])
#         print(torch.sum(torch.abs(model0_params[i][1]-model1_params[i][1])))
 
#     # for name, param in model.named_parameters():
#     #     print(name)
#     #     print(param)


#     # # print("----------------------")
#     # # print("load vqgan weights from {}".format(checkpoint_paths[-1]))
#     # model.load_state_dict(torch.load(checkpoint_paths[-1]), strict=False)
#     # for name, param in model.named_parameters():
#     #     print(name)
#     #     print(param)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, required=True)
#     args = parser.parse_args()

#     config = yaml.safe_load(open(args.config, 'r'))
#     args_d = vars(args)
#     args_d.update(config)
#     config = args
#     main()



import pickle

class_map = {'acrobot_swingup': 0, 'cartpole_balance': 1, 'cartpole_swingup': 2, 'cheetah_run': 3, 
                'cup_catch': 4, 'finger_spin': 5, 'finger_turn_hard': 6, 'hopper_stand': 7, 
                'manipulator_bring_ball': 8, 'pendulum_swingup': 9, 'pointmass_easy': 10, 
                'pointmass_hard': 11, 'quadruped_run': 12, 'quadruped_walk': 13, 'reacher_easy': 14, 
                'reacher_hard': 15, 'walker_walk': 16}

with open('class_map.pkl', 'wb') as f:
    pickle.dump(class_map, f)