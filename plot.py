import numpy as np
import matplotlib.pyplot as plt
import json
import ast

import argparse


if __name__ == "__main__":
    # env = "memorymaze_15x15"
    # env = "crafter_reward"
    # env = 'minecraft_diamond'

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", dest='env', type=str, default="crafter_reward")
    parser.add_argument("--mode", dest='mode', type=str,  default="eval")
    args = parser.parse_args()
    
    env = args.env
    mode = args.mode

    return_mode = "{}_return".format(mode)
    
    steps = []
    rewards = []

    eval_txt = 'logdir/{0}/metrics.jsonl'.format(env)
    plot_txt = 'plots/{0}_rewards.png'.format(env)
    
    with open(eval_txt) as f:
        data = f.readlines()
        l = len(data)
        print(l)
        for i in range(l):
            # print(i)
            d = ast.literal_eval(data[i])
            
            if return_mode in d:
                rewards.append(d[return_mode])
                steps.append(d["step"])
    
    plt.grid(True)
    plt.xlabel('Step', fontsize = 15)
    plt.ylabel('Rewards', fontsize = 15)

    plt.plot(steps, rewards, '-o', markersize=5)

    plt.title("{0} {1} return curve".format(env, mode), fontsize = 20)

    plt.savefig(plot_txt)
    plt.close()
    

        

        
