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
    plot_txt = 'plots/{0}_rewards.png'.format(env)
    
    methods = ["vanilla", "0.05-greedy", "0.5-0.05 decay"]
    seeds = ["", "1", "2"]

    for k in range(len(seeds)):
        steps = []
        rewards = []

        eval_txt = 'logdir/{0}{1}/metrics.jsonl'.format(env, seeds[k])
        
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

        plt.plot(steps, rewards, '-o', markersize=5, label="seed{0} {1}".format(seeds[k], methods[k]))


    plt.grid(True)
    plt.xlabel('Step', fontsize = 15)
    plt.ylabel('Rewards', fontsize = 15)
    plt.legend(loc='lower right', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.title("{0} {1} return curve".format(env, mode), fontsize = 20)

    plt.savefig(plot_txt)
    plt.close()
    

        

        
