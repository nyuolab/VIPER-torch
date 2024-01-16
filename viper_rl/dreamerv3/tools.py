import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
# import uuid
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from PIL import Image
# from accelerate import Accelerator
# accelerator = Accelerator()

import multiprocessing

from gym.spaces import MultiDiscrete


def for_loop_parallel(input_list, fn):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        res = pool.map(fn, input_list)
    return res

to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)
    
    
    # def vector(self, name, value):
    #     self._vectors[name] = value

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._vectors = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def save_image(img, outdir, filename):
    # transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    output_path = outdir / filename
    Image.fromarray(img.astype('uint8')).save(output_path)



# compare real & imaginary trajectories
def eval_rollout(
    agent,
    envs,
    cache, # trajectories/transitions
    directory,
    logger,
    limit=None,
    episodes=0,
    steps=0,
    state=None,
    real=True
):  
    # initialize or unpack simulation state
    print("Number of envs = {}".format(len(envs)))
    # obs: dict_keys(['orientations', 'height', 'velocity', 'image', 'is_terminal', 'is_first', 'reward'])
    
    step, episode = 0, 0
    done = np.ones(len(envs), bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
    reward = [0] * len(envs)


    indices = [index for index, d in enumerate(done) if d] # indices of terminated envs
    results = [envs[i].reset() for i in indices]
    results = [r() for r in results]

    save_image(results[0]['image'], logger._logdir, agent._config.task + "/real_imgs/0.jpg".format())
    save_image(results[0]['image'], logger._logdir, agent._config.task + "/fake_imgs/0.jpg".format())

    initial_states = results.copy()
    
    tb = agent._task_behavior
    # H = tb._config.imag_horizon
    H = 500
    # Real rollout
    while True:
        if done[0]:

            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}

        action, agent_state = agent(obs, done, agent_state, training=False)
        # type(action) = dict
        # print(action.keys())
        
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)


        assert len(action) == len(envs)
        # print(action) dictionary with keys action, log_prob
        # step envs one step per episode
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        # print(results)
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1

        # print(obs[0]['image'].shape)
        if step < H:
            save_image(obs[0]['image'], logger._logdir, agent._config.task + "/real_imgs/{}.jpg".format(step+1)) 


        step += len(envs)
        length *= 1 - done
        # add to cache
        # print(obs[0].keys())
        # print(obs[0]['image'].shape)

        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            # PROGRESS
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)



        if done[0]:
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                # save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                
                if not "eval_lengths" in locals():
                    eval_lengths = []
                    eval_scores = []
                    eval_done = False
                # start counting scores for evaluation
                eval_scores.append(score)
                eval_lengths.append(length)

                score = sum(eval_scores) / len(eval_scores)
                length = sum(eval_lengths) / len(eval_lengths)
                logger.video(f"eval_policy", np.array(video)[None])

            
            logger.scalar(f"eval_return", score)
            logger.scalar(f"eval_length", length)
            logger.scalar(f"eval_episodes", len(eval_scores))
            logger.write(step=logger.step)

            break

    # keep only last item for saving memory. this cache is used for video_pred later
    while len(cache) > 1:
        # FIFO
        cache.popitem(last=False)
    print("Total number of steps = {}".format(step))
    
    
    # Imaginary rollout
    for index, result in zip([0], initial_states):
        t = result.copy()
        t = {k: convert(v) for k, v in t.items()}
        # action will be added to transition in add_to_cache
        t["reward"] = 0.0
        t["discount"] = 1.0
        # replace obs with done by initial state
        obs[index] = result
    # step agents
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}

    # Might be the reason for weird colors for fake imgs 
    obs = agent._wm.preprocess(obs)
    embed = agent._wm.encoder(obs) # The embedding might still need normalized input 

    # print(obs)
    print(embed)

    # pseudo action?
    action = torch.zeros((len(envs), agent._config.num_actions)).to(agent._config.device)

    state = agent._wm.dynamics.initial(len(envs))
    start, _ = agent._wm.dynamics.obs_step(state, action, embed, obs["is_first"])
    
    
    start = {k: v.detach() for k, v in start.items()}

    imag_feat, imag_state, imag_action = \
        agent._task_behavior._imagine(start, tb.actor, H)
    
    print(imag_action)
    # reward = tb._reward(imag_feat, imag_state, imag_action)
    recon_imgs = agent._wm.heads["decoder"](imag_feat)["image"].mode()
    for i in range(H):
        save_image((recon_imgs[i].cpu().numpy()+0.5)*255, logger._logdir, agent._config.task + "/fake_imgs/{}.jpg".format(i+1))
    print(recon_imgs.shape)

    # return recon_imgs, imag_action, reward




def simulate(
    agent,
    # random_agent,
    envs,
    cache, # trajectories/transitions
    directory, # config.traindir
    logger, 
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    # initialize or unpack simulation state
    reward_model = agent.func.reward_model if is_eval else agent.reward_model
    # video_len = agent.func._config.video_len if is_eval else agent._config.video_len 
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)

    else:
        step, episode, done, length, obs, agent_state, reward = state
        
    # if agent._config.video_len > 1:
    #     video_seg = [[] for _ in range(len(envs))]
    
    # print(envs[0].action_space.shape[0])
    # minecraft max time steps = 36000, steps = 10000
    
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset all terminated envs
        # print("Doing step {}".format(step))
        if done.any():
            indices = [index for index, d in enumerate(done) if d] # indices of terminated envs
            
            results = [envs[i].reset() for i in indices]
            # print(results)
            results = [r() for r in results]
            
            for index, result in zip(indices, results):
                # video_seg = np.stack([result["image"]] * video_len) # np.expand_dims(np.stack([obs["image"]] * 5), axis=0)
                # print(video_seg.shape)
                # result["video"] = video_encoder(video_seg).flatten()
                # result.pop("image")

                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
            
            obs_dict = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}    
            action, agent_state = agent(obs_dict, done, agent_state, training=not is_eval, train=True)
        else:
            obs_dict = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
            action, agent_state = agent(obs_dict, done, agent_state, training=not is_eval, train=False)

        # obs_dict = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}    
        # action, agent_state = agent(obs_dict, done, agent_state, training=not is_eval, train=True)
    

        
        # action: dict(action_dim_list(torch.tensor(len(envs) * one hot actions)))
        # action.keys() = [action, log_prob]
        # print(action)
        
        # separate treatment for actions in minedojo
        if isinstance(envs[0].action_space, MultiDiscrete):
            # convert to index from onehot straightaway
            if isinstance(action, dict):
                action_list = []
                for i in range(len(envs)):
                    action_list.append({})
                    for k in action.keys():
                        # action_list[i][k] = np.zeros(envs[0].action_space.shape[0], dtype=int)
                        action_list[i][k] = []
                        for a in range(envs[0].action_space.shape[0]):
                            # action_list[i][k][a] = np.argmax(action[k][a][i].detach().cpu().numpy()).astype(int)
                            action_list[i][k].append(action[k][a][i].detach().cpu().numpy().astype(int))
                action = action_list
            else:
                raise TypeError("Multidiscrete action has to use dictionary")
        else:
            if isinstance(action, dict):  
                # if video_len > 1:
                #     # (num_envs, video_length, action_dim)
                #     action_seq = [
                #         [{k: action[k][i][j].detach().cpu().numpy() for k in action} for j in range(video_len)]
                #         for i in range(len(envs))
                #     ]
                #     # print(action["action"][0].shape)
                #     action = [
                #         {k: action[k][i].flatten(start_dim=-2).detach().cpu().numpy() if len(action[k][i].shape) > 2 else action[k][i].view(-1).detach().cpu().numpy() for k in action}
                #         for i in range(len(envs))
                #     ]
                # else:
                action = [
                    {k: action[k][i].detach().cpu().numpy() for k in action}
                    for i in range(len(envs))
                ]
            else:
                action = np.array(action)

        # print(action)

        assert len(action) == len(envs)
        
        # step envs
        # if video_len > 1:
        #     video_segs = [[] for _ in range(len(envs))]
        #     reward = [0] * len(envs)

        #     for j in range(video_len):
        #         results = []
        #         for i in range(len(action_seq)):
        #             result = envs[i].step(action_seq[i][j])
        #             results.append(result())
                
        #         frame_obs, frame_reward, done = zip(*[p[:3] for p in results])

        #         # frame_obs = list(frame_obs)
        #         # frame_reward = list(frame_reward)
        #         done = np.stack(done)
        #         episode += int(done.sum())
        #         length += 1
        #         step += len(envs)
        #         length *= 1 - done
                
        #         # if "log_inventory/variant" in cache[envs[0].id]:
        #         #     print("The inventory variants are {}".format(cache[envs[0].id]["log_inventory/variant"]))
        #         # if "log_equipment/variant" in cache[envs[0].id]:
        #         #     print("The equipped variants are {}".format(cache[envs[0].id]["log_equipment/variant"]))
        #         # add to cache

        #         for i in range(len(envs)):
        #             o, r, d, info = results[i]
        #             o = {k: convert(v) for k, v in o.items()}

        #             reward[i] += r
        #             video_segs[i].append(o["image"])

        #             # transition = o.copy()
        #             # if isinstance(a, dict):
        #             #     transition.update(a)
        #             # else:
        #             #     transition["action"] = a
        #             # transition["reward"] = r
        #             # transition["discount"] = info.get("discount", np.array(1 - float(d)))
                    
        #             # add_to_cache(cache, env.id, transition)
                    

        #         # only works for one env so far
        #         if done.any():
        #             indices = [index for index, d in enumerate(done) if d]
        #             # logging for done episode

        #             for i in indices:
        #                 for _ in range(j+1, video_len):
        #                     video_segs[i].append(video_segs[i][-1].copy())
                        

        #                 o, r, d, info = results[i]
        #                 # o = {k: convert(v) for k, v in o.items()}
        #                 transition = o.copy()
        #                 transition = {k: convert(v) for k, v in transition.items()}
        #                 transition["video"] = video_encoder(np.stack(video_segs[i], axis=0)).flatten()
        #                 transition.pop("image")
                        
        #                 if isinstance(action[i], dict):
        #                     transition.update(action[i])
        #                 else:
        #                     transition["action"] = action[i]
        #                 transition["reward"] = reward[i]
        #                 transition["discount"] = info.get("discount", np.array(1 - float(d)))
        #                 add_to_cache(cache, envs[i].id, transition)
                    
        #                 save_episodes(directory, {envs[i].id: cache[envs[i].id]})
        #                 length = len(cache[envs[i].id]["reward"]) - 1
        #                 score = float(np.array(cache[envs[i].id]["reward"]).sum())
        #                 # video = cache[envs[i].id]["image"]

        #                 # print("The inventory variants are {}".format(cache[envs[i].id]["log_inventory/variant"]))
        #                 # print("The equipped variants are {}".format(cache[envs[i].id]["log_equipment/variant"]))
        #                 # record logs given from environments
        #                 # variant_dict = {}

        #                 for key in list(cache[envs[i].id].keys()):
        #                     if "log_" in key:
        #                         logger.scalar(
        #                             key, float(np.array(cache[envs[i].id][key]).sum())
        #                             # key, list(cache[envs[i].id][key])
        #                         )
        #                         # variant_dict[key] = np.array(cache[envs[i].id][key]).flatten().tolist()
        #                         # print("{}: {}".format(key, cache[envs[i].id][key]))
        #                         # log items won't be used later
        #                         cache[envs[i].id].pop(key)
                        
        #                 # with open("logdir/minecraft_diamond/metrics.jsonl", "a") as f:
        #                 #     json.dump(variant_dict, f)

        #                 if not is_eval:
        #                     step_in_dataset = erase_over_episodes(cache, limit)
        #                     logger.scalar(f"dataset_size", step_in_dataset)
        #                     logger.scalar(f"train_return", score)
        #                     logger.scalar(f"train_length", length)
        #                     logger.scalar(f"train_episodes", len(cache))
        #                     logger.write(step=logger.step)
        #                 else:
        #                     if not "eval_lengths" in locals():
        #                         eval_lengths = []
        #                         eval_scores = []
        #                         eval_done = False
        #                     # start counting scores for evaluation
        #                     eval_scores.append(score)
        #                     eval_lengths.append(length)

        #                     score = sum(eval_scores) / len(eval_scores)
        #                     length = sum(eval_lengths) / len(eval_lengths)
        #                     # logger.video(f"eval_policy", np.array(video)[None])

        #                     if len(eval_scores) >= episodes and not eval_done:
        #                         logger.scalar(f"eval_return", score)
        #                         logger.scalar(f"eval_length", length)
        #                         logger.scalar(f"eval_episodes", len(eval_scores))
        #                         logger.write(step=logger.step)
        #                         eval_done = True
                    
        #             # obs, reward, done = zip(*[p[:3] for p in results])
        #             # obs = list(obs)
        #             # obs[index] = result
        #             # print(obs[0].keys())
        #             # obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}

        #             break

        #     # all envs not finished      
        #     if not done.any():
        #         for i in range(len(envs)):
        #             o, r, d, info = results[i]
        #             o = {k: convert(v) for k, v in o.items()}
        #             o["video"] = video_encoder(np.stack(video_segs[i], axis=0)).flatten()
        #             o.pop("image")
        #             transition = o.copy()
        #             # print(np.stack(video_segs[i]).shape)
        #             # print(transition["video"].shape)
        #             # transition.pop("image")
                    
        #             if isinstance(action[i], dict):
        #                 transition.update(action[i])
        #             else:
        #                 transition["action"] = action[i]
        #             transition["reward"] = reward[i]
        #             transition["discount"] = info.get("discount", np.array(1 - float(d)))
        #             add_to_cache(cache, envs[i].id, transition)

        #             obs[i] = o


        # else:
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                # separate treatment for actions in minedojo
                if isinstance(envs[0].action_space, MultiDiscrete):
                    for k in a.keys():
                        assert len(a[k])
                        if len(a[k][0].shape):    
                            a[k] = np.concatenate(a[k], axis=-1)
                        else:
                            a[k] = np.mean(np.stack(a[k], axis=0))
                transition.update(a)
            else:
                # separate treatment for actions in minedojo
                if isinstance(envs[0].action_space, MultiDiscrete):
                    a = np.concatenate(a, axis=-1)
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)
            
        
        # if "log_inventory/variant" in cache[envs[0].id]:
        #     print("The inventory variants are {}".format(cache[envs[0].id]["log_inventory/variant"]))
        # if "log_equipment/variant" in cache[envs[0].id]:
        #     print("The equipped variants are {}".format(cache[envs[0].id]["log_equipment/variant"]))
    
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                # print("The cache length is {}".format(len(cache[envs[i].id]["image"])))
                # print("The number of images is {}".format(len(video)))
                # print("The length of reward seq is {}".format(len(cache[envs[i].id]["reward"])))

                if reward_model is not None and (not is_eval):
                    # print(cache[envs[i].id]["reward"])
                    with torch.no_grad():
                        cache[envs[i].id] = reward_model(cache[envs[i].id])
                    # print("The length of density seq is {}".format(len(cache[envs[i].id]["density"])))
                # print(cache[envs[i].id].keys())
                # print("The inventory variants are {}".format(cache[envs[i].id]["log_inventory/variant"]))
                # print("The equipped variants are {}".format(cache[envs[i].id]["log_equipment/variant"]))
                # record logs given from environments
                # variant_dict = {}

                save_episodes(directory, {envs[i].id: cache[envs[i].id]})

                if reward_model and not is_eval:
                    if 'density' not in cache[envs[i].id].keys():
                        raise Exception("Has to have density for complete trajectories")
                    else:
                        print("Complete trajectory with density")

                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                            # key, list(cache[envs[i].id][key])
                        )
                        # variant_dict[key] = np.array(cache[envs[i].id][key]).flatten().tolist()
                        # print("{}: {}".format(key, cache[envs[i].id][key]))
                        # log items won't be used later
                        cache[envs[i].id].pop(key)
                
                # with open("logdir/minecraft_diamond/metrics.jsonl", "a") as f:
                #     json.dump(variant_dict, f)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True

    if not is_eval:
        if not done.any():
            indices = [index for index, d in enumerate(done) if not d]
            # logging for done episode
            for i in indices:
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]

                if reward_model is not None:
                    # print(cache[envs[i].id]["reward"])
                    with torch.no_grad():
                        cache[envs[i].id] = reward_model(cache[envs[i].id])
                    # print(cache[envs[i].id]["density"])
                # print(cache[envs[i].id].keys())
                # print("The inventory variants are {}".format(cache[envs[i].id]["log_inventory/variant"]))
                # print("The equipped variants are {}".format(cache[envs[i].id]["log_equipment/variant"]))
                # record logs given from environments
                # variant_dict = {}
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})

                if reward_model and 'density' not in cache[envs[i].id].keys():
                    raise Exception("Has to have density for incomplete trajectories")
                else:
                    print("Incomplete trajectory with density")


                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                            # key, list(cache[envs[i].id][key])
                        )
                        # variant_dict[key] = np.array(cache[envs[i].id][key]).flatten().tolist()
                        # print("{}: {}".format(key, cache[envs[i].id][key]))
                        # log items won't be used later
                        cache[envs[i].id].pop(key)
            
                # with open("logdir/minecraft_diamond/metrics.jsonl", "a") as f:
                #     json.dump(variant_dict, f)

                step_in_dataset = erase_over_episodes(cache, limit)
                logger.scalar(f"dataset_size", step_in_dataset)
                logger.scalar(f"train_return", score)
                logger.scalar(f"train_length", length)
                logger.scalar(f"train_episodes", len(cache))
                logger.write(step=logger.step) 
        else:
            print("All trajectories done for this episode")
    else:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
            
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


class CollectDataset:
    def __init__(
        self, env, mode, train_eps, eval_eps=dict(), callbacks=None, precision=32
    ):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None
        self._cache = dict(train=train_eps, eval=eval_eps)[mode]
        self._temp_name = str(uuid.uuid4())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        if isinstance(action, dict):
            transition.update(action)
        else:
            transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        self._episode.append(transition)
        self.add_to_cache(transition)
        if done:
            # detele transitions before whole episode is stored
            del self._cache[self._temp_name]
            self._temp_name = str(uuid.uuid4())
            for key, value in self._episode[1].items():
                if key not in self._episode[0]:
                    self._episode[0][key] = 0 * value
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        # missing keys will be filled with a zeroed out version of the first
        # transition, because we do not know what action information the agent will
        # pass yet.
        transition["reward"] = 0.0
        transition["discount"] = 1.0
        self._episode = [transition]
        self.add_to_cache(transition)
        return obs


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    # print(cache[list(cache.keys())[-1]].keys())
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    # print(cache.keys())
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
        # elif np.issubdtype(value.dtype, dtype.str):
        #     dtype = str
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    # print(episodes['20240115T032602-a2ac21d75bdd4c6ca13477a3b708b295-501'].keys())
    # print(episodes.keys())

    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        
        err_count = 0.0
        count = 0
        while size < length:
            count += 1
            episode = np_random.choice(list(episodes.values()), p=p)
            # print("episode keys: {}".format(episode.keys()))
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            # if "density" not in episode.keys():
            #     err_count += 1
            #     continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)]
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                # ret_new = {}
                # for k, v in episode.items():
                #     if "log_" not in k:
                #         print(k)
                #         print(np.array(ret[k]).shape)
                #         print(np.array(v[index : min(index + possible, total)]).shape)
                #         ret_new[k] = np.append(
                #             ret[k], v[index : min(index + possible, total)], axis=0
                #         )
                # ret = ret_new
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index+possible, total)], axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            # print("ret keys: {}".format(ret.keys()))
            size = len(next(iter(ret.values())))
        # if err_count > 0:
        #     print("missing keys occurrence frequency for this batch: {}".format(err_count/count))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes

def tensor_seg_reshape(x, num_seg=5):
    assert not (x.size(-1) % num_seg)
    seg_len = x.size(-1)//num_seg
    old_shape = x.size()
    new_shape = list(old_shape[:-1]) + [num_seg, seg_len]
    # print(new_shape)
    
    x = x.view(*new_shape)
    return x

class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)



class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0, num_seg=5):
        self.num_seg = num_seg
        self.orig_shape = None
        self.new_shape = None
        self.num_seg = num_seg
        # print(logits.shape)
        if logits is not None:
            self.orig_shape = logits.size()
            self.new_shape = list(self.orig_shape[:-1]) + [num_seg, logits.size(-1)//self.num_seg]

            if num_seg > 1:
                logits = tensor_seg_reshape(logits, num_seg=self.num_seg)
            # print(logits.shape)

            if unimix_ratio > 0.0:
                probs = F.softmax(logits, dim=-1)
                probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
                logits = torch.log(probs)
            
            # print(logits.shape)
            # if num_segments > 1:
            #     logits = logits.view(*self.orig_shape)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs) # input probs already reshaped w.r.t. separate actions in a sequence

    def mode(self):
        # print(super().logits.shape)
        # if self.num_seg > 1:
        #     logits = tensor_seg_reshape(super().logits)
        # else:
        #    logits = super().logits
        logits = super().logits

        _mode = F.one_hot(
            torch.argmax(logits, axis=-1), logits.shape[-1]
        )

        # if num_segments > 1:
        #     _mode = _mode.view(*self.orig_shape)
        #     logits = logits.view(*self.orig_shape)

        return _mode.detach() + logits - logits.detach()

    def sample(self, sample_shape=(), seed=None, flatten=False):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        # if self.num_seg > 1:
        # print(sample.shape)
        if flatten and self.num_seg > 1:
            sample = sample.view(*self.orig_shape)
            # sample.flatten(start_dim=-2)
        # if not flatten and self.new_shape and self.num_seg > 1:
        #     sample = sample.view(*self.new_shape)
            # print(sample.shape)
        return sample


class MultiOneHotDist():
    def __init__(
        self, 
        logits_list=None, 
        probs_list=None, 
        unimix_ratio=0.0):
        super().__init__()
        # if probs_list is None:   
        #     self.dist_list = for_loop_parallel(logits_list, 
        #                     lambda logit: OneHotDist(logit, None, unimix_ratio=unimix_ratio))
        # else:
        #     self.dist_list = for_loop_parallel(zip(logits_list, probs_list), 
        #                     lambda logit, probs: OneHotDist(logit, probs, unimix_ratio=unimix_ratio))
        
        if probs_list is None:   
            self.dist_list = [OneHotDist(logit, None, unimix_ratio=unimix_ratio) 
                                for logit in logits_list]
        
        else:
            self.dist_list = [OneHotDist(None, probs, unimix_ratio=unimix_ratio) 
                                for probs in probs_list]
                        
            # self.dist_list = [OneHotDist(logit, probs, unimix_ratio=unimix_ratio) 
            #                     for logit, probs in zip(logits_list, probs_list)]
                        

    def sample(self, sample_shape=(), seed=None):
        return [dist.sample(sample_shape, seed) for dist in self.dist_list]
        # return for_loop_parallel(self.dist_list, lambda dist: dist.sample(sample_shape, seed))
    
    def mode(self):
        return [dist.mode() for dist in self.dist_list]
        # return for_loop_parallel(self.dist_list, lambda dist: dist.mode())
    
    def log_prob(self, x, i=None):
        if i is None:
            return [dist.log_prob(xi) for xi, dist in zip(x, self.dist_list)]
            # return for_loop_parallel(zip(xlist, self.dist_list), lambda x, dist: dist.log_prob(x))
        return self.dist_list[i].log_prob(x)

    def entropy(self):
        return [dist.entropy() for dist in self.dist_list]
        # return for_loop_parallel(self.dist_list, lambda dist: dist.entropy())




class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        # assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        # (16, 100, 128, 128, 3) vs (16, 100, 3, 128, 128)
        if self._mode.shape != value.shape and (self._mode.shape[-1] == 3):
            distance = (self._mode.permute(0,1,4,2,3) - value) ** 2
        else:
            distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        # print(self._mode.shape)
        # print(value.shape)
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2)
            - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        # accelerator.backward(self._scaler.scale(loss))
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

'''
It performs a static scan operation over a sequence of inputs, 
applying a provided function fn at each step and collecting the outputs. 
It maintains an internal state last that gets updated in each iteration. 
The collected outputs are structured based on the type of the intermediate state and 
whether it's the first iteration or not
fn(start, )
'''
def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


# Original version
# def static_scan2(fn, inputs, start, reverse=False):
#  last = start
#  outputs = [[] for _ in range(len([start] if type(start)==type({}) else start))]
#  indices = range(inputs[0].shape[0])
#  if reverse:
#    indices = reversed(indices)
#  for index in indices:
#    inp = lambda x: (_input[x] for _input in inputs)
#    last = fn(last, *inp(index))
#    [o.append(l) for o, l in zip(outputs, [last] if type(last)==type({}) else last)]
#  if reverse:
#    outputs = [list(reversed(x)) for x in outputs]
#  res = [[]] * len(outputs)
#  for i in range(len(outputs)):
#    if type(outputs[i][0]) == type({}):
#      _res = {}
#      for key in outputs[i][0].keys():
#        _res[key] = []
#        for j in range(len(outputs[i])):
#          _res[key].append(outputs[i][j][key])
#        #_res[key] = torch.stack(_res[key], 0)
#        _res[key] = faster_stack(_res[key], 0)
#    else:
#      _res = outputs[i]
#      #_res = torch.stack(_res, 0)
#      _res = faster_stack(_res, 0)
#    res[i] = _res
#  return res


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(torch.Tensor([step / duration]), 0, 1)[0]
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)

# Initialize model weights
def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)