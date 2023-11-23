import gym
# import minedojo
# import logging
# logging.basicConfig(level=logging.DEBUG)

# env = gym.make('MineRLBasaltFindCave-v0')

# done = False

# obs = env.reset()
# while not done:
#     # Take a random action
#     action = env.action_space.sample()
#     # In BASALT environments, sending ESC action will end the episode
#     # Lets not do that
#     action["ESC"] = 0
#     obs, reward, done, _ = env.step(action)
#     env.render()

# env = minedojo.make(
#     task_id="harvest_wool_with_shears_and_sheep",
#     image_size=(160, 160)
# )

# obs = env.reset()

# print(env.action_space.nvec)
# for i in range(50):
#     act = env.action_space.no_op()
#     act[0] = 1    # forward/backward
#     if i % 10 == 0:
#         act[2] = 1    # jump
#     obs, reward, done, info = env.step(act)
#     # print(obs)
# env.close()

# env_list = minedojo.tasks.ALL_TASK_IDS

# with open("mdj", "w") as file:
#     # Iterate over the strings in the list
#     for string in env_list:
#         # Write each string followed by a newline character
#         file.write(string + "\n")

import minerl

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)
ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)


env = gym.make("MineRLBasaltFindCave-v0", break_speed_multiplier=100.0)

print(env.action_space)
# Dict(ESC:Discrete(2), attack:Discrete(2), back:Discrete(2), 
# camera:Box(low=-180.0, high=180.0, shape=(2,)), drop:Discrete(2), 
# forward:Discrete(2), hotbar.1:Discrete(2), 
# hotbar.2:Discrete(2), hotbar.3:Discrete(2), 
# hotbar.4:Discrete(2), hotbar.5:Discrete(2), 
# hotbar.6:Discrete(2), hotbar.7:Discrete(2), 
# hotbar.8:Discrete(2), hotbar.9:Discrete(2), 
# inventory:Discrete(2), jump:Discrete(2), 
# left:Discrete(2), pickItem:Discrete(2), 
# right:Discrete(2), sneak:Discrete(2), 
# sprint:Discrete(2), swapHands:Discrete(2), use:Discrete(2)

obs = env.reset()


done = False
while not done:
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 3]
    obs, reward, done, info = env.step(ac)
    env.render()
env.close()
