import gym
# import minerl
import minedojo
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

env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 160)
)

obs = env.reset()

print(env.action_space.nvec)
for i in range(50):
    act = env.action_space.no_op()
    act[0] = 1    # forward/backward
    if i % 10 == 0:
        act[2] = 1    # jump
    obs, reward, done, info = env.step(act)
    # print(obs)
env.close()

# env_list = minedojo.tasks.ALL_TASK_IDS

# with open("mdj", "w") as file:
#     # Iterate over the strings in the list
#     for string in env_list:
#         # Write each string followed by a newline character
#         file.write(string + "\n")
