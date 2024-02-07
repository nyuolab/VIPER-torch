import gym
import minerl

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()

print(env.action_space.noop())

env.close()