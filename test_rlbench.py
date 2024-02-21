import os
import numpy as np
from PIL import Image

import gym
from gym import spaces
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
# from pyrep.objects.vision_sensor import VisionSensor

# Define observation configuration for third-person view
class RLBenchGym(gym.Env):
    def __init__(self, task_name, observation_config):
        super(RLBenchGym, self).__init__()
        
        self.env = Environment(observation_config=observation_config)
        self.env.launch()
        self.task = self.env.get_task(task_name)

        # Set action and observation space according to your task and observation_config
        # This is an example; you'll need to adjust this according to your specific task
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.task.action_size,), dtype='float32')
        # Adjust observation space based on the camera configuration
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype='uint8')

    def reset(self):
        descriptions, obs = self.task.reset()
        # Here, process obs to fit your observation_space and return
        return obs.get_low_dim_data()

    def step(self, action):
        obs, reward, terminate, _ = self.task.step(action)
        # Process obs to fit your observation_space
        return obs.get_low_dim_data(), reward, terminate, {}

    def close(self):
        self.env.shutdown()


def rollout_and_save(env_wrapper, steps, save_dir, env_type='train'):
    """
    Generate and save images for a given environment.
    
    Parameters:
    - env_wrapper: An instance of RLBenchGymWrapper.
    - steps: Number of steps to generate in the trajectory.
    - save_dir: Base directory to save the images.
    - env_type: 'train' or 'test', specifies the subdirectory.
    """
    # Ensure the save directory exists
    env_save_dir = os.path.join(save_dir, env_type)
    os.makedirs(env_save_dir, exist_ok=True)
    
    # Reset the environment
    obs = env_wrapper.reset()
    
    for step in range(steps):
        action = env_wrapper.action_space.sample()  # Sample a random action
        obs, _, _, _ = env_wrapper.step(action)
        
        # Assuming obs is a dictionary and contains an image at key 'left_shoulder_camera_rgb'
        # Adjust this according to how your observations are structured
        image = obs['left_shoulder_camera_rgb']  # This line will vary based on your observation space configuration
        
        # Convert the image to a PIL image and save
        img = Image.fromarray(image)
        img.save(os.path.join(env_save_dir, f'step_{step:03d}.png'))

if __name__ == "__main__":
    # Setup observation configuration for third-person camera view
    obs_config_train = ObservationConfig()
    third_person_camera = CameraConfig(
        image_size=(64, 64),
        position=[0, -2, 2],  # Example position
        quaternion=[0, 0, 0, 1],  # Example orientation
        active=True
    )
    obs_config_train.third_party_camera = [third_person_camera]
    obs_config_train.set_all(False)  # Disable other observations if not needed

    # Instantiate the training environment
    train_env = RLBenchGym('reach_target', observation_config=obs_config_train)

    # Setup observation configuration for first-person camera view
    obs_config_test = ObservationConfig()
    obs_config_test.set_all(False)
    obs_config_test.left_shoulder_camera.rgb = True  # Enable RGB for first-person view
    obs_config_test.left_shoulder_camera.image_size = (64, 64)  # Set image size to 64x64 for first-person view

    # Instantiate the testing environment
    test_env = RLBenchGym('reach_target', observation_config=obs_config_test)

    save_dir = 'logdir/rlbench'

    # Generate and save images for the training environment
    rollout_and_save(train_env, 50, save_dir, env_type='train')

    # Generate and save images for the testing environment
    rollout_and_save(test_env, 50, save_dir, env_type='test')

    


