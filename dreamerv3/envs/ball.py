import numpy as np 
import cv2 
import math
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces, wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

def translate_points(points, translation_vector):
    return points - translation_vector

def rotate_points_around_origin(points, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    
    # Apply rotation to the points
    rotated_points = np.dot(points, rotation_matrix)
    
    return rotated_points

def rotate_points_around_center(points, center, angle_degrees):
    # Translate the points so that the center becomes the origin
    translated_points = translate_points(points, center)
    
    # Rotate the translated points around the origin
    rotated_points = rotate_points_around_origin(translated_points, angle_degrees)
    
    # Translate the rotated points back to their original position
    rotated_points = translate_points(rotated_points, -center)
    
    return rotated_points

def generate_random_poly_segment(x_range, y_range, degree=5, num_points=256):
    # Generate random coefficients for the polynomial
    coefs = np.random.rand(degree+1)
    
    # Generate x values within the specified range
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate y values using the polynomial equation
    y_values = np.polyval(coefs, x_values)
    
    # Scale and shift y values to fit within the specified y range
    y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    y_values = y_values * (y_range[1] - y_range[0]) + y_range[0]
    
    # Combine x and y values into a 2D array
    polynomial_segment = np.column_stack((x_values, y_values))
    
    return polynomial_segment

def video_callable(episode_id):
    return not (episode_id % 1)

class BallEnv(Env):
    def __init__(self, shape=(64, 64, 3), grid_w=4):
        super(BallEnv, self).__init__()
    
        # Define a 2-D observation space
        self.observation_shape = shape

        self.xgrid = self.observation_shape[1]//grid_w
        self.ygrid = self.observation_shape[0]//grid_w
        
        self.max_steps = self.observation_shape[0]
        
        self.grid_w = grid_w
        self.rad = self.grid_w//2
        
        # Define elements present inside the environment
        # self.elements = ["ball"]

        # Permissible area of ball to be 
        self.y_min = int (self.observation_shape[0]+self.rad)
        self.x_min = int (self.observation_shape[1]+self.rad)
        self.y_max = int (self.observation_shape[0]-self.rad)
        self.x_max = int (self.observation_shape[1]-self.rad)

        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
    
    @property
    def observation_space(self):
        return spaces.Box(low = np.zeros(self.observation_shape), 
                    high = np.ones(self.observation_shape)*255,
                    dtype = np.float64)

    @property
    def action_space(self):
        return spaces.Box(low = np.zeros(2), 
                          high = np.array([self.xgrid, self.ygrid]),
                          dtype = np.float64)

    def draw_on_img(self, xid, yid):
        self.img = np.ones(self.observation_shape) * 1
        cv2.circle(self.img, (yid*self.grid_w + self.rad, xid*self.grid_w + self.rad), self.rad, self.red, -1)
        return self.img
    
    def flatten2d_idx(self, x, y):
        return self.ygrid*min(math.floor(y/self.grid_w), self.ygrid-1) + \
                min(math.floor(x/self.grid_w), self.xgrid-1)
    
    def to2d_idx(self, idx):
        yid = idx//self.ygrid
        xid = idx%self.ygrid
        return xid, yid
    
    def reset(self, degree=5, num_points=256):
        self.steps = 0
        self.bc_cells = {}
        
        polynomial_segment = generate_random_poly_segment((self.grid_w, self.observation_shape[1]-self.grid_w), (self.grid_w, self.observation_shape[0]-self.grid_w), degree=degree, num_points=num_points)
        
        rotation_center = np.array([self.observation_shape[1]//2, self.observation_shape[0]//2])

        rotated_segment = rotate_points_around_center(polynomial_segment, rotation_center, random.uniform(0, 360))
        
        traj_points = []

        for p in rotated_segment:
            if (p[0] >= self.grid_w) and (p[0] <= self.observation_shape[1]-self.grid_w) and (p[1] >= self.grid_w) and (p[1] <= self.observation_shape[0]-self.grid_w):
                traj_points.append(p)
        
        num_points = len(traj_points)
        
        cells = {}

        for i in range(num_points):
            cells[self.flatten2d_idx(traj_points[i][1], traj_points[i][0])] = i
        
        cells = sorted(cells, key=lambda k: cells[k])

        self.num_cells = len(cells)

        # for i in range(self.num_cells//2):
        #     self.bc_cells[cells[i]] = ["green", 0]
        
        # for i in range(self.num_cells//2, self.num_cells):
        #     self.bc_cells[cells[i]] = ["red", 0]

        self.bc_cells[cells[0]] = ["green", 1]
        for i in range(1, self.num_cells):
            self.bc_cells[cells[i]] = ["black", 0]

        self.obs = {
                    "image": np.ones(self.observation_shape) * 255,
                    "pos": list(self.to2d_idx(cells[0])),
                    "is_first": True,
                    "is_last": False,
                    "is_terminal": False,
                }
        
        self.path_cover = 1


        for color_cell_idx, value in self.bc_cells.items():
            yidx = color_cell_idx//self.ygrid
            xidx = color_cell_idx%self.ygrid
            cv2.circle(self.obs["image"], (yidx*self.grid_w + self.rad, xidx*self.grid_w + self.rad), self.rad, self.black, -1)
        
        cv2.circle(self.obs["image"], (self.obs["pos"][1]*self.grid_w + self.rad, self.obs["pos"][0]*self.grid_w + self.rad), self.rad, self.green, -1)
        
        # cv2.imwrite("../plots/ball_env.jpg", self.obs["image"])

        return self.obs
    
    def render(self, mode = "rgb_array"):
        return self.obs["image"].astype("uint8")

    def step(self, action):
        # action = (x, y) coordinates

        xid = min(math.floor(action[0]/self.grid_w), self.xgrid-1)
        yid = min(math.floor(action[1]/self.grid_w), self.ygrid-1)

        
        flat_id = self.flatten2d_idx(action[0], action[1])

        if self.steps < self.max_steps//2:
            color_name = "green"
        else:
            color_name = "red"

        color = self.green if color_name == "green" else self.red
        

        if flat_id in self.bc_cells:
            color = self.blue
            # only first visit to a critical cell counts
            if not self.bc_cells[flat_id][1]:
                self.path_cover += 1
            reward = -self.bc_cells[flat_id][1]
            self.bc_cells[flat_id][1] += 1

        else:
            reward = -1


        obs = self.obs.copy()

        # whiten the previous ball
        pre_flat_id = self.ygrid*min(obs["pos"][1], self.ygrid-1) + min(obs["pos"][0], self.xgrid-1)
        
        if pre_flat_id in self.bc_cells:
            fill = self.black
        else:
            fill = self.white
        cv2.circle(obs["image"], (obs["pos"][1]*self.grid_w + self.rad, obs["pos"][0]*self.grid_w + self.rad), self.rad, fill, -1)

        obs["pos"] = [xid, yid]
        obs["is_first"] = False
        obs["is_last"] = False
        obs["is_terminal"] = False

        done = False
        info = {"status":"Ball moving..."}


        cv2.circle(obs["image"], (yid*self.grid_w + self.rad, xid*self.grid_w + self.rad), self.rad, color, -1)

        if self.steps >= self.max_steps:
            done = True
            info["status"] = "Max length reached :("
        
        if self.path_cover >= self.num_cells:
            done = True
            info["status"] = "Entire path covered :)"

        if done:
            obs["is_last"] = True
            obs["is_terminal"] = True
        
        self.obs = obs
        self.steps += 1
            
        return obs, reward, done, info

    def close(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Register the custom environment with Gym
    env_name = 'BallEnv-v0'  # Replace 'CustomEnv' with your environment's name
    gym.envs.register(id=env_name, entry_point='ball:BallEnv')
    # env = BallEnv()
    env = gym.make('BallEnv-v0')
    video_dir = "../experiments/ball/ball_trial.mp4"
    env = wrappers.Monitor(env, video_dir, force=True, video_callable=video_callable)

    # env.metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 30}

    obs = env.reset()
    done = False
    cv2.imwrite("../experiments/ball/step{}.jpg".format(env.steps), obs["image"])

    while not done:
        action = np.random.uniform(0, env.observation_shape[0], size=2)
        obs, reward, done, _ = env.step(action)

        cv2.imwrite("../experiments/ball/step{}.jpg".format(env.steps), obs["image"])
    
    env.close()



        


    




        
