import logging
import threading

import numpy as np
import gym
from . import minecraft_minedojo

def print_in_file(d):
    import json
    with open("record.txt", 'a') as f:
        f.write(json.dumps(d))
        f.write('\n')


class MinecraftBase(gym.Env):
    _LOCK = threading.Lock()

    def __init__(
        self,
        # actions,
        task,
        repeat=1,
        size=(64, 64),
        break_speed=100.0,
        gamma=10.0,
        sticky_attack=30,
        sticky_jump=10,
        pitch_limit=(-60, 60),
        # logs=True,
        logs=False,
    ):
        if logs:
            logging.basicConfig(level=logging.DEBUG)
        self._repeat = repeat
        self._size = size
        if break_speed != 1.0:
            sticky_attack = 0

        # Make env
        with self._LOCK:
            # from minedojo.sim import MineDojoSim
            # self._env = MineDojoSim(break_speed_multiplier=break_speed, image_size=size)
            import minedojo
            self._env = minedojo.make(task_id=task, image_size=size, break_speed_multiplier=break_speed)
        
        self.inv_exclude = ["inventory/name", "inventory/variant", "inventory/cur_durability", "inventory/max_durability"]
        self.equip_exclude = ["equipment/name", "equipment/variant", "equipment/cur_durability", "equipment/max_durability"]

        self._inventory = {}

        # Observations
        self._inv_keys = [
            k
            for k in self._flatten(self._env.observation_space.spaces)
            if k.startswith("inventory/")
            if k not in self.inv_exclude
        ]

        # print(self._inv_keys)
        # self._step = 0
        self._max_inventory = None
        # self._equip_enum = self._env.observation_space["equipped_items"]["mainhand"][
        #     "type"
        # ].values.tolist()
        self._equip_keys = [
            k
            for k in self._flatten(self._env.observation_space.spaces)
            if k.startswith("equipment/")
            if k not in self.equip_exclude
        ]

        # Actions
        self._noop_action = self._env.action_space.no_op()
        # actions = self._insert_defaults(actions)
        # self._action_names = tuple(actions.keys())
        # self._action_values = tuple(actions.values())
        # message = f"Minecraft action space ({len(self._action_values)}):"
        # print(message, ", ".join(self._action_names))
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
                "inventory": gym.spaces.Box(
                    -1.0, np.inf, (len(self._inv_keys)*36,), dtype=np.float32
                ),
                "inventory_max": gym.spaces.Box(
                    -1.0, np.inf, (len(self._inv_keys)*36,), dtype=np.float32
                ),

                "equipped": gym.spaces.Box(
                    -1.0, np.inf, (len(self._equip_keys)*6,), dtype=np.float32
                ),
                
                "block_meta": gym.spaces.Box(-np.inf, np.inf, (27,), dtype=np.float32),
                "block_collidable": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_tool": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_movement": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_liquid": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_solid": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_burn": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "block_light": gym.spaces.Box(0.0, 1.0, (27,), dtype=np.float32),
                "look_angle": gym.spaces.Box(-1.0, 1.0, (27,), dtype=np.float32),

                # "reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "health": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "hunger": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "armor": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "breath": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "xp": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                
                "yaw": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "pitch": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),  
                "rain": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "temperature": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "light" : gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "sky_light" : gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "sun": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "sea": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                
                "damage_amount": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "damage_dist": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "damage_yaw": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
                "damage_hunger": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "is_explosive": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "is_fire": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "is_projectile": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "is_unblockable": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                
                "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int8),
                "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int8),
                "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int8),
                **{
                    f"log_{k}": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int64)
                    for k in (self._inv_keys + self._equip_keys)
                },
                "pos": gym.spaces.Box(
                    -np.inf, np.inf, (3,), dtype=np.float32
                ),
            }
        )

    @property
    def action_space(self):
        # space = gym.spaces.discrete.Discrete(len(self._action_values))
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        # action = action.copy()
        # action = self._action_values[action]
        following = self._noop_action.copy()
        # for key in ("attack", "forward", "back", "left", "right"):
        #     following[key] = action[key]
        # print(action)
        # print(following)
        for i in range(3):
            following[i] = action[i]

        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self._env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", done))

        obs = self._obs(obs)
        # print(obs["inventory_names"])
        # self._step += 1
        if obs["health"] <= 0.0:
            done = True
            
        assert "pov" not in obs, list(obs.keys())
        return obs, reward, done, info

    @property
    def inventory(self):
        return self._inventory

    def reset(self):
        # inventory will be added in _obs
        self._inventory = {}
        self._max_inventory = None

        with self._LOCK:
            obs = self._env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)

        # self._step = 0
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def _obs(self, obs):
        obs = self._flatten(obs)
        # print(obs.keys())
        # obs["inventory/log"] += obs.pop("inventory/log2")
        self._inventory = {
            k.split("/", 1)[1]: obs[k] for k in (self._inv_keys + ["inventory/name"])
        }
        # ValueError: setting an array element with a sequence. 
        # The requested array has an inhomogeneous shape after 1 dimensions. 
        # The detected shape was (126,) + inhomogeneous part.
        # o = {k:obs[k] for k in self._inv_keys}
        # print(o)
        # print_in_file(o)
        inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
        # print(inventory.shape)
        if self._max_inventory is None:
            self._max_inventory = inventory
        else:
            self._max_inventory = np.maximum(self._max_inventory, inventory)
        # index = self._equip_enum.index(obs["equipped_items/mainhand/type"])
        # equipped = np.zeros(len(self._equip_enum), np.float32)
        # equipped[index] = 1.0
        equipped = np.array([obs[k] for k in self._equip_keys], np.float32)
        
        # player_x = obs["location_stats/xpos"]
        # player_y = obs["location_stats/ypos"]
        # player_z = obs["location_stats/zpos"]
        # print(obs["inventory/name"])
        # print(obs["inventory/id"])

        obs = {
            "image": obs["rgb"],
            # "inventory_name": obs["inventory/name"],
            "inventory": inventory.reshape(-1),
            "inventory_max": self._max_inventory.copy().reshape(-1),
            
            # "equipment_name": obs["equipment/name"],
            "equipped": equipped.reshape(-1),

            "block_meta": np.float32(obs["voxels/block_meta"]).reshape(-1),
            "block_collidable": np.float32(obs["voxels/is_collidable"]).reshape(-1),
            "block_tool": np.float32(obs["voxels/is_tool_not_required"]).reshape(-1),
            "block_movement": np.float32(obs["voxels/blocks_movement"]).reshape(-1),
            "block_liquid": np.float32(obs["voxels/is_liquid"]).reshape(-1),
            "block_solid": np.float32(obs["voxels/is_solid"]).reshape(-1),
            "block_burn": np.float32(obs["voxels/can_burn"]).reshape(-1),
            "block_light": np.float32(obs["voxels/blocks_light"]).reshape(-1),
            "look_angle": np.float32(obs["voxels/cos_look_vec_angle"]).reshape(-1),
            
            "health": np.float32(obs["life_stats/life"]) / 20,
            "hunger": np.float32(obs["life_stats/food"]) / 20,
            "breath": np.float32(obs["life_stats/oxygen"]) / 300,
            "armor": np.float32(obs["life_stats/armor"]) / 20,
            "xp": np.float32(obs["life_stats/xp"]) / 1395,
            
            "yaw": np.float32(obs["location_stats/yaw"]) / 180,
            "pitch": np.float32(obs["location_stats/pitch"]) / 180,
            "rain": np.float32(obs["location_stats/rainfall"]),
            "temperature": np.float32(obs["location_stats/temperature"]),
            "light" : np.float32(obs["location_stats/light_level"]) / 15,
            "sky_light" : np.float32(obs["location_stats/sky_light_level"]),
            "sun": np.float32(obs["location_stats/sun_brightness"]),
            "sea": np.float32(obs["location_stats/sea_level"]) / 255,

            "damage_amount": np.float32(obs["damage_source/damage_amount"]) / 40,
            "damage_dist": np.float32(obs["damage_source/damage_distance"]),
            "damage_yaw": np.float32(obs["damage_source/damage_yaw"]) / 180,
            "damage_hunger": np.float32(obs["damage_source/hunger_damage"]) / 20,
            "is_explosive": np.array([obs["damage_source/is_explosion"]], np.float32),
            "is_fire": np.array([obs["damage_source/is_fire_damage"]], np.float32),
            "is_projectile": np.array([obs["damage_source/is_projectile"]], np.float32),
            "is_unblockable": np.array([obs["damage_source/is_unblockable"]], np.float32),
            # "reward": [0.0],
            "is_first": obs["is_first"],
            "is_last": obs["is_last"],
            "is_terminal": obs["is_terminal"],
            **{f"log_{k}": obs[k] for k in (self._inv_keys + self.inv_exclude[1:] + self._equip_keys + self.equip_exclude[1:])},
            "pos": obs['location_stats/pos'],
        }
        # print(obs["log_inventory/variant"])
        # print(obs["log_equipment/variant"])

        for key, value in obs.items():
            # print("{}: {}".format(key, value.shape if isinstance(value, np.ndarray) else value))
            # space = self.observation_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            # assert (key, value, value.dtype, value.shape, space)

        return obs

    # def _action(self, action):
    #     if self._sticky_attack_length:
    #         if action["attack"]:
    #             self._sticky_attack_counter = self._sticky_attack_length
    #         if self._sticky_attack_counter > 0:
    #             action["attack"] = 1
    #             action["jump"] = 0
    #             self._sticky_attack_counter -= 1
    #     if self._sticky_jump_length:
    #         if action["jump"]:
    #             self._sticky_jump_counter = self._sticky_jump_length
    #         if self._sticky_jump_counter > 0:
    #             action["jump"] = 1
    #             action["forward"] = 1
    #             self._sticky_jump_counter -= 1
    #     if self._pitch_limit and action["camera"][0]:
    #         lo, hi = self._pitch_limit
    #         if not (lo <= self._pitch + action["camera"][0] <= hi):
    #             action["camera"] = (0, action["camera"][1])
    #         self._pitch += action["camera"][0]
    #     return action

    # def _insert_defaults(self, actions):
    #     actions = {name: action.copy() for name, action in actions.items()}
    #     for key, default in self._noop_action.items():
    #         for action in actions.values():
    #             if key not in action:
    #                 action[key] = default
    #     return actions

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result
