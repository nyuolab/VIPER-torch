import numpy as np
from . import minecraft_base

import gym

def find_index(arr, x):
    idxs,  = np.where(arr == x)
    if len(idxs):    
        return idxs[0]
    return -1
    

MC_ITEM_IDS = [
    "minecraft:air",
    "minecraft:acacia_boat",
    "minecraft:acacia_door",
    "minecraft:acacia_fence",
    "minecraft:acacia_fence_gate",
    "minecraft:acacia_stairs",
    "minecraft:activator_rail",
    "minecraft:anvil",
    "minecraft:apple",
    "minecraft:armor_stand",
    "minecraft:arrow",
    "minecraft:baked_potato",
    "minecraft:banner",
    "minecraft:barrier",
    "minecraft:beacon",
    "minecraft:bed",
    "minecraft:bedrock",
    "minecraft:beef",
    "minecraft:beetroot",
    "minecraft:beetroot_seeds",
    "minecraft:beetroot_soup",
    "minecraft:birch_boat",
    "minecraft:birch_door",
    "minecraft:birch_fence",
    "minecraft:birch_fence_gate",
    "minecraft:birch_stairs",
    "minecraft:black_glazed_terracotta",
    "minecraft:black_shulker_box",
    "minecraft:blaze_powder",
    "minecraft:blaze_rod",
    "minecraft:blue_glazed_terracotta",
    "minecraft:blue_shulker_box",
    "minecraft:boat",
    "minecraft:bone",
    "minecraft:bone_block",
    "minecraft:book",
    "minecraft:bookshelf",
    "minecraft:bow",
    "minecraft:bowl",
    "minecraft:bread",
    "minecraft:brewing_stand",
    "minecraft:brick",
    "minecraft:brick_block",
    "minecraft:brick_stairs",
    "minecraft:brown_glazed_terracotta",
    "minecraft:brown_mushroom",
    "minecraft:brown_mushroom_block",
    "minecraft:brown_shulker_box",
    "minecraft:bucket",
    "minecraft:cactus",
    "minecraft:cake",
    "minecraft:carpet",
    "minecraft:carrot",
    "minecraft:carrot_on_a_stick",
    "minecraft:cauldron",
    "minecraft:chain_command_block",
    "minecraft:chainmail_boots",
    "minecraft:chainmail_chestplate",
    "minecraft:chainmail_helmet",
    "minecraft:chainmail_leggings",
    "minecraft:chest",
    "minecraft:chest_minecart",
    "minecraft:chicken",
    "minecraft:chorus_flower",
    "minecraft:chorus_fruit",
    "minecraft:chorus_fruit_popped",
    "minecraft:chorus_plant",
    "minecraft:clay",
    "minecraft:clay_ball",
    "minecraft:clock",
    "minecraft:coal",
    "minecraft:coal_block",
    "minecraft:coal_ore",
    "minecraft:cobblestone",
    "minecraft:cobblestone_wall",
    "minecraft:command_block",
    "minecraft:command_block_minecart",
    "minecraft:comparator",
    "minecraft:compass",
    "minecraft:concrete",
    "minecraft:concrete_powder",
    "minecraft:cooked_beef",
    "minecraft:cooked_chicken",
    "minecraft:cooked_fish",
    "minecraft:cooked_mutton",
    "minecraft:cooked_porkchop",
    "minecraft:cooked_rabbit",
    "minecraft:cookie",
    "minecraft:crafting_table",
    "minecraft:cyan_glazed_terracotta",
    "minecraft:cyan_shulker_box",
    "minecraft:dark_oak_boat",
    "minecraft:dark_oak_door",
    "minecraft:dark_oak_fence",
    "minecraft:dark_oak_fence_gate",
    "minecraft:dark_oak_stairs",
    "minecraft:daylight_detector",
    "minecraft:deadbush",
    "minecraft:detector_rail",
    "minecraft:diamond",
    "minecraft:diamond_axe",
    "minecraft:diamond_block",
    "minecraft:diamond_boots",
    "minecraft:diamond_chestplate",
    "minecraft:diamond_helmet",
    "minecraft:diamond_hoe",
    "minecraft:diamond_horse_armor",
    "minecraft:diamond_leggings",
    "minecraft:diamond_ore",
    "minecraft:diamond_pickaxe",
    "minecraft:diamond_shovel",
    "minecraft:diamond_sword",
    "minecraft:dirt",
    "minecraft:dispenser",
    "minecraft:double_plant",
    "minecraft:dragon_breath",
    "minecraft:dragon_egg",
    "minecraft:dropper",
    "minecraft:dye",
    "minecraft:egg",
    "minecraft:elytra",
    "minecraft:emerald",
    "minecraft:emerald_block",
    "minecraft:emerald_ore",
    "minecraft:enchanted_book",
    "minecraft:enchanting_table",
    "minecraft:end_bricks",
    "minecraft:end_crystal",
    "minecraft:end_portal_frame",
    "minecraft:end_rod",
    "minecraft:end_stone",
    "minecraft:ender_chest",
    "minecraft:ender_eye",
    "minecraft:ender_pearl",
    "minecraft:experience_bottle",
    "minecraft:farmland",
    "minecraft:feather",
    "minecraft:fence",
    "minecraft:fence_gate",
    "minecraft:fermented_spider_eye",
    "minecraft:filled_map",
    "minecraft:fire_charge",
    "minecraft:firework_charge",
    "minecraft:fireworks",
    "minecraft:fish",
    "minecraft:fishing_rod",
    "minecraft:flint",
    "minecraft:flint_and_steel",
    "minecraft:flower_pot",
    "minecraft:furnace",
    "minecraft:furnace_minecart",
    "minecraft:ghast_tear",
    "minecraft:glass",
    "minecraft:glass_bottle",
    "minecraft:glass_pane",
    "minecraft:glowstone",
    "minecraft:glowstone_dust",
    "minecraft:gold_block",
    "minecraft:gold_ingot",
    "minecraft:gold_nugget",
    "minecraft:gold_ore",
    "minecraft:golden_apple",
    "minecraft:golden_axe",
    "minecraft:golden_boots",
    "minecraft:golden_carrot",
    "minecraft:golden_chestplate",
    "minecraft:golden_helmet",
    "minecraft:golden_hoe",
    "minecraft:golden_horse_armor",
    "minecraft:golden_leggings",
    "minecraft:golden_pickaxe",
    "minecraft:golden_rail",
    "minecraft:golden_shovel",
    "minecraft:golden_sword",
    "minecraft:grass",
    "minecraft:grass_path",
    "minecraft:gravel",
    "minecraft:gray_glazed_terracotta",
    "minecraft:gray_shulker_box",
    "minecraft:green_glazed_terracotta",
    "minecraft:green_shulker_box",
    "minecraft:gunpowder",
    "minecraft:hardened_clay",
    "minecraft:hay_block",
    "minecraft:heavy_weighted_pressure_plate",
    "minecraft:hopper",
    "minecraft:hopper_minecart",
    "minecraft:ice",
    "minecraft:iron_axe",
    "minecraft:iron_bars",
    "minecraft:iron_block",
    "minecraft:iron_boots",
    "minecraft:iron_chestplate",
    "minecraft:iron_door",
    "minecraft:iron_helmet",
    "minecraft:iron_hoe",
    "minecraft:iron_horse_armor",
    "minecraft:iron_ingot",
    "minecraft:iron_leggings",
    "minecraft:iron_nugget",
    "minecraft:iron_ore",
    "minecraft:iron_pickaxe",
    "minecraft:iron_shovel",
    "minecraft:iron_sword",
    "minecraft:iron_trapdoor",
    "minecraft:item_frame",
    "minecraft:jukebox",
    "minecraft:jungle_boat",
    "minecraft:jungle_door",
    "minecraft:jungle_fence",
    "minecraft:jungle_fence_gate",
    "minecraft:jungle_stairs",
    "minecraft:ladder",
    "minecraft:lapis_block",
    "minecraft:lapis_ore",
    "minecraft:lava_bucket",
    "minecraft:lead",
    "minecraft:leather",
    "minecraft:leather_boots",
    "minecraft:leather_chestplate",
    "minecraft:leather_helmet",
    "minecraft:leather_leggings",
    "minecraft:leaves",
    "minecraft:leaves2",
    "minecraft:lever",
    "minecraft:light_blue_glazed_terracotta",
    "minecraft:light_blue_shulker_box",
    "minecraft:light_weighted_pressure_plate",
    "minecraft:lime_glazed_terracotta",
    "minecraft:lime_shulker_box",
    "minecraft:lingering_potion",
    "minecraft:lit_pumpkin",
    "minecraft:log",
    "minecraft:log2",
    "minecraft:magenta_glazed_terracotta",
    "minecraft:magenta_shulker_box",
    "minecraft:magma",
    "minecraft:magma_cream",
    "minecraft:map",
    "minecraft:melon",
    "minecraft:melon_block",
    "minecraft:melon_seeds",
    "minecraft:milk_bucket",
    "minecraft:minecart",
    "minecraft:mob_spawner",
    "minecraft:monster_egg",
    "minecraft:mossy_cobblestone",
    "minecraft:mushroom_stew",
    "minecraft:mutton",
    "minecraft:mycelium",
    "minecraft:name_tag",
    "minecraft:nether_brick",
    "minecraft:nether_brick_fence",
    "minecraft:nether_brick_stairs",
    "minecraft:nether_star",
    "minecraft:nether_wart",
    "minecraft:nether_wart_block",
    "minecraft:netherbrick",
    "minecraft:netherrack",
    "minecraft:noteblock",
    "minecraft:oak_stairs",
    "minecraft:observer",
    "minecraft:obsidian",
    "minecraft:orange_glazed_terracotta",
    "minecraft:orange_shulker_box",
    "minecraft:packed_ice",
    "minecraft:painting",
    "minecraft:paper",
    "minecraft:pink_glazed_terracotta",
    "minecraft:pink_shulker_box",
    "minecraft:piston",
    "minecraft:planks",
    "minecraft:poisonous_potato",
    "minecraft:porkchop",
    "minecraft:potato",
    "minecraft:potion",
    "minecraft:prismarine",
    "minecraft:prismarine_crystals",
    "minecraft:prismarine_shard",
    "minecraft:pumpkin",
    "minecraft:pumpkin_pie",
    "minecraft:pumpkin_seeds",
    "minecraft:purple_glazed_terracotta",
    "minecraft:purple_shulker_box",
    "minecraft:purpur_block",
    "minecraft:purpur_pillar",
    "minecraft:purpur_slab",
    "minecraft:purpur_stairs",
    "minecraft:quartz",
    "minecraft:quartz_block",
    "minecraft:quartz_ore",
    "minecraft:quartz_stairs",
    "minecraft:rabbit",
    "minecraft:rabbit_foot",
    "minecraft:rabbit_hide",
    "minecraft:rabbit_stew",
    "minecraft:rail",
    "minecraft:record_11",
    "minecraft:record_13",
    "minecraft:record_blocks",
    "minecraft:record_cat",
    "minecraft:record_chirp",
    "minecraft:record_far",
    "minecraft:record_mall",
    "minecraft:record_mellohi",
    "minecraft:record_stal",
    "minecraft:record_strad",
    "minecraft:record_wait",
    "minecraft:record_ward",
    "minecraft:red_flower",
    "minecraft:red_glazed_terracotta",
    "minecraft:red_mushroom",
    "minecraft:red_mushroom_block",
    "minecraft:red_nether_brick",
    "minecraft:red_sandstone",
    "minecraft:red_sandstone_stairs",
    "minecraft:red_shulker_box",
    "minecraft:redstone",
    "minecraft:redstone_block",
    "minecraft:redstone_lamp",
    "minecraft:redstone_ore",
    "minecraft:redstone_torch",
    "minecraft:reeds",
    "minecraft:repeater",
    "minecraft:repeating_command_block",
    "minecraft:rotten_flesh",
    "minecraft:saddle",
    "minecraft:sand",
    "minecraft:sandstone",
    "minecraft:sandstone_stairs",
    "minecraft:sapling",
    "minecraft:sea_lantern",
    "minecraft:shears",
    "minecraft:shield",
    "minecraft:shulker_shell",
    "minecraft:sign",
    "minecraft:silver_glazed_terracotta",
    "minecraft:silver_shulker_box",
    "minecraft:skull",
    "minecraft:slime",
    "minecraft:slime_ball",
    "minecraft:snow",
    "minecraft:snow_layer",
    "minecraft:snowball",
    "minecraft:soul_sand",
    "minecraft:spawn_egg",
    "minecraft:speckled_melon",
    "minecraft:spectral_arrow",
    "minecraft:spider_eye",
    "minecraft:splash_potion",
    "minecraft:sponge",
    "minecraft:spruce_boat",
    "minecraft:spruce_door",
    "minecraft:spruce_fence",
    "minecraft:spruce_fence_gate",
    "minecraft:spruce_stairs",
    "minecraft:stained_glass",
    "minecraft:stained_glass_pane",
    "minecraft:stained_hardened_clay",
    "minecraft:stick",
    "minecraft:sticky_piston",
    "minecraft:stone",
    "minecraft:stone_axe",
    "minecraft:stone_brick_stairs",
    "minecraft:stone_button",
    "minecraft:stone_hoe",
    "minecraft:stone_pickaxe",
    "minecraft:stone_pressure_plate",
    "minecraft:stone_shovel",
    "minecraft:stone_slab",
    "minecraft:stone_slab2",
    "minecraft:stone_stairs",
    "minecraft:stone_sword",
    "minecraft:stonebrick",
    "minecraft:string",
    "minecraft:structure_block",
    "minecraft:structure_void",
    "minecraft:sugar",
    "minecraft:tallgrass",
    "minecraft:tipped_arrow",
    "minecraft:tnt",
    "minecraft:tnt_minecart",
    "minecraft:torch",
    "minecraft:totem_of_undying",
    "minecraft:trapdoor",
    "minecraft:trapped_chest",
    "minecraft:tripwire_hook",
    "minecraft:vine",
    "minecraft:water_bucket",
    "minecraft:waterlily",
    "minecraft:web",
    "minecraft:wheat",
    "minecraft:wheat_seeds",
    "minecraft:white_glazed_terracotta",
    "minecraft:white_shulker_box",
    "minecraft:wooden_axe",
    "minecraft:wooden_button",
    "minecraft:wooden_door",
    "minecraft:wooden_hoe",
    "minecraft:wooden_pickaxe",
    "minecraft:wooden_pressure_plate",
    "minecraft:wooden_shovel",
    "minecraft:wooden_slab",
    "minecraft:wooden_sword",
    "minecraft:wool",
    "minecraft:writable_book",
    "minecraft:written_book",
    "minecraft:yellow_flower",
    "minecraft:yellow_glazed_terracotta",
    "minecraft:yellow_shulker_box",
]

N = len(MC_ITEM_IDS)

mc_item2id = {}
for i in range(N):
    mc_item2id[MC_ITEM_IDS[i][10:]] = i

# def make_env(task, *args, **kwargs):
#     return {
#         "wood": MinecraftWood,
#         "climb": MinecraftClimb,
#         "diamond": MinecraftDiamond,
#     }[task](*args, **kwargs)

def make_env(task, *args, **kwargs):
    return Minecraft(task, *args, **kwargs)


class MinecraftWood:
    def __init__(self, *args, **kwargs):
        actions = BASIC_ACTIONS
        self.rewards = [
            CollectReward("log", repeated=1),
            HealthReward(),
        ]
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        # obs["reward"] = reward
        return obs, reward, done, info


class MinecraftClimb:
    def __init__(self, *args, **kwargs):
        actions = BASIC_ACTIONS
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
        self._previous = None
        self._health_reward = HealthReward()
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x, y, z = obs["pos"]
        height = np.float32(y)
        if obs["is_first"]:
            self._previous = height
        reward = height - self._previous
        reward += self._health_reward(obs)
        # obs["reward"] = reward
        self._previous = height
        return obs, reward, done, info


# class MinecraftDiamond(gym.Wrapper):
#     def __init__(self, task, *args, **kwargs):
#         # actions = {
#         #     **BASIC_ACTIONS,
#         #     "craft_planks": dict(craft="planks"),
#         #     "craft_stick": dict(craft="stick"),
#         #     "craft_crafting_table": dict(craft="crafting_table"),
#         #     "place_crafting_table": dict(place="crafting_table"),
#         #     "craft_wooden_pickaxe": dict(nearbyCraft="wooden_pickaxe"),
#         #     "craft_stone_pickaxe": dict(nearbyCraft="stone_pickaxe"),
#         #     "craft_iron_pickaxe": dict(nearbyCraft="iron_pickaxe"),
#         #     "equip_stone_pickaxe": dict(equip="stone_pickaxe"),
#         #     "equip_wooden_pickaxe": dict(equip="wooden_pickaxe"),
#         #     "equip_iron_pickaxe": dict(equip="iron_pickaxe"),
#         #     "craft_furnace": dict(nearbyCraft="furnace"),
#         #     "place_furnace": dict(place="furnace"),
#         #     "smelt_iron_ingot": dict(nearbySmelt="iron_ingot"),
#         # }

#         self.items = [
#             "log",
#             "planks",
#             "stick",
#             "crafting_table",
#             "wooden_pickaxe",
#             "cobblestone",
#             "stone_pickaxe",
#             "iron_ore",
#             "furnace",
#             "iron_ingot",
#             "iron_pickaxe",
#             "diamond",
#         ]
#         self.rewards = [CollectReward(item, once=1) for item in self.items] + [
#             HealthReward()
#         ]
#         env = minecraft_base.MinecraftBase(task, *args, **kwargs)
#         super().__init__(env)

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
#         # obs["reward"] = reward
#         # restrict log for memory save
#         obs = {
#             k: v
#             for k, v in obs.items()
#             # if "log" not in k or k.split("/")[-1] in self.items
#         }
#         return obs, reward, done, info

#     def reset(self):
#         obs = self.env.reset()
#         # called for reset of reward calculations
#         # _ = sum([fn(obs, self.env.inventory) for fn in self.rewards])
#         # restrict log for memory save
#         obs = {
#             k: v
#             for k, v in obs.items()
#             # if "log" not in k or k.split("/")[-1] in self.items
#         }
#         return obs

class Minecraft(gym.Wrapper):
    def __init__(self, task, *args, **kwargs):
        env = minecraft_base.MinecraftBase(task, *args, **kwargs)
        super().__init__(env)
        self.items = [
            "log",
            "planks",
            "stick",
            "crafting_table",
            "wooden_pickaxe",
            "cobblestone",
            "stone_pickaxe",
            "iron_ore",
            "furnace",
            "iron_ingot",
            "iron_pickaxe",
            "diamond",
        ]
        
        self.other_rewards = [CollectReward(item, once=1) for item in self.items] + [HealthReward()]


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += sum([fn(obs, self.env.inventory) for fn in self.other_rewards])
        # restrict log for memory save
        obs = {
            k: v
            for k, v in obs.items()
            # if "log" not in k or k.split("/")[-1] in self.items
        }
        if obs["health"] <= 0.0:
            reward = -10
            print("WASTED!!!")
            # done = True

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # called for reset of reward calculations
        _ = sum([fn(obs, self.env.inventory) for fn in self.other_rewards])
        
        # restrict log for memory save
        obs = {
            k: v
            for k, v in obs.items()
            # if "log" not in k or k.split("/")[-1] in self.items
        }
        return obs


class CollectReward:
    def __init__(self, item, once=0, repeated=0):
        self.item = item
        self.item_id = mc_item2id[self.item]
        self.once = once
        self.repeated = repeated
        self.previous = 0
        # self.maximum = 0

    def __call__(self, obs, inventory):
        item_inv_idx = find_index(inventory["id"], self.item_id)
        current = inventory["quantity"][item_inv_idx] if item_inv_idx >= 0 else 0
        
        # if the useful item is found
        if item_inv_idx >= 0:
            print("The useful item name is {}".format(inventory["name"][item_inv_idx]))
            print("The useful item id is {}".format(inventory["id"][item_inv_idx]))
            print("The useful item quantity is {}".format(inventory["quantity"][item_inv_idx]))
        
        if obs["is_first"]:
            self.previous = current
            # self.maximum = current
            return 0
        
        reward = self.repeated * max(0, current - self.previous)
        # if self.maximum == 0 and current > 0:

        # if this item was not collected before
        if (current > self.previous) and (not self.previous):
            reward += self.once
        
        # if we lose the item
        elif (not current) and self.previous:
            print("The item {} is lost!".format(self.item))
            reward -= self.once
        self.previous = current
        # self.maximum = max(self.maximum, current)
        return reward


class HealthReward:
    def __init__(self, scale=0.01):
        self.scale = scale
        self.previous = None

    def __call__(self, obs, inventory=None):
        health = obs["health"]

        # env rest or just revived
        if obs["is_first"] or (self.previous <= 0):
            self.previous = health
            return 0
        delta_health = health - self.previous
        reward = self.scale * delta_health
        if reward > 0:
            print("Gain {} health!!!".format(delta_health))
        elif reward < 0:
            print("Lose {} health!!!".format(delta_health))
        self.previous = health
        return sum(reward)


BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=1),
    "turn_up": dict(camera=(-15, 0)),
    "turn_down": dict(camera=(15, 0)),
    "turn_left": dict(camera=(0, -15)),
    "turn_right": dict(camera=(0, 15)),
    "forward": dict(forward=1),
    "back": dict(back=1),
    "left": dict(left=1),
    "right": dict(right=1),
    "jump": dict(jump=1, forward=1),
    "place_dirt": dict(place="dirt"),
}
