"""
六边形地图策略游戏 - 强化学习训练模块（优化版）
使用Deep Q-Learning (DQN)算法训练智能体寻找最优策略
重点优化：奖励函数、动作空间、训练策略
"""
import sys
import os

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import json
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import math


# ============================================================================
# 地形类型定义
# ============================================================================
class TerrainType(Enum):
    """地形类型枚举"""
    START_POSITION = 0
    NORMAL_LV1 = 1
    NORMAL_LV2 = 2
    NORMAL_LV3 = 3
    NORMAL_LV4 = 4
    NORMAL_LV5 = 5
    NORMAL_LV6 = 6
    DUMMY_LV1 = 7
    DUMMY_LV2 = 8
    DUMMY_LV3 = 9
    DUMMY_LV4 = 10
    DUMMY_LV5 = 11
    DUMMY_LV6 = 12
    TRAINING_GROUND = 13
    WATCHTOWER_LV1 = 14
    WATCHTOWER_LV2 = 15
    WATCHTOWER_LV3 = 16
    WATCHTOWER_LV4 = 17
    WATCHTOWER_LV5 = 18
    WATCHTOWER_LV6 = 19
    BLACK_MARKET = 20
    RELIC_STONE = 21
    TREASURE_1 = 22
    TREASURE_2 = 23
    TREASURE_3 = 24
    TREASURE_4 = 25
    TREASURE_5 = 26
    TREASURE_6 = 27
    TREASURE_7 = 28
    TREASURE_8 = 29
    WALL = 30
    BOSS_GAARA = 31
    BOSS_ZETSU = 32
    BOSS_DART = 33
    BOSS_SHIRA = 34
    BOSS_KUSHINA = 35
    BOSS_KISAME = 36
    BOSS_HANA = 37
    TENT = 38
    AKATSUKI_TREASURE = 39
    KONOHA_TREASURE_1 = 40
    KONOHA_TREASURE_2 = 41
    BOUNDARY = 42


# ============================================================================
# 改进的游戏环境
# ============================================================================
class ImprovedHexGameEnvironment:
    """改进的六边形地图游戏环境"""

    # 地块属性定义
    TERRAIN_PROPERTIES = {
        TerrainType.START_POSITION: {"food_cost": 0, "exp_gain": 0, "score_cost": 0},
        TerrainType.NORMAL_LV1: {"food_cost": 100, "exp_gain": 35, "score_cost": 0},
        TerrainType.NORMAL_LV2: {"food_cost": 110, "exp_gain": 40, "score_cost": 0},
        TerrainType.NORMAL_LV3: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.NORMAL_LV4: {"food_cost": 130, "exp_gain": 50, "score_cost": 0},
        TerrainType.NORMAL_LV5: {"food_cost": 140, "exp_gain": 55, "score_cost": 0},
        TerrainType.NORMAL_LV6: {"food_cost": 150, "exp_gain": 60, "score_cost": 0},
        TerrainType.DUMMY_LV1: {"food_cost": 100, "exp_gain": 35, "score_cost": 0},
        TerrainType.DUMMY_LV2: {"food_cost": 100, "exp_gain": 40, "score_cost": 0},
        TerrainType.DUMMY_LV3: {"food_cost": 100, "exp_gain": 45, "score_cost": 0},
        TerrainType.DUMMY_LV4: {"food_cost": 100, "exp_gain": 50, "score_cost": 0},
        TerrainType.DUMMY_LV5: {"food_cost": 100, "exp_gain": 55, "score_cost": 0},
        TerrainType.DUMMY_LV6: {"food_cost": 100, "exp_gain": 60, "score_cost": 0},
        TerrainType.TRAINING_GROUND: {"food_cost": 0, "exp_gain": 520, "score_cost": 0},
        TerrainType.WATCHTOWER_LV1: {"food_cost": 100, "exp_gain": 35, "score_cost": 0},
        TerrainType.WATCHTOWER_LV2: {"food_cost": 100, "exp_gain": 40, "score_cost": 0},
        TerrainType.WATCHTOWER_LV3: {"food_cost": 100, "exp_gain": 45, "score_cost": 0},
        TerrainType.WATCHTOWER_LV4: {"food_cost": 100, "exp_gain": 50, "score_cost": 0},
        TerrainType.WATCHTOWER_LV5: {"food_cost": 100, "exp_gain": 55, "score_cost": 0},
        TerrainType.WATCHTOWER_LV6: {"food_cost": 100, "exp_gain": 60, "score_cost": 0},
        TerrainType.BLACK_MARKET: {"food_cost": 0, "exp_gain": 30, "score_cost": 1000},
        TerrainType.RELIC_STONE: {"food_cost": 100, "exp_gain": 40, "score_cost": 0},
        TerrainType.TREASURE_1: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.TREASURE_2: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.TREASURE_3: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.TREASURE_4: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.TREASURE_5: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.TREASURE_6: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.TREASURE_7: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.TREASURE_8: {"food_cost": 120, "exp_gain": 45, "score_cost": 0},
        TerrainType.WALL: {"food_cost": -1, "exp_gain": 0, "score_cost": 0},
        TerrainType.BOUNDARY: {"food_cost": -1, "exp_gain": 0, "score_cost": 0},
        TerrainType.BOSS_GAARA: {"food_cost": 200, "exp_gain": 500, "score_cost": 0},
        TerrainType.BOSS_ZETSU: {"food_cost": 200, "exp_gain": 1000, "score_cost": 0, "has_thunder": True},
        TerrainType.BOSS_KUSHINA: {"food_cost": 200, "exp_gain": 1000, "score_cost": 0, "has_thunder": True},
        TerrainType.BOSS_KISAME: {"food_cost": 200, "exp_gain": 500, "score_cost": 0},
        TerrainType.BOSS_DART: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.BOSS_SHIRA: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.BOSS_HANA: {"food_cost": 200, "exp_gain": 300, "score_cost": 0},
        TerrainType.TENT: {"food_cost": 0, "exp_gain": 0, "score_cost": 0, "is_tent": True},
        TerrainType.AKATSUKI_TREASURE: {"food_cost": 130, "exp_gain": 50, "score_cost": 0},
        TerrainType.KONOHA_TREASURE_1: {"food_cost": 110, "exp_gain": 40, "score_cost": 0},
        TerrainType.KONOHA_TREASURE_2: {"food_cost": 100, "exp_gain": 35, "score_cost": 0},
    }

    def __init__(self, map_file_path="map_save.json"):
        self.map_data = self.load_map(map_file_path)
        self.action_space = 50
        self.key_positions = self._identify_key_positions()
        self.exp_milestones = set(range(3000, 63000, 3000))  # 每3000经验的里程碑

        # 添加队伍切换跟踪 - 确保所有属性都初始化
        self.daily_switch_count = 0  # 每日切换次数
        self.max_daily_switches = 5  # 每日最大切换次数
        self.last_switch_team = -1  # 上次切换的队伍ID
        self.switch_action_tracking = {}  # 跟踪切换后的行动
        self.valid_actions_cache = {}
        self.reset()

    def _identify_key_positions(self):
        """识别地图上的关键位置"""
        positions = {
            'strong_bosses': [],
            'normal_bosses': [],
            'treasures': [],
            'training': [],
            'tents': [],
            'black_markets': [],
            'high_value': [],
            'mid_value': []
        }

        for pos, tile_type in self.map_data.items():
            props = self.TERRAIN_PROPERTIES.get(tile_type, {})
            exp_gain = props.get('exp_gain', 0)

            if tile_type in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
                positions['strong_bosses'].append(pos)
            elif 'BOSS' in tile_type.name:
                positions['normal_bosses'].append(pos)
            elif 22 <= tile_type.value <= 29:
                positions['treasures'].append(pos)
            elif tile_type == TerrainType.TRAINING_GROUND:
                positions['training'].append(pos)
            elif tile_type == TerrainType.TENT:
                positions['tents'].append(pos)
            elif tile_type == TerrainType.BLACK_MARKET:
                positions['black_markets'].append(pos)

            # 按经验值分类
            if exp_gain >= 500:
                positions['high_value'].append(pos)
            elif exp_gain >= 100:
                positions['mid_value'].append(pos)

        return positions

    def load_map(self, map_file_path):
        """加载地图数据"""
        if not os.path.exists(map_file_path):
            raise FileNotFoundError(f"地图文件 {map_file_path} 不存在！请先使用地图编辑器创建地图。")

        with open(map_file_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
            hex_map = {}
            for tile_data in save_data['tiles']:
                q, r = tile_data['q'], tile_data['r']
                terrain_type = TerrainType(tile_data['terrain_type'])
                hex_map[(q, r)] = terrain_type

            print(f"成功加载地图：{len(hex_map)} 个地块")
            return hex_map

    def reset(self):
        """重置游戏状态"""
        start_pos = None
        for pos, tile_type in self.map_data.items():
            if tile_type == TerrainType.START_POSITION:
                start_pos = pos
                break

        if not start_pos:
            print("WARNING: No START_POSITION found in map!")
            for pos, tile_type in self.map_data.items():
                if tile_type != TerrainType.WALL and tile_type != TerrainType.BOUNDARY:
                    start_pos = pos
                    print(f"Using fallback position: {pos}")
                    break

        self.state = {
            'day': 1,
            'level': 1,
            'experience': 0,
            'food': 6800,
            'conquest_score': 1000,
            'thunder_god_items': 1,
            'teams': [
                {'id': 0, 'position': start_pos, 'action_points': 6}
            ],
            'current_team': 0,
            'num_teams': 1,
            'conquered_tiles': {start_pos},
            'treasures_conquered': set(),
            'has_treasure_buff': False,
            'weekly_exp_quota': 500,
            'weekly_exp_claimed': 0,
            'total_conquered': 1,
            'reached_milestones': set(),  # 记录已达到的经验里程碑
            'high_value_conquered': 0,  # 高价值地块征服数
            'efficiency_score': 0,  # 效率分数
            # 添加食物跟踪变量
            'food_income_total': 0,
            'food_from_tents': 0,
            'food_spent_move': 0,
            'food_spent_conquer': 0,
            'move_attempts': 0,
            'conquer_attempts': 0
        }
        # 初始化历史记录（跨episode保持）
        if not hasattr(self, 'historical_best_exp'):
            self.historical_best_exp = 0
        if not hasattr(self, 'historical_best_efficiency'):
            self.historical_best_efficiency = 0.0

        # 重置episode统计
        self.episode_action_stats = {
            'total_actions': 0,
            'effective_actions': 0,
            'successful_moves': 0,
            'successful_conquers': 0,
            'next_day_actions': 0,
            'invalid_actions': 0
        }
        # 重置切换跟踪
        self.daily_switch_count = 0
        self.last_switch_team = -1
        self.switch_action_tracking = {}

        self.valid_actions_cache = {}
        if not hasattr(self, 'historical_best_exp'):
            self.historical_best_exp = 0
        return self.get_observation()

    def get_observation(self):
        """获取状态观察向量 - 添加切换相关特征"""
        # 基础特征
        base_features = np.array([
            self.state['day'] / 91.0,
            self.state['level'] / 200.0,
            self.state['experience'] / 60000.0,
            self.state['food'] / 10000.0,
            self.state['conquest_score'] / 10000.0,
            self.state['thunder_god_items'] / 5.0,
            len(self.state['treasures_conquered']) / 8.0,
            float(self.state['has_treasure_buff']),
            self.state['weekly_exp_quota'] / 500.0,
            self.state['total_conquered'] / len(self.map_data),
            self.state['high_value_conquered'] / max(len(self.key_positions['high_value']), 1),
            len(self.state['reached_milestones']) / 20.0,
            # 新增切换相关特征
            self.daily_switch_count / self.max_daily_switches,  # 今日切换使用比例
            len(self.switch_action_tracking) / self.state['num_teams'],  # 待验证切换比例
        ])

        # 队伍特征
        team_features = []
        for i in range(3):
            if i < len(self.state['teams']):
                team = self.state['teams'][i]
                team_features.extend([
                    team['action_points'] / 18.0,
                    float(i == self.state['current_team']),
                    1.0
                ])
            else:
                team_features.extend([0.0, 0.0, 0.0])

        # 地图特征
        map_features = self.get_map_features()

        # 距离特征
        distance_features = self.get_distance_features()

        return np.concatenate([
            base_features,
            team_features,
            map_features,
            distance_features
        ])

    def get_map_features(self):
        """获取地图统计特征"""
        features = []

        # 各类地块的征服进度
        for category in ['high_value', 'mid_value', 'strong_bosses', 'tents', 'black_markets']:
            if self.key_positions.get(category):
                conquered = sum(1 for pos in self.key_positions[category]
                              if pos in self.state['conquered_tiles'])
                features.append(conquered / len(self.key_positions[category]))
            else:
                features.append(0.0)

        # 当前位置附近的未征服高价值地块数
        current_team = self.state['teams'][self.state['current_team']]
        nearby_high_value = self.count_nearby_high_value(current_team['position'], radius=5)
        features.append(nearby_high_value / 10.0)

        # 效率指标
        if self.state['total_conquered'] > 0:
            avg_exp_per_tile = self.state['experience'] / self.state['total_conquered']
            features.append(min(avg_exp_per_tile / 200.0, 1.0))
        else:
            features.append(0.0)

        return np.array(features)

    def get_distance_features(self):
        """获取到关键位置的距离特征"""
        current_team = self.state['teams'][self.state['current_team']]
        q, r = current_team['position']
        features = []

        # 到最近高价值未征服地块的距离
        high_value_unconquered = [pos for pos in self.key_positions['high_value']
                                 if pos not in self.state['conquered_tiles']]
        if high_value_unconquered:
            min_dist = min(self.hex_distance(q, r, tq, tr) for tq, tr in high_value_unconquered)
            features.append(np.exp(-min_dist / 10.0))
        else:
            features.append(0.0)

        # 到最近BOSS的距离
        all_bosses = self.key_positions['strong_bosses'] + self.key_positions['normal_bosses']
        unconquered_bosses = [pos for pos in all_bosses if pos not in self.state['conquered_tiles']]
        if unconquered_bosses:
            min_dist = min(self.hex_distance(q, r, tq, tr) for tq, tr in unconquered_bosses)
            features.append(np.exp(-min_dist / 10.0))
        else:
            features.append(0.0)

        # 到最近训练场的距离
        training_unconquered = [pos for pos in self.key_positions['training']
                              if pos not in self.state['conquered_tiles']]
        if training_unconquered:
            min_dist = min(self.hex_distance(q, r, tq, tr) for tq, tr in training_unconquered)
            features.append(np.exp(-min_dist / 10.0))
        else:
            features.append(0.0)

        # 填充到固定长度
        while len(features) < 5:
            features.append(0.0)

        return np.array(features[:5])

    def hex_distance(self, q1, r1, q2, r2):
        """计算六边形距离"""
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

    def count_nearby_high_value(self, position, radius=5):
        """统计附近高价值未征服地块数量"""
        q, r = position
        count = 0
        for pos in self.key_positions['high_value']:
            if pos not in self.state['conquered_tiles']:
                if self.hex_distance(q, r, pos[0], pos[1]) <= radius:
                    count += 1
        return count

    def execute_team_switch(self, team_idx):
        """更智能的队伍切换机制 - 基于机会价值判断"""
        # 基本条件检查
        if self.state['num_teams'] <= 1:
            return -1.0  # 只有一个队伍时不能切换

        if team_idx >= self.state['num_teams'] or team_idx == self.state['current_team']:
            return -0.8  # 无效切换

        # 检查每日切换次数限制
        if self.daily_switch_count >= self.max_daily_switches:
            return -1.5  # 超过每日切换上限

        current_team = self.state['teams'][self.state['current_team']]
        target_team = self.state['teams'][team_idx]

        # 目标队伍没有行动点时严重惩罚
        if target_team['action_points'] <= 0:
            return -1.2

        # 评估当前队伍的机会价值
        current_opportunity_value = self._evaluate_team_opportunity(self.state['current_team'])

        # 评估目标队伍的机会价值
        target_opportunity_value = self._evaluate_team_opportunity(team_idx)

        # 计算切换的价值差
        value_difference = target_opportunity_value - current_opportunity_value

        # 如果当前队伍有高价值机会而目标队伍价值更低，重惩罚
        if current_opportunity_value >= 2.0 and value_difference < -1.0:
            return -2.0

        # 如果目标队伍有明显更高的价值，给予奖励
        if value_difference >= 1.5:
            switch_reward = min(value_difference * 0.5, 2.0)
        elif value_difference >= 0.5:
            switch_reward = value_difference * 0.3
        else:
            switch_reward = -0.4  # 没有明显价值提升的切换给予小惩罚

        # 检查未完成的切换惩罚
        switch_penalty = 0
        if self.last_switch_team in self.switch_action_tracking:
            switch_penalty = -0.4  # 上次切换后没有进行有效行动
            del self.switch_action_tracking[self.last_switch_team]

        # 执行切换
        old_team = self.state['current_team']
        self.state['current_team'] = team_idx
        self.daily_switch_count += 1

        # 记录这次切换，等待后续行动验证
        self.last_switch_team = old_team
        self.switch_action_tracking[old_team] = True

        # 频繁切换的额外惩罚
        frequency_penalty = 0
        if self.daily_switch_count > 5:
            frequency_penalty = (self.daily_switch_count - 5) * 0.2

        final_reward = switch_reward + switch_penalty - frequency_penalty
        return np.clip(final_reward, -3.0, 2.0)

    def _evaluate_team_opportunity(self, team_idx):
        """评估队伍的机会价值"""
        if team_idx >= len(self.state['teams']):
            return 0.0

        team = self.state['teams'][team_idx]
        if team['action_points'] <= 0:
            return 0.0

        pos = team['position']
        max_value = 0.0

        # 检查当前位置的征服价值
        if pos not in self.state['conquered_tiles']:
            tile_props = self.TERRAIN_PROPERTIES.get(self.map_data[pos], {})
            food_cost = tile_props.get('food_cost', 0)
            exp_gain = tile_props.get('exp_gain', 0)

            if self.state['has_treasure_buff']:
                food_cost = int(food_cost * 0.8)

            if food_cost >= 0 and food_cost <= self.state['food']:
                # 将经验转换为价值分数
                if exp_gain >= 500:
                    max_value = 3.0
                elif exp_gain >= 300:
                    max_value = 2.5
                elif exp_gain >= 100:
                    max_value = 1.5
                elif exp_gain >= 50:
                    max_value = 1.0
                else:
                    max_value = 0.5

        # 检查移动到高价值目标的机会
        elif pos in self.state['conquered_tiles']:
            reachable = self.get_reachable_tiles(team_idx)

            for target_pos in reachable[:5]:  # 检查前5个目标
                target_type = self.map_data[target_pos]
                target_props = self.TERRAIN_PROPERTIES.get(target_type, {})
                target_exp = target_props.get('exp_gain', 0)

                # 计算移动成本
                if target_pos in self.get_neighbors(*pos):
                    move_cost = 40 if self.state['has_treasure_buff'] else 50
                else:
                    distance = self.hex_distance(pos[0], pos[1], target_pos[0], target_pos[1])
                    move_cost = 30 + 10 * distance
                    if self.state['has_treasure_buff']:
                        move_cost = int(move_cost * 0.8)

                if move_cost <= self.state['food']:
                    # 计算移动+征服的综合价值
                    if target_exp >= 500:
                        move_value = 2.5
                    elif target_exp >= 300:
                        move_value = 2.0
                    elif target_exp >= 150:
                        move_value = 1.3
                    elif target_exp >= 100:
                        move_value = 1.0
                    else:
                        move_value = 0.3

                    # 考虑移动成本对价值的影响
                    cost_factor = max(0.5, 1.0 - (move_cost / 500.0))
                    adjusted_value = move_value * cost_factor

                    max_value = max(max_value, adjusted_value)

        # 行动点加成
        action_bonus = min(team['action_points'] / 18.0, 1.0) * 0.3

        return max_value + action_bonus

    def get_neighbors(self, q, r):
        """获取六边形的邻居坐标"""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in directions:
            neighbor = (q + dq, r + dr)
            if neighbor in self.map_data:
                neighbors.append(neighbor)
        return neighbors

    def get_reachable_tiles(self, team_idx):
        """获取队伍可到达的所有地块（优化版）"""
        team = self.state['teams'][team_idx]
        q, r = team['position']
        reachable = []

        # 1. 相邻的未征服地块
        for neighbor in self.get_neighbors(q, r):
            if neighbor not in self.state['conquered_tiles'] and \
               self.map_data.get(neighbor) != TerrainType.WALL:
                reachable.append(neighbor)

        # 2. 可跳跃到的地块（与已征服区域相邻的未征服地块）
        max_jump_distance = 8  # 增加搜索范围
        for pos in self.map_data:
            if pos not in self.state['conquered_tiles'] and \
               self.map_data[pos] != TerrainType.WALL:
                dist = self.hex_distance(q, r, pos[0], pos[1])
                if 2 <= dist <= max_jump_distance:
                    if any(n in self.state['conquered_tiles'] for n in self.get_neighbors(pos[0], pos[1])):
                        reachable.append(pos)

        # 按潜在经验值排序，优先高价值目标
        def get_tile_value(pos):
            tile_type = self.map_data[pos]
            props = self.TERRAIN_PROPERTIES.get(tile_type, {})
            exp_gain = props.get('exp_gain', 0)

            # 给特殊地块额外权重
            if tile_type in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
                return exp_gain * 2
            elif tile_type == TerrainType.TRAINING_GROUND:
                return exp_gain * 1.5
            elif tile_type == TerrainType.TENT:
                return 300  # 帐篷也很重要
            else:
                return exp_gain

        reachable.sort(key=get_tile_value, reverse=True)

        # 返回前30个最高价值目标
        return reachable[:30]

    def step(self, action):
        """执行动作并返回新状态 - 针对激进扩张策略优化的突破奖励机制"""
        reward = 0
        done = False
        info = {}

        old_conquered = self.state['conquered_tiles'].copy()
        old_exp = self.state['experience']
        old_level = self.state['level']
        old_milestones = self.state['reached_milestones'].copy()

        current_team = self.state['teams'][self.state['current_team']]

        # 跟踪动作效果（用于效率计算）
        if not hasattr(self, 'episode_action_stats'):
            self.episode_action_stats = {
                'total_actions': 0,
                'effective_actions': 0,
                'successful_moves': 0,
                'successful_conquers': 0,
                'next_day_actions': 0,
                'invalid_actions': 0
            }

        self.episode_action_stats['total_actions'] += 1
        action_was_effective = False

        # 检查是否是切换后的首次有效行动
        switch_bonus = 0
        if self.last_switch_team in self.switch_action_tracking:
            if action < 30 or action == 30:  # 移动或征服动作
                switch_bonus = 0.5
                del self.switch_action_tracking[self.last_switch_team]
                self.last_switch_team = -1

        # 记录执行前状态用于效果判断
        old_position = current_team['position']
        old_conquered_count = len(self.state['conquered_tiles'])

        # 执行动作
        if action < 30:  # 移动动作
            reachable = self.get_reachable_tiles(self.state['current_team'])
            if action < len(reachable):
                target = reachable[action]
                if target not in self.state['conquered_tiles']:
                    reward = self.execute_move(self.state['current_team'], target)
                    # 检查是否成功移动
                    if current_team['position'] != old_position:
                        self.episode_action_stats['successful_moves'] += 1
                        action_was_effective = True
                    else:
                        self.episode_action_stats['invalid_actions'] += 1
                else:
                    reward = -0.5
                    self.episode_action_stats['invalid_actions'] += 1
            else:
                reward = -0.2
                self.episode_action_stats['invalid_actions'] += 1

        elif action == 30:  # 征服
            reward = self.execute_conquer(self.state['current_team'])
            # 检查是否成功征服
            if len(self.state['conquered_tiles']) > old_conquered_count:
                self.episode_action_stats['successful_conquers'] += 1
                action_was_effective = True
            else:
                self.episode_action_stats['invalid_actions'] += 1

        elif action == 31:  # 飞雷神
            old_thunder = self.state['thunder_god_items']
            reward = self.execute_thunder_god(self.state['current_team'])
            # 飞雷神使用成功就算有效
            if self.state['thunder_god_items'] < old_thunder:
                action_was_effective = True

        elif action == 32:  # 下一天
            reward = self.next_day()
            self.episode_action_stats['next_day_actions'] += 1
            action_was_effective = True  # 下一天算有效动作

        elif action == 33:  # 领取周经验
            old_quota = self.state['weekly_exp_quota']
            reward = self.claim_weekly_exp()
            # 成功领取经验就算有效
            if self.state['weekly_exp_quota'] < old_quota:
                action_was_effective = True

        elif action >= 34 and action < 37:  # 切换队伍
            reward = self.execute_team_switch(action - 34)
            # 切换队伍一般不算特别有效，除非奖励为正
            if reward > 0:
                action_was_effective = True

        # 更新有效动作计数
        if action_was_effective:
            self.episode_action_stats['effective_actions'] += 1

        # 添加切换后行动奖励
        reward += switch_bonus

        # 计算综合奖励
        reward = self.calculate_reward(reward, old_conquered, old_exp, old_level, old_milestones)

        # ========== 基于激进扩张策略的突破奖励 ==========
        current_exp = self.state['experience']
        current_day = self.state['day']

        # 计算日均经验
        daily_avg_exp = current_exp / max(current_day, 1)

        # 初始化历史最佳日均经验
        if not hasattr(self, 'historical_best_daily_avg'):
            self.historical_best_daily_avg = 0

        # 检测日均经验突破
        if daily_avg_exp > self.historical_best_daily_avg:
            improvement = daily_avg_exp - self.historical_best_daily_avg

            # 激进扩张下的合理档位
            if improvement >= 100:  # 日均提升100+（重大策略改进）
                breakthrough_bonus = 15.0
            elif improvement >= 50:  # 日均提升50-100
                breakthrough_bonus = 10.0
            elif improvement >= 25:  # 日均提升25-50
                breakthrough_bonus = 5.0 + (improvement - 25) / 10.0
            elif improvement >= 10:  # 日均提升10-25
                breakthrough_bonus = 2.0 + (improvement - 10) / 5.0
            else:  # 日均提升0-10
                breakthrough_bonus = improvement / 5.0

            breakthrough_bonus = min(breakthrough_bonus, 20.0)
            reward += breakthrough_bonus
            self.historical_best_daily_avg = daily_avg_exp

            info['daily_avg_breakthrough'] = {
                'improvement': improvement,
                'bonus': breakthrough_bonus,
                'new_record': daily_avg_exp,
                'day': current_day,
                'total_exp': current_exp
            }

        # 检查游戏结束
        if self.state['day'] > 91:
            done = True

            # 基于激进扩张的期望值
            # 假设理想情况：前30天日均300+，中期200+，后期150+
            # 91天总期望：~18000经验，日均~200

            if daily_avg_exp >= 250:  # 极度激进且成功
                final_bonus = 20.0
            elif daily_avg_exp >= 200:  # 优秀的激进扩张
                final_bonus = 15.0
            elif daily_avg_exp >= 150:  # 良好扩张
                final_bonus = 10.0
            elif daily_avg_exp >= 120:  # 合格水平
                final_bonus = 5.0
            elif daily_avg_exp >= 100:  # 略低于期望
                final_bonus = 2.0
            else:  # 保守策略（不应该）
                final_bonus = -2.0  # 轻微惩罚保守玩法

            reward += final_bonus

            # Episode结束时计算效率突破奖励
            if hasattr(self, 'episode_action_stats') and self.episode_action_stats['total_actions'] > 0:
                current_efficiency = (self.episode_action_stats['effective_actions'] /
                                      self.episode_action_stats['total_actions']) * 100

                if not hasattr(self, 'historical_best_efficiency'):
                    self.historical_best_efficiency = 0.0

                if current_efficiency > self.historical_best_efficiency:
                    efficiency_improvement = current_efficiency - self.historical_best_efficiency

                    if efficiency_improvement >= 20:
                        efficiency_bonus = 8.0
                    elif efficiency_improvement >= 10:
                        efficiency_bonus = 4.0 + (efficiency_improvement - 10) / 5.0
                    elif efficiency_improvement >= 5:
                        efficiency_bonus = 2.0 + (efficiency_improvement - 5) / 2.5
                    else:
                        efficiency_bonus = efficiency_improvement / 2.5

                    efficiency_bonus = min(efficiency_bonus, 10.0)
                    reward += efficiency_bonus
                    self.historical_best_efficiency = current_efficiency

                    info['efficiency_breakthrough'] = {
                        'improvement': efficiency_improvement,
                        'bonus': efficiency_bonus,
                        'new_record': current_efficiency,
                        'stats': self.episode_action_stats.copy()
                    }

            info['final_experience'] = self.state['experience']
            info['final_level'] = self.state['level']
            info['final_conquered'] = self.state['total_conquered']
            info['final_daily_avg'] = daily_avg_exp

            # 重置episode统计
            self.episode_action_stats = {
                'total_actions': 0,
                'effective_actions': 0,
                'successful_moves': 0,
                'successful_conquers': 0,
                'next_day_actions': 0,
                'invalid_actions': 0
            }

        elif self.state['food'] < 0:
            done = True
            # 食物耗尽意味着过度激进或规划失败
            # 根据持续天数评估
            if current_day >= 70:  # 坚持到后期
                penalty = -2.0
            elif current_day >= 40:  # 中期崩溃
                penalty = -5.0
            else:  # 早期崩溃
                penalty = -10.0

            # 但如果日均经验很高，减轻惩罚
            if daily_avg_exp >= 300:  # 虽然崩溃但效率极高
                penalty += 5.0
            elif daily_avg_exp >= 200:
                penalty += 2.0

            reward = penalty
            info['termination_reason'] = 'food_exhausted'
            info['final_daily_avg'] = daily_avg_exp

        self.valid_actions_cache = {}
        return self.get_observation(), reward, done, info

    def calculate_reward(self, base_reward, old_conquered, old_exp, old_level, old_milestones):
        """优化的奖励函数 - 专注于经验获取"""
        total_reward = base_reward

        # 1. 经验奖励 - 作为核心目标
        exp_gained = self.state['experience'] - old_exp
        if exp_gained > 0:
            # 根据经验量级给予不同奖励
            if exp_gained >= 1000:  # 超高价值（BOSS等）
                total_reward += exp_gained / 200.0 + 5.0
            elif exp_gained >= 500:  # 高价值
                total_reward += exp_gained / 250.0 + 2.0
            elif exp_gained >= 100:  # 中价值
                total_reward += exp_gained / 300.0 + 0.5
            else:  # 低价值
                total_reward += exp_gained / 500.0

        # 2. 征服奖励 - 根据地块价值
        new_conquered = self.state['conquered_tiles'] - old_conquered
        for pos in new_conquered:
            tile_type = self.map_data[pos]
            props = self.TERRAIN_PROPERTIES.get(tile_type, {})
            exp_value = props.get('exp_gain', 0)

            if exp_value >= 500:
                total_reward += 2.0
                self.state['high_value_conquered'] += 1
            elif exp_value >= 100:
                total_reward += 0.8
            else:
                total_reward += 0.2

            # 特殊地块额外奖励
            if tile_type in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
                total_reward += 5.0
            elif tile_type == TerrainType.TRAINING_GROUND:
                total_reward += 3.0
            elif tile_type == TerrainType.TENT:
                total_reward += 1.0

        # 3. 里程碑奖励 - 每3000经验
        for milestone in self.exp_milestones:
            if self.state['experience'] >= milestone and milestone not in old_milestones:
                self.state['reached_milestones'].add(milestone)
                milestone_reward = 10.0 + (milestone / 3000) * 2.0  # 递增奖励
                total_reward += milestone_reward

        # 4. 等级提升奖励
        if self.state['level'] > old_level:
            total_reward += 3.0 * (self.state['level'] - old_level)

        # 5. 效率奖励
        if self.state['total_conquered'] > 0:
            exp_per_tile = self.state['experience'] / self.state['total_conquered']
            if exp_per_tile > 200:  # 平均每地块超过200经验
                total_reward += 2.0
            elif exp_per_tile > 100:
                total_reward += 1.0

        # 6. 秘宝收集奖励
        if len(self.state['treasures_conquered']) == 8 and self.state['has_treasure_buff']:
            total_reward += 5.0

        # 7. 轻微的时间压力（不要太重）
        if self.state['day'] > 70 and self.state['experience'] < 30000:
            total_reward -= 0.3

        return np.clip(total_reward, -5.0, 30.0)  # 扩大正向奖励范围

    def execute_move(self, team_idx, target):
        """执行移动（修正版）"""
        team = self.state['teams'][team_idx]
        current = team['position']

        if target in self.state['conquered_tiles']:
            return -0.5

        if self.map_data.get(target) == TerrainType.WALL:
            return -0.5

        # 计算移动成本
        if target in self.get_neighbors(*current):
            # 相邻地块直接移动
            cost = 50
        else:
            # 跳跃移动 - 必须目标与已征服区域相邻
            if not any(n in self.state['conquered_tiles'] for n in self.get_neighbors(target[0], target[1])):
                return -0.5
            distance = self.hex_distance(current[0], current[1], target[0], target[1])
            # 修正：步数包括自身（距离+1）
            cost = 30 + 10 * (distance + 1)

        if self.state['has_treasure_buff']:
            cost = int(cost * 0.8)

        if cost > self.state['food']:
            return -0.3

        # 执行移动并跟踪成本
        self.state['food'] -= cost
        self.state['food_spent_move'] += cost  # 跟踪移动成本
        self.state['move_attempts'] += 1  # 跟踪移动次数
        team['position'] = target

        # 基于目标潜在价值的移动奖励
        target_props = self.TERRAIN_PROPERTIES.get(self.map_data[target], {})
        potential_exp = target_props.get('exp_gain', 0)

        # 根据经验潜力计算移动奖励
        if potential_exp >= 1000:
            move_reward = 2.0
        elif potential_exp >= 500:
            move_reward = 1.0
        elif potential_exp >= 100:
            move_reward = 0.5
        else:
            move_reward = potential_exp / 500.0

        # 特殊目标额外奖励
        if self.map_data[target] in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
            move_reward += 2.0
        elif self.map_data[target] == TerrainType.TRAINING_GROUND:
            move_reward += 1.5
        elif self.map_data[target] == TerrainType.TENT:
            move_reward += 0.5

        # 距离成本（降低权重）
        distance_penalty = (cost - 50) / 10000.0

        return move_reward - distance_penalty

    def execute_conquer(self, team_idx):
        """执行征服"""
        team = self.state['teams'][team_idx]
        pos = team['position']

        if pos in self.state['conquered_tiles']:
            return -0.2

        if team['action_points'] <= 0:
            return -0.3

        tile_type = self.map_data[pos]
        props = self.TERRAIN_PROPERTIES.get(tile_type, {})

        food_cost = props.get('food_cost', 0)
        if food_cost < 0:
            return -0.5

        if self.state['has_treasure_buff']:
            food_cost = int(food_cost * 0.8)

        exp_gain = props.get('exp_gain', 0)
        score_cost = props.get('score_cost', 0)

        if food_cost > self.state['food'] or score_cost > self.state['conquest_score']:
            return -0.5

        # 执行征服并跟踪成本
        self.state['food'] -= food_cost
        self.state['food_spent_conquer'] += food_cost  # 跟踪征服成本
        self.state['conquer_attempts'] += 1            # 跟踪征服次数
        self.state['conquest_score'] -= score_cost
        self.state['experience'] += exp_gain
        self.state['conquered_tiles'].add(pos)
        self.state['total_conquered'] += 1

        self.state['level'] = 1 + self.state['experience'] // 100

        # 处理特殊地块
        if not props.get('is_tent', False):
            team['action_points'] -= 1
        else:
            team['action_points'] = min(team['action_points'] + 1, 18)
            tent_food = self.get_tent_food_reward()
            self.state['food'] += tent_food

        if 22 <= tile_type.value <= 29:
            self.state['treasures_conquered'].add(tile_type)
            if len(self.state['treasures_conquered']) == 8:
                self.state['has_treasure_buff'] = True

        if props.get('has_thunder', False):
            self.state['thunder_god_items'] += 1

        # 解锁新队伍
        if self.state['level'] >= 20 and self.state['num_teams'] == 1:
            self.state['num_teams'] = 2
            self.state['teams'].append({'id': 1, 'position': pos, 'action_points': 6})
        elif self.state['level'] >= 60 and self.state['num_teams'] == 2:
            self.state['num_teams'] = 3
            self.state['teams'].append({'id': 2, 'position': pos, 'action_points': 6})

        # 征服奖励已在calculate_reward中处理
        return 0

    def get_tent_food_reward(self):
        """获取帐篷粮食奖励"""
        day = self.state['day']
        if day <= 10:
            tent_reward = 300
        elif day <= 20:
            tent_reward = 250
        elif day <= 35:
            tent_reward = 200
        elif day <= 50:
            tent_reward = 150
        else:
            tent_reward = 100

        self.state['food_from_tents'] += tent_reward  # 跟踪帐篷收入
        return tent_reward

    def execute_thunder_god(self, team_idx):
        """执行飞雷神 - 修正版，严厉惩罚无效使用"""
        if self.state['thunder_god_items'] <= 0:
            return -5.0  # 严重惩罚：没有道具还尝试使用

        team = self.state['teams'][team_idx]

        # 必须站在已征服地块上才能使用飞雷神
        if team['position'] not in self.state['conquered_tiles']:
            return -3.0  # 加重惩罚：位置不对

        # 寻找高价值目标
        high_value_targets = []
        for pos in self.map_data:
            if pos not in self.state['conquered_tiles'] and \
                    self.map_data[pos] != TerrainType.WALL:
                if any(n in self.state['conquered_tiles'] for n in self.get_neighbors(pos[0], pos[1])):
                    props = self.TERRAIN_PROPERTIES.get(self.map_data[pos], {})
                    exp = props.get('exp_gain', 0)
                    if exp >= 100:  # 降低门槛
                        high_value_targets.append((pos, exp))

        if high_value_targets:
            high_value_targets.sort(key=lambda x: x[1], reverse=True)
            target = high_value_targets[0][0]
            team['position'] = target
            self.state['thunder_god_items'] -= 1

        return -1.0  # 没找到合适目标也要惩罚

    def next_day(self):
        """下一天 - 完全重构的资源平衡判断"""
        if self.state['day'] >= 91:
            return 0

        waste_penalty = 0
        opportunity_penalty = 0

        # 计算当前资源状况
        total_action_points = sum(team['action_points'] for team in self.state['teams'])
        daily_food_income = self.get_daily_food_income()
        current_food = self.state['food']
        num_teams = self.state['num_teams']

        # 估算行动点的理论消耗
        avg_action_cost = 100 if not self.state['has_treasure_buff'] else 80
        theoretical_food_needed = total_action_points * avg_action_cost

        # 判断是否是资源限制
        resource_limited = (current_food + daily_food_income) < theoretical_food_needed

        # 1. 行动点浪费惩罚（仅在资源充足时）
        if not resource_limited:
            if self.state['day'] <= 20 and num_teams == 1:
                # 早期单队伍，只有大量行动点剩余才惩罚
                if total_action_points >= 4:  # 剩余4个以上才算浪费
                    waste_penalty += (total_action_points - 3) * 0.2

            elif self.state['day'] <= 60 and num_teams == 2:
                # 中期双队伍
                if total_action_points >= 8:  # 剩余8个以上才惩罚
                    waste_penalty += (total_action_points - 6) * 0.15

            # 后期三队伍：不惩罚行动点剩余（资源限制是常态）

        # 2. 食物"过剩"惩罚（需要考虑队伍数量和未来需求）
        # 计算合理的食物储备
        if num_teams == 1:
            # 单队伍时期：为20级解锁第二队伍储备
            if self.state['level'] < 20:
                levels_to_20 = 20 - self.state['level']
                exp_to_20 = levels_to_20 * 100
                # 假设每天获得200经验，估算到20级的天数
                days_to_unlock = min(exp_to_20 / 200, 10)
                reasonable_food = 3000 + days_to_unlock * 500
            else:
                reasonable_food = 3000

        elif num_teams == 2:
            # 双队伍时期：为60级解锁第三队伍储备
            if self.state['level'] < 60:
                reasonable_food = 2500
            else:
                reasonable_food = 2000

        else:  # 3个队伍
            # 三队伍时期：保持最小储备即可
            reasonable_food = 1500

        # 只有远超合理储备才惩罚
        if current_food > reasonable_food * 2:  # 超过合理储备的2倍
            excess_ratio = current_food / reasonable_food
            waste_penalty += min((excess_ratio - 2.0) * 0.5, 2.0)

        # 3. 机会成本惩罚（只关注超高价值且能负担的机会）
        for i, team in enumerate(self.state['teams']):
            if team['action_points'] <= 0:
                continue

            pos = team['position']

            # 当前位置的征服机会
            if pos not in self.state['conquered_tiles']:
                tile_props = self.TERRAIN_PROPERTIES.get(self.map_data[pos], {})
                food_cost = tile_props.get('food_cost', 0)
                exp_gain = tile_props.get('exp_gain', 0)

                if self.state['has_treasure_buff']:
                    food_cost = int(food_cost * 0.8)

                # 只惩罚放弃超高价值目标
                if food_cost >= 0 and food_cost <= self.state['food']:
                    if exp_gain >= 1000:  # BOSS级别
                        opportunity_penalty += 5.0
                    elif exp_gain >= 500:  # 高价值BOSS
                        opportunity_penalty += 3.0
                    elif exp_gain >= 300:  # 中高价值
                        opportunity_penalty += 1.0
                    # 低价值不惩罚

        # 4. 游戏阶段和队伍数量综合判断
        stage_penalty = 0

        # 早期快速发展检查
        if self.state['day'] <= 10:
            if self.state['experience'] < 1000:  # 10天内应该有1000+经验
                stage_penalty += 1.0
        elif self.state['day'] <= 20:
            if self.state['level'] < 15:  # 20天内应该接近20级
                stage_penalty += 0.5

        # 周日自动领取（保持原逻辑）
        if self.state['day'] % 7 == 3:  # 周日
            if self.state['weekly_exp_quota'] > 0:
                auto_claimed = self.state['weekly_exp_quota']
                self.state['experience'] += auto_claimed
                self.state['weekly_exp_claimed'] += auto_claimed
                self.state['weekly_exp_quota'] = 0
                self.state['level'] = 1 + self.state['experience'] // 100

        # 执行日期推进
        self.state['day'] += 1

        # 清理切换追踪
        switch_penalty = 0
        if self.switch_action_tracking:
            switch_penalty = -0.2 * len(self.switch_action_tracking)
            self.switch_action_tracking.clear()
            self.last_switch_team = -1

        # 重置每日切换计数
        self.daily_switch_count = 0

        # 日常资源更新
        daily_food = self.get_daily_food_income()
        self.state['food'] += daily_food
        self.state['food_income_total'] += daily_food
        self.state['conquest_score'] += 1000

        # 恢复行动点
        for team in self.state['teams']:
            team['action_points'] = min(team['action_points'] + 6, 18)

        # 周一重置
        if self.state['day'] % 7 == 4:  # 周一
            self.state['weekly_exp_quota'] = 500
            self.state['weekly_exp_claimed'] = 0

        # 动态惩罚上限
        if num_teams == 1:
            max_penalty = 3.0  # 单队伍时期宽松
        elif num_teams == 2:
            max_penalty = 5.0  # 双队伍时期适中
        else:
            max_penalty = 3.0  # 三队伍时期宽松（资源紧张）

        total_penalty = waste_penalty + opportunity_penalty + stage_penalty + switch_penalty
        final_reward = -min(total_penalty, max_penalty)

        return final_reward

    def _check_can_act(self):
        """检查是否还能行动"""
        for team in self.state['teams']:
            if team['action_points'] <= 0:
                continue

            # 检查能否征服当前位置
            pos = team['position']
            if pos not in self.state['conquered_tiles']:
                tile_type = self.map_data[pos]
                props = self.TERRAIN_PROPERTIES.get(tile_type, {})
                food_cost = props.get('food_cost', 0)

                if self.state['has_treasure_buff']:
                    food_cost = int(food_cost * 0.8)

                if food_cost >= 0 and food_cost <= self.state['food']:
                    return True

            # 检查能否移动到有价值的地块
            for neighbor in self.get_neighbors(*pos):
                if neighbor not in self.state['conquered_tiles']:
                    move_cost = 50 if self.state['has_treasure_buff'] else 40
                    if move_cost <= self.state['food']:
                        return True

        return False

    def get_daily_food_income(self):
        """获取每日粮草收入"""
        level = self.state['level']
        if level <= 4:
            return 800
        elif level <= 14:
            return 900
        elif level <= 29:
            return 1000
        elif level <= 44:
            return 1100
        elif level <= 69:
            return 1200
        elif level <= 99:
            return 1300
        elif level <= 139:
            return 1400
        elif level <= 189:
            return 1500
        else:
            return 1600

    def claim_weekly_exp(self):
        if self.state['weekly_exp_quota'] <= 0:
            return -1.0

        amount = min(100, self.state['weekly_exp_quota'])
        self.state['experience'] += amount
        self.state['weekly_exp_quota'] -= amount
        self.state['weekly_exp_claimed'] += amount
        self.state['level'] = 1 + self.state['experience'] // 100

        return amount / 200.0

    def get_valid_actions(self):
        """获取有效动作 - 极度限制"下一天"使用，鼓励激进扩张"""
        valid = []
        current_team = self.state['teams'][self.state['current_team']]

        # 移动动作
        if current_team['position'] in self.state['conquered_tiles']:
            reachable = self.get_reachable_tiles(self.state['current_team'])
            for i in range(min(len(reachable), 30)):
                valid.append(i)

        # 征服动作
        if current_team['position'] not in self.state['conquered_tiles'] and \
                current_team['action_points'] > 0:
            tile_type = self.map_data.get(current_team['position'])
            props = self.TERRAIN_PROPERTIES.get(tile_type, {})
            food_cost = props.get('food_cost', 0)
            score_cost = props.get('score_cost', 0)
            if self.state['has_treasure_buff']:
                food_cost = int(food_cost * 0.8)
            if food_cost >= 0 and food_cost <= self.state['food'] and \
                    score_cost <= self.state['conquest_score']:
                valid.append(30)

        # 飞雷神 - 双重检查
        if self.state['thunder_god_items'] > 0:  # 必须有道具
            if current_team['position'] in self.state['conquered_tiles']:  # 必须在已征服地块
                # 额外检查：是否有合适的目标
                has_valid_target = False
                for pos in self.map_data:
                    if pos not in self.state['conquered_tiles'] and \
                            self.map_data[pos] != TerrainType.WALL:
                        if any(n in self.state['conquered_tiles'] for n in self.get_neighbors(pos[0], pos[1])):
                            props = self.TERRAIN_PROPERTIES.get(self.map_data[pos], {})
                            if props.get('exp_gain', 0) >= 100:
                                has_valid_target = True
                                break

                if has_valid_target:  # 只有存在有效目标时才添加
                    valid.append(31)

        # 领取周经验
        if self.state['weekly_exp_quota'] > 0:
            valid.append(33)

        # 切换队伍
        if self.state['num_teams'] > 1 and self.daily_switch_count < self.max_daily_switches:
            for i in range(self.state['num_teams']):
                if i != self.state['current_team']:
                    valid.append(34 + i)

        # 下一天 - 只有在真正无法行动时才允许
        next_day_allowed = False

        # 检查是否还有任何有价值的行动
        total_action_points = sum(team['action_points'] for team in self.state['teams'])

        if total_action_points == 0:  # 所有队伍都没有行动点
            next_day_allowed = True
        elif self.state['food'] < 50:  # 食物极度匮乏（无法支持基本移动）
            next_day_allowed = True
        else:
            # 检查是否还有任何可执行的有意义行动
            has_opportunity = False

            for team_idx, team in enumerate(self.state['teams']):
                if team['action_points'] <= 0:
                    continue

                pos = team['position']

                # 检查当前位置是否可征服
                if pos not in self.state['conquered_tiles']:
                    tile_props = self.TERRAIN_PROPERTIES.get(self.map_data[pos], {})
                    food_cost = tile_props.get('food_cost', 0)
                    if self.state['has_treasure_buff']:
                        food_cost = int(food_cost * 0.8)
                    if food_cost >= 0 and food_cost <= self.state['food']:
                        has_opportunity = True
                        break

                # 检查是否可移动到未征服地块
                else:
                    # 检查相邻未征服地块
                    for neighbor in self.get_neighbors(*pos):
                        if neighbor not in self.state['conquered_tiles']:
                            move_cost = 50 if not self.state['has_treasure_buff'] else 40
                            if move_cost <= self.state['food']:
                                has_opportunity = True
                                break

                    # 如果没有相邻的，检查是否能跳跃到有价值的地块
                    if not has_opportunity and self.state['food'] >= 40:
                        # 简化检查：只要有可达的未征服地块就认为有机会
                        reachable = self.get_reachable_tiles(team_idx)
                        if reachable and len(reachable) > 0:
                            # 检查最近的几个目标
                            for target in reachable[:3]:
                                if target in self.get_neighbors(*pos):
                                    move_cost = 50 if not self.state['has_treasure_buff'] else 40
                                else:
                                    distance = self.hex_distance(pos[0], pos[1], target[0], target[1])
                                    move_cost = 30 + 10 * (distance + 1)  # 包括自身
                                    if self.state['has_treasure_buff']:
                                        move_cost = int(move_cost * 0.8)

                                if move_cost <= self.state['food']:
                                    has_opportunity = True
                                    break

                if has_opportunity:
                    break

            # 只有确实没有任何有意义的行动机会时才允许下一天
            if not has_opportunity:
                next_day_allowed = True

        # 额外检查：游戏早期（前20天）如果有大量行动点和充足食物，禁止下一天
        if self.state['day'] <= 20 and total_action_points >= 6 and self.state['food'] >= 500:
            next_day_allowed = False

        # 游戏中期（21-60天）也要严格控制
        elif self.state['day'] <= 60 and total_action_points >= 10 and self.state['food'] >= 1000:
            next_day_allowed = False

        if next_day_allowed:
            valid.append(32)

        # 保底机制（应该极少触发）
        if not valid:
            if self.state['day'] < 91:
                valid = [32]
            else:
                # 游戏已经结束，返回一个无害的动作
                valid = [33] if self.state['weekly_exp_quota'] > 0 else [32]

        return valid

    def get_expected_food_by_day(self):
        """根据游戏阶段和队伍数量计算合理的粮草水平"""
        base_daily = self.get_daily_food_income()
        days_passed = self.state['day'] - 1
        num_teams = self.state['num_teams']

        # 根据队伍数量估算每日合理消耗
        if num_teams == 1:
            daily_consumption = 600  # 单队伍期望消耗
        elif num_teams == 2:
            daily_consumption = 1100  # 双队伍期望消耗
        else:  # 3个队伍
            daily_consumption = 1600  # 三队伍期望消耗

        # 考虑秘宝buff
        if self.state['has_treasure_buff']:
            daily_consumption = int(daily_consumption * 0.8)

        # 计算理论剩余
        total_income = 6800 + base_daily * days_passed
        total_expected_spent = daily_consumption * days_passed
        expected_remaining = total_income - total_expected_spent

        # 根据游戏阶段设定合理的储备量
        if self.state['day'] <= 20:
            # 早期：需要大量储备为解锁新队伍做准备
            reasonable_reserve = 4000 + (20 - self.state['day']) * 200
        elif self.state['day'] <= 60:
            # 中期：适度储备
            reasonable_reserve = 2000 + (60 - self.state['day']) * 50
        else:
            # 后期：最小储备
            reasonable_reserve = 1000
        return max(expected_remaining, reasonable_reserve)
# ============================================================================
# 经验回放缓冲区
# ============================================================================
class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # 先转换为numpy数组，再转换为tensor（避免性能警告）
        states = np.array([s[0] for s in samples], dtype=np.float32)
        actions = np.array([s[1] for s in samples], dtype=np.int64)
        rewards = np.array([s[2] for s in samples], dtype=np.float32)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)
        dones = np.array([s[4] for s in samples], dtype=np.float32)

        # 转换为tensor
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 神经网络模型
# ============================================================================
class ImprovedDuelingDQN(nn.Module):
    """改进的Dueling DQN网络架构"""

    def __init__(self, state_size, action_size, hidden_sizes=[512, 256, 128]):
        super(ImprovedDuelingDQN, self).__init__()

        # 共享层
        layers = []
        prev_size = state_size

        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if i == 0:
                layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )

        # 初始化权重
        self._initialize_weights()
        # 添加队伍切换跟踪
        self.daily_switch_count = 0  # 每日切换次数
        self.max_daily_switches = 10  # 每日最大切换次数
        self.last_switch_team = -1  # 上次切换的队伍ID
        self.switch_action_tracking = {}  # 跟踪切换后的行动

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state):
        features = self.shared(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# ============================================================================
# 训练器
# ============================================================================
class ImprovedHexGameDQNTrainer:
    """改进的DQN训练器"""

    def __init__(self, env, state_size, action_size, device='cpu'):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.action_stats = {
            'move_actions': 0,  # 0-29: 移动动作
            'conquer_actions': 0,  # 30: 征服
            'thunder_god_actions': 0,  # 31: 雷神飞
            'next_day_actions': 0,  # 32: 下一天
            'claim_exp_actions': 0,  # 33: 领取周经验
            'switch_team_actions': 0,  # 34-36: 切换队伍
            'invalid_actions': 0,  # 无效动作（被环境拒绝）
            'successful_moves': 0,  # 成功移动次数
            'successful_conquers': 0,  # 成功征服次数
            'failed_moves': 0,  # 失败移动次数
            'failed_conquers': 0,  # 失败征服次数
        }
        # 网络
        self.q_network = ImprovedDuelingDQN(state_size, action_size).to(device)
        self.target_network = ImprovedDuelingDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        # 经验回放
        self.memory = PrioritizedReplayBuffer(capacity=100000)

        # 超参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # 更慢的衰减
        self.epsilon_min = 0.01
        self.batch_size = 128  # 增大batch size
        self.update_target_every = 500
        self.train_step_counter = 0

        # 训练统计
        self.episode_rewards = []
        self.episode_experiences = []
        self.episode_conquests = []
        self.losses = []
        self.best_experience = 0

    def select_action(self, state, training=True):
        valid_actions = self.env.get_valid_actions()

        if training and random.random() < self.epsilon:
            # 智能随机选择：强烈偏向有价值的行动
            non_next_day = [a for a in valid_actions if a != 32]

            if non_next_day:
                # 95%概率选择非"下一天"动作
                if random.random() < 0.95:
                    # 进一步细分动作价值
                    high_value_actions = []
                    medium_value_actions = []
                    low_value_actions = []

                    for action in non_next_day:
                        if action < 30:  # 移动动作
                            reachable = self.env.get_reachable_tiles(self.env.state['current_team'])
                            if action < len(reachable):
                                target_pos = reachable[action]
                                target_exp = self.env.TERRAIN_PROPERTIES.get(
                                    self.env.map_data[target_pos], {}
                                ).get('exp_gain', 0)

                                if target_exp >= 300:
                                    high_value_actions.append(action)
                                elif target_exp >= 100:
                                    medium_value_actions.append(action)
                                else:
                                    low_value_actions.append(action)

                        elif action == 30:  # 征服动作
                            pos = self.env.state['teams'][self.env.state['current_team']]['position']
                            if pos not in self.env.state['conquered_tiles']:
                                exp_gain = self.env.TERRAIN_PROPERTIES.get(
                                    self.env.map_data[pos], {}
                                ).get('exp_gain', 0)

                                if exp_gain >= 300:
                                    high_value_actions.append(action)
                                elif exp_gain >= 100:
                                    medium_value_actions.append(action)
                                elif exp_gain >= 30:
                                    low_value_actions.append(action)

                        elif action == 31:  # 飞雷神
                            # 飞雷神通常是高价值的
                            high_value_actions.append(action)

                        elif action == 33:  # 领取周经验
                            # 周经验也是中等价值
                            medium_value_actions.append(action)

                        elif 34 <= action <= 36:  # 切换队伍
                            # 队伍切换通常价值较低，但有时必要
                            team_idx = action - 34
                            if team_idx < len(self.env.state['teams']):
                                target_team = self.env.state['teams'][team_idx]
                                if target_team['action_points'] > 3:  # 目标队伍有足够行动点
                                    low_value_actions.append(action)

                    # 分层选择：80%高价值，15%中价值，5%低价值
                    rand = random.random()
                    if high_value_actions and rand < 0.8:
                        return random.choice(high_value_actions)
                    elif medium_value_actions and rand < 0.95:
                        return random.choice(medium_value_actions)
                    elif low_value_actions:
                        return random.choice(low_value_actions)
                    else:
                        return random.choice(non_next_day)
                else:
                    # 5%概率还是可能选择"下一天"
                    return random.choice(valid_actions)
            else:
                # 如果没有非"下一天"动作，只能选择它
                return random.choice(valid_actions)

        # 贪婪选择（训练时epsilon较低或非训练时）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze()

            # 动作掩码 - 确保只选择有效动作
            masked_q_values = torch.full_like(q_values, float('-inf'))
            for action in valid_actions:
                if action < len(masked_q_values):
                    masked_q_values[action] = q_values[action]

            # 对"下一天"动作施加轻微惩罚，鼓励其他行动
            if 32 in valid_actions and len(valid_actions) > 1:
                masked_q_values[32] -= 0.1  # 轻微降低"下一天"的Q值

            # 如果所有Q值都是-inf，返回第一个有效动作
            if torch.all(masked_q_values == float('-inf')):
                return valid_actions[0] if valid_actions else 32

            return masked_q_values.argmax().item()

    def reset_action_stats(self):
        """重置动作统计"""
        for key in self.action_stats:
            self.action_stats[key] = 0

    def update_action_stats(self, action, old_state, new_state, reward):
        """更新动作统计"""
        # 基本动作分类统计
        if 0 <= action <= 29:
            self.action_stats['move_actions'] += 1
        elif action == 30:
            self.action_stats['conquer_actions'] += 1
        elif action == 31:
            self.action_stats['thunder_god_actions'] += 1
        elif action == 32:
            self.action_stats['next_day_actions'] += 1
        elif action == 33:
            self.action_stats['claim_exp_actions'] += 1
        elif 34 <= action <= 36:
            self.action_stats['switch_team_actions'] += 1

        # 检测动作结果
        if 0 <= action <= 29:  # 移动动作
            # 检查位置是否发生变化
            old_pos = old_state['teams'][old_state['current_team']]['position']
            new_pos = new_state['teams'][new_state['current_team']]['position']
            if old_pos != new_pos:
                self.action_stats['successful_moves'] += 1
            else:
                self.action_stats['failed_moves'] += 1

        elif action == 30:  # 征服动作
            # 检查征服地块数量是否增加
            if new_state['total_conquered'] > old_state['total_conquered']:
                self.action_stats['successful_conquers'] += 1
            else:
                self.action_stats['failed_conquers'] += 1

        # 检测无效动作（奖励为负且较小的情况）
        if reward <= -0.2:
            self.action_stats['invalid_actions'] += 1

    def print_action_stats(self, episode_steps):
        """打印动作统计信息"""
        stats = self.action_stats
        total_actions = sum(stats.values()) - stats['successful_moves'] - stats['successful_conquers'] - stats[
            'failed_moves'] - stats['failed_conquers'] - stats['invalid_actions']

        print(f"\n[ACTION ANALYSIS - {episode_steps} steps]")
        print(
            f"Move actions: {stats['move_actions']:4d} (Success: {stats['successful_moves']:3d}, Failed: {stats['failed_moves']:3d})")
        print(
            f"Conquer actions: {stats['conquer_actions']:4d} (Success: {stats['successful_conquers']:3d}, Failed: {stats['failed_conquers']:3d})")
        print(f"Thunder God: {stats['thunder_god_actions']:4d}")
        print(f"Next Day: {stats['next_day_actions']:4d}")
        print(f"Claim Exp: {stats['claim_exp_actions']:4d}")
        print(f"Switch Team: {stats['switch_team_actions']:4d}")
        print(f"Invalid/Rejected: {stats['invalid_actions']:4d}")

        # 计算效率
        effective_actions = stats['successful_moves'] + stats['successful_conquers'] + stats['next_day_actions']
        wasted_actions = episode_steps - effective_actions
        efficiency = (effective_actions / episode_steps) * 100 if episode_steps > 0 else 0

        print(f"\nEfficiency Analysis:")
        print(f"Effective actions: {effective_actions:4d} ({efficiency:.1f}%)")
        print(f"Wasted actions: {wasted_actions:4d} ({100 - efficiency:.1f}%)")

        # 分析主要浪费来源
        waste_sources = []
        if stats['failed_moves'] > 50:
            waste_sources.append(f"Failed moves ({stats['failed_moves']})")
        if stats['failed_conquers'] > 20:
            waste_sources.append(f"Failed conquers ({stats['failed_conquers']})")
        if stats['invalid_actions'] > 100:
            waste_sources.append(f"Invalid actions ({stats['invalid_actions']})")
        if stats['switch_team_actions'] > 50:
            waste_sources.append(f"Excessive team switching ({stats['switch_team_actions']})")

        if waste_sources:
            print(f"Main waste sources: {', '.join(waste_sources)}")
        print("-" * 60)
    def train_step(self):
        """训练一步"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size, beta=0.4)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD误差
        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # 损失
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.losses.append(loss.item())
        self.train_step_counter += 1

        # 更新目标网络
        if self.train_step_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, episodes=10000, save_interval=100):
        """训练主循环 - 简化版，突破奖励由环境处理"""
        import time

        best_exp = 0
        start_time = time.time()
        first_episode_analyzed = False

        print(f"Starting {episodes} training episodes...", flush=True)
        print("-" * 80, flush=True)

        # 使用普通列表避免切片问题
        recent_experiences = []
        recent_rewards = []
        recent_conquests = []
        breakthrough_count = 0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            max_steps = 2000
            episode_steps = 0
            episode_breakthroughs = 0  # 本episode的突破次数

            # 重置动作统计
            self.reset_action_stats()

            # 为了跟踪状态变化，需要保存旧状态
            old_state = None

            for step in range(max_steps):
                # 保存执行动作前的状态
                old_state = {
                    'teams': [team.copy() for team in self.env.state['teams']],
                    'current_team': self.env.state['current_team'],
                    'total_conquered': self.env.state['total_conquered'],
                    'position': self.env.state['teams'][self.env.state['current_team']]['position']
                }

                action = self.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                if 'efficiency_breakthrough' in info:
                    eff_info = info['efficiency_breakthrough']
                    print(f"  EFFICIENCY RECORD! {eff_info['new_record']:.1f}% "
                          f"(+{eff_info['improvement']:.1f}%, reward: +{eff_info['bonus']:.2f})", flush=True)
                # 检查是否有突破
                if 'breakthrough' in info:
                    breakthrough_info = info['breakthrough']
                    episode_breakthroughs += 1
                    breakthrough_count += 1
                    print(f"  Step {step + 1}: NEW RECORD! {breakthrough_info['new_record']:.0f} exp "
                          f"(+{breakthrough_info['improvement']:.0f}, reward: +{breakthrough_info['bonus']:.2f})",
                          flush=True)

                # 更新动作统计
                self.update_action_stats(action, old_state, self.env.state, reward)

                self.memory.push(state, action, reward, next_state, done)

                # 开始训练
                if len(self.memory) >= self.batch_size * 10:
                    self.train_step()

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # Episode结束后的分析
            final_exp = self.env.state['experience']
            final_conquered = self.env.state['total_conquered']
            final_day = self.env.state['day']
            final_food = self.env.state['food']
            final_level = self.env.state['level']

            # 食物分析和动作统计 - 只在前几个episode或定期打印
            if not first_episode_analyzed or episode < 3 or (episode + 1) % 100 == 0:
                if not first_episode_analyzed:
                    first_episode_analyzed = True
                    print("\n[FOOD ANALYSIS - First Episode]", flush=True)
                    print(f"Day: {final_day}, Food: {final_food}", flush=True)
                    print(f"Experience: {final_exp}, Level: {final_level}", flush=True)
                    print(f"Conquered tiles: {final_conquered}", flush=True)
                    print(f"Total steps: {episode_steps}", flush=True)
                    print(f"Initial food: 6800", flush=True)
                    print(f"Daily income total: {self.env.state.get('food_income_total', 0)}", flush=True)
                    print(f"Tent income total: {self.env.state.get('food_from_tents', 0)}", flush=True)
                    print(f"Move costs total: {self.env.state.get('food_spent_move', 0)}", flush=True)
                    print(f"Conquer costs total: {self.env.state.get('food_spent_conquer', 0)}", flush=True)

                    total_income = 6800 + self.env.state.get('food_income_total', 0) + self.env.state.get(
                        'food_from_tents', 0)
                    total_spent = self.env.state.get('food_spent_move', 0) + self.env.state.get('food_spent_conquer', 0)
                    expected = total_income - total_spent
                    print(f"\nTotal income: {total_income}")
                    print(f"Total spent: {total_spent}")
                    print(f"Expected remaining: {expected}")
                    print(f"Actual remaining: {final_food}")
                    missing_food = expected - final_food
                    if abs(missing_food) > 100:
                        print(f"FOOD DISCREPANCY: {missing_food}")

                # 打印动作统计
                self.print_action_stats(episode_steps)

            # 更新epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_experiences.append(final_exp)
            self.episode_conquests.append(final_conquered)

            # 安全地添加到列表并维护最大长度
            recent_experiences.append(final_exp)
            recent_rewards.append(episode_reward)
            recent_conquests.append(final_conquered)

            # 手动维护列表长度（保持最近100个）
            if len(recent_experiences) > 100:
                recent_experiences.pop(0)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            if len(recent_conquests) > 100:
                recent_conquests.pop(0)

            # 更新全局最佳并保存
            if final_exp > best_exp:
                best_exp = final_exp
                self.best_experience = best_exp
                self.save_model('best_model.pth')

            # 每个episode都打印结果
            effective_actions = (self.action_stats['successful_moves'] +
                                 self.action_stats['successful_conquers'] +
                                 self.action_stats['next_day_actions'])
            efficiency = (effective_actions / episode_steps) * 100 if episode_steps > 0 else 0
            # 检查是否已经打印过详细分析
            detailed_analysis_printed = False

            # 食物分析和动作统计 - 只在前几个episode或定期打印
            if not first_episode_analyzed or episode < 3 or (episode + 1) % 100 == 0:
                if not first_episode_analyzed:
                    first_episode_analyzed = True
                    print("\n[FOOD ANALYSIS - First Episode]", flush=True)
                    print(f"Day: {final_day}, Food: {final_food}", flush=True)
                    print(f"Experience: {final_exp}, Level: {final_level}", flush=True)
                    print(f"Conquered tiles: {final_conquered}", flush=True)
                    print(f"Total steps: {episode_steps}", flush=True)
                    print(f"Initial food: 6800", flush=True)
                    print(f"Daily income total: {self.env.state.get('food_income_total', 0)}", flush=True)
                    print(f"Tent income total: {self.env.state.get('food_from_tents', 0)}", flush=True)
                    print(f"Move costs total: {self.env.state.get('food_spent_move', 0)}", flush=True)
                    print(f"Conquer costs total: {self.env.state.get('food_spent_conquer', 0)}", flush=True)

                    total_income = 6800 + self.env.state.get('food_income_total', 0) + self.env.state.get(
                        'food_from_tents', 0)
                    total_spent = self.env.state.get('food_spent_move', 0) + self.env.state.get('food_spent_conquer', 0)
                    expected = total_income - total_spent
                    print(f"\nTotal income: {total_income}")
                    print(f"Total spent: {total_spent}")
                    print(f"Expected remaining: {expected}")
                    print(f"Actual remaining: {final_food}")
                    missing_food = expected - final_food
                    if abs(missing_food) > 100:
                        print(f"FOOD DISCREPANCY: {missing_food}")

                # 打印动作统计
                self.print_action_stats(episode_steps)
                detailed_analysis_printed = True

            # 新增：如果效率低于50%且还没有打印过详细分析，则打印
            if efficiency < 90.0 and not detailed_analysis_printed:
                print(f"\n⚠️  LOW EFFICIENCY DETECTED (Episode {episode + 1}): {efficiency:.1f}%", flush=True)
                self.print_action_stats(episode_steps)
                detailed_analysis_printed = True

            # 如果显示突破，显示特殊标记
            breakthrough_marker = f" [x{episode_breakthroughs}]" if episode_breakthroughs > 0 else ""
            # 如果有突破，显示特殊标记
            breakthrough_marker = f" [x{episode_breakthroughs}]" if episode_breakthroughs > 0 else ""

            print(f"[Ep {episode + 1:4d}] "
                  f"EXP: {final_exp:6.0f}{breakthrough_marker} | "
                  f"Day: {final_day:2d} | "
                  f"Food: {final_food:5.0f} | "
                  f"Tiles: {final_conquered:3d} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Eff: {efficiency:4.1f}%", flush=True)

            # 每10个episodes打印汇总
            if (episode + 1) % 10 == 0:
                # 安全地处理列表切片
                if len(recent_experiences) >= 10:
                    avg_exp = np.mean(recent_experiences[-10:])
                    avg_reward = np.mean(recent_rewards[-10:])
                    avg_conquests = np.mean(recent_conquests[-10:])
                else:
                    avg_exp = np.mean(recent_experiences) if recent_experiences else 0
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_conquests = np.mean(recent_conquests) if recent_conquests else 0

                # 计算最近10个episode的突破次数
                recent_breakthroughs = 0
                start_idx = max(0, episode - 9)
                for i in range(start_idx, episode + 1):
                    if i < len(self.episode_experiences):
                        # 检查这个episode是否创造了当时的新记录
                        if i == 0 or self.episode_experiences[i] > max(self.episode_experiences[:i]):
                            recent_breakthroughs += 1

                print(f"[Summary Ep {max(1, episode - 8):4d}-{episode + 1:4d}] "
                      f"Avg EXP: {avg_exp:7.1f}/{best_exp:7.0f} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Tiles: {avg_conquests:5.1f} | "
                      f"Breakthroughs: {recent_breakthroughs} | "
                      f"ε: {self.epsilon:.3f}", flush=True)
                print("-" * 80, flush=True)

            # 定期保存
            if (episode + 1) % save_interval == 0:
                print(f"\n{'=' * 60}")
                print(f"Checkpoint: Episode {episode + 1}")
                print(f"Best Experience: {best_exp:.0f}")
                print(f"Total Breakthroughs: {breakthrough_count}")
                elapsed = time.time() - start_time
                print(f"Time elapsed: {elapsed / 60:.1f} minutes")
                print(f"Avg time/episode: {elapsed / (episode + 1):.2f}s")

                # 突破率统计
                breakthrough_rate = (breakthrough_count / (episode + 1)) * 100 if episode + 1 > 0 else 0
                print(f"Breakthrough rate: {breakthrough_rate:.1f}%")

                # 学习效率分析
                if len(self.episode_experiences) >= 100:
                    recent_100_max = max(self.episode_experiences[-100:])
                    recent_100_avg = np.mean(self.episode_experiences[-100:])
                    print(f"Recent 100 episodes: Max {recent_100_max:.0f}, Avg {recent_100_avg:.1f}")

                print(f"{'=' * 60}\n")

                self.save_model(f'checkpoint_{episode + 1}.pth')
                self.plot_training_curves()

            # 性能监控：检测学习停滞
            if (episode + 1) % 1000 == 0 and episode > 0:
                recent_1000_max = max(self.episode_experiences[-1000:]) if len(
                    self.episode_experiences) >= 1000 else max(self.episode_experiences)
                if recent_1000_max <= best_exp * 0.95:  # 如果最近1000轮最佳表现不到历史最佳的95%
                    print(f"⚠️  Performance plateau detected after episode {episode + 1}")
                    print(f"   Current best: {best_exp:.0f}, Recent 1000 max: {recent_1000_max:.0f}")
                    print(f"   Consider adjusting learning parameters if this persists.")

        # 训练完成总结
        print(f"\n{'=' * 80}")
        print(f"TRAINING COMPLETE")
        print(f"Best Experience: {best_exp:.0f}")
        print(f"Total Breakthroughs: {breakthrough_count}")
        if episodes > 0:
            print(f"Final Breakthrough Rate: {(breakthrough_count / episodes) * 100:.1f}%")
            print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"Average time per episode: {(time.time() - start_time) / episodes:.2f}s")

        # 最终表现分析
        if len(self.episode_experiences) >= 100:
            final_100_avg = np.mean(self.episode_experiences[-100:])
            final_100_max = max(self.episode_experiences[-100:])
            print(f"Final 100 episodes: Max {final_100_max:.0f}, Avg {final_100_avg:.1f}")

            if final_100_max >= best_exp * 0.9:
                print("✅ Training ended with strong recent performance")
            else:
                print("⚠️  Recent performance below historical best - may need longer training")

        print(f"{'=' * 80}")

        # 最终保存
        self.save_model('final_model.pth')
        self.plot_training_curves()

        # 训练完成总结
        print(f"\n{'=' * 80}")
        print(f"TRAINING COMPLETE")
        print(f"Best Experience: {best_exp:.0f}")
        print(f"Total Breakthroughs: {breakthrough_count}")
        print(f"Final Breakthrough Rate: {(breakthrough_count / episodes) * 100:.1f}% if episodes > 0 else 0")
        print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"Average time per episode: {(time.time() - start_time) / episodes:.2f}s if episodes > 0 else 0")
        print(f"{'=' * 80}")

        self.save_model('final_model.pth')
        self.plot_training_curves()

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_experience': self.best_experience,
            'episode_rewards': self.episode_rewards,
            'episode_experiences': self.episode_experiences
        }, filepath)

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.best_experience = checkpoint.get('best_experience', 0)

    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.episode_experiences) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 经验值曲线
        axes[0, 0].plot(self.episode_experiences, alpha=0.3, label='Episode Experience')
        if len(self.episode_experiences) >= 100:
            moving_avg = np.convolve(self.episode_experiences,
                                    np.ones(100) / 100, mode='valid')
            axes[0, 0].plot(range(99, len(self.episode_experiences)),
                          moving_avg, label='100-Episode Average', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Experience')
        axes[0, 0].set_title('Experience Gained')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 奖励曲线
        axes[0, 1].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards,
                                    np.ones(100) / 100, mode='valid')
            axes[0, 1].plot(range(99, len(self.episode_rewards)),
                          moving_avg, label='100-Episode Average', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 征服数量曲线
        axes[1, 0].plot(self.episode_conquests, alpha=0.3, label='Tiles Conquered')
        if len(self.episode_conquests) >= 100:
            moving_avg = np.convolve(self.episode_conquests,
                                    np.ones(100) / 100, mode='valid')
            axes[1, 0].plot(range(99, len(self.episode_conquests)),
                          moving_avg, label='100-Episode Average', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Tiles')
        axes[1, 0].set_title('Tiles Conquered')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 损失曲线
        if self.losses:
            axes[1, 1].plot(self.losses[-1000:], alpha=0.5)
            axes[1, 1].set_xlabel('Training Steps (last 1000)')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主训练流程"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"\n{'=' * 60}")
    print("Hex Game RL Training System")
    print(f"{'=' * 60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✗ Using CPU")
    print(f"{'=' * 60}\n")

    # 创建环境
    env = ImprovedHexGameEnvironment()
    state_size = len(env.get_observation())
    action_size = env.action_space

    print(f"Environment initialized:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Map tiles: {len(env.map_data)}")
    print(f"\nKey positions identified:")
    for category, positions in env.key_positions.items():
        if positions:
            print(f"  {category}: {len(positions)} tiles")
    print()

    # 创建训练器
    trainer = ImprovedHexGameDQNTrainer(env, state_size, action_size, device)

    # 开始训练
    print("Starting training with optimized reward function...")
    print("Focus: Maximizing experience gain")
    print("=" * 60 + "\n")

    trainer.train(episodes=10000, save_interval=100)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()