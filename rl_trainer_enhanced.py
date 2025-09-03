"""
六边形地图策略游戏 - PPO强化学习训练器（修复版v5）
完全匹配实际游戏规则，包含周经验系统
修复所有缩进问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("GPU不可用，使用CPU训练")
    return device


@dataclass
class DayAction:
    """记录每天的行动"""
    day: int
    team_id: int
    action_type: str
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    food_cost: int
    exp_gain: int
    resources_after: Dict


class GameCore:
    """无GUI的游戏核心逻辑，完全匹配实际游戏规则"""

    def __init__(self, map_file='map_save.json'):
        with open(map_file, 'r') as f:
            self.map_data = json.load(f)

        # 初始化地块属性
        self.terrain_properties = {
            'NORMAL_LV1': {'food_cost': 100, 'exp_gain': 35, 'score_cost': 0},
            'NORMAL_LV2': {'food_cost': 110, 'exp_gain': 40, 'score_cost': 0},
            'NORMAL_LV3': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'NORMAL_LV4': {'food_cost': 130, 'exp_gain': 50, 'score_cost': 0},
            'NORMAL_LV5': {'food_cost': 140, 'exp_gain': 55, 'score_cost': 0},
            'NORMAL_LV6': {'food_cost': 150, 'exp_gain': 60, 'score_cost': 0},
            'DUMMY_LV1': {'food_cost': 100, 'exp_gain': 35, 'score_cost': 0},
            'DUMMY_LV2': {'food_cost': 100, 'exp_gain': 40, 'score_cost': 0},
            'DUMMY_LV3': {'food_cost': 100, 'exp_gain': 45, 'score_cost': 0},
            'DUMMY_LV4': {'food_cost': 100, 'exp_gain': 50, 'score_cost': 0},
            'DUMMY_LV5': {'food_cost': 100, 'exp_gain': 55, 'score_cost': 0},
            'DUMMY_LV6': {'food_cost': 100, 'exp_gain': 60, 'score_cost': 0},
            'TOWER_LV1': {'food_cost': 100, 'exp_gain': 35, 'score_cost': 0},
            'TOWER_LV2': {'food_cost': 100, 'exp_gain': 40, 'score_cost': 0},
            'TOWER_LV3': {'food_cost': 100, 'exp_gain': 45, 'score_cost': 0},
            'TOWER_LV4': {'food_cost': 100, 'exp_gain': 50, 'score_cost': 0},
            'TOWER_LV5': {'food_cost': 100, 'exp_gain': 55, 'score_cost': 0},
            'TOWER_LV6': {'food_cost': 100, 'exp_gain': 60, 'score_cost': 0},
            'TRAINING_GROUND': {'food_cost': 0, 'exp_gain': 520, 'score_cost': 0},
            'BLACK_MARKET': {'food_cost': 0, 'exp_gain': 30, 'score_cost': 1000},
            'STONE_TABLET': {'food_cost': 100, 'exp_gain': 40, 'score_cost': 0},
            'TENT': {'food_cost': 0, 'exp_gain': 0, 'score_cost': 0},
            'TREASURE_1': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'TREASURE_2': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'TREASURE_3': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'TREASURE_4': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'TREASURE_5': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'TREASURE_6': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'TREASURE_7': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'TREASURE_8': {'food_cost': 120, 'exp_gain': 45, 'score_cost': 0},
            'BOSS_GAARA': {'food_cost': 200, 'exp_gain': 500, 'score_cost': 0},
            'BOSS_ZETSU': {'food_cost': 200, 'exp_gain': 1000, 'score_cost': 0},
            'BOSS_DARTMAN': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'BOSS_SHIRA': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'BOSS_KUSHINA': {'food_cost': 200, 'exp_gain': 1000, 'score_cost': 0},
            'BOSS_KISAME': {'food_cost': 200, 'exp_gain': 500, 'score_cost': 0},
            'BOSS_HANA': {'food_cost': 200, 'exp_gain': 300, 'score_cost': 0},
            'WALL': {'food_cost': -1, 'exp_gain': 0, 'score_cost': 0},
            'START_POSITION': {'food_cost': 0, 'exp_gain': 0, 'score_cost': 0},
        }

        self.action_history = []
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.current_day = 1
        self.experience = 0
        self.level = 1
        self.food = 6800
        self.conquest_score = 1000
        self.thunder_god_items = 1
        self.treasures_conquered = set()
        self.has_treasure_buff = False
        self.conquered_tiles = set()

        # 周经验系统
        self.weekly_exp_quota = 500
        self.weekly_exp_claimed = 0
        self.weekly_claim_count = 0
        self.current_week = 1

        # 效率追踪
        self.last_conquer_day = 0
        self.consecutive_conquers = 0
        self.total_wasted_actions = 0

        # 队伍管理
        self.teams = {
            1: {'pos': None, 'action_points': 6, 'max_action_points': 18, 'active': True}
        }
        self.current_team = 1
        self.max_teams = 1

        # 清空历史记录
        self.action_history.clear()

        # 初始化地图
        self.hex_map = {}
        for tile_data in self.map_data['tiles']:
            q, r = tile_data['q'], tile_data['r']
            self.hex_map[(q, r)] = tile_data

        # 找起始位置
        self.start_pos = None
        for pos, tile in self.hex_map.items():
            terrain_type = str(tile.get('terrain_type', ''))
            if terrain_type == 'START_POSITION' or terrain_type == '0':
                self.start_pos = pos
                self.conquered_tiles.add(pos)
                break

        if not self.start_pos:
            for pos, tile in self.hex_map.items():
                terrain_type = str(tile.get('terrain_type', ''))
                if terrain_type != 'WALL' and terrain_type != '30':
                    self.start_pos = pos
                    self.conquered_tiles.add(pos)
                    break

        if not self.start_pos:
            self.start_pos = next(iter(self.hex_map.keys()))
            self.conquered_tiles.add(self.start_pos)

        self.teams[1]['pos'] = self.start_pos
        self.check_team_unlock()

    def get_day_of_week(self, day):
        """获取星期几"""
        base_day = 5
        days_passed = day - 1
        return (base_day + days_passed) % 7

    def get_week_number(self, day):
        """获取当前是第几周"""
        if day <= 3:
            return 1
        else:
            days_from_first_monday = day - 4
            return 2 + (days_from_first_monday // 7)

    def calculate_level(self):
        """计算当前等级"""
        self.level = 1 + (self.experience // 100)

    def check_team_unlock(self):
        """检查是否解锁新队伍"""
        self.calculate_level()

        if self.level >= 20 and self.max_teams == 1:
            self.max_teams = 2
            if len(self.teams) == 1:
                self.teams[2] = {
                    'pos': self.teams[1]['pos'],
                    'action_points': 6,
                    'max_action_points': 18,
                    'active': True
                }

        if self.level >= 60 and self.max_teams == 2:
            self.max_teams = 3
            if len(self.teams) == 2:
                self.teams[3] = {
                    'pos': self.teams[self.current_team]['pos'],
                    'action_points': 6,
                    'max_action_points': 18,
                    'active': True
                }

    def get_daily_food(self):
        """根据等级获取每日粮草"""
        if self.level <= 4:
            return 800
        elif self.level <= 14:
            return 900
        elif self.level <= 29:
            return 1000
        elif self.level <= 44:
            return 1100
        elif self.level <= 69:
            return 1200
        elif self.level <= 99:
            return 1300
        elif self.level <= 139:
            return 1400
        elif self.level <= 189:
            return 1500
        else:
            return 1600

    def get_tent_food_gain(self, day):
        """获取帐篷粮草收益"""
        if day <= 10:
            return 300
        elif day <= 20:
            return 250
        elif day <= 35:
            return 200
        elif day <= 50:
            return 150
        else:
            return 100

    def apply_cost_reduction(self, cost):
        """应用秘宝buff减免"""
        if self.has_treasure_buff:
            return int(cost * 0.8)
        return cost

    def get_neighbors(self, pos):
        """获取存在于地图中的相邻位置"""
        if not pos:
            return []
        q, r = pos
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []

        for dq, dr in directions:
            neighbor_pos = (q + dq, r + dr)
            if neighbor_pos in self.hex_map:
                neighbors.append(neighbor_pos)

        return neighbors

    def find_path_to_unconquered(self, start, end):
        """寻找到未征服地块的路径"""
        if start == end:
            return [start]

        from heapq import heappush, heappop

        def heuristic(a, b):
            return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) / 2

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while open_set:
            current = heappop(open_set)[1]

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tile = self.hex_map.get(neighbor)
                if not tile:
                    continue

                terrain_type = str(tile.get('terrain_type', ''))
                if terrain_type == 'WALL' or terrain_type == '30':
                    continue

                if neighbor != end and neighbor not in self.conquered_tiles:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def calculate_move_cost(self, target):
        """计算移动成本"""
        team = self.teams[self.current_team]

        if target not in self.hex_map:
            return -1

        if target in self.conquered_tiles:
            return -1

        target_tile = self.hex_map.get(target)
        if not target_tile:
            return -1

        terrain_type = str(target_tile.get('terrain_type', ''))
        if terrain_type == 'WALL' or terrain_type == '30':
            return -1

        neighbors = self.get_neighbors(team['pos'])
        if target in neighbors:
            cost = 50
        else:
            path = self.find_path_to_unconquered(team['pos'], target)
            if path:
                steps = len(path)
                cost = 30 + 10 * steps
            else:
                return -1

        return self.apply_cost_reduction(cost)

    def get_tile_properties(self, tile):
        """获取地块属性"""
        terrain_type = str(tile.get('terrain_type', ''))

        # 处理数字类型的terrain_type
        terrain_map = {
            '0': 'START_POSITION', '1': 'NORMAL_LV1', '2': 'NORMAL_LV2',
            '3': 'NORMAL_LV3', '4': 'NORMAL_LV4', '5': 'NORMAL_LV5',
            '6': 'NORMAL_LV6', '7': 'DUMMY_LV1', '8': 'DUMMY_LV2',
            '9': 'DUMMY_LV3', '10': 'DUMMY_LV4', '11': 'DUMMY_LV5',
            '12': 'DUMMY_LV6', '13': 'TRAINING_GROUND', '14': 'TOWER_LV1',
            '15': 'TOWER_LV2', '16': 'TOWER_LV3', '17': 'TOWER_LV4',
            '18': 'TOWER_LV5', '19': 'TOWER_LV6', '20': 'BLACK_MARKET',
            '21': 'STONE_TABLET', '22': 'TREASURE_1', '23': 'TREASURE_2',
            '24': 'TREASURE_3', '25': 'TREASURE_4', '26': 'TREASURE_5',
            '27': 'TREASURE_6', '28': 'TREASURE_7', '29': 'TREASURE_8',
            '30': 'WALL', '31': 'BOSS_GAARA', '32': 'BOSS_ZETSU',
            '33': 'BOSS_DARTMAN', '34': 'BOSS_SHIRA', '35': 'BOSS_KUSHINA',
            '36': 'BOSS_KISAME', '37': 'BOSS_HANA', '38': 'TENT'
        }

        if terrain_type in terrain_map:
            terrain_type = terrain_map[terrain_type]

        return self.terrain_properties.get(terrain_type, {'food_cost': 100, 'exp_gain': 10, 'score_cost': 0})

    def step(self, action):
        """执行动作"""
        reward = 0
        done = False
        old_exp = self.experience
        old_food = self.food

        team = self.teams[self.current_team]
        from_pos = team['pos']
        action_type = None
        to_pos = from_pos

        if team['pos'] and team['pos'] not in self.hex_map:
            team['pos'] = self.start_pos
            from_pos = team['pos']

        if action == 0:
            action_type = 'rest'
            can_conquer_any = False

            if team['action_points'] > 0:
                if team['pos'] not in self.conquered_tiles:
                    tile = self.hex_map.get(team['pos'])
                    if tile:
                        props = self.get_tile_properties(tile)
                        food_cost = self.apply_cost_reduction(props['food_cost'])
                        if food_cost <= self.food and props['score_cost'] <= self.conquest_score:
                            can_conquer_any = True

                if not can_conquer_any:
                    neighbors = self.get_neighbors(team['pos'])
                    for neighbor in neighbors:
                        if neighbor not in self.conquered_tiles:
                            move_cost = self.calculate_move_cost(neighbor)
                            if move_cost > 0 and move_cost <= self.food:
                                tile = self.hex_map.get(neighbor)
                                if tile:
                                    props = self.get_tile_properties(tile)
                                    total_cost = move_cost + self.apply_cost_reduction(props['food_cost'])
                                    if total_cost <= self.food and props['score_cost'] <= self.conquest_score:
                                        can_conquer_any = True
                                        break

            if can_conquer_any:
                reward = -50
                wasted_actions = sum(t['action_points'] for t in self.teams.values())
                reward -= wasted_actions * 5
            else:
                reward = -1

            self.next_day()

        elif 1 <= action <= 6:
            neighbors = self.get_neighbors(team['pos'])
            if action - 1 < len(neighbors):
                target = neighbors[action - 1]
                if target in self.hex_map:
                    cost = self.calculate_move_cost(target)
                    if cost > 0 and cost <= self.food:
                        self.food -= cost
                        team['pos'] = target
                        to_pos = target
                        action_type = 'move'
                        if target not in self.conquered_tiles:
                            reward = 5
                        else:
                            reward = -10
                    else:
                        reward = -20
                else:
                    reward = -20

        elif action == 7:
            current_pos = team['pos']
            if current_pos and current_pos not in self.conquered_tiles:
                if current_pos in self.hex_map:
                    tile = self.hex_map[current_pos]
                    props = self.get_tile_properties(tile)
                    exp_gain = props['exp_gain']
                    food_cost = self.apply_cost_reduction(props['food_cost'])
                    score_cost = props['score_cost']

                    terrain_type = str(tile.get('terrain_type', ''))
                    if terrain_type in ['38', 'TENT']:
                        action_point_cost = 0
                    else:
                        action_point_cost = 1

                    if (food_cost <= self.food and
                        score_cost <= self.conquest_score and
                        team['action_points'] >= action_point_cost):

                        self.food -= food_cost
                        self.conquest_score -= score_cost
                        self.experience += exp_gain
                        self.conquered_tiles.add(current_pos)
                        team['action_points'] -= action_point_cost
                        action_type = 'conquer'

                        base_reward = 20
                        exp_reward = exp_gain / 20
                        food_penalty = food_cost / 10000 if food_cost > 0 else 0
                        reward = base_reward + exp_reward - food_penalty

                        time_bonus = max(0, (91 - self.current_day) / 91 * 10)
                        reward += time_bonus

                        if hasattr(self, 'last_conquer_day'):
                            if self.current_day == self.last_conquer_day:
                                reward += 5
                        self.last_conquer_day = self.current_day

                        if terrain_type in ['38', 'TENT']:
                            tent_food = self.get_tent_food_gain(self.current_day)
                            self.food += tent_food
                            team['action_points'] = min(team['action_points'] + 1, team['max_action_points'])
                            reward += tent_food / 1000

                        if terrain_type in ['22', '23', '24', '25', '26', '27', '28', '29'] or 'TREASURE' in terrain_type:
                            try:
                                if '_' in terrain_type:
                                    treasure_id = int(terrain_type.split('_')[-1])
                                else:
                                    treasure_id = int(terrain_type) - 21
                                self.treasures_conquered.add(treasure_id)
                                reward += 10

                                if len(self.treasures_conquered) == 8:
                                    self.has_treasure_buff = True
                                    reward += 100
                            except:
                                pass

                        if terrain_type in ['32', '35', 'BOSS_ZETSU', 'BOSS_KUSHINA']:
                            self.thunder_god_items += 1
                            reward += 5

                        if terrain_type in ['13', 'TRAINING_GROUND']:
                            reward += 20
                    else:
                        reward = -5
                else:
                    reward = -10
                    team['pos'] = self.start_pos

        elif action == 8:
            if self.thunder_god_items > 0:
                target = self.find_nearest_unconquered_treasure()
                if target and target in self.hex_map:
                    neighbors = self.get_neighbors(target)
                    has_conquered_neighbor = any(n in self.conquered_tiles for n in neighbors)

                    if has_conquered_neighbor:
                        team['pos'] = target
                        to_pos = target
                        self.thunder_god_items -= 1
                        action_type = 'thunder_god'
                        reward = 5
                    else:
                        reward = -2
                else:
                    reward = -2

        elif action == 9:
            if self.weekly_exp_quota >= 100 and self.weekly_claim_count < 5:
                claim_amount = min(100, self.weekly_exp_quota)

                self.experience += claim_amount
                self.weekly_exp_quota -= claim_amount
                self.weekly_exp_claimed += claim_amount
                self.weekly_claim_count += 1
                action_type = 'claim_weekly_exp'

                reward = claim_amount / 20

                old_level = self.level
                self.calculate_level()
                if self.level > old_level:
                    reward += 10
            else:
                reward = -1

        if action_type:
            day_action = DayAction(
                day=self.current_day,
                team_id=self.current_team,
                action_type=action_type,
                from_pos=from_pos,
                to_pos=to_pos,
                food_cost=old_food - self.food,
                exp_gain=self.experience - old_exp,
                resources_after={
                    'food': self.food,
                    'exp': self.experience,
                    'score': self.conquest_score,
                    'level': self.level
                }
            )
            self.action_history.append(day_action)

        if team['action_points'] <= 0:
            self.switch_to_next_team()

        self.check_team_unlock()

        if self.current_day >= 91 or self.food <= 0:
            done = True

            reward += self.experience / 100

            tiles_conquered = len(self.conquered_tiles) - 1
            expected_tiles = min(200, self.current_day * 2)

            if tiles_conquered < expected_tiles:
                reward -= (expected_tiles - tiles_conquered) * 10
            else:
                reward += (tiles_conquered - expected_tiles) * 5

            if self.food > 1000:
                reward -= self.food / 100

            reward += len(self.treasures_conquered) * 20
            reward += self.level * 10

        return reward, done

    def switch_to_next_team(self):
        """切换到下一个有行动点的队伍"""
        team_ids = sorted(self.teams.keys())
        for _ in range(len(team_ids)):
            next_idx = (team_ids.index(self.current_team) + 1) % len(team_ids)
            next_team_id = team_ids[next_idx]
            if self.teams[next_team_id]['action_points'] > 0:
                self.current_team = next_team_id
                break

    def next_day(self):
        """进入下一天"""
        self.current_day += 1

        day_of_week = self.get_day_of_week(self.current_day)

        if day_of_week == 1:
            self.weekly_exp_quota = 500
            self.weekly_exp_claimed = 0
            self.weekly_claim_count = 0
            self.current_week = self.get_week_number(self.current_day)

        elif day_of_week == 0:
            if self.weekly_exp_quota > 0:
                self.experience += self.weekly_exp_quota
                self.weekly_exp_quota = 0
                self.weekly_exp_claimed = 500

        daily_food = self.get_daily_food()
        self.food += daily_food

        self.conquest_score += 1000

        for team in self.teams.values():
            team['action_points'] = min(team['action_points'] + 6, team['max_action_points'])

        self.calculate_level()

    def find_nearest_unconquered_treasure(self):
        """找最近的未征服秘宝"""
        for pos, tile in self.hex_map.items():
            terrain_type = str(tile.get('terrain_type', ''))
            if (terrain_type in ['22', '23', '24', '25', '26', '27', '28', '29'] or
                'TREASURE' in terrain_type) and pos not in self.conquered_tiles:
                return pos
        return None

    def get_state(self):
        """获取状态向量"""
        state = np.zeros(160, dtype=np.float32)

        state[0] = self.current_day / 91
        state[1] = self.experience / 10000
        state[2] = self.food / 10000
        state[3] = self.conquest_score / 10000
        state[4] = self.thunder_god_items / 10
        state[5] = len(self.treasures_conquered) / 8
        state[6] = float(self.has_treasure_buff)

        state[7] = self.weekly_exp_quota / 500
        state[8] = self.weekly_exp_claimed / 500
        state[9] = self.weekly_claim_count / 5
        state[10] = self.get_day_of_week(self.current_day) / 7
        state[11] = self.get_week_number(self.current_day) / 13

        team = self.teams[self.current_team]
        state[12] = team['action_points'] / 18
        state[13] = team['pos'][0] / 30 if team['pos'] else 0
        state[14] = team['pos'][1] / 30 if team['pos'] else 0

        idx = 15
        for tid, t in self.teams.items():
            if tid != self.current_team and t['pos']:
                state[idx] = t['pos'][0] / 30
                state[idx + 1] = t['pos'][1] / 30
                state[idx + 2] = t['action_points'] / 18
                idx += 3

        state[idx] = self.level / 100

        return state

    def get_valid_actions(self):
        """获取有效动作列表"""
        actions = [0]
        team = self.teams[self.current_team]

        if team['pos']:
            neighbors = self.get_neighbors(team['pos'])
            for i, pos in enumerate(neighbors[:6]):
                if pos not in self.conquered_tiles:
                    tile = self.hex_map.get(pos)
                    if tile:
                        terrain_type = str(tile.get('terrain_type', ''))
                        if terrain_type != 'WALL' and terrain_type != '30':
                            actions.append(i + 1)

            if team['pos'] not in self.conquered_tiles and team['action_points'] > 0:
                actions.append(7)

        if self.thunder_god_items > 0:
            actions.append(8)

        if self.weekly_exp_quota >= 100 and self.weekly_claim_count < 5:
            actions.append(9)

        return actions


class PPOModel(nn.Module):
    def __init__(self, state_dim=160, action_dim=10, hidden_dim=512, num_layers=3):
        super(PPOModel, self).__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        self.shared = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

    def get_action(self, state, valid_actions=None):
        logits, value = self.forward(state)

        if valid_actions is not None:
            mask = torch.ones_like(logits) * -1e8
            for i, action in enumerate(valid_actions):
                mask[i, action] = 0
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), value


class PPOTrainer:
    def __init__(self, device, lr=5e-4, gamma=0.90, eps_clip=0.3, epochs=15,
                 batch_size=64, n_workers=4, use_amp=True):
        self.device = device
        self.game = GameCore()

        self.model = PPOModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        self.use_amp = use_amp and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.buffer_size = 4096
        self.reset_buffer()

        self.writer = SummaryWriter(f'runs/ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        self.episodes = []
        self.best_exp = 0
        self.episode_count = 0
        self.recent_exp = []

    def reset_buffer(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.valid_actions_list = []

    def collect_trajectory(self, n_steps=2048):
        """收集轨迹数据"""
        self.model.eval()

        with torch.no_grad():
            for _ in range(n_steps):
                state = torch.FloatTensor(self.game.get_state()).unsqueeze(0).to(self.device)
                valid_actions = self.game.get_valid_actions()

                action, log_prob, value = self.model.get_action(state, [valid_actions])
                action_cpu = action.item()

                reward, done = self.game.step(action_cpu)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.log_probs.append(log_prob)
                self.values.append(value)
                self.dones.append(done)
                self.valid_actions_list.append(valid_actions)

                if done:
                    self.episode_count += 1
                    exp = self.game.experience
                    self.recent_exp.append(exp)
                    if len(self.recent_exp) > 100:
                        self.recent_exp.pop(0)

                    self.writer.add_scalar('Episode/Experience', exp, self.episode_count)
                    self.writer.add_scalar('Episode/Day', self.game.current_day, self.episode_count)
                    self.writer.add_scalar('Episode/Level', self.game.level, self.episode_count)

                    if exp > self.best_exp:
                        self.best_exp = exp
                        self.save_model('best_model.pth')
                        print(f"\n新纪录！经验值: {exp} (第{self.episode_count}轮)")

                    self.game.reset()

    def compute_returns(self):
        """计算折扣回报"""
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        returns = torch.zeros_like(rewards)
        discounted_reward = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_reward = 0
            discounted_reward = rewards[i] + self.gamma * discounted_reward
            returns[i] = discounted_reward

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        """PPO更新"""
        self.model.train()

        returns = self.compute_returns()
        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        old_log_probs = torch.cat(self.log_probs).detach()
        old_values = torch.cat(self.values).detach().squeeze()
        advantages = returns - old_values

        dataset_size = len(states)

        for epoch in range(self.epochs):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                if self.use_amp:
                    with autocast():
                        logits, values = self.model(batch_states)
                        probs = torch.softmax(logits, dim=-1)
                        dist = Categorical(probs)
                        new_log_probs = dist.log_prob(batch_actions)

                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                        entropy = dist.entropy().mean()

                        loss = actor_loss + 0.5 * critic_loss - 0.1 * entropy

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, values = self.model(batch_states)
                    probs = torch.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(batch_actions)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                    entropy = dist.entropy().mean()

                    loss = actor_loss + 0.5 * critic_loss - 0.1 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

        self.scheduler.step()
        self.reset_buffer()

        self.writer.add_scalar('Loss/Total', loss.item(), self.episode_count)
        self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], self.episode_count)

    def train(self, total_episodes=1000):
        """训练主循环"""
        print("=" * 60)
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        print(f"批大小: {self.batch_size}")
        print("=" * 60)

        start_time = time.time()

        with tqdm(total=total_episodes, desc="训练进度") as pbar:
            while self.episode_count < total_episodes:
                self.collect_trajectory(n_steps=self.buffer_size)
                self.update()

                pbar.n = min(self.episode_count, total_episodes)
                avg_exp = np.mean(self.recent_exp) if self.recent_exp else 0
                pbar.set_postfix({
                    'Best': self.best_exp,
                    'Avg': f"{avg_exp:.0f}",
                    'LR': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                pbar.refresh()

                if self.episode_count % 100 == 0:
                    self.save_model(f'checkpoint_{self.episode_count}.pth')

        elapsed = time.time() - start_time
        print(f"\n训练完成！用时: {elapsed/60:.2f}分钟")
        print(f"最佳经验值: {self.best_exp}")
        print(f"平均经验值: {np.mean(self.recent_exp):.0f}")

        self.writer.close()

    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_exp': self.best_exp,
            'episode_count': self.episode_count
        }, filename)

    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_exp = checkpoint['best_exp']
        self.episode_count = checkpoint['episode_count']


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = setup_device()

    trainer = PPOTrainer(
        device=device,
        lr=5e-4,
        gamma=0.90,
        eps_clip=0.3,
        epochs=15,
        batch_size=128,
        n_workers=4,
        use_amp=True
    )

    trainer.train(total_episodes=1000)

    print("\n" + "=" * 60)
    print("训练完成！生成的文件：")
    print("1. best_model.pth - 最佳模型权重")
    print("2. checkpoint_*.pth - 检查点文件")
    print("3. runs/ - TensorBoard日志")
    print("\n查看训练曲线：tensorboard --logdir=runs")
    print("=" * 60)