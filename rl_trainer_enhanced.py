"""
六边形地图策略游戏 - PPO强化学习训练器（GPU加速版）
支持CUDA加速、混合精度训练、并行数据收集
根据实际游戏规则修正版本
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import time
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


# ========== 设备配置 ==========
def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # 清理GPU缓存
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("GPU不可用，使用CPU训练")

    return device


# ========== 数据类（保持不变） ==========
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


@dataclass
class Episode:
    """记录完整的一局游戏"""
    episode_id: int
    total_reward: float
    final_exp: int
    final_day: int
    actions: List[DayAction] = field(default_factory=list)
    conquered_tiles: List[Tuple[int, int]] = field(default_factory=list)
    treasure_order: List[int] = field(default_factory=list)


# ========== 游戏核心逻辑（根据实际规则修正） ==========
class GameCore:
    """无GUI的游戏核心逻辑，完全匹配实际游戏规则"""

    def __init__(self, map_file='map_save.json'):
        with open(map_file, 'r') as f:
            self.map_data = json.load(f)

        self.action_history = []
        self.reset()

    def reset(self):
        """重置游戏状态 - 使用实际游戏的初始值"""
        self.current_day = 1
        self.experience = 0  # 修正：实际游戏初始经验为0
        self.level = 1
        self.food = 6800  # 修正：实际游戏初始粮草为6800
        self.conquest_score = 1000  # 修正：实际游戏初始征服积分为1000
        self.thunder_god_items = 1  # 修正：实际游戏初始1个飞雷神道具
        self.treasures_conquered = set()
        self.has_treasure_buff = False
        self.conquered_tiles = set()

        # 队伍管理 - 根据实际游戏规则
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

        # 找起始位置 - 修复版本
        self.start_pos = None
        for pos, tile in self.hex_map.items():
            terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串
            if terrain_type == 'START_POSITION':
                self.start_pos = pos
                self.conquered_tiles.add(pos)
                print(f"找到起始位置: {pos}")
                break

        if not self.start_pos:
            # 如果没有找到起始位置，使用地图中的第一个非墙壁位置
            for pos, tile in self.hex_map.items():
                terrain_type = str(tile.get('terrain_type', ''))
                if terrain_type != 'WALL':
                    self.start_pos = pos
                    self.conquered_tiles.add(pos)
                    print(f"使用默认起始位置: {pos}")
                    break

        if not self.start_pos:
            # 最后的备选方案：使用地图中的任意位置
            self.start_pos = next(iter(self.hex_map.keys()))
            self.conquered_tiles.add(self.start_pos)
            print(f"使用地图首个位置作为起始点: {self.start_pos}")

        self.teams[1]['pos'] = self.start_pos

        # 检查队伍解锁
        self.check_team_unlock()

    def calculate_level(self):
        """计算当前等级 - 根据实际游戏规则"""
        self.level = 1 + (self.experience // 100)

    def check_team_unlock(self):
        """检查是否解锁新队伍 - 根据实际游戏规则"""
        self.calculate_level()

        # 20级解锁二队
        if self.level >= 20 and self.max_teams == 1:
            self.max_teams = 2
            if len(self.teams) == 1:
                self.teams[2] = {
                    'pos': self.teams[1]['pos'],
                    'action_points': 6,
                    'max_action_points': 18,
                    'active': True
                }

        # 60级解锁三队
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
        """根据等级获取每日粮草 - 根据实际游戏规则"""
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

    def apply_cost_reduction(self, cost):
        """应用秘宝buff减免 - 根据实际游戏规则"""
        if self.has_treasure_buff:
            return int(cost * 0.8)  # 20%减免
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
            # 只返回存在于地图中的邻居
            if neighbor_pos in self.hex_map:
                neighbors.append(neighbor_pos)

        return neighbors

    def find_path_to_unconquered(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """寻找到未征服地块的路径（路径上的中间点必须是已征服的） - 根据实际游戏规则"""
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
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                # 检查是否可通过
                tile = self.hex_map.get(neighbor)
                if not tile:
                    continue

                terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串
                if terrain_type == 'WALL':
                    continue

                # 中间节点必须是已征服的（除非是目标节点）
                if neighbor != end and neighbor not in self.conquered_tiles:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # 没有找到路径

    def calculate_move_cost(self, target):
        """计算移动成本 - 根据实际游戏规则"""
        team = self.teams[self.current_team]

        # 检查目标位置是否存在于地图中
        if target not in self.hex_map:
            return -1  # 无法移动到不存在的位置

        # 不能停留在已征服地块
        if target in self.conquered_tiles:
            return -1

        # 检查是否是墙壁
        target_tile = self.hex_map.get(target)
        if not target_tile:
            return -1

        terrain_type = str(target_tile.get('terrain_type', ''))  # 强制转换为字符串
        if terrain_type == 'WALL':
            return -1

        # 首先检查是否是直接相邻
        neighbors = self.get_neighbors(team['pos'])
        if target in neighbors:
            # 直接相邻的未征服地块固定消耗50
            cost = 50
        else:
            # 不相邻，尝试通过已征服地块到达
            path = self.find_path_to_unconquered(team['pos'], target)
            if path:
                # 路径长度大于2，说明是跳跃移动
                steps = len(path)  # 包括起点和终点
                cost = 30 + 10 * steps
            else:
                return -1  # 无法到达

        # 应用秘宝buff
        return self.apply_cost_reduction(cost)

    def get_tile_properties(self, tile):
        """获取地块属性 - 修复数据类型问题"""
        terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串

        # 根据地形类型返回属性
        terrain_props = {
            'NORMAL_LV1': {'food_cost': 100, 'exp_gain': 10, 'score_cost': 0},
            'NORMAL_LV2': {'food_cost': 200, 'exp_gain': 20, 'score_cost': 0},
            'DUMMY_LV1': {'food_cost': 150, 'exp_gain': 15, 'score_cost': 0},
            'TREASURE_1': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_2': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_3': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_4': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_5': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_6': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_7': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TREASURE_8': {'food_cost': 500, 'exp_gain': 100, 'score_cost': 0},
            'TENT': {'food_cost': 0, 'exp_gain': 50, 'score_cost': 0},
            'BLACK_MARKET': {'food_cost': 300, 'exp_gain': 30, 'score_cost': 0},
            'BOSS_ZETSU': {'food_cost': 1000, 'exp_gain': 200, 'score_cost': 0},
            'BOSS_KUSHINA': {'food_cost': 1000, 'exp_gain': 200, 'score_cost': 0},
        }

        return terrain_props.get(terrain_type, {'food_cost': 100, 'exp_gain': 10, 'score_cost': 0})

    def step(self, action):
        """执行动作并记录详细信息 - 根据实际游戏规则"""
        reward = 0
        done = False
        old_exp = self.experience
        old_food = self.food

        team = self.teams[self.current_team]
        from_pos = team['pos']
        action_type = None
        to_pos = from_pos

        # 安全检查：确保当前位置存在
        if team['pos'] and team['pos'] not in self.hex_map:
            print(f"警告：队伍位置 {team['pos']} 不存在于地图中，重置到起始位置")
            team['pos'] = self.start_pos
            from_pos = team['pos']

        if action == 0:  # 结束回合
            action_type = 'rest'
            self.next_day()
            reward = -1

        elif 1 <= action <= 6:  # 移动
            neighbors = self.get_neighbors(team['pos'])
            if action - 1 < len(neighbors):
                target = neighbors[action - 1]
                # 双重检查目标位置
                if target in self.hex_map:
                    cost = self.calculate_move_cost(target)
                    if cost > 0 and cost <= self.food:
                        self.food -= cost
                        team['pos'] = target
                        to_pos = target
                        action_type = 'move'
                        reward = -cost / 1000
                    else:
                        reward = -10  # 无效移动的惩罚
                else:
                    print(f"错误：尝试移动到不存在的位置 {target}")
                    reward = -10

        elif action == 7:  # 征服
            current_pos = team['pos']
            if current_pos and current_pos not in self.conquered_tiles:
                # 安全检查：确保位置存在于地图中
                if current_pos in self.hex_map:
                    tile = self.hex_map[current_pos]
                    props = self.get_tile_properties(tile)
                    exp_gain = props['exp_gain']
                    food_cost = self.apply_cost_reduction(props['food_cost'])

                    # 检查资源和行动点
                    terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串
                    action_point_cost = 0 if terrain_type == 'TENT' else 1

                    if food_cost <= self.food and team['action_points'] >= action_point_cost:
                        self.food -= food_cost
                        self.experience += exp_gain
                        self.conquered_tiles.add(current_pos)
                        team['action_points'] -= action_point_cost
                        action_type = 'conquer'

                        reward = exp_gain / 100 - food_cost / 1000

                        # 特殊地块效果
                        if terrain_type == 'TENT':
                            # 帐篷效果：+粮草 +行动点
                            tent_food = min(800 + self.current_day * 10, 1600)  # 简化的帐篷粮草计算
                            self.food += tent_food
                            team['action_points'] = min(team['action_points'] + 1, team['max_action_points'])
                            reward += tent_food / 1000

                        # 检查秘宝
                        if 'TREASURE' in terrain_type:
                            try:
                                treasure_id = int(terrain_type.split('_')[-1]) if '_' in terrain_type else 0
                                self.treasures_conquered.add(treasure_id)
                                reward += 10

                                if len(self.treasures_conquered) == 8:
                                    self.has_treasure_buff = True
                                    reward += 100
                            except (ValueError, IndexError):
                                pass  # 忽略解析错误

                        # BOSS掉落飞雷神
                        if terrain_type in ['BOSS_ZETSU', 'BOSS_KUSHINA']:
                            self.thunder_god_items += 1
                            reward += 5

                    else:
                        reward = -5  # 资源不足的惩罚
                else:
                    print(f"错误：尝试征服不存在的位置 {current_pos}")
                    reward = -10
                    # 重置到起始位置
                    team['pos'] = self.start_pos

        elif action == 8:  # 飞雷神
            if self.thunder_god_items > 0:
                target = self.find_nearest_unconquered_treasure()
                if target and target in self.hex_map:
                    # 检查是否与已征服地块相邻
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
                    reward = -2  # 没有有效目标的小惩罚

        # 记录行动
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

        # 切换队伍
        if team['action_points'] <= 0:
            self.switch_to_next_team()

        # 检查队伍解锁
        self.check_team_unlock()

        # 检查游戏结束
        if self.current_day >= 91 or self.food <= 0:
            done = True
            reward += self.experience / 1000

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
        """进入下一天 - 根据实际游戏规则"""
        self.current_day += 1

        # 发放每日粮草
        daily_food = self.get_daily_food()
        self.food += daily_food

        # 发放每日征服积分
        self.conquest_score += 1000

        # 恢复所有队伍的行动点数
        for team in self.teams.values():
            team['action_points'] = min(team['action_points'] + 6, team['max_action_points'])

    def find_nearest_unconquered_treasure(self):
        """找最近的未征服秘宝 - 修复数据类型问题"""
        for pos, tile in self.hex_map.items():
            terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串
            if 'TREASURE' in terrain_type and pos not in self.conquered_tiles:
                return pos
        return None

    def get_state(self):
        """获取状态向量（numpy数组）"""
        state = np.zeros(150, dtype=np.float32)

        # 全局信息
        state[0] = self.current_day / 91
        state[1] = self.experience / 10000
        state[2] = self.food / 10000
        state[3] = self.conquest_score / 10000
        state[4] = self.thunder_god_items / 10
        state[5] = len(self.treasures_conquered) / 8
        state[6] = float(self.has_treasure_buff)

        # 当前队伍信息
        team = self.teams[self.current_team]
        state[7] = team['action_points'] / 18
        state[8] = team['pos'][0] / 30 if team['pos'] else 0
        state[9] = team['pos'][1] / 30 if team['pos'] else 0

        # 其他队伍位置
        idx = 10
        for tid, t in self.teams.items():
            if tid != self.current_team and t['pos']:
                state[idx] = t['pos'][0] / 30
                state[idx + 1] = t['pos'][1] / 30
                state[idx + 2] = t['action_points'] / 18
                idx += 3

        return state

    def get_valid_actions(self):
        """获取有效动作列表 - 修复数据类型问题"""
        actions = [0]  # 总是可以结束回合
        team = self.teams[self.current_team]

        if team['pos']:
            neighbors = self.get_neighbors(team['pos'])
            for i, pos in enumerate(neighbors[:6]):
                if pos not in self.conquered_tiles:
                    # 检查是否是墙壁
                    tile = self.hex_map.get(pos)
                    if tile:
                        terrain_type = str(tile.get('terrain_type', ''))  # 强制转换为字符串
                        if terrain_type != 'WALL':
                            actions.append(i + 1)

            if team['pos'] not in self.conquered_tiles and team['action_points'] > 0:
                actions.append(7)

        if self.thunder_god_items > 0:
            actions.append(8)

        return actions


# ========== PPO模型（保持不变） ==========
class PPOModel(nn.Module):
    def __init__(self, state_dim=150, action_dim=9, hidden_dim=512, num_layers=3):
        super(PPOModel, self).__init__()

        # 使用更深的网络
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

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化权重
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


# ========== GPU加速PPO训练器（保持主要结构不变） ==========
class PPOTrainer:
    def __init__(self, device, lr=1e-4, gamma=0.99, eps_clip=0.2, epochs=10,
                 batch_size=64, n_workers=4, use_amp=True):
        self.device = device
        self.game = GameCore()

        # 模型和优化器
        self.model = PPOModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # 混合精度训练
        self.use_amp = use_amp and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers

        # 经验缓冲区（在GPU上）
        self.buffer_size = 4096
        self.reset_buffer()

        # TensorBoard
        self.writer = SummaryWriter(f'runs/ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # 监控
        self.episodes = []
        self.best_exp = 0
        self.episode_count = 0

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
        """收集轨迹数据（GPU优化）"""
        self.model.eval()

        with torch.no_grad():
            for _ in range(n_steps):
                # 获取状态
                state = torch.FloatTensor(self.game.get_state()).unsqueeze(0).to(self.device)
                valid_actions = self.game.get_valid_actions()

                # 获取动作
                action, log_prob, value = self.model.get_action(state, [valid_actions])
                action_cpu = action.item()

                # 执行动作
                reward, done = self.game.step(action_cpu)

                # 存储到缓冲区
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.log_probs.append(log_prob)
                self.values.append(value)
                self.dones.append(done)
                self.valid_actions_list.append(valid_actions)

                if done:
                    # 记录episode
                    self.episode_count += 1
                    exp = self.game.experience

                    self.writer.add_scalar('Episode/Experience', exp, self.episode_count)
                    self.writer.add_scalar('Episode/Day', self.game.current_day, self.episode_count)

                    if exp > self.best_exp:
                        self.best_exp = exp
                        self.save_model('best_model.pth')

                    self.game.reset()

    def compute_returns(self):
        """计算折扣回报（GPU加速）"""
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        returns = torch.zeros_like(rewards)
        discounted_reward = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_reward = 0
            discounted_reward = rewards[i] + self.gamma * discounted_reward
            returns[i] = discounted_reward

        # 标准化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        """PPO更新（GPU批处理）"""
        self.model.train()

        # 准备数据
        returns = self.compute_returns()
        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        old_log_probs = torch.cat(self.log_probs).detach()
        old_values = torch.cat(self.values).detach().squeeze()
        advantages = returns - old_values

        # 创建数据集
        dataset_size = len(states)

        for epoch in range(self.epochs):
            # 打乱数据
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 前向传播（混合精度）
                if self.use_amp:
                    with autocast():
                        logits, values = self.model(batch_states)
                        probs = torch.softmax(logits, dim=-1)
                        dist = Categorical(probs)
                        new_log_probs = dist.log_prob(batch_actions)

                        # PPO损失
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                        entropy = dist.entropy().mean()

                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                    # 反向传播
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准训练
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

                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

        # 更新学习率
        self.scheduler.step()

        # 清空缓冲区
        self.reset_buffer()

        # 记录到TensorBoard
        self.writer.add_scalar('Loss/Total', loss.item(), self.episode_count)
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.episode_count)
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.episode_count)
        self.writer.add_scalar('Loss/Entropy', entropy.item(), self.episode_count)
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
                # 收集数据
                self.collect_trajectory(n_steps=self.buffer_size)

                # 更新模型
                self.update()

                # 更新进度条
                pbar.n = min(self.episode_count, total_episodes)
                pbar.set_postfix({
                    'Best Exp': self.best_exp,
                    'LR': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                pbar.refresh()

                # 定期保存
                if self.episode_count % 100 == 0:
                    self.save_model(f'checkpoint_{self.episode_count}.pth')

        # 训练完成
        elapsed = time.time() - start_time
        print(f"\n训练完成！用时: {elapsed/60:.2f}分钟")
        print(f"最佳经验值: {self.best_exp}")

        # 关闭TensorBoard
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


# ========== 主函数 ==========
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置设备
    device = setup_device()

    # 创建训练器
    trainer = PPOTrainer(
        device=device,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2,
        epochs=10,
        batch_size=128,      # GPU可以处理更大的批次
        n_workers=4,
        use_amp=True         # 启用混合精度训练
    )

    # 开始训练
    trainer.train(total_episodes=1000)

    print("\n" + "=" * 60)
    print("训练完成！生成的文件：")
    print("1. best_model.pth - 最佳模型权重")
    print("2. checkpoint_*.pth - 检查点文件")
    print("3. runs/ - TensorBoard日志")
    print("\n查看训练曲线：tensorboard --logdir=runs")
    print("=" * 60)