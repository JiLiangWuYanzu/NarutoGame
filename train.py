"""
六边形地图策略游戏 - PPO强化学习训练脚本 V6
修复版：解决AI不使用二队和螺旋困住问题
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Set
import pygame
import json
import os
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 导入游戏核心模块
from game_play_system import GamePlaySystem, Team, GameState, TerrainType
from map_style_config import StyleConfig


class HexGameEnv(gym.Env):
    """六边形游戏的Gymnasium环境封装 - V6修复版"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=2000):
        super().__init__()

        # 初始化pygame
        if not pygame.get_init():
            pygame.init()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # 创建屏幕
        if render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Hex Game AI Training V6")
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.Surface((1200, 800))

        # 初始化游戏
        self.game = None
        self._init_game()

        # 动作空间 - 修改：[动作类型0-7, 目标选择0-19]
        self.action_space = spaces.MultiDiscrete([8, 20])

        # 观察空间 - 扩展到200维
        self.observation_space = spaces.Box(
            low=-1000,
            high=100000,
            shape=(200,),
            dtype=np.float32
        )

        # 改进的奖励权重 V6
        self.reward_weights = {
            'exp_gain': 0.001,
            'level_up': 1.0,
            'conquer': 5.0,
            'first_move': 10.0,
            'exploration': 8.0,
            'efficiency': 0.05,
            'treasure': 20.0,
            'boss_defeat': 50.0,
            'tent_capture': 10.0,
            'invalid_action': -2.0,
            'no_action_points': -5.0,
            'idle_penalty': -5.0,
            'stuck_penalty': -10.0,
            'waste_action': -3.0,
            'smart_next_day': 2.0,
            'game_over': -100.0,
            'completion_bonus': 300.0,
            'progress_bonus': 1.0,
            # 新增奖励
            'team_spread': 3.0,  # 队伍分散奖励
            'multi_team_conquest': 10.0,  # 多队协同征服
            'smart_switch': 1.0,  # 智能切换队伍
            'jump_move': 2.0,  # 跳跃移动奖励
        }

        # 追踪变量
        self.idle_steps = 0
        self.last_position = None
        self.exploration_history = set()
        self.position_history = []
        self.has_moved = False
        self.stuck_counter = 0
        self.invalid_action_counter = 0
        self.daily_actions_done = 0
        self.weekly_exp_used = 0
        self.last_action_type = -1
        self.repeated_action_count = 0

        # 新增：多队伍追踪
        self.team_positions_history = []
        self.daily_team_conquest = {}  # 每日征服数（用于奖励计算）
        self.total_team_conquest = {}  # 总征服数（用于最终统计）
        self.last_team_spread = 0  # 上一步的队伍分散度

        # 记录初始状态
        self.last_exp = 0
        self.last_level = 1
        self.last_conquered = 0
        self.last_treasures = 0
        self.last_day = 1

    def _init_game(self):
        """初始化游戏系统"""
        try:
            # 确保无论什么模式都传入ai_mode=True
            self.game = GamePlaySystem(self.screen, 1200, 800, ai_mode=True)

            # 禁用游戏内部日志（训练时不需要）
            if self.render_mode != "human":
                self.game.add_message = lambda text, msg_type="info": None

            if not self.game.teams:
                print("警告：游戏初始化失败，没有队伍")
        except Exception as e:
            print(f"游戏初始化错误: {e}")
            self.game = GamePlaySystem(self.screen, 1200, 800, ai_mode=True)

    def _get_all_reachable_targets(self, team: Team, max_distance: int = 10) -> List[Tuple[int, int]]:
        """获取所有可达的未征服地块（包括跳跃）"""
        reachable = []

        # 扫描范围内所有地块
        for dq in range(-max_distance, max_distance + 1):
            for dr in range(-max_distance, max_distance + 1):
                if abs(dq + dr) <= max_distance:
                    target = (team.position[0] + dq, team.position[1] + dr)

                    # 必须是未征服地块
                    if target in self.game.conquered_tiles:
                        continue

                    # 必须在地图上
                    if target not in self.game.hex_map:
                        continue

                    tile = self.game.hex_map[target]
                    # 不能是墙或边界
                    if tile.terrain_type == TerrainType.WALL or tile.terrain_type == TerrainType.BOUNDARY:
                        continue

                    # 检查是否可达（直接相邻或通过已征服地块）
                    if target in self.game.get_neighbors(*team.position):
                        # 直接相邻
                        reachable.append(target)
                    else:
                        # 尝试寻路
                        path = self.game.find_path_to_unconquered(team.position, target)
                        if path:
                            reachable.append(target)

        return reachable

    def _get_valid_actions(self) -> Dict[str, any]:
        """获取当前所有合法动作 - V6增强版"""
        valid_actions = {
            'adjacent_targets': [],  # 相邻可移动地块
            'jump_targets': [],  # 跳跃可达地块
            'thunder_targets': [],  # 飞雷神目标
            'can_conquer': False,
            'can_switch': False,
            'can_claim_exp': False,
            'has_action_points': False,
            'total_action_points': 0,
            'can_next_day': False,
            # 新增：各队伍状态
            'team_states': [],  # 每个队伍的状态信息
        }

        if not self.game.teams:
            return valid_actions

        team = self.game.teams[self.game.current_team_index]

        # 检查行动点
        valid_actions['has_action_points'] = team.action_points > 0
        valid_actions['total_action_points'] = sum(t.action_points for t in self.game.teams)

        # 是否可以进入下一天
        valid_actions['can_next_day'] = self.game.current_day < self.game.max_days

        # 检查是否可以征服当前位置
        if team.position not in self.game.conquered_tiles:
            tile = self.game.hex_map.get(team.position)
            if tile and tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                props = self.game.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})
                food_cost = self.game.apply_cost_reduction(props.get('food_cost', 0))
                score_cost = props.get('score_cost', 0)

                # 帐篷不需要行动点，其他地块需要
                if tile.terrain_type == TerrainType.TENT:
                    if food_cost >= 0 and food_cost <= self.game.food and score_cost <= self.game.conquest_score:
                        valid_actions['can_conquer'] = True
                else:
                    if (team.action_points > 0 and
                            food_cost >= 0 and
                            food_cost <= self.game.food and
                            score_cost <= self.game.conquest_score):
                        valid_actions['can_conquer'] = True

        # 获取移动目标 - 分离相邻和跳跃
        if team.position in self.game.conquered_tiles:
            # 相邻地块
            neighbors = self.game.get_neighbors(*team.position)
            for n in neighbors:
                if n in self.game.hex_map and n not in self.game.conquered_tiles:
                    tile = self.game.hex_map[n]
                    if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                        cost = 50
                        if self.game.has_treasure_buff:
                            cost = int(cost * 0.8)
                        if cost <= self.game.food:
                            valid_actions['adjacent_targets'].append(n)

            # 跳跃目标（更远的可达地块）
            all_reachable = self._get_all_reachable_targets(team, max_distance=10)
            for target in all_reachable:
                if target not in valid_actions['adjacent_targets']:  # 排除相邻的
                    # 计算跳跃成本
                    path = self.game.find_path_to_unconquered(team.position, target)
                    if path:
                        steps = len(path)
                        cost = 30 + 10 * steps
                        if self.game.has_treasure_buff:
                            cost = int(cost * 0.8)
                        if cost <= self.game.food:
                            valid_actions['jump_targets'].append(target)

            # 限制跳跃目标数量，按价值排序
            if len(valid_actions['jump_targets']) > 15:
                # 按地形价值排序
                def get_tile_value(pos):
                    tile = self.game.hex_map[pos]
                    terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(tile.terrain_type)
                    if 'BOSS' in terrain_str:
                        return 100
                    elif 'TREASURE' in terrain_str:
                        return 80
                    elif 'TENT' in terrain_str:
                        return 60
                    elif 'BLACK_MARKET' in terrain_str:
                        return 50
                    else:
                        props = self.game.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})
                        return props.get('exp_gain', 0)

                valid_actions['jump_targets'].sort(key=get_tile_value, reverse=True)
                valid_actions['jump_targets'] = valid_actions['jump_targets'][:15]

        # 飞雷神目标
        if self.game.thunder_god_items > 0:
            thunder_targets = self.game.get_valid_thunder_targets()
            if thunder_targets:
                valid_actions['thunder_targets'] = list(thunder_targets)[:5]

        # 其他动作
        valid_actions['can_switch'] = len(self.game.teams) > 1
        valid_actions['can_claim_exp'] = (
                self.game.weekly_exp_quota > 0 and
                self.game.weekly_claim_count < 5 and
                self.game.game_state == GameState.PLAYING
        )

        # 收集各队伍状态信息
        for i, t in enumerate(self.game.teams):
            team_info = {
                'index': i,
                'position': t.position,
                'action_points': t.action_points,
                'is_current': i == self.game.current_team_index,
                'nearby_unconquered': 0,  # 周围未征服地块数
            }

            # 计算周围未征服地块
            for n in self.game.get_neighbors(*t.position):
                if n in self.game.hex_map and n not in self.game.conquered_tiles:
                    tile = self.game.hex_map[n]
                    if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                        team_info['nearby_unconquered'] += 1

            valid_actions['team_states'].append(team_info)

        return valid_actions

    def _calculate_team_spread(self) -> float:
        """计算队伍分散度（平均距离）"""
        if len(self.game.teams) <= 1:
            return 0.0

        total_distance = 0
        count = 0

        for i in range(len(self.game.teams)):
            for j in range(i + 1, len(self.game.teams)):
                pos1 = self.game.teams[i].position
                pos2 = self.game.teams[j].position
                # 六边形距离
                distance = (abs(pos1[0] - pos2[0]) +
                           abs(pos1[0] + pos1[1] - pos2[0] - pos2[1]) +
                           abs(pos1[1] - pos2[1])) / 2
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        self.current_step = 0
        self.idle_steps = 0
        self.weekly_exp_used = 0
        self.last_action_type = -1
        self.repeated_action_count = 0
        self.exploration_history.clear()
        self.position_history = []
        self.has_moved = False
        self.stuck_counter = 0
        self.invalid_action_counter = 0
        self.daily_actions_done = 0

        # 重置多队伍追踪
        self.team_positions_history = []
        self.daily_team_conquest = {}
        self.total_team_conquest = {}
        self.last_team_spread = 0

        # 重新初始化游戏
        self._init_game()

        # 记录初始状态
        self.last_exp = self.game.experience
        self.last_level = self.game.level
        self.last_conquered = len(self.game.conquered_tiles)
        self.last_treasures = len(self.game.treasures_conquered)
        self.last_day = self.game.current_day

        if self.game.teams:
            self.last_position = self.game.teams[0].position
            self.exploration_history.add(self.last_position)
            self.position_history = [self.last_position]
            self.last_team_spread = self._calculate_team_spread()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self):
        """获取增强的观察向量 V6 - 200维"""
        obs = np.zeros(200, dtype=np.float32)

        try:
            # 基础资源信息 (0-5)
            obs[0] = self.game.food / 150000.0
            obs[1] = self.game.conquest_score / 91000.0
            obs[2] = self.game.experience / 65000.0
            obs[3] = self.game.level / 630.0
            obs[4] = self.game.thunder_god_items / 3.0
            obs[5] = self.game.weekly_exp_quota / 500.0

            # 时间信息 (6-9)
            obs[6] = self.game.current_day / 91.0
            obs[7] = self.game.get_day_of_week(self.game.current_day) / 7.0
            obs[8] = len(self.game.teams) / 3.0
            obs[9] = (91 - self.game.current_day) / 91.0

            # 获取合法动作
            valid_actions = self._get_valid_actions()

            # 当前队伍信息 (10-29)
            if self.game.teams:
                team = self.game.teams[self.game.current_team_index]
                obs[10] = (team.position[0] + 30) / 60.0
                obs[11] = (team.position[1] + 30) / 60.0
                obs[12] = team.action_points / 18.0
                obs[13] = float(self.game.current_team_index) / 3.0
                obs[14] = 1.0 if team.position in self.game.conquered_tiles else 0.0

                # 行动点相关信息
                obs[15] = 1.0 if valid_actions['has_action_points'] else 0.0
                obs[16] = 0.0 if valid_actions['total_action_points'] > 0 else 1.0
                obs[17] = self.daily_actions_done / 50.0

                # 可执行的动作种类数量
                obs[18] = 1.0 if valid_actions['can_conquer'] else 0.0
                obs[19] = len(valid_actions['adjacent_targets']) / 6.0
                obs[20] = len(valid_actions['jump_targets']) / 15.0
                obs[21] = len(valid_actions['thunder_targets']) / 5.0
                obs[22] = 1.0 if valid_actions['can_switch'] else 0.0
                obs[23] = 1.0 if valid_actions['can_claim_exp'] else 0.0
                obs[24] = 1.0 if self.game.current_day < self.game.max_days else 0.0

                # 队伍分散度 (25-26)
                current_spread = self._calculate_team_spread()
                obs[25] = current_spread / 20.0
                obs[26] = (current_spread - self.last_team_spread) / 10.0

                # 周期特殊状态 (27-29)
                is_sunday_spent = (self.game.get_day_of_week(self.game.current_day) == 0 and
                                   self.game.weekly_exp_quota == 0)
                obs[27] = 1.0 if is_sunday_spent else 0.0
                is_monday = self.game.get_day_of_week(self.game.current_day) == 1
                obs[28] = 1.0 if is_monday else 0.0
                obs[29] = (len(valid_actions['adjacent_targets']) + len(valid_actions['jump_targets'])) / 20.0

            # 所有队伍详细信息 (30-59)
            for i in range(3):
                base_idx = 30 + i * 10
                if i < len(self.game.teams):
                    t = self.game.teams[i]
                    obs[base_idx] = (t.position[0] + 30) / 60.0
                    obs[base_idx + 1] = (t.position[1] + 30) / 60.0
                    obs[base_idx + 2] = t.action_points / 18.0
                    obs[base_idx + 3] = 1.0 if i == self.game.current_team_index else 0.0

                    # 周围未征服地块数
                    nearby_unconquered = 0
                    for n in self.game.get_neighbors(*t.position):
                        if n in self.game.hex_map and n not in self.game.conquered_tiles:
                            tile = self.game.hex_map[n]
                            if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                                nearby_unconquered += 1
                    obs[base_idx + 4] = nearby_unconquered / 6.0

                    # 到最近未征服地块的距离
                    min_dist = 100
                    for pos in self.game.hex_map:
                        if pos not in self.game.conquered_tiles:
                            dist = (abs(pos[0] - t.position[0]) +
                                    abs(pos[0] + pos[1] - t.position[0] - t.position[1]) +
                                    abs(pos[1] - t.position[1])) / 2
                            min_dist = min(min_dist, dist)
                    obs[base_idx + 5] = min_dist / 20.0

                    # 队伍征服贡献
                    if i in self.total_team_conquest:
                        obs[base_idx + 6] = self.total_team_conquest[i] / 250

                    # 位置是否已征服
                    obs[base_idx + 7] = 1.0 if t.position in self.game.conquered_tiles else 0.0

                    # 队伍是否被困
                    is_trapped = nearby_unconquered == 0 and t.position in self.game.conquered_tiles
                    obs[base_idx + 8] = 1.0 if is_trapped else 0.0

                    obs[base_idx + 9] = 0.0

            # 游戏进度信息 (60-69)
            obs[60] = len(self.game.conquered_tiles) / 900.0
            obs[61] = len(self.game.treasures_conquered) / 8.0
            obs[62] = 1.0 if self.game.has_treasure_buff else 0.0
            obs[63] = self.game.weekly_exp_claimed / 500.0
            obs[64] = self.game.weekly_claim_count / 5.0

            # 效率指标
            if self.game.current_day > 0:
                obs[65] = len(self.game.conquered_tiles) / self.game.current_day / 10.0
                obs[66] = self.game.experience / self.game.current_day / 600.0

            # 行为追踪
            obs[67] = min(self.idle_steps / 10.0, 1.0)
            obs[68] = min(self.stuck_counter / 5.0, 1.0)
            obs[69] = min(self.invalid_action_counter / 5.0, 1.0)

            # 全局未征服地块分布 (70-77)
            if self.game.teams:
                team = self.game.teams[self.game.current_team_index]
                directions = [
                    (1, 0), (1, -1), (0, -1), (-1, 0),
                    (-1, 1), (0, 1), (2, -1), (-2, 1)
                ]
                for i, (dq, dr) in enumerate(directions):
                    count = 0
                    for dist in range(1, 15):
                        pos = (team.position[0] + dq * dist, team.position[1] + dr * dist)
                        if pos in self.game.hex_map and pos not in self.game.conquered_tiles:
                            tile = self.game.hex_map[pos]
                            if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                                count += 1
                    obs[70 + i] = min(count / 10.0, 1.0)

            # 局部地图信息 (78-199) - 11x11网格
            if self.game.teams:
                team = self.game.teams[self.game.current_team_index]
                idx = 78

                for dq in range(-5, 6):
                    for dr in range(-5, 6):
                        if idx >= 199:
                            break
                        pos = (team.position[0] + dq, team.position[1] + dr)

                        if pos in self.game.hex_map:
                            tile = self.game.hex_map[pos]
                            terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(
                                tile.terrain_type)

                            # 障碍物
                            if 'WALL' in terrain_str or 'BOUNDARY' in terrain_str:
                                obs[idx] = -1.0
                            # 已征服
                            elif pos in self.game.conquered_tiles:
                                obs[idx] = 0.2
                            # 当前位置
                            elif pos == team.position:
                                obs[idx] = 0.1
                            # 起始位置（特殊处理）
                            elif 'START' in terrain_str:
                                obs[idx] = 0.3
                            # BOSS（最高价值）
                            elif 'BOSS' in terrain_str:
                                obs[idx] = 0.95
                            # 秘宝（必须收集）
                            elif 'TREASURE' in terrain_str:
                                obs[idx] = 0.9
                            # 历练之地（0消耗520经验）
                            elif '历练' in terrain_str or 'TRAINING' in terrain_str:
                                obs[idx] = 0.85
                            # 帐篷（粮草+行动点）
                            elif 'TENT' in terrain_str:
                                obs[idx] = 0.8
                            # 黑商（需要积分）
                            elif 'BLACK_MARKET' in terrain_str:
                                obs[idx] = 0.7
                            # 遗迹石板（特殊处理）
                            elif '遗迹' in terrain_str or 'RELIC' in terrain_str:
                                obs[idx] = 0.5  # 性价比一般（100粮草40经验）
                            # 其他所有可征服地块（根据性价比）
                            else:
                                props = self.game.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})
                                exp_gain = props.get('exp_gain', 0)
                                food_cost = props.get('food_cost', 1)

                                if food_cost > 0 and exp_gain > 0:
                                    efficiency = exp_gain / food_cost
                                    # 性价比映射
                                    # 0.35 (35/100) -> 0.35
                                    # 0.60 (60/100) -> 0.65
                                    obs[idx] = min(0.35 + efficiency * 0.5, 0.65)
                                elif food_cost == 0 and exp_gain > 0:
                                    # 其他0消耗地块
                                    obs[idx] = 0.75
                                else:
                                    obs[idx] = 0.5
                        else:
                            obs[idx] = -0.5
                        idx += 1

                # 最后的特征
                obs[199] = len(self.exploration_history) / 900.0

        except Exception as e:
            print(f"观察获取错误: {e}")

        return obs

    def _get_info(self):
        """获取信息字典"""
        team_actions = []
        for team in self.game.teams:
            team_actions.append(team.action_points)

        return {
            'exp': self.game.experience,
            'level': self.game.level,
            'day': self.game.current_day,
            'conquered': len(self.game.conquered_tiles),
            'food': self.game.food,
            'treasures': len(self.game.treasures_conquered),
            'exploration': len(self.exploration_history),
            'team_actions': team_actions,
            'num_teams': len(self.game.teams),
            'has_moved': self.has_moved,
            'invalid_actions': self.invalid_action_counter,
            'team_spread': self._calculate_team_spread(),
            'team_conquests': self.total_team_conquest.copy(),  # 返回总征服数
        }

    def step(self, action):
        """执行动作 - V6修复版（带完整调试）"""
        self.current_step += 1

        # 解析动作
        action_type = int(action[0])
        target_idx = int(action[1])

        # 初始化
        reward = 0
        terminated = False
        truncated = False

        # 防止卡在周经验界面
        if self.game.game_state == GameState.WEEKLY_EXP_CLAIM:
            self.game.game_state = GameState.PLAYING
            print("警告：检测到周经验界面，已强制关闭")

        # 记录动作前的状态
        old_food = self.game.food
        old_conquered = len(self.game.conquered_tiles)
        old_position = self.game.teams[self.game.current_team_index].position if self.game.teams else None
        old_action_points = self.game.teams[self.game.current_team_index].action_points if self.game.teams else 0
        old_day = self.game.current_day
        old_team_spread = self._calculate_team_spread()
        old_team_index = self.game.current_team_index

        # 获取合法动作
        valid_actions = self._get_valid_actions()

        # ========== 调试代码：检查无行动点征服问题 ==========
        if self.game.teams and action_type == 3:  # 如果选择征服
            team = self.game.teams[self.game.current_team_index]

            # 调试输出（每次都输出，帮助分析）
            if self.render_mode == "human" and (team.action_points == 0 or team.position in self.game.conquered_tiles):
                current_obs = self._get_observation()

                print(f"\n=== 征服动作分析 Step {self.current_step} ===")
                print(f"队伍状态：")
                print(f"  位置: {team.position}")
                print(f"  行动点: {team.action_points}/{team.max_action_points}")
                print(f"  位置已征服: {team.position in self.game.conquered_tiles}")
                print(f"  粮草: {self.game.food}")

                print(f"合法动作判断：")
                print(f"  can_conquer: {valid_actions['can_conquer']}")
                print(f"  has_action_points: {valid_actions['has_action_points']}")
                print(f"  相邻可移动数: {len(valid_actions['adjacent_targets'])}")

                print(f"观察向量关键值：")
                print(f"  obs[12](行动点/18): {current_obs[12]:.3f}")
                print(f"  obs[14](位置已征服): {current_obs[14]}")
                print(f"  obs[15](有行动点): {current_obs[15]}")
                print(f"  obs[18](可征服): {current_obs[18]}")

                if team.action_points == 0:
                    print("❌ 错误：无行动点还在尝试征服！")
                if team.position in self.game.conquered_tiles:
                    print("❌ 错误：位置已征服还在尝试征服！")
        # ========== 调试代码结束 ==========

        # 智能决策：如果所有队伍都没有行动点且无法执行其他有效动作，自动进入下一天
        if (not valid_actions['has_action_points'] and
                valid_actions['total_action_points'] == 0 and
                not valid_actions['can_claim_exp'] and
                self.game.current_day < self.game.max_days):
            action_type = 7  # 修改为7（下一天）
            reward += self.reward_weights['smart_next_day']
            print(f"智能决策：所有队伍无行动点，自动进入下一天")

        # 防止在周日结束后AI无意义循环
        if (self.game.get_day_of_week(self.game.current_day) == 0 and
                self.game.weekly_exp_quota == 0 and
                valid_actions['total_action_points'] == 0):
            action_type = 7
            reward += self.reward_weights['smart_next_day'] * 0.5

        # ========== 修正无效征服动作 ==========
        if action_type == 3 and self.game.teams:
            team = self.game.teams[self.game.current_team_index]

            # 检查是否可以征服
            should_correct = False
            correct_reason = ""

            if team.position in self.game.conquered_tiles:
                should_correct = True
                correct_reason = "位置已征服"
            elif team.action_points == 0:
                # 特殊检查：如果是帐篷，不需要行动点
                tile = self.game.hex_map.get(team.position)
                if tile and tile.terrain_type != TerrainType.TENT:
                    should_correct = True
                    correct_reason = "无行动点"

            if should_correct:
                print(f"修正：{correct_reason}，将征服改为其他动作")

                # 按优先级尝试其他动作
                if valid_actions['adjacent_targets']:
                    action_type = 1  # 改为相邻移动
                    target_idx = 0
                    print(f"  -> 改为相邻移动")
                elif valid_actions['jump_targets']:
                    action_type = 2  # 改为跳跃移动
                    target_idx = 0
                    print(f"  -> 改为跳跃移动")
                elif valid_actions['can_switch'] and len(self.game.teams) > 1:
                    action_type = 5  # 改为切换队伍
                    print(f"  -> 改为切换队伍")
                elif valid_actions['can_claim_exp']:
                    action_type = 6  # 改为领取经验
                    target_idx = 0
                    print(f"  -> 改为领取经验")
                else:
                    action_type = 7  # 改为下一天
                    print(f"  -> 改为进入下一天")
        # ========== 修正结束 ==========

        # 检查重复动作
        if action_type == self.last_action_type:
            self.repeated_action_count += 1
            if self.repeated_action_count > 10 and action_type != 1:
                reward += self.reward_weights['stuck_penalty'] * 0.5
        else:
            self.repeated_action_count = 0
        self.last_action_type = action_type

        try:
            # ========== 执行动作 =========
            if action_type == 1:  # 相邻移动
                if self.game.teams and valid_actions['adjacent_targets']:
                    team = self.game.teams[self.game.current_team_index]
                    if target_idx < len(valid_actions['adjacent_targets']):
                        target_pos = valid_actions['adjacent_targets'][target_idx]
                    else:
                        target_pos = valid_actions['adjacent_targets'][0] if valid_actions['adjacent_targets'] else None

                    if target_pos:
                        success = self.game.move_team(team, target_pos)
                        if success:
                            reward += 2.0
                            self.daily_actions_done += 1
                            if not self.has_moved:
                                reward += self.reward_weights['first_move']
                                self.has_moved = True
                            if target_pos not in self.exploration_history:
                                reward += self.reward_weights['exploration']
                                self.exploration_history.add(target_pos)

                            tile = self.game.hex_map.get(target_pos)
                            if tile:
                                terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type,
                                                                                      'value') else str(
                                    tile.terrain_type)
                                if 'TREASURE' in terrain_str:
                                    reward += 5.0
                                elif 'BOSS' in terrain_str:
                                    reward += 8.0
                                elif 'TENT' in terrain_str:
                                    reward += 3.0

                            self.idle_steps = 0
                            self.stuck_counter = 0
                            self.invalid_action_counter = 0
                            self.position_history.append(target_pos)
                        else:
                            reward += self.reward_weights['invalid_action'] * 0.5
                    else:
                        reward += self.reward_weights['invalid_action']
                else:
                    reward += self.reward_weights['invalid_action'] * 0.3
                    self.stuck_counter += 1

            elif action_type == 2:  # 跳跃移动
                if self.game.teams and valid_actions['jump_targets']:
                    team = self.game.teams[self.game.current_team_index]
                    if target_idx < len(valid_actions['jump_targets']):
                        target_pos = valid_actions['jump_targets'][target_idx]
                    else:
                        target_pos = valid_actions['jump_targets'][0] if valid_actions['jump_targets'] else None

                    if target_pos:
                        success = self.game.move_team(team, target_pos)
                        if success:
                            reward += self.reward_weights['jump_move']
                            self.daily_actions_done += 1
                            distance = (abs(target_pos[0] - old_position[0]) +
                                        abs(target_pos[1] - old_position[1]))
                            reward += distance * 0.5

                            if not self.has_moved:
                                reward += self.reward_weights['first_move']
                                self.has_moved = True
                            if target_pos not in self.exploration_history:
                                reward += self.reward_weights['exploration'] * 1.5
                                self.exploration_history.add(target_pos)

                            tile = self.game.hex_map.get(target_pos)
                            if tile:
                                terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type,
                                                                                      'value') else str(
                                    tile.terrain_type)
                                if 'TREASURE' in terrain_str:
                                    reward += 10.0
                                elif 'BOSS' in terrain_str:
                                    reward += 15.0
                                elif 'TENT' in terrain_str:
                                    reward += 5.0

                            self.idle_steps = 0
                            self.stuck_counter = 0
                            self.invalid_action_counter = 0
                            self.position_history.append(target_pos)
                        else:
                            reward += self.reward_weights['invalid_action']
                    else:
                        reward += self.reward_weights['invalid_action']
                else:
                    reward += self.reward_weights['invalid_action'] * 0.1

            elif action_type == 3:  # 征服当前地块
                if self.game.teams:
                    team = self.game.teams[self.game.current_team_index]

                    if not valid_actions['can_conquer']:
                        reward += self.reward_weights['invalid_action']
                        self.invalid_action_counter += 1
                        if not valid_actions['has_action_points']:
                            reward += self.reward_weights['no_action_points']
                    else:
                        old_treasures = len(self.game.treasures_conquered)
                        success = self.game.conquer_tile(team)

                        if success:
                            reward += self.reward_weights['conquer']
                            self.daily_actions_done += 1
                            self.invalid_action_counter = 0

                            if self.game.current_team_index not in self.daily_team_conquest:
                                self.daily_team_conquest[self.game.current_team_index] = 0
                            self.daily_team_conquest[self.game.current_team_index] += 1

                            if self.game.current_team_index not in self.total_team_conquest:
                                self.total_team_conquest[self.game.current_team_index] = 0
                            self.total_team_conquest[self.game.current_team_index] += 1

                            tile = self.game.hex_map.get(team.position)
                            if tile:
                                terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type,
                                                                                      'value') else str(
                                    tile.terrain_type)
                                if 'TREASURE' in terrain_str:
                                    reward += self.reward_weights['treasure']
                                    if len(self.game.treasures_conquered) == 8:
                                        reward += 100.0
                                elif 'BOSS' in terrain_str:
                                    reward += self.reward_weights['boss_defeat']
                                elif 'TENT' in terrain_str:
                                    reward += self.reward_weights['tent_capture']
                                elif 'BLACK_MARKET' in terrain_str:
                                    reward += 8.0

                            self.idle_steps = 0
                            self.stuck_counter = 0
                        else:
                            reward += self.reward_weights['invalid_action']
                            self.invalid_action_counter += 1

            elif action_type == 4:  # 使用飞雷神
                if valid_actions['thunder_targets']:
                    team = self.game.teams[self.game.current_team_index]
                    if target_idx < len(valid_actions['thunder_targets']):
                        target_pos = valid_actions['thunder_targets'][target_idx]
                    else:
                        target_pos = valid_actions['thunder_targets'][0]

                    success = self.game.use_thunder_god(team, target_pos)
                    if success:
                        reward += 10.0
                        self.daily_actions_done += 1
                        if target_pos not in self.exploration_history:
                            reward += self.reward_weights['exploration'] * 2
                            self.exploration_history.add(target_pos)
                        self.has_moved = True
                        self.stuck_counter = 0
                        self.invalid_action_counter = 0
                else:
                    reward += self.reward_weights['invalid_action'] * 0.2

            elif action_type == 5:  # 切换队伍
                if valid_actions['can_switch']:
                    current_team = self.game.teams[self.game.current_team_index]
                    next_index = (self.game.current_team_index + 1) % len(self.game.teams)
                    next_team = self.game.teams[next_index]

                    if current_team.action_points == 0 and next_team.action_points > 0:
                        reward += self.reward_weights['smart_switch']
                    elif current_team.action_points > 0:
                        current_nearby = 0
                        for n in self.game.get_neighbors(*current_team.position):
                            if n in self.game.hex_map and n not in self.game.conquered_tiles:
                                tile = self.game.hex_map[n]
                                if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                                    current_nearby += 1

                        next_nearby = 0
                        for n in self.game.get_neighbors(*next_team.position):
                            if n in self.game.hex_map and n not in self.game.conquered_tiles:
                                tile = self.game.hex_map[n]
                                if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                                    next_nearby += 1

                        if current_nearby == 0 and next_nearby > 0:
                            reward += self.reward_weights['smart_switch'] * 0.5
                        else:
                            reward += -0.1

                    self.game.current_team_index = next_index
                    self.invalid_action_counter = 0
                else:
                    reward += self.reward_weights['invalid_action'] * 0.1

            elif action_type == 6:  # 领取经验
                if self.game.weekly_exp_quota > 0 and self.game.weekly_claim_count < 5:
                    amounts = [100, 200, 300, 400, 500]
                    amount_idx = min(target_idx % 5, len(amounts) - 1)
                    amount = min(amounts[amount_idx], self.game.weekly_exp_quota)

                    if self.game.claim_weekly_exp(amount):
                        reward += 0.05 * (amount / 100)
                        self.weekly_exp_used = self.game.weekly_claim_count

                    self.game.game_state = GameState.PLAYING
                else:
                    reward += self.reward_weights['invalid_action']
                    self.invalid_action_counter += 1

            elif action_type == 7:  # 下一天
                if self.game.current_day < self.game.max_days:
                    total_action_points = sum(team.action_points for team in self.game.teams)

                    if total_action_points == 0:
                        reward += self.reward_weights['smart_next_day'] * 2
                    elif total_action_points <= 3:
                        reward += self.reward_weights['smart_next_day']
                    elif total_action_points > 12:
                        reward += self.reward_weights['waste_action'] * (total_action_points / 18.0)
                    else:
                        reward += 0.5

                    if not self.has_moved and self.game.current_day <= 10:
                        reward += self.reward_weights['stuck_penalty']

                    old_week = self.game.get_week_number(self.game.current_day)
                    self.game.next_day()
                    new_week = self.game.get_week_number(self.game.current_day)

                    self.daily_actions_done = 0
                    self.invalid_action_counter = 0
                    self.daily_team_conquest.clear()

                    if new_week > old_week:
                        self.weekly_exp_used = 0

        except Exception as e:
            print(f"动作执行错误: {e}")
            reward += self.reward_weights['invalid_action']

        # 如果一天结束了，重置每日计数
        if self.game.current_day > old_day:
            self.daily_actions_done = 0
            self.invalid_action_counter = 0

        # 计算增量奖励
        exp_gain = self.game.experience - self.last_exp
        if exp_gain > 0:
            if action_type == 3:
                reward += exp_gain * self.reward_weights['exp_gain'] * 5
            else:
                reward += exp_gain * self.reward_weights['exp_gain']

        level_gain = self.game.level - self.last_level
        if level_gain > 0:
            if self.has_moved:
                reward += level_gain * self.reward_weights['level_up']
            else:
                reward -= level_gain * 0.5

        # 征服进度奖励
        conquered_gain = len(self.game.conquered_tiles) - old_conquered
        if conquered_gain > 0:
            reward += conquered_gain * 5.0
            total_conquered = len(self.game.conquered_tiles)
            if total_conquered in [5, 10, 20, 30, 50, 75, 100, 150]:
                reward += 30.0

        # 多队协同奖励
        if len(self.game.teams) > 1:
            teams_with_conquest = sum(1 for count in self.daily_team_conquest.values() if count > 0)
            if teams_with_conquest > 1:
                reward += self.reward_weights['multi_team_conquest'] * teams_with_conquest

            new_spread = self._calculate_team_spread()
            if new_spread > old_team_spread:
                reward += self.reward_weights['team_spread'] * (new_spread - old_team_spread)
            self.last_team_spread = new_spread

        # 效率奖励
        if old_food > self.game.food and exp_gain > 0:
            efficiency = exp_gain / (old_food - self.game.food + 1)
            reward += efficiency * self.reward_weights['efficiency']

        # 进度奖励
        if self.game.current_day > 0 and self.has_moved:
            progress_rate = len(self.game.conquered_tiles) / (self.game.current_day + 1)
            reward += progress_rate * self.reward_weights['progress_bonus']

        # 检查是否卡住
        if len(self.position_history) > 10:
            recent_positions = self.position_history[-10:]
            if len(set(recent_positions)) <= 2:
                self.stuck_counter += 1
                reward += self.reward_weights['stuck_penalty'] * (self.stuck_counter / 10.0)

        # 如果连续无效动作太多，额外惩罚
        if self.invalid_action_counter > 5:
            reward += self.reward_weights['stuck_penalty']

        # 检查终止条件
        if self.game.game_state == GameState.GAME_OVER:
            terminated = True
            if self.game.current_day >= self.game.max_days:
                if self.has_moved:
                    final_bonus = 0
                    final_bonus += self.game.level * 1.0
                    final_bonus += len(self.game.conquered_tiles) * 1.0
                    final_bonus += len(self.game.treasures_conquered) * 15.0
                    if len(self.game.teams) > 1:
                        final_bonus *= 1.2
                    reward += min(final_bonus, self.reward_weights['completion_bonus'])
                else:
                    reward += self.reward_weights['game_over'] * 2
            elif self.game.food <= 0:
                reward += self.reward_weights['game_over']

        # 步数限制
        if self.current_step >= self.max_steps:
            truncated = True
            if not self.has_moved:
                reward += self.reward_weights['game_over']

        # 在执行动作后，如果是渲染模式，让镜头跟随当前队伍
        if self.render_mode == "human" and self.game.teams:
            current_team = self.game.teams[self.game.current_team_index]
            if (action_type == 5 or action_type == 1 or action_type == 2 or
                    action_type == 4 or self.current_step % 20 == 0):
                self.game.center_camera_on_position(current_team.position)

        # 更新记录
        self.last_exp = self.game.experience
        self.last_level = self.game.level
        self.last_conquered = len(self.game.conquered_tiles)
        self.last_treasures = len(self.game.treasures_conquered)
        self.last_day = self.game.current_day
        if self.game.teams:
            self.last_position = self.game.teams[self.game.current_team_index].position

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染游戏画面"""
        if self.render_mode == "human":
            try:
                self.game.draw()
                pygame.display.flip()
            except Exception as e:
                print(f"渲染错误: {e}")

    def close(self):
        """关闭环境"""
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()


class TrainingCallback(BaseCallback):
    """增强的训练回调 - V6版本"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_exp = []
        self.episode_level = []
        self.episode_conquered = []
        self.episode_exploration = []
        self.episode_food = []
        self.episode_days = []
        self.episode_team_actions = []
        self.episode_has_moved = []
        self.episode_invalid_actions = []
        self.episode_team_spread = []  # 新增：队伍分散度
        self.episode_team_conquests = []  # 新增：各队伍征服数
        self.best_exp = 0
        self.best_level = 0
        self.best_conquered = 0
        self.best_exploration = 0
        self.best_spread = 0  # 新增：最佳分散度

    def _on_step(self) -> bool:
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]
            exp = info.get('exp', 0)
            level = info.get('level', 0)
            conquered = info.get('conquered', 0)
            exploration = info.get('exploration', 0)
            food = info.get('food', 0)
            day = info.get('day', 0)
            team_actions = info.get('team_actions', [])
            has_moved = info.get('has_moved', False)
            invalid_actions = info.get('invalid_actions', 0)
            team_spread = info.get('team_spread', 0)
            team_conquests = info.get('team_conquests', {})

            # 添加到历史记录
            self.episode_exp.append(exp)
            self.episode_level.append(level)
            self.episode_conquered.append(conquered)
            self.episode_exploration.append(exploration)
            self.episode_food.append(food)
            self.episode_days.append(day)
            self.episode_team_actions.append(team_actions)
            self.episode_has_moved.append(has_moved)
            self.episode_invalid_actions.append(invalid_actions)
            self.episode_team_spread.append(team_spread)
            self.episode_team_conquests.append(team_conquests)

            # 显示每个episode的结果
            ep_num = len(self.episode_exp)
            print(f"\n📊 Episode {ep_num}:")
            print(f"   等级: {level} | 经验: {exp} | 征服: {conquered} | 探索: {exploration}")
            print(f"   第{day}天结束 | 剩余粮草: {food} | 移动: {'是' if has_moved else '否'}")

            # 显示队伍信息
            if team_actions:
                actions_str = " ".join([f"队{i + 1}:{ap}点" for i, ap in enumerate(team_actions)])
                print(f"   队伍行动点: {actions_str}")

            if team_spread > 0:
                print(f"   队伍分散度: {team_spread:.1f}")

            if team_conquests:
                conquests_str = " ".join([f"队{k + 1}:{v}个" for k, v in team_conquests.items()])
                print(f"   各队征服: {conquests_str}")

            if invalid_actions > 0:
                print(f"   ⚠️ 无效动作: {invalid_actions}次")

            # 检查新纪录
            # 检查新纪录部分
            new_record = False
            record_type = []

            if exp > self.best_exp:
                record_type.append(f"经验({self.best_exp}→{exp})")
                self.best_exp = exp
                new_record = True

            if level > self.best_level:
                record_type.append(f"等级({self.best_level}→{level})")
                self.best_level = level
                new_record = True

            if conquered > self.best_conquered:
                record_type.append(f"征服({self.best_conquered}→{conquered})")
                self.best_conquered = conquered
                new_record = True

            if exploration > self.best_exploration:
                record_type.append(f"探索({self.best_exploration}→{exploration})")
                self.best_exploration = exploration
                new_record = True

            if team_spread > self.best_spread:
                record_type.append(f"分散度({self.best_spread:.1f}→{team_spread:.1f})")
                self.best_spread = team_spread
                new_record = True

            # 如果是新纪录，额外强调
            if new_record:
                print(f"   🏆 新纪录! {', '.join(record_type)}")

            # 每20个episode打印详细统计
            if len(self.episode_exp) % 20 == 0:
                recent_exp = self.episode_exp[-20:]
                recent_level = self.episode_level[-20:]
                recent_conquered = self.episode_conquered[-20:]
                recent_exploration = self.episode_exploration[-20:]
                recent_food = self.episode_food[-20:]
                recent_days = self.episode_days[-20:]
                recent_team_actions = self.episode_team_actions[-20:]
                recent_has_moved = self.episode_has_moved[-20:]
                recent_invalid = self.episode_invalid_actions[-20:]
                recent_spread = self.episode_team_spread[-20:]
                recent_conquests = self.episode_team_conquests[-20:]

                print(f"\n{'=' * 70}")
                print(f"📈 Episode {len(self.episode_exp)} 统计汇总 (最近20局)")
                print(f"{'=' * 70}")

                # 移动和无效动作统计
                move_rate = sum(recent_has_moved) / len(recent_has_moved) * 100
                avg_invalid = np.mean(recent_invalid)
                print(f"\n🚶 行为统计:")
                print(f"   移动率: {move_rate:.1f}% ({sum(recent_has_moved)}/{len(recent_has_moved)}局有移动)")
                print(f"   平均无效动作: {avg_invalid:.1f} 次/局")
                if avg_invalid > 5:
                    print(f"   ⚠️ 警告：无效动作过多，AI可能卡住了！")

                # 基础统计
                print(f"\n📊 基础数据:")
                print(
                    f"   等级: 平均 {np.mean(recent_level):.1f} | 最高 {max(recent_level)} | 最低 {min(recent_level)}")
                print(f"   经验: 平均 {np.mean(recent_exp):.0f} | 最高 {max(recent_exp)} | 最低 {min(recent_exp)}")
                print(
                    f"   征服: 平均 {np.mean(recent_conquered):.1f} | 最高 {max(recent_conquered)} | 最低 {min(recent_conquered)}")
                print(f"   探索: 平均 {np.mean(recent_exploration):.1f} | 最高 {max(recent_exploration)}")

                # 多队伍统计
                avg_spread = np.mean([s for s in recent_spread if s > 0])
                if avg_spread > 0:
                    print(f"\n👥 多队伍协同:")
                    print(f"   平均分散度: {avg_spread:.1f}")

                    # 统计多队征服情况
                    multi_team_games = 0
                    for conquests in recent_conquests:
                        if len(conquests) > 1:
                            multi_team_games += 1
                    if multi_team_games > 0:
                        print(f"   多队协同征服: {multi_team_games}/{len(recent_conquests)}局")

                # 资源统计
                print(f"\n💰 资源管理:")
                print(
                    f"   剩余粮草: 平均 {np.mean(recent_food):.0f} | 最高 {max(recent_food)} | 最低 {min(recent_food)}")
                print(
                    f"   结束天数: 平均 {np.mean(recent_days):.1f} | 最早 {min(recent_days)} | 最晚 {max(recent_days)}")

                # 效率分析
                moved_games = [(c, d, e, f) for c, d, e, f, m in
                               zip(recent_conquered, recent_days, recent_exp, recent_food, recent_has_moved) if m]

                if moved_games:
                    print(f"\n📈 效率分析 (仅统计有移动的{len(moved_games)}局):")
                    moved_conquered, moved_days, moved_exp, moved_food = zip(*moved_games)

                    conquest_eff = [c / d if d > 0 else 0 for c, d in zip(moved_conquered, moved_days)]
                    print(f"   日均征服: {np.mean(conquest_eff):.2f} 个地块/天")

                    exp_eff = [e / d if d > 0 else 0 for e, d in zip(moved_exp, moved_days)]
                    print(f"   日均经验: {np.mean(exp_eff):.1f} 经验/天")

                print(f"\n🏆 历史最佳:")
                print(f"   等级: {self.best_level} | 经验: {self.best_exp}")
                print(f"   征服: {self.best_conquered} | 探索: {self.best_exploration}")
                print(f"   分散度: {self.best_spread:.1f}")
                print(f"{'=' * 70}")

        return True


def make_env():
    """创建环境的工厂函数"""
    def _init():
        env = HexGameEnv(render_mode=None, max_steps=2000)
        env = Monitor(env)
        return env
    return _init


def train_ppo_agent(
    total_timesteps=1000000,
    n_envs=1,
    save_path="hex_game_ppo_v6",
    log_path="./logs/"
):
    """训练PPO智能体 - V6版本"""

    print("=" * 70)
    print("PPO训练 V6 - 修复二队不用和螺旋困住问题")
    print("=" * 70)
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"并行环境数: {n_envs}")
    print(f"总训练步数: {total_timesteps}")
    print("=" * 70)

    # 创建环境
    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])

    # PPO模型 - 增大网络容量
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_path,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256, 256], vf=[512, 256, 256])]  # 增大网络
        )
    )

    # 设置回调
    training_callback = TrainingCallback()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=f"{save_path}/eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="ppo_hex_v6"
    )

    callbacks = [training_callback, eval_callback, checkpoint_callback]

    print("\n开始训练... (Ctrl+C 中断并保存)")
    print("提示: 使用 tensorboard --logdir ./logs 查看训练曲线")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False
        )

        model.save(f"{save_path}/final_model")
        print(f"\n✅ 训练完成！模型保存至 {save_path}")

    except KeyboardInterrupt:
        print("\n⚠️ 训练中断，保存模型...")
        model.save(f"{save_path}/interrupted_model")

    finally:
        env.close()
        eval_env.close()

    return model


def test_agent(model_path, n_episodes=10, render=False):
    """测试训练好的智能体 - 增强调试版本"""

    print(f"\n加载模型: {model_path}")

    render_mode = "human" if render else None
    env = HexGameEnv(render_mode=render_mode, max_steps=3000)

    # 检查模型文件
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            print(f"❌ 找不到模型文件: {model_path}")
            return

    model = PPO.load(model_path)

    results = {
        'rewards': [],
        'exp': [],
        'levels': [],
        'days': [],
        'conquered': [],
        'treasures': [],
        'exploration': [],
        'food': [],
        'team_actions': [],
        'has_moved': [],
        'invalid_actions': [],
        'team_spread': [],
        'team_conquests': [],
    }

    print("\n开始测试...")
    print("=" * 70)

    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        # 防卡死机制
        consecutive_invalid = 0
        max_consecutive_invalid = 20
        last_day = 1
        stuck_count = 0

        # 动作统计
        from collections import Counter
        daily_action_counts = {}  # 记录每天的动作统计
        current_day_actions = []  # 当前天的所有动作
        action_names = {
            0: "等待", 1: "相邻移动", 2: "跳跃移动", 3: "征服",
            4: "飞雷神", 5: "切换队伍", 6: "领取经验", 7: "下一天"
        }

        while not (terminated or truncated):
            # 记录旧的天数
            old_day = env.game.current_day

            # 处理pygame事件（防止窗口无响应）
            if render:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                pygame.event.pump()

            # AI决策
            action, _ = model.predict(obs, deterministic=True)

            # 记录动作
            action_type = int(action[0])
            current_day_actions.append(action_type)

            # 每50步显示调试信息和动作统计
            if steps % 50 == 0:
                # 计算最近50步的动作统计
                recent_actions_info = ""
                if len(current_day_actions) >= 50:
                    recent_50 = current_day_actions[-50:]
                    recent_counter = Counter(recent_50)
                    most_recent = recent_counter.most_common(1)[0]
                    recent_actions_info = f", 最近50步最多: {action_names[most_recent[0]]}({most_recent[1]}次)"

                print(f"Step {steps}, Day {env.game.current_day}, Teams {len(env.game.teams)}, "
                      f"State: {env.game.game_state}{recent_actions_info}")

            # 检查游戏状态
            old_state = env.game.game_state

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 检查是否换天了
            if old_day != env.game.current_day:
                # 统计上一天的动作
                if current_day_actions:
                    action_counter = Counter(current_day_actions)
                    most_common = action_counter.most_common(3)  # 获取前3个最常见的动作

                    print(f"\n第{old_day}天动作统计:")
                    print(f"  总动作数: {len(current_day_actions)}")
                    for action_id, count in most_common:
                        percentage = (count / len(current_day_actions)) * 100
                        print(f"  - {action_names.get(action_id, '未知')}: {count}次 ({percentage:.1f}%)")

                    # 如果某个动作占比超过80%，可能有问题
                    if most_common[0][1] / len(current_day_actions) > 0.8:
                        print(f"  ⚠️ 警告：{action_names[most_common[0][0]]}动作占比过高！")

                    daily_action_counts[old_day] = dict(action_counter)

                # 重置为新的一天
                current_day_actions = []
                stuck_count = 0
                last_day = env.game.current_day

            # 检查是否卡在某一天
            if env.game.current_day == last_day:
                stuck_count += 1
                if stuck_count > 100:  # 如果在同一天超过100步
                    print(f"警告：卡在第{last_day}天超过100步，强制进入下一天")
                    # 分析卡住的原因
                    if current_day_actions:
                        recent_100 = current_day_actions[-100:]
                        stuck_counter = Counter(recent_100)
                        stuck_most = stuck_counter.most_common(1)[0]
                        print(f"  卡住时主要执行: {action_names[stuck_most[0]]}({stuck_most[1]}次)")

                    # 强制执行下一天动作
                    force_next_day_action = np.array([7, 0])
                    obs, reward, terminated, truncated, info = env.step(force_next_day_action)
                    stuck_count = 0

            # 检查游戏状态变化
            if old_state != env.game.game_state:
                print(f"游戏状态变化: {old_state} -> {env.game.game_state}")
                # 如果进入了周经验界面，立即退出
                if env.game.game_state == GameState.WEEKLY_EXP_CLAIM:
                    print("检测到周经验界面，强制关闭")
                    env.game.game_state = GameState.PLAYING

            total_reward += reward
            steps += 1

            if render:
                env.render()
                pygame.time.wait(10)

            # 防止无限循环
            if steps > 5000:
                print(f"警告：步数超过5000，强制结束")
                truncated = True
                break

        # 保存结果
        results['rewards'].append(total_reward)
        results['exp'].append(info['exp'])
        results['levels'].append(info['level'])
        results['days'].append(info['day'])
        results['conquered'].append(info['conquered'])
        results['treasures'].append(info.get('treasures', 0))
        results['exploration'].append(info.get('exploration', 0))
        results['food'].append(info.get('food', 0))
        results['team_actions'].append(info.get('team_actions', []))
        results['has_moved'].append(info.get('has_moved', False))
        results['invalid_actions'].append(info.get('invalid_actions', 0))
        results['team_spread'].append(info.get('team_spread', 0))
        results['team_conquests'].append(info.get('team_conquests', {}))

        # 详细输出
        print(f"\n🎮 Episode {episode + 1}/{n_episodes}:")
        print(f"   等级: {info['level']} | 经验: {info['exp']} | 第{info['day']}天结束")
        print(f"   征服: {info['conquered']}个地块 | 探索: {info.get('exploration', 0)}个位置")
        print(f"   剩余粮草: {info.get('food', 0)} | 移动: {'是' if info.get('has_moved', False) else '否'}")
        print(f"   无效动作: {info.get('invalid_actions', 0)}次")

        team_actions = info.get('team_actions', [])
        if team_actions:
            actions_str = " ".join([f"队{i + 1}:{ap}点" for i, ap in enumerate(team_actions)])
            print(f"   队伍行动点: {actions_str}")

        if info.get('team_spread', 0) > 0:
            print(f"   队伍分散度: {info['team_spread']:.1f}")

        team_conquests = info.get('team_conquests', {})
        if team_conquests:
            conquests_str = " ".join([f"队{k + 1}:{v}个" for k, v in team_conquests.items()])
            print(f"   各队征服: {conquests_str}")

        print(f"   总奖励: {total_reward:.1f} | 总步数: {steps}")

        # 显示整局的动作分布
        if daily_action_counts:
            print(f"\n   整局动作分布摘要:")
            all_actions = []
            for day_actions in daily_action_counts.values():
                for action_id, count in day_actions.items():
                    all_actions.extend([action_id] * count)

            if all_actions:
                total_counter = Counter(all_actions)
                for action_id, count in total_counter.most_common():
                    percentage = (count / len(all_actions)) * 100
                    print(f"     {action_names.get(action_id, '未知')}: {count}次 ({percentage:.1f}%)")

    # 统计
    print("\n" + "=" * 70)
    print("📊 测试结果统计")
    print("=" * 70)

    # 行为统计
    move_rate = sum(results['has_moved']) / len(results['has_moved']) * 100
    avg_invalid = np.mean(results['invalid_actions'])
    print(f"\n🚶 行为统计:")
    print(f"   移动率: {move_rate:.1f}%")
    print(f"   平均无效动作: {avg_invalid:.1f} 次/局")

    # 基础统计
    print(f"\n📈 基础数据:")
    print(f"   等级: 平均 {np.mean(results['levels']):.1f} | 最高 {max(results['levels'])}")
    print(f"   征服: 平均 {np.mean(results['conquered']):.1f} | 最高 {max(results['conquered'])}")
    print(f"   探索: 平均 {np.mean(results['exploration']):.1f} | 最高 {max(results['exploration'])}")

    # 多队伍统计
    avg_spread = np.mean([s for s in results['team_spread'] if s > 0])
    if avg_spread > 0:
        print(f"\n👥 多队伍协同:")
        print(f"   平均分散度: {avg_spread:.1f}")

        # 统计多队征服
        multi_team_games = 0
        for conquests in results['team_conquests']:
            if len(conquests) > 1:
                multi_team_games += 1
        if multi_team_games > 0:
            print(f"   多队协同征服率: {multi_team_games / len(results['team_conquests']) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="六边形游戏PPO训练脚本 V6")
    parser.add_argument("--mode", choices=["train", "test", "check"], default="train")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="hex_game_ppo_v6")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n_test_episodes", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "check":
        print("检查环境...")
        env = HexGameEnv()
        check_env(env)
        print("✅ 环境检查通过!")
        obs, _ = env.reset()
        print(f"观察空间: {obs.shape}")
        print(f"动作空间: {env.action_space}")

        # 测试合法动作
        valid = env._get_valid_actions()
        print(f"初始合法动作: {valid}")
        env.close()

    elif args.mode == "train":
        model = train_ppo_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.model_path
        )

    else:  # test
        model_file = f"{args.model_path}/best/best_model"
        if not os.path.exists(model_file + ".zip"):
            model_file = f"{args.model_path}/final_model"

        test_agent(
            model_path=model_file,
            n_episodes=args.n_test_episodes,
            render=args.render
        )