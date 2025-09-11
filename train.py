"""
六边形地图策略游戏 - PPO强化学习训练脚本 V7
修复版：解决征服逻辑、观察空间、队伍切换、周经验等问题
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Set
import pygame
import json
import os
from collections import deque, Counter
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
    """六边形游戏的Gymnasium环境封装 - V7修复版"""

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
            pygame.display.set_caption("Hex Game AI Training V7")
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.Surface((1200, 800))

        # 初始化游戏
        self.game = None
        self._init_game()

        # 动作空间 - 修改：[动作类型0-7, 目标选择0-29]
        # 增加目标选择空间以分别处理相邻和跳跃目标
        self.action_space = spaces.MultiDiscrete([8, 30])

        # 观察空间 - 扩展到250维以包含更多信息
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(250,),
            dtype=np.float32
        )

        # 改进的奖励权重 V7
        self.reward_weights = {
            'exp_gain': 0.002,
            'level_up': 2.0,
            'conquer': 5.0,
            'first_move': 15.0,
            'exploration': 8.0,
            'efficiency': 0.1,
            'treasure': 25.0,
            'treasure_complete': 100.0,  # 集齐8个秘宝
            'boss_defeat': 50.0,
            'tent_capture': 10.0,
            'tent_early_bonus': 5.0,  # 早期帐篷额外奖励
            'invalid_action': -5.0,
            'no_action_points': -10.0,
            'idle_penalty': -8.0,
            'stuck_penalty': -15.0,
            'waste_action': -5.0,
            'smart_next_day': 3.0,
            'game_over': -100.0,
            'completion_bonus': 500.0,
            'progress_bonus': 2.0,
            # 新增/调整奖励
            'team_spread': 5.0,
            'multi_team_conquest': 15.0,
            'smart_switch': 3.0,
            'jump_move': 3.0,
            'weekly_exp_early': 2.0,  # 早期领取周经验
            'weekly_exp_late': -5.0,  # 周日被动发放
            'high_value_target': 10.0,  # 高价值目标
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
        self.consecutive_invalid = 0

        # 多队伍追踪
        self.team_positions_history = []
        self.daily_team_conquest = {}
        self.total_team_conquest = {}
        self.last_team_spread = 0
        self.team_opportunity_scores = {}  # 各队伍的机会值

        # 记录初始状态
        self.last_exp = 0
        self.last_level = 1
        self.last_conquered = 0
        self.last_treasures = 0
        self.last_day = 1

    def _init_game(self):
        """初始化游戏系统"""
        try:
            self.game = GamePlaySystem(self.screen, 1200, 800, ai_mode=True)

            # 禁用游戏内部日志
            if self.render_mode != "human":
                self.game.add_message = lambda text, msg_type="info": None

            if not self.game.teams:
                print("警告：游戏初始化失败，没有队伍")
        except Exception as e:
            print(f"游戏初始化错误: {e}")
            self.game = GamePlaySystem(self.screen, 1200, 800, ai_mode=True)

    def _calculate_tile_value(self, pos: Tuple[int, int], current_day: int) -> float:
        """计算地块的价值（用于性价比判断）"""
        if pos not in self.game.hex_map:
            return 0

        tile = self.game.hex_map[pos]
        terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(tile.terrain_type)
        props = self.game.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})

        exp_gain = props.get('exp_gain', 0)
        food_cost = props.get('food_cost', 1)

        # 应用秘宝buff
        if self.game.has_treasure_buff and food_cost > 0:
            food_cost = int(food_cost * 0.8)

        # 基础价值 = 经验/消耗
        if food_cost > 0:
            base_value = exp_gain / food_cost
        elif food_cost == 0 and exp_gain > 0:
            base_value = exp_gain / 10.0  # 0消耗地块的特殊处理
        else:
            base_value = 0

        # 特殊地块加成
        if 'BOSS' in terrain_str:
            base_value += 100  # BOSS高价值（经验+飞雷神）
        elif 'TREASURE' in terrain_str:
            treasure_num = 8 - len(self.game.treasures_conquered)
            base_value += 50 + treasure_num * 10  # 越接近集齐越重要
        elif 'TENT' in terrain_str:
            # 帐篷价值随天数递减
            tent_value = 200 / (1 + current_day / 10)
            base_value += tent_value
        elif '历练' in terrain_str or 'TRAINING' in terrain_str:
            base_value += 300  # 520经验0消耗
        elif 'BLACK_MARKET' in terrain_str:
            if self.game.conquest_score >= 1000:
                base_value += 30

        return base_value

    def _calculate_move_efficiency(self, team: Team, target: Tuple[int, int]) -> float:
        """计算移动到目标的性价比"""
        # 计算移动成本
        if target in self.game.get_neighbors(*team.position):
            move_cost = 50
        else:
            path = self.game.find_path_to_unconquered(team.position, target)
            if not path:
                return 0
            steps = len(path)
            move_cost = 30 + 10 * steps

        # 应用秘宝buff
        if self.game.has_treasure_buff:
            move_cost = int(move_cost * 0.8)

        # 计算征服成本
        tile = self.game.hex_map[target]
        props = self.game.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})
        conquer_cost = props.get('food_cost', 0)
        if self.game.has_treasure_buff and conquer_cost > 0:
            conquer_cost = int(conquer_cost * 0.8)

        total_cost = move_cost + conquer_cost

        # 计算价值
        tile_value = self._calculate_tile_value(target, self.game.current_day)

        # 性价比 = 价值 / 总成本
        if total_cost > 0:
            return tile_value / total_cost
        else:
            return tile_value

    def _calculate_team_opportunity(self, team: Team) -> float:
        """计算队伍的机会值（周围可征服地块的总价值）"""
        opportunity = 0

        # 检查队伍位置状态
        if team.position not in self.game.conquered_tiles:
            # 队伍在未征服地块上，需要先征服
            return 1000  # 高优先级

        # 计算周围未征服地块的价值
        neighbors = self.game.get_neighbors(*team.position)
        nearby_unconquered = 0

        for n in neighbors:
            if n in self.game.hex_map and n not in self.game.conquered_tiles:
                tile = self.game.hex_map[n]
                if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                    value = self._calculate_tile_value(n, self.game.current_day)
                    opportunity += value
                    nearby_unconquered += 1

        # 位置优势系数
        if nearby_unconquered >= 3:
            opportunity *= 1.5  # 前线位置
        elif nearby_unconquered == 0:
            opportunity *= 0.1  # 被包围

        # 考虑行动点
        if team.action_points == 0:
            opportunity *= 0.1

        return opportunity

    def _get_valid_actions(self) -> Dict[str, any]:
        """获取当前所有合法动作 - V7增强版"""
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
            'should_next_day': False,  # 新增：是否应该进入下一天
            'team_opportunities': {},  # 各队伍机会值
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

                if tile.terrain_type == TerrainType.TENT:
                    if food_cost >= 0 and food_cost <= self.game.food and score_cost <= self.game.conquest_score:
                        valid_actions['can_conquer'] = True
                else:
                    if (team.action_points > 0 and food_cost >= 0 and
                        food_cost <= self.game.food and score_cost <= self.game.conquest_score):
                        valid_actions['can_conquer'] = True

        # 获取移动目标 - 分离相邻和跳跃，并计算性价比
        if team.position in self.game.conquered_tiles:
            # 相邻地块（保留所有，最多6个）
            neighbors = self.game.get_neighbors(*team.position)
            adjacent_with_efficiency = []

            for n in neighbors:
                if n in self.game.hex_map and n not in self.game.conquered_tiles:
                    tile = self.game.hex_map[n]
                    if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                        cost = 50
                        if self.game.has_treasure_buff:
                            cost = int(cost * 0.8)
                        if cost <= self.game.food:
                            efficiency = self._calculate_move_efficiency(team, n)
                            adjacent_with_efficiency.append((n, efficiency))

            # 按性价比排序相邻目标
            adjacent_with_efficiency.sort(key=lambda x: x[1], reverse=True)
            valid_actions['adjacent_targets'] = [pos for pos, _ in adjacent_with_efficiency]

            # 跳跃目标（限制数量）
            jump_with_efficiency = []

            # 扫描更大范围
            for dq in range(-10, 11):
                for dr in range(-10, 11):
                    if abs(dq + dr) <= 10:
                        target = (team.position[0] + dq, team.position[1] + dr)

                        # 跳过已处理的相邻目标
                        if target in valid_actions['adjacent_targets']:
                            continue

                        if target not in self.game.conquered_tiles and target in self.game.hex_map:
                            tile = self.game.hex_map[target]
                            if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                                path = self.game.find_path_to_unconquered(team.position, target)
                                if path and len(path) > 2:  # 确实是跳跃
                                    steps = len(path)
                                    cost = 30 + 10 * steps
                                    if self.game.has_treasure_buff:
                                        cost = int(cost * 0.8)
                                    if cost <= self.game.food:
                                        efficiency = self._calculate_move_efficiency(team, target)
                                        jump_with_efficiency.append((target, efficiency))

            # 按性价比排序，只保留前15个
            jump_with_efficiency.sort(key=lambda x: x[1], reverse=True)
            valid_actions['jump_targets'] = [pos for pos, _ in jump_with_efficiency[:15]]

        # 飞雷神目标（按价值排序）
        if self.game.thunder_god_items > 0:
            thunder_targets = list(self.game.get_valid_thunder_targets())
            thunder_with_value = []
            for target in thunder_targets:
                value = self._calculate_tile_value(target, self.game.current_day)
                thunder_with_value.append((target, value))
            thunder_with_value.sort(key=lambda x: x[1], reverse=True)
            valid_actions['thunder_targets'] = [pos for pos, _ in thunder_with_value[:5]]

        # 其他动作
        valid_actions['can_switch'] = len(self.game.teams) > 1
        valid_actions['can_claim_exp'] = (
            self.game.weekly_exp_quota > 0 and
            self.game.weekly_claim_count < 5 and
            self.game.game_state == GameState.PLAYING
        )

        # 计算各队伍机会值
        for i, t in enumerate(self.game.teams):
            opportunity = self._calculate_team_opportunity(t)
            valid_actions['team_opportunities'][i] = opportunity

        # 判断是否应该进入下一天
        no_action_points = valid_actions['total_action_points'] == 0
        no_affordable_targets = (len(valid_actions['adjacent_targets']) == 0 and
                                len(valid_actions['jump_targets']) == 0)
        all_teams_trapped = all(opp < 1 for opp in valid_actions['team_opportunities'].values())

        valid_actions['should_next_day'] = (no_action_points or no_affordable_targets or all_teams_trapped)

        return valid_actions

    def _calculate_team_spread(self) -> float:
        """计算队伍分散度"""
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
        self.consecutive_invalid = 0
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
        self.team_opportunity_scores = {}

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
        """获取增强的观察向量 V7 - 250维"""
        obs = np.zeros(250, dtype=np.float32)

        try:
            # 基础资源信息 (0-5)
            obs[0] = np.clip(self.game.food / 15000.0, 0, 10)
            obs[1] = np.clip(self.game.conquest_score / 10000.0, 0, 10)
            obs[2] = np.clip(self.game.experience / 10000.0, 0, 10)
            obs[3] = self.game.level / 100.0
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

                # 行动状态
                obs[15] = 1.0 if valid_actions['has_action_points'] else 0.0
                obs[16] = 1.0 if valid_actions['should_next_day'] else 0.0
                obs[17] = self.daily_actions_done / 50.0

                # 可执行动作
                obs[18] = 1.0 if valid_actions['can_conquer'] else 0.0
                obs[19] = len(valid_actions['adjacent_targets']) / 6.0
                obs[20] = len(valid_actions['jump_targets']) / 15.0
                obs[21] = len(valid_actions['thunder_targets']) / 5.0
                obs[22] = 1.0 if valid_actions['can_switch'] else 0.0
                obs[23] = 1.0 if valid_actions['can_claim_exp'] else 0.0

                # 队伍机会值
                if self.game.current_team_index in valid_actions['team_opportunities']:
                    obs[24] = np.clip(valid_actions['team_opportunities'][self.game.current_team_index] / 1000.0, 0, 1)

                # 队伍分散度
                current_spread = self._calculate_team_spread()
                obs[25] = current_spread / 20.0
                obs[26] = (current_spread - self.last_team_spread) / 10.0

                # 周期特殊状态
                day_of_week = self.game.get_day_of_week(self.game.current_day)
                obs[27] = 1.0 if day_of_week <= 2 else 0.0  # 周一到周三
                obs[28] = 1.0 if day_of_week == 0 else 0.0  # 周日
                obs[29] = self.game.weekly_claim_count / 5.0

            # 所有队伍详细信息 (30-59)
            for i in range(3):
                base_idx = 30 + i * 10
                if i < len(self.game.teams):
                    t = self.game.teams[i]
                    obs[base_idx] = (t.position[0] + 30) / 60.0
                    obs[base_idx + 1] = (t.position[1] + 30) / 60.0
                    obs[base_idx + 2] = t.action_points / 18.0
                    obs[base_idx + 3] = 1.0 if i == self.game.current_team_index else 0.0

                    # 队伍机会值
                    if i in valid_actions['team_opportunities']:
                        obs[base_idx + 4] = np.clip(valid_actions['team_opportunities'][i] / 1000.0, 0, 1)

                    # 队伍征服贡献
                    if i in self.total_team_conquest:
                        obs[base_idx + 5] = self.total_team_conquest[i] / 250.0

                    # 位置状态
                    obs[base_idx + 6] = 1.0 if t.position in self.game.conquered_tiles else 0.0

                    # 是否被困
                    is_trapped = (i in valid_actions['team_opportunities'] and
                                 valid_actions['team_opportunities'][i] < 1)
                    obs[base_idx + 7] = 1.0 if is_trapped else 0.0

            # 游戏进度信息 (60-79)
            obs[60] = len(self.game.conquered_tiles) / 500.0
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
            obs[70] = min(self.consecutive_invalid / 3.0, 1.0)

            # 探索统计
            obs[71] = len(self.exploration_history) / 500.0

            # 秘宝收集详情 (72-79)
            for i in range(8):
                treasure_type = getattr(TerrainType, f'TREASURE_{i+1}', None)
                if treasure_type and treasure_type in self.game.treasures_conquered:
                    obs[72 + i] = 1.0

            # 局部地图信息 - 改进的编码方式 (80-249)
            if self.game.teams:
                team = self.game.teams[self.game.current_team_index]
                idx = 80

                # 使用两层编码：地形类型 + 特征
                for dq in range(-6, 7):  # 13x13网格
                    for dr in range(-6, 7):
                        if idx >= 250:
                            break

                        pos = (team.position[0] + dq, team.position[1] + dr)

                        # 第一层：地形大类编码
                        if pos not in self.game.hex_map:
                            obs[idx] = -1  # 未探索
                        else:
                            tile = self.game.hex_map[pos]
                            terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(tile.terrain_type)

                            # 特殊状态优先
                            if pos == team.position:
                                if pos in self.game.conquered_tiles:
                                    obs[idx] = 9  # 当前位置已征服（需要移动）
                                else:
                                    obs[idx] = 8  # 当前位置未征服（可以征服）
                            elif 'WALL' in terrain_str or 'BOUNDARY' in terrain_str:
                                obs[idx] = -1  # 障碍
                            elif pos in self.game.conquered_tiles:
                                obs[idx] = 0  # 已征服
                            elif 'BOSS' in terrain_str:
                                obs[idx] = 7  # BOSS
                            elif 'TREASURE' in terrain_str:
                                obs[idx] = 6  # 秘宝
                            elif 'TENT' in terrain_str:
                                obs[idx] = 5  # 帐篷
                            elif '历练' in terrain_str or 'TRAINING' in terrain_str:
                                obs[idx] = 4  # 历练之地
                            elif 'BLACK_MARKET' in terrain_str:
                                obs[idx] = 3.5  # 黑商
                            elif 'DUMMY' in terrain_str or '木人' in terrain_str:
                                obs[idx] = 2  # 木人桩
                            elif 'WATCHTOWER' in terrain_str or '瞭望' in terrain_str:
                                obs[idx] = 2.5  # 瞭望塔
                            else:
                                obs[idx] = 1  # 普通地块

                        idx += 1
                        if idx >= 250:
                            break

        except Exception as e:
            print(f"观察获取错误: {e}")

        return obs

    def _get_info(self):
        """获取信息字典"""
        return {
            'exp': self.game.experience,
            'level': self.game.level,
            'day': self.game.current_day,
            'conquered': len(self.game.conquered_tiles),
            'food': self.game.food,
            'treasures': len(self.game.treasures_conquered),
            'exploration': len(self.exploration_history),
            'team_actions': [t.action_points for t in self.game.teams],
            'num_teams': len(self.game.teams),
            'has_moved': self.has_moved,
            'invalid_actions': self.invalid_action_counter,
            'team_spread': self._calculate_team_spread(),
            'team_conquests': self.total_team_conquest.copy(),
        }

    def step(self, action):
        """执行动作 - V7修复版"""
        self.current_step += 1

        # 解析动作
        action_type = int(action[0])
        target_idx = int(action[1])

        # 初始化
        reward = 0
        terminated = False
        truncated = False

        # 记录动作前的状态
        old_food = self.game.food
        old_conquered = len(self.game.conquered_tiles)
        old_position = self.game.teams[self.game.current_team_index].position if self.game.teams else None
        old_day = self.game.current_day
        old_team_spread = self._calculate_team_spread()
        old_team_index = self.game.current_team_index
        old_weekly_exp = self.game.weekly_exp_quota

        # 获取合法动作
        valid_actions = self._get_valid_actions()

        # 检查重复动作
        if action_type == self.last_action_type and action_type not in [1, 2, 3]:  # 移动和征服不算重复
            self.repeated_action_count += 1
            if self.repeated_action_count > 10:
                reward += self.reward_weights['stuck_penalty'] * 0.5
        else:
            self.repeated_action_count = 0
        self.last_action_type = action_type

        try:
            # ========== 执行动作 ==========
            if action_type == 0:  # 无效动作
                reward += self.reward_weights['invalid_action']
                self.invalid_action_counter += 1
                self.consecutive_invalid += 1

            elif action_type == 1:  # 相邻移动
                if self.game.teams and valid_actions['adjacent_targets']:
                    team = self.game.teams[self.game.current_team_index]

                    # 确保索引有效
                    if target_idx < len(valid_actions['adjacent_targets']):
                        target_pos = valid_actions['adjacent_targets'][target_idx]
                    else:
                        target_pos = valid_actions['adjacent_targets'][0] if valid_actions['adjacent_targets'] else None

                    if target_pos:
                        success = self.game.move_team(team, target_pos)
                        if success:
                            reward += 3.0
                            self.daily_actions_done += 1
                            self.consecutive_invalid = 0

                            # 首次移动奖励
                            if not self.has_moved:
                                reward += self.reward_weights['first_move']
                                self.has_moved = True

                            # 探索奖励
                            if target_pos not in self.exploration_history:
                                reward += self.reward_weights['exploration']
                                self.exploration_history.add(target_pos)

                            # 高价值目标奖励
                            tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                            if tile_value > 100:
                                reward += self.reward_weights['high_value_target']

                            self.idle_steps = 0
                            self.stuck_counter = 0
                            self.position_history.append(target_pos)
                        else:
                            reward += self.reward_weights['invalid_action'] * 0.5
                            self.consecutive_invalid += 1
                    else:
                        reward += self.reward_weights['invalid_action']
                        self.consecutive_invalid += 1
                else:
                    reward += self.reward_weights['invalid_action'] * 0.3
                    self.stuck_counter += 1
                    self.consecutive_invalid += 1

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
                            self.consecutive_invalid = 0

                            # 距离奖励
                            distance = (abs(target_pos[0] - old_position[0]) +
                                       abs(target_pos[1] - old_position[1]))
                            reward += distance * 0.5

                            if not self.has_moved:
                                reward += self.reward_weights['first_move']
                                self.has_moved = True

                            if target_pos not in self.exploration_history:
                                reward += self.reward_weights['exploration'] * 1.5
                                self.exploration_history.add(target_pos)

                            # 高价值目标额外奖励
                            tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                            if tile_value > 150:
                                reward += self.reward_weights['high_value_target'] * 1.5

                            self.idle_steps = 0
                            self.stuck_counter = 0
                            self.position_history.append(target_pos)
                        else:
                            reward += self.reward_weights['invalid_action']
                            self.consecutive_invalid += 1
                    else:
                        reward += self.reward_weights['invalid_action']
                        self.consecutive_invalid += 1
                else:
                    reward += self.reward_weights['invalid_action'] * 0.1
                    self.consecutive_invalid += 1

            elif action_type == 3:  # 征服当前地块
                if self.game.teams:
                    team = self.game.teams[self.game.current_team_index]

                    # 检查是否可以征服
                    if team.position in self.game.conquered_tiles:
                        # 已征服的地块，不执行征服，给予惩罚
                        reward += self.reward_weights['invalid_action'] * 2
                        self.invalid_action_counter += 1
                        self.consecutive_invalid += 1
                    elif not valid_actions['can_conquer']:
                        # 不满足征服条件，不执行征服
                        reward += self.reward_weights['invalid_action']
                        self.invalid_action_counter += 1
                        self.consecutive_invalid += 1
                        if not valid_actions['has_action_points']:
                            reward += self.reward_weights['no_action_points']
                    else:
                        # 可以征服，执行征服
                        old_treasures = len(self.game.treasures_conquered)
                        success = self.game.conquer_tile(team)

                        if success:
                            reward += self.reward_weights['conquer']
                            self.daily_actions_done += 1
                            self.consecutive_invalid = 0

                            # 更新队伍征服统计
                            if self.game.current_team_index not in self.daily_team_conquest:
                                self.daily_team_conquest[self.game.current_team_index] = 0
                            self.daily_team_conquest[self.game.current_team_index] += 1

                            if self.game.current_team_index not in self.total_team_conquest:
                                self.total_team_conquest[self.game.current_team_index] = 0
                            self.total_team_conquest[self.game.current_team_index] += 1

                            # 特殊地块奖励
                            tile = self.game.hex_map.get(team.position)
                            if tile:
                                terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(tile.terrain_type)

                                if 'TREASURE' in terrain_str:
                                    reward += self.reward_weights['treasure']
                                    # 集齐8个秘宝的额外奖励
                                    if len(self.game.treasures_conquered) == 8:
                                        reward += self.reward_weights['treasure_complete']
                                elif 'BOSS' in terrain_str:
                                    reward += self.reward_weights['boss_defeat']
                                elif 'TENT' in terrain_str:
                                    reward += self.reward_weights['tent_capture']
                                    # 早期帐篷额外奖励
                                    if self.game.current_day <= 20:
                                        reward += self.reward_weights['tent_early_bonus']
                                elif '历练' in terrain_str:
                                    reward += 30  # 520经验0消耗
                                elif 'BLACK_MARKET' in terrain_str:
                                    reward += 10

                            self.idle_steps = 0
                            self.stuck_counter = 0
                        else:
                            # 征服失败（其他原因）
                            reward += self.reward_weights['invalid_action'] * 0.5
                            self.invalid_action_counter += 1
                            self.consecutive_invalid += 1

            elif action_type == 4:  # 使用飞雷神
                if valid_actions['thunder_targets']:
                    team = self.game.teams[self.game.current_team_index]

                    if target_idx < len(valid_actions['thunder_targets']):
                        target_pos = valid_actions['thunder_targets'][target_idx]
                    else:
                        target_pos = valid_actions['thunder_targets'][0] if valid_actions['thunder_targets'] else None

                    if target_pos:
                        success = self.game.use_thunder_god(team, target_pos)
                        if success:
                            reward += 15.0  # 飞雷神使用奖励
                            self.daily_actions_done += 1
                            self.consecutive_invalid = 0

                            # 高价值目标额外奖励
                            tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                            if tile_value > 200:
                                reward += 20

                            if target_pos not in self.exploration_history:
                                reward += self.reward_weights['exploration'] * 2
                                self.exploration_history.add(target_pos)

                            self.has_moved = True
                            self.stuck_counter = 0
                        else:
                            reward += self.reward_weights['invalid_action']
                            self.consecutive_invalid += 1
                else:
                    reward += self.reward_weights['invalid_action'] * 0.2
                    self.consecutive_invalid += 1

            elif action_type == 5:  # 切换队伍
                if valid_actions['can_switch']:
                    current_team = self.game.teams[self.game.current_team_index]
                    next_index = (self.game.current_team_index + 1) % len(self.game.teams)

                    # 基于机会值判断切换价值
                    current_opportunity = valid_actions['team_opportunities'].get(self.game.current_team_index, 0)
                    next_opportunity = valid_actions['team_opportunities'].get(next_index, 0)

                    # 智能切换判断
                    if current_team.action_points == 0 and self.game.teams[next_index].action_points > 0:
                        reward += self.reward_weights['smart_switch']
                    elif next_opportunity > current_opportunity * 1.5:  # 下一个队伍机会值明显更高
                        reward += self.reward_weights['smart_switch'] * 0.5
                    elif current_opportunity < 1 and next_opportunity > 10:  # 当前队伍被困，下一个队伍有机会
                        reward += self.reward_weights['smart_switch']
                    else:
                        reward += -0.5  # 不必要的切换

                    self.game.current_team_index = next_index
                    self.consecutive_invalid = 0
                else:
                    reward += self.reward_weights['invalid_action'] * 0.1
                    self.consecutive_invalid += 1

            elif action_type == 6:  # 领取经验
                if valid_actions['can_claim_exp']:
                    # 根据target_idx决定领取金额
                    amounts = [100, 200, 300, 400, 500]
                    amount_idx = min(target_idx % 5, len(amounts) - 1)
                    amount = min(amounts[amount_idx], self.game.weekly_exp_quota)

                    old_level = self.game.level
                    success = self.game.claim_weekly_exp(amount)

                    if success:
                        # 基础奖励
                        reward += 0.1 * (amount / 100)

                        # 早期领取奖励（周一到周三）
                        day_of_week = self.game.get_day_of_week(self.game.current_day)
                        if day_of_week <= 3:
                            reward += self.reward_weights['weekly_exp_early']

                        # 升级奖励
                        if self.game.level > old_level:
                            reward += self.reward_weights['level_up'] * (self.game.level - old_level)

                        self.consecutive_invalid = 0

                    # 确保回到游戏状态
                    if self.game.game_state == GameState.WEEKLY_EXP_CLAIM:
                        self.game.game_state = GameState.PLAYING
                else:
                    reward += self.reward_weights['invalid_action']
                    self.consecutive_invalid += 1

            elif action_type == 7:  # 下一天
                if valid_actions['can_next_day']:
                    # 判断是否明智地进入下一天
                    if valid_actions['should_next_day']:
                        # 应该进入下一天
                        reward += self.reward_weights['smart_next_day']
                    else:
                        # 浪费了行动点
                        wasted_points = valid_actions['total_action_points']
                        reward += self.reward_weights['waste_action'] * min(wasted_points, 5)

                    # 检查是否完全没有移动
                    if not self.has_moved and self.game.current_day <= 10:
                        reward += self.reward_weights['stuck_penalty']

                    # 检查周日未领取周经验
                    day_of_week = self.game.get_day_of_week(self.game.current_day)
                    if day_of_week == 6 and self.game.weekly_exp_quota > 0:  # 周六晚上
                        reward += self.reward_weights['weekly_exp_late']  # 即将被动发放

                    old_week = self.game.get_week_number(self.game.current_day)
                    self.game.next_day()
                    new_week = self.game.get_week_number(self.game.current_day)

                    self.daily_actions_done = 0
                    self.daily_team_conquest.clear()
                    self.consecutive_invalid = 0

                    if new_week > old_week:
                        self.weekly_exp_used = 0
                else:
                    reward += self.reward_weights['invalid_action']
                    self.consecutive_invalid += 1

        except Exception as e:
            print(f"动作执行错误: {e}")
            reward += self.reward_weights['invalid_action']
            self.consecutive_invalid += 1

        # 计算增量奖励
        exp_gain = self.game.experience - self.last_exp
        if exp_gain > 0:
            reward += exp_gain * self.reward_weights['exp_gain']

        level_gain = self.game.level - self.last_level
        if level_gain > 0:
            reward += level_gain * self.reward_weights['level_up']

        # 征服进度奖励
        conquered_gain = len(self.game.conquered_tiles) - old_conquered
        if conquered_gain > 0:
            reward += conquered_gain * 5.0
            # 里程碑奖励
            total_conquered = len(self.game.conquered_tiles)
            if total_conquered in [10, 25, 50, 100, 150, 200]:
                reward += 50.0

        # 多队伍协作奖励
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

        # 检查卡住
        if len(self.position_history) > 10:
            recent_positions = self.position_history[-10:]
            if len(set(recent_positions)) <= 2:
                self.stuck_counter += 1
                reward += self.reward_weights['stuck_penalty'] * (self.stuck_counter / 10.0)

        # 连续无效动作惩罚
        if self.consecutive_invalid > 5:
            reward += self.reward_weights['stuck_penalty']
            if self.consecutive_invalid > 10:
                # 严重卡住，考虑提前结束
                truncated = True
                reward += self.reward_weights['game_over'] * 0.5

        # 检查终止条件
        if self.game.game_state == GameState.GAME_OVER:
            terminated = True
            if self.game.current_day >= self.game.max_days:
                # 正常结束
                if self.has_moved:
                    final_bonus = 0
                    final_bonus += self.game.level * 2.0
                    final_bonus += len(self.game.conquered_tiles) * 1.0
                    final_bonus += len(self.game.treasures_conquered) * 20.0
                    if len(self.game.teams) > 1:
                        final_bonus *= 1.5
                    reward += min(final_bonus, self.reward_weights['completion_bonus'])
                else:
                    reward += self.reward_weights['game_over']

        # 步数限制
        if self.current_step >= self.max_steps:
            truncated = True
            if not self.has_moved:
                reward += self.reward_weights['game_over']

        # 更新镜头（如果需要）
        if self.render_mode == "human" and self.game.teams:
            if action_type in [1, 2, 4, 5] or self.current_step % 20 == 0:
                current_team = self.game.teams[self.game.current_team_index]
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
    """增强的训练回调 - V7版本"""

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
        self.episode_team_spread = []
        self.episode_team_conquests = []
        self.episode_steps = []  # 新增：记录每个episode的步数
        self.best_exp = 0
        self.best_level = 0
        self.best_conquered = 0
        self.best_exploration = 0
        self.best_spread = 0

        # 新增：当前episode的步数计数器
        self.current_episode_steps = 0

    def _on_step(self) -> bool:
        # 每步都增加计数
        self.current_episode_steps += 1

        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]

            # 收集数据
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
            self.episode_steps.append(self.current_episode_steps)  # 记录本episode步数

            # 显示结果
            ep_num = len(self.episode_exp)
            print(f"\n📊 Episode {ep_num}:")
            print(f"   等级: {level} | 经验: {exp} | 征服: {conquered} | 探索: {exploration}")
            print(f"   第{day}天结束 | 剩余粮草: {food} | 移动: {'是' if has_moved else '否'}")

            # 新增：显示步数信息和结束原因
            print(f"   ⏱️ 步数: {self.current_episode_steps}/2000 | 平均步/天: {self.current_episode_steps / day:.1f}")

            if self.current_episode_steps >= 2000:
                print(f"   ❌ 因步数达到上限结束 (仅完成{day}/91天)")
            elif day >= 91:
                print(f"   ✅ 正常完成91天 (用时{self.current_episode_steps}步)")
            else:
                print(f"   ⚠️ 其他原因结束")

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

            if new_record:
                print(f"   🏆 新纪录! {', '.join(record_type)}")

            # 重置当前episode步数计数器
            self.current_episode_steps = 0

            # 每20个episode打印详细统计
            if len(self.episode_exp) % 20 == 0:
                self._print_statistics()

        return True

    def _print_statistics(self):
        """打印详细统计信息"""
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
        recent_steps = self.episode_steps[-20:]  # 新增

        print(f"\n{'=' * 70}")
        print(f"📈 Episode {len(self.episode_exp)} 统计汇总 (最近20局)")
        print(f"{'=' * 70}")

        # 新增：步数统计
        print(f"\n⏱️ 步数统计:")
        print(f"   平均步数: {np.mean(recent_steps):.0f} | 最少: {min(recent_steps)} | 最多: {max(recent_steps)}")

        # 统计完成情况
        completed_games = sum(1 for d in recent_days if d >= 91)
        step_limited_games = sum(1 for s in recent_steps if s >= 2000)
        print(f"   完成91天: {completed_games}/20 ({completed_games * 5}%)")
        print(f"   步数耗尽: {step_limited_games}/20 ({step_limited_games * 5}%)")

        # 效率分析
        if completed_games > 0:
            completed_steps = [s for s, d in zip(recent_steps, recent_days) if d >= 91]
            print(f"   完成游戏平均步数: {np.mean(completed_steps):.0f}")

        avg_steps_per_day = [s / d for s, d in zip(recent_steps, recent_days) if d > 0]
        print(f"   平均步/天: {np.mean(avg_steps_per_day):.1f}")

        # 如果步数效率太低，给出警告
        if np.mean(avg_steps_per_day) > 50:
            print(f"   ⚠️ 警告：步数效率过低，可能存在大量无效动作！")

        # 行为统计
        move_rate = sum(recent_has_moved) / len(recent_has_moved) * 100
        avg_invalid = np.mean(recent_invalid)
        print(f"\n🚶 行为统计:")
        print(f"   移动率: {move_rate:.1f}% ({sum(recent_has_moved)}/{len(recent_has_moved)}局有移动)")
        print(f"   平均无效动作: {avg_invalid:.1f} 次/局")
        if avg_invalid > 5:
            print(f"   ⚠️ 警告：无效动作过多，AI可能卡住了！")

        # 基础统计
        print(f"\n📊 基础数据:")
        print(f"   等级: 平均 {np.mean(recent_level):.1f} | 最高 {max(recent_level)} | 最低 {min(recent_level)}")
        print(f"   经验: 平均 {np.mean(recent_exp):.0f} | 最高 {max(recent_exp)} | 最低 {min(recent_exp)}")
        print(
            f"   征服: 平均 {np.mean(recent_conquered):.1f} | 最高 {max(recent_conquered)} | 最低 {min(recent_conquered)}")
        print(f"   探索: 平均 {np.mean(recent_exploration):.1f} | 最高 {max(recent_exploration)}")

        # 多队伍统计
        avg_spread = np.mean([s for s in recent_spread if s > 0])
        if avg_spread > 0:
            print(f"\n👥 多队伍协作:")
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
        print(f"   剩余粮草: 平均 {np.mean(recent_food):.0f} | 最高 {max(recent_food)} | 最低 {min(recent_food)}")
        print(f"   结束天数: 平均 {np.mean(recent_days):.1f} | 最早 {min(recent_days)} | 最晚 {max(recent_days)}")

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
    save_path="hex_game_ppo_v7",
    log_path="./logs/"
):
    """训练PPO智能体 - V7版本"""

    print("=" * 70)
    print("PPO训练 V7 - 修复征服逻辑、观察空间、队伍切换等问题")
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
        learning_rate=5e-4,  # 降低学习率提高稳定性
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,  # 降低熵系数，减少随机探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_path,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]  # 增大网络
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
        name_prefix="ppo_hex_v7"
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
    """测试训练好的智能体 - V7增强版"""

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
        max_consecutive_invalid = 30
        last_day = 1
        stuck_count = 0

        # 动作统计
        action_counts = Counter()
        action_names = {
            0: "等待", 1: "相邻移动", 2: "跳跃移动", 3: "征服",
            4: "飞雷神", 5: "切换队伍", 6: "领取经验", 7: "下一天"
        }

        while not (terminated or truncated):
            old_day = env.game.current_day

            # 处理pygame事件
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
            action_type = int(action[0])
            action_counts[action_type] += 1

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 检查是否换天了
            if old_day != env.game.current_day:
                stuck_count = 0
                last_day = env.game.current_day

                # 显示当天进度
                if steps % 100 == 0 or env.game.current_day % 10 == 0:
                    print(f"  第{env.game.current_day}天: 等级{env.game.level}, "
                          f"征服{info['conquered']}个, 队伍{len(env.game.teams)}个")

            # 检查是否卡住
            if env.game.current_day == last_day:
                stuck_count += 1
                if stuck_count > 150:
                    print(f"  警告：卡在第{last_day}天超过150步！")
                    truncated = True
                    break

            # 检查连续无效动作
            if info.get('invalid_actions', 0) > consecutive_invalid:
                consecutive_invalid = info['invalid_actions']
                if consecutive_invalid > max_consecutive_invalid:
                    print(f"  警告：连续{consecutive_invalid}次无效动作！")
                    truncated = True
                    break

            total_reward += reward
            steps += 1

            if render:
                env.render()
                pygame.time.wait(10)

            # 防止无限循环
            if steps > 5000:
                print(f"  警告：步数超过5000，强制结束")
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

        # 输出结果
        print(f"\n🎮 Episode {episode + 1}/{n_episodes}:")
        print(f"   等级: {info['level']} | 经验: {info['exp']} | 第{info['day']}天结束")
        print(f"   征服: {info['conquered']}个地块 | 探索: {info.get('exploration', 0)}个位置")
        print(f"   剩余粮草: {info.get('food', 0)} | 移动: {'是' if info.get('has_moved', False) else '否'}")
        print(f"   无效动作: {info.get('invalid_actions', 0)}次")

        team_actions = info.get('team_actions', [])
        if team_actions:
            actions_str = " ".join([f"队{i+1}:{ap}点" for i, ap in enumerate(team_actions)])
            print(f"   队伍行动点: {actions_str}")

        if info.get('team_spread', 0) > 0:
            print(f"   队伍分散度: {info['team_spread']:.1f}")

        team_conquests = info.get('team_conquests', {})
        if team_conquests:
            conquests_str = " ".join([f"队{k+1}:{v}个" for k, v in team_conquests.items()])
            print(f"   各队征服: {conquests_str}")

        print(f"   总奖励: {total_reward:.1f} | 总步数: {steps}")

        # 显示动作分布
        print(f"   动作分布:")
        for action_id, count in action_counts.most_common():
            percentage = (count / steps) * 100
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
    print(f"   秘宝: 平均 {np.mean(results['treasures']):.1f} | 最高 {max(results['treasures'])}")

    # 多队伍统计
    avg_spread = np.mean([s for s in results['team_spread'] if s > 0])
    if avg_spread > 0:
        print(f"\n👥 多队伍协作:")
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

    parser = argparse.ArgumentParser(description="六边形游戏PPO训练脚本 V7")
    parser.add_argument("--mode", choices=["train", "test", "check"], default="train")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="hex_game_ppo_v7")
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