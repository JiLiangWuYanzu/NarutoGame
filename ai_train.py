"""
六边形地图策略游戏 - PPO强化学习训练脚本 V8
使用动作掩码(Action Masking)解决无效动作问题
需要先安装: pip install sb3-contrib
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

# 使用MaskablePPO替代普通PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 检查CUDA环境
print("=" * 70)
print("检查CUDA环境...")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA不可用，将使用CPU训练（速度较慢）")
print("=" * 70)

# 导入游戏核心模块
from game_play_system import GamePlaySystem, Team, GameState, TerrainType
from map_style_config import StyleConfig


class HexGameEnv(gym.Env):
    """六边形游戏的Gymnasium环境封装 - V8 Action Masking版本"""

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
            pygame.display.set_caption("Hex Game AI Training V8 - Action Masking")
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.Surface((1200, 800))

        # 初始化游戏
        self.game = None
        self._init_game()

        # 修改动作空间为单一维度，便于掩码处理
        # 动作映射：
        # 0-29: 相邻移动到6个方向（实际最多6个）
        # 30-59: 跳跃移动（最多30个目标）
        # 60: 征服当前地块
        # 61-65: 飞雷神（最多5个目标）
        # 66: 切换队伍
        # 67-71: 领取经验（100/200/300/400/500）
        # 72: 下一天
        self.action_space = spaces.Discrete(73)

        # 观察空间保持不变
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(250,),
            dtype=np.float32
        )

        # 奖励权重
        self.reward_weights = {
            'exp_gain': 0.003,
            'level_up': 3.0,
            'conquer': 8.0,
            'first_move': 20.0,
            'exploration': 10.0,
            'efficiency': 0.15,
            'treasure': 30.0,
            'treasure_complete': 150.0,
            'boss_defeat': 60.0,
            'tent_capture': 15.0,
            'tent_early_bonus': 8.0,
            'invalid_action': -0.5,  # 大幅减少，因为动作掩码会避免大部分无效动作
            'no_action_points': -2.0,
            'idle_penalty': -5.0,
            'stuck_penalty': -10.0,
            'waste_action': -3.0,
            'smart_next_day': 5.0,
            'game_over': -50.0,
            'completion_bonus': 600.0,
            'progress_bonus': 3.0,
            'team_spread': 6.0,
            'multi_team_conquest': 18.0,
            'smart_switch': 4.0,
            'jump_move': 4.0,
            'weekly_exp_early': 3.0,
            'weekly_exp_late': -3.0,
            'high_value_target': 12.0,
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
        self.team_opportunity_scores = {}

        # 记录初始状态
        self.last_exp = 0
        self.last_level = 1
        self.last_conquered = 0
        self.last_treasures = 0
        self.last_day = 1

        # 动作统计
        self.action_stats = {
            'adjacent_move': 0,
            'jump_move': 0,
            'conquer': 0,
            'thunder_god': 0,
            'switch_team': 0,
            'claim_exp': 0,
            'next_day': 0,
            'masked': 0,  # 被掩码阻止的动作
        }
        self.action_success = {}

        # 缓存有效动作
        self.cached_valid_actions = None
        self.cache_dirty = True

    def action_masks(self) -> np.ndarray:
        """返回当前状态下的动作掩码

        Returns:
            np.ndarray: 布尔数组，True表示动作可用，False表示不可用
        """
        mask = np.zeros(self.action_space.n, dtype=bool)

        # 获取有效动作
        if self.cache_dirty:
            self.cached_valid_actions = self._get_valid_actions()
            self.cache_dirty = False

        valid = self.cached_valid_actions

        if not self.game.teams:
            # 如果没有队伍，只允许下一天
            mask[72] = True
            return mask

        team = self.game.teams[self.game.current_team_index]

        # 1. 相邻移动 (0-29)
        if team.position in self.game.conquered_tiles:  # 只有在已征服地块上才能移动
            for i, target in enumerate(valid['adjacent_targets'][:30]):
                mask[i] = True

        # 2. 跳跃移动 (30-59)
        if team.position in self.game.conquered_tiles:
            for i, target in enumerate(valid['jump_targets'][:30]):
                mask[30 + i] = True

        # 3. 征服 (60)
        if valid['can_conquer']:
            mask[60] = True

        # 4. 飞雷神 (61-65)
        if self.game.thunder_god_items > 0 and valid['thunder_targets']:
            for i in range(min(len(valid['thunder_targets']), 5)):
                mask[61 + i] = True

        # 5. 切换队伍 (66) - 添加每天6次限制
        if valid['can_switch'] and self.daily_switch_count < 6:
            mask[66] = True

        # 6. 领取经验 (67-71)
        if valid['can_claim_exp']:
            amounts = [100, 200, 300, 400, 500]
            for i, amount in enumerate(amounts):
                if amount <= self.game.weekly_exp_quota:
                    mask[67 + i] = True

        # 7. 下一天 (72) - 简化逻辑
        if valid['can_next_day']:
            can_proceed = False
            # 强制条件：必须满足其一
            if self.daily_conquered_count >= 6:
                can_proceed = True
            elif self.game.food < 50:
                can_proceed = True
            elif valid['total_action_points'] == 0:
                can_proceed = True
            elif (not valid['adjacent_targets'] and
                  not valid['jump_targets'] and
                  not valid['can_conquer']):
                can_proceed = True
            if can_proceed:
                mask[72] = True

        # 防止死锁：如果没有任何其他可用动作，允许下一天
        if not mask[:72].any() and valid['can_next_day']:
            mask[72] = True

        return mask

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
            base_value = exp_gain / 10.0
        else:
            base_value = 0

        # 特殊地块加成
        if 'BOSS' in terrain_str:
            base_value += 100
        elif 'TREASURE' in terrain_str:
            treasure_num = 8 - len(self.game.treasures_conquered)
            base_value += 50 + treasure_num * 10
        elif 'TENT' in terrain_str:
            tent_value = 200 / (1 + current_day / 10)
            base_value += tent_value
        elif '历练' in terrain_str or 'TRAINING' in terrain_str:
            base_value += 300
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
            opportunity *= 1.5
        elif nearby_unconquered == 0:
            opportunity *= 0.1

        # 考虑行动点
        if team.action_points == 0:
            opportunity *= 0.1

        return opportunity

    def _get_valid_actions(self) -> Dict[str, any]:
        """获取当前所有合法动作 - V8优化版"""
        valid_actions = {
            'adjacent_targets': [],
            'jump_targets': [],
            'thunder_targets': [],
            'can_conquer': False,
            'can_switch': False,
            'can_claim_exp': False,
            'has_action_points': False,
            'total_action_points': 0,
            'can_next_day': False,
            'should_next_day': False,
            'team_opportunities': {},
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

        # 获取移动目标
        if team.position in self.game.conquered_tiles:
            # 相邻地块
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

            # 跳跃目标
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

            # 按性价比排序，只保留前30个
            jump_with_efficiency.sort(key=lambda x: x[1], reverse=True)
            valid_actions['jump_targets'] = [pos for pos, _ in jump_with_efficiency[:30]]

        # 飞雷神目标
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

        # 新增：重置切换队伍追踪
        self.daily_switch_count = 0
        self.last_switch_step = -10
        self.switch_count_recent = 0

        # 新增：重置每天征服计数
        self.daily_conquered_count = 0

        # 重置动作统计
        self.action_stats = {
            'adjacent_move': 0,
            'jump_move': 0,
            'conquer': 0,
            'thunder_god': 0,
            'switch_team': 0,
            'claim_exp': 0,
            'next_day': 0,
            'masked': 0,
        }
        self.action_success = {
            'adjacent_move': 0,
            'jump_move': 0,
            'conquer': 0,
            'thunder_god': 0,
            'switch_team': 0,
            'claim_exp': 0,
            'next_day': 0,
        }

        # 重新初始化游戏
        self._init_game()

        # 标记缓存需要更新
        self.cache_dirty = True

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
        """获取增强的观察向量 V8 - 250维"""
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
                obs[20] = len(valid_actions['jump_targets']) / 30.0
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
                obs[27] = 1.0 if day_of_week <= 2 else 0.0
                obs[28] = 1.0 if day_of_week == 0 else 0.0
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
                treasure_type = getattr(TerrainType, f'TREASURE_{i + 1}', None)
                if treasure_type and treasure_type in self.game.treasures_conquered:
                    obs[72 + i] = 1.0

            # 局部地图信息 (80-249)
            if self.game.teams:
                team = self.game.teams[self.game.current_team_index]
                idx = 80

                for dq in range(-6, 7):
                    for dr in range(-6, 7):
                        if idx >= 250:
                            break

                        pos = (team.position[0] + dq, team.position[1] + dr)

                        if pos not in self.game.hex_map:
                            obs[idx] = -1
                        else:
                            tile = self.game.hex_map[pos]
                            terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(
                                tile.terrain_type)

                            # 特殊状态优先
                            if pos == team.position:
                                if pos in self.game.conquered_tiles:
                                    obs[idx] = 9
                                else:
                                    obs[idx] = 8
                            elif 'WALL' in terrain_str or 'BOUNDARY' in terrain_str:
                                obs[idx] = -1
                            elif pos in self.game.conquered_tiles:
                                obs[idx] = 0
                            elif 'BOSS' in terrain_str:
                                obs[idx] = 7
                            elif 'TREASURE' in terrain_str:
                                obs[idx] = 6
                            elif 'TENT' in terrain_str:
                                obs[idx] = 5
                            elif '历练' in terrain_str or 'TRAINING' in terrain_str:
                                obs[idx] = 4
                            elif 'BLACK_MARKET' in terrain_str:
                                obs[idx] = 3.5
                            elif 'DUMMY' in terrain_str or '木人' in terrain_str:
                                obs[idx] = 2
                            elif 'WATCHTOWER' in terrain_str or '瞭望' in terrain_str:
                                obs[idx] = 2.5
                            else:
                                obs[idx] = 1

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
            'action_stats': self.action_stats.copy(),
            'action_success': self.action_success.copy(),
            'total_steps': self.current_step,
        }

    def step(self, action):
        """执行动作 - V8动作掩码版本（调试版）"""
        self.current_step += 1
        self.cache_dirty = True

        # ============ 调试代码开始 ============
        mask = self.action_masks()
        mask_violated = False

        # 原有的掩码检查逻辑
        if not mask[action]:
            self.action_stats['masked'] += 1
            self.consecutive_invalid += 1

            # 强制选择一个有效动作
            if mask[72]:
                action = 72
            else:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
                else:
                    action = 72

        # 解析动作
        action_succeeded = False
        action_name = 'unknown'

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

        # 获取合法动作
        valid_actions = self._get_valid_actions()

        try:
            # ========== 执行动作 ==========
            if action < 30:  # 相邻移动
                action_name = 'adjacent_move'
                self.action_stats[action_name] += 1

                if valid_actions['adjacent_targets'] and action < len(valid_actions['adjacent_targets']):
                    team = self.game.teams[self.game.current_team_index]
                    target_pos = valid_actions['adjacent_targets'][action]

                    success = self.game.move_team(team, target_pos)
                    if success:
                        action_succeeded = True
                        reward += 5.0
                        self.daily_actions_done += 1
                        self.consecutive_invalid = 0

                        if not self.has_moved:
                            reward += self.reward_weights['first_move']
                            self.has_moved = True

                        if target_pos not in self.exploration_history:
                            reward += self.reward_weights['exploration']
                            self.exploration_history.add(target_pos)

                        tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                        if tile_value > 100:
                            reward += self.reward_weights['high_value_target']

                        self.idle_steps = 0
                        self.stuck_counter = 0
                        self.position_history.append(target_pos)

            elif action < 60:  # 跳跃移动
                action_name = 'jump_move'
                self.action_stats[action_name] += 1

                idx = action - 30
                if valid_actions['jump_targets'] and idx < len(valid_actions['jump_targets']):
                    team = self.game.teams[self.game.current_team_index]
                    target_pos = valid_actions['jump_targets'][idx]

                    success = self.game.move_team(team, target_pos)
                    if success:
                        action_succeeded = True
                        reward += self.reward_weights['jump_move']
                        self.daily_actions_done += 1
                        self.consecutive_invalid = 0

                        distance = (abs(target_pos[0] - old_position[0]) +
                                    abs(target_pos[1] - old_position[1]))
                        reward += distance * 0.5

                        if not self.has_moved:
                            reward += self.reward_weights['first_move']
                            self.has_moved = True

                        if target_pos not in self.exploration_history:
                            reward += self.reward_weights['exploration'] * 1.5
                            self.exploration_history.add(target_pos)

                        tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                        if tile_value > 150:
                            reward += self.reward_weights['high_value_target'] * 1.5

                        self.idle_steps = 0
                        self.stuck_counter = 0
                        self.position_history.append(target_pos)


            elif action == 60:  # 征服

                action_name = 'conquer'

                self.action_stats[action_name] += 1

                if valid_actions['can_conquer']:

                    team = self.game.teams[self.game.current_team_index]

                    success = self.game.conquer_tile(team)

                    if success:

                        action_succeeded = True

                        reward += self.reward_weights['conquer']

                        self.daily_actions_done += 1

                        self.consecutive_invalid = 0

                        # 新增：更新每天征服计数

                        self.daily_conquered_count += 1

                        if self.game.current_team_index not in self.daily_team_conquest:
                            self.daily_team_conquest[self.game.current_team_index] = 0

                        self.daily_team_conquest[self.game.current_team_index] += 1

                        if self.game.current_team_index not in self.total_team_conquest:
                            self.total_team_conquest[self.game.current_team_index] = 0

                        self.total_team_conquest[self.game.current_team_index] += 1

                        tile = self.game.hex_map.get(team.position)

                        if tile:

                            terrain_str = str(tile.terrain_type.value) if hasattr(tile.terrain_type, 'value') else str(
                                tile.terrain_type)

                            if 'TREASURE' in terrain_str:

                                reward += self.reward_weights['treasure']

                                if len(self.game.treasures_conquered) == 8:
                                    reward += self.reward_weights['treasure_complete']

                            elif 'BOSS' in terrain_str:

                                reward += self.reward_weights['boss_defeat']

                            elif 'TENT' in terrain_str:

                                reward += self.reward_weights['tent_capture']

                                if self.game.current_day <= 20:
                                    reward += self.reward_weights['tent_early_bonus']

                            elif '历练' in terrain_str:

                                reward += 30

                            elif 'BLACK_MARKET' in terrain_str:

                                reward += 10

                        self.idle_steps = 0

                        self.stuck_counter = 0

            elif 61 <= action <= 65:  # 飞雷神
                action_name = 'thunder_god'
                self.action_stats[action_name] += 1

                idx = action - 61
                if valid_actions['thunder_targets'] and idx < len(valid_actions['thunder_targets']):
                    team = self.game.teams[self.game.current_team_index]
                    target_pos = valid_actions['thunder_targets'][idx]

                    success = self.game.use_thunder_god(team, target_pos)
                    if success:
                        action_succeeded = True
                        reward += 20.0
                        self.daily_actions_done += 1
                        self.consecutive_invalid = 0

                        tile_value = self._calculate_tile_value(target_pos, self.game.current_day)
                        if tile_value > 200:
                            reward += 25

                        if target_pos not in self.exploration_history:
                            reward += self.reward_weights['exploration'] * 2
                            self.exploration_history.add(target_pos)

                        self.has_moved = True
                        self.stuck_counter = 0


            elif action == 66:  # 切换队伍
                action_name = 'switch_team'
                self.action_stats[action_name] += 1
                if valid_actions['can_switch']:
                    current_team = self.game.teams[self.game.current_team_index]
                    next_index = (self.game.current_team_index + 1) % len(self.game.teams)
                    # 检查短期频繁切换
                    steps_since_switch = self.current_step - self.last_switch_step
                    # 惩罚频繁切换
                    if steps_since_switch < 3:  # 3步内再次切换
                        reward += -5.0  # 重惩罚
                        self.switch_count_recent += 1
                        if self.switch_count_recent > 5:  # 短期内切换超过5次
                            reward += -10.0  # 额外重惩罚
                    else:
                        self.switch_count_recent = 0  # 重置计数
                        # 获取机会值
                        current_opportunity = valid_actions['team_opportunities'].get(self.game.current_team_index, 0)
                        next_opportunity = valid_actions['team_opportunities'].get(next_index, 0)
                        # 更严格的奖励条件
                        if current_team.action_points == 0 and self.game.teams[next_index].action_points > 0:
                            # 当前队伍无行动点，下个队伍有行动点
                            reward += self.reward_weights['smart_switch']
                        elif current_opportunity < 5 and next_opportunity > current_opportunity * 2:
                            # 当前队伍机会很少，下个队伍机会至少2倍
                            reward += self.reward_weights['smart_switch'] * 0.5
                        elif current_team.action_points > 0:
                            # 当前队伍还有行动点却切换 - 惩罚
                            reward += -3.0
                        else:
                            # 其他情况给予负奖励
                            reward += -1.0
                    # 执行切换
                    action_succeeded = True
                    self.game.current_team_index = next_index
                    self.last_switch_step = self.current_step
                    self.daily_switch_count += 1  # 更新每天切换计数
                    self.consecutive_invalid = 0
            elif 67 <= action <= 71:  # 领取经验
                action_name = 'claim_exp'
                self.action_stats[action_name] += 1
                if valid_actions['can_claim_exp']:
                    amounts = [100, 200, 300, 400, 500]
                    amount_idx = action - 67
                    amount = min(amounts[amount_idx], self.game.weekly_exp_quota)
                    old_level = self.game.level
                    success = self.game.claim_weekly_exp(amount)
                    if success:
                        action_succeeded = True
                        reward += 0.1 * (amount / 100)
                        day_of_week = self.game.get_day_of_week(self.game.current_day)
                        if day_of_week <= 3:
                            reward += self.reward_weights['weekly_exp_early']
                        if self.game.level > old_level:
                            reward += self.reward_weights['level_up'] * (self.game.level - old_level)
                        self.consecutive_invalid = 0
                    if self.game.game_state == GameState.WEEKLY_EXP_CLAIM:
                        self.game.game_state = GameState.PLAYING
            elif action == 72:  # 下一天
                action_name = 'next_day'
                self.action_stats[action_name] += 1

                if valid_actions['can_next_day']:
                    # 简化判断逻辑
                    can_proceed = False

                    # 条件1：征服数量达标
                    if self.daily_conquered_count >= 6:
                        can_proceed = True
                        proceed_reason = "conquest_quota"
                    # 条件2：粮草不足以移动
                    elif self.game.food < 50:
                        can_proceed = True
                        proceed_reason = "no_food_move"
                    # 条件3：无法征服任何地块
                    elif not valid_actions['adjacent_targets'] and not valid_actions['jump_targets'] and not \
                    valid_actions['can_conquer']:
                        can_proceed = True
                        proceed_reason = "no_targets"
                    # 条件4：所有队伍都没有行动点
                    elif valid_actions['total_action_points'] == 0:
                        can_proceed = True
                        proceed_reason = "no_action_points"

                    if can_proceed:
                        # ★★★ 实际执行下一天 ★★★
                        old_day = self.game.current_day
                        self.game.next_day()  # 这行代码缺失了！！！

                        action_succeeded = True
                        self.consecutive_invalid = 0

                        # 重置每日计数器
                        self.daily_conquered_count = 0
                        self.daily_switch_count = 0
                        self.daily_actions_done = 0

                        # 奖励计算
                        if proceed_reason == "conquest_quota":
                            reward += self.reward_weights['smart_next_day']
                        elif proceed_reason in ["no_food_move", "no_action_points", "no_targets"]:
                            reward += 2.0  # 合理进入下一天

                        # 检查游戏是否结束
                        if self.game.current_day >= self.game.max_days:
                            terminated = True
        except Exception as e:
            print(f"动作执行错误: {e}")
            import traceback
            traceback.print_exc()
            reward += self.reward_weights['invalid_action']
            self.consecutive_invalid += 1

        # 记录成功的动作
        if action_succeeded and action_name != 'unknown':
            self.action_success[action_name] = self.action_success.get(action_name, 0) + 1

        # 计算增量奖励
        exp_gain = self.game.experience - self.last_exp
        if exp_gain > 0:
            reward += exp_gain * self.reward_weights['exp_gain']

        level_gain = self.game.level - self.last_level
        if level_gain > 0:
            reward += level_gain * self.reward_weights['level_up']

        conquered_gain = len(self.game.conquered_tiles) - old_conquered
        if conquered_gain > 0:
            reward += conquered_gain * 8.0
            total_conquered = len(self.game.conquered_tiles)
            if total_conquered in [10, 25, 50, 100, 150, 200]:
                reward += 60.0

        if len(self.game.teams) > 1:
            teams_with_conquest = sum(1 for count in self.daily_team_conquest.values() if count > 0)
            if teams_with_conquest > 1:
                reward += self.reward_weights['multi_team_conquest'] * teams_with_conquest

            new_spread = self._calculate_team_spread()
            if new_spread > old_team_spread:
                reward += self.reward_weights['team_spread'] * (new_spread - old_team_spread)
            self.last_team_spread = new_spread

        if old_food > self.game.food and exp_gain > 0:
            efficiency = exp_gain / (old_food - self.game.food + 1)
            reward += efficiency * self.reward_weights['efficiency']

        if self.game.current_day > 0 and self.has_moved:
            progress_rate = len(self.game.conquered_tiles) / (self.game.current_day + 1)
            reward += progress_rate * self.reward_weights['progress_bonus']

        if len(self.position_history) > 10:
            recent_positions = self.position_history[-10:]
            if len(set(recent_positions)) <= 2:
                self.stuck_counter += 1
                reward += self.reward_weights['stuck_penalty'] * (self.stuck_counter / 10.0)

        # 终止条件检查
        if self.consecutive_invalid > 50:  # 提高阈值，因为动作掩码会减少无效动作
            truncated = True
            reward += self.reward_weights['game_over'] * 0.5

        if self.game.game_state == GameState.GAME_OVER:
            terminated = True
            if self.game.current_day >= self.game.max_days:
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

        if self.current_step >= self.max_steps:
            truncated = True
            if not self.has_moved:
                reward += self.reward_weights['game_over']

        # 更新镜头（如果需要）
        if self.render_mode == "human" and self.game.teams:
            if action < 60 or action == 66 or self.current_step % 20 == 0:
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
    """增强的训练回调 - V8版本"""

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
        self.episode_steps = []
        self.best_exp = 0
        self.best_level = 0
        self.best_conquered = 0
        self.best_exploration = 0
        self.best_spread = 0
        self.current_episode_steps = 0
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

        # 新增：切换队伍追踪
        self.daily_switch_count = 0
        self.last_switch_step = -10
        self.switch_count_recent = 0

        # 新增：每天征服追踪
        self.daily_conquered_count = 0

    def _on_step(self) -> bool:
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
            action_stats = info.get('action_stats', {})
            action_success = info.get('action_success', {})
            total_steps = info.get('total_steps', self.current_episode_steps)

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
            self.episode_steps.append(self.current_episode_steps)

            # 显示结果
            ep_num = len(self.episode_exp)
            print(f"\n📊 Episode {ep_num}:")
            print(f"   等级: {level} | 经验: {exp} | 征服: {conquered} | 探索: {exploration}")
            print(f"   第{day}天结束 | 剩余粮草: {food} | 移动: {'是' if has_moved else '否'}")
            print(f"   ⏱️ 步数: {self.current_episode_steps}/2000 | 平均步/天: {self.current_episode_steps / max(day, 1):.1f}")

            # 结束原因
            if self.current_episode_steps >= 2000:
                print(f"   ⌛ 因步数达到上限结束 (仅完成{day}/91天)")
            elif day >= 91:
                print(f"   ✅ 正常完成91天 (用时{self.current_episode_steps}步)")
            else:
                print(f"   ⚠️ 其他原因结束")

            # 队伍信息
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

            # 步数消耗详情
            if action_stats and total_steps > 0:
                print(f"\n   📈 步数消耗详情 (共{total_steps}步):")
                action_names = {
                    'adjacent_move': '相邻移动',
                    'jump_move': '跳跃移动',
                    'conquer': '征服',
                    'thunder_god': '飞雷神',
                    'switch_team': '切换队伍',
                    'claim_exp': '领取经验',
                    'next_day': '下一天',
                    'masked': '被掩码阻止',
                }

                sorted_actions = sorted(action_stats.items(), key=lambda x: x[1], reverse=True)

                for action_key, count in sorted_actions:
                    if count > 0:
                        percentage = (count / total_steps) * 100
                        success_count = action_success.get(action_key, 0)

                        if action_key == 'masked':
                            action_name = action_names.get(action_key, action_key)
                            print(f"      {action_name}: {count}次 ({percentage:.1f}%)")
                        else:
                            success_rate = (success_count / count * 100) if count > 0 else 0
                            action_name = action_names.get(action_key, action_key)
                            print(f"      {action_name}: {count}次 ({percentage:.1f}%) | 成功: {success_count}次 ({success_rate:.0f}%)")

                # 效率分析
                total_moves = action_stats.get('adjacent_move', 0) + action_stats.get('jump_move', 0)
                successful_moves = action_success.get('adjacent_move', 0) + action_success.get('jump_move', 0)

                if total_moves > 0:
                    move_success_rate = (successful_moves / total_moves * 100)
                    print(f"      移动成功率: {move_success_rate:.1f}% ({successful_moves}/{total_moves})")

                    if successful_moves > 0:
                        move_efficiency = conquered / successful_moves
                        print(f"      移动效率: {move_efficiency:.2f} 征服/成功移动")

                if action_stats.get('next_day', 0) > 0:
                    avg_actions_per_day = (total_steps - action_stats['next_day']) / action_stats['next_day']
                    print(f"      平均每天动作数: {avg_actions_per_day:.1f}")

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
        recent_steps = self.episode_steps[-20:]

        print(f"\n{'=' * 70}")
        print(f"📈 Episode {len(self.episode_exp)} 统计汇总 (最近20局)")
        print(f"{'=' * 70}")

        # 步数统计
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
        print(f"   征服: 平均 {np.mean(recent_conquered):.1f} | 最高 {max(recent_conquered)} | 最低 {min(recent_conquered)}")
        print(f"   探索: 平均 {np.mean(recent_exploration):.1f} | 最高 {max(recent_exploration)}")

        # 多队伍统计
        avg_spread = np.mean([s for s in recent_spread if s > 0])
        if avg_spread > 0:
            print(f"\n👥 多队伍协作:")
            print(f"   平均分散度: {avg_spread:.1f}")

            multi_team_games = 0
            for conquests in recent_conquests:
                if len(conquests) > 1:
                    multi_team_games += 1
            if multi_team_games > 0:
                print(f"   多队共同征服: {multi_team_games}/{len(recent_conquests)}局")

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
        # 删除ActionMasker包装器，直接使用环境
        # env = ActionMasker(env, "action_masks")  # 删除这行
        env = Monitor(env)
        return env
    return _init


def train_maskable_ppo_agent(
        total_timesteps=1000000,
        n_envs=1,
        save_path="hex_game_maskable_ppo_v8",
        log_path="./logs/"
):
    """训练使用动作掩码的PPO智能体"""

    # ============ 调试：检查版本 ============
    import sb3_contrib
    import stable_baselines3
    print("\n📦 包版本检查:")
    print(f"  sb3-contrib: {sb3_contrib.__version__}")
    print(f"  stable-baselines3: {stable_baselines3.__version__}")
    print("=" * 70)
    # ============ 调试结束 ============

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and n_envs == 1:
        print("💡 提示：检测到GPU，建议使用 --n_envs 4 或更多来充分利用GPU性能")

    print("=" * 70)
    print("PPO训练 V8 - 动作掩码版本")
    print("=" * 70)
    print(f"设备: {device.upper()}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"并行环境数: {n_envs}")
    print(f"总训练步数: {total_timesteps}")
    print("=" * 70)

    # 创建环境
    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])

    # 使用MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_path,
        verbose=0,
        device=device,
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]
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
        name_prefix="maskable_ppo_hex_v8"
    )

    callbacks = [training_callback, eval_callback, checkpoint_callback]

    print("\n开始训练... (Ctrl+C 中断并保存)")
    print("提示: 使用 tensorboard --logdir ./logs 查看训练曲线")
    print("动作掩码已启用，无效动作将被自动阻止")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="六边形游戏PPO训练脚本 V8 - 动作掩码版")
    parser.add_argument("--mode", choices=["train", "test", "check"], default="train")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="hex_game_maskable_ppo_v8")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n_test_episodes", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "check":
        print("检查环境...")
        env = HexGameEnv()
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")

        # 测试动作掩码
        obs, _ = env.reset()
        mask = env.action_masks()
        print(f"动作掩码形状: {mask.shape}")
        print(f"可用动作数: {mask.sum()}/{len(mask)}")
        print(f"可用动作: {np.where(mask)[0]}")
        env.close()
        print("✅ 环境检查通过!")

    elif args.mode == "train":
        model = train_maskable_ppo_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.model_path
        )

    else:  # test
        print("测试模式暂未实现，请使用原版测试函数")