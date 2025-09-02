"""
六边形地图策略游戏 - PPO强化学习训练器（增强版）
包含训练可视化、最优路线记录和回放功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


# ========== 数据类 ==========
@dataclass
class DayAction:
    """记录每天的行动"""
    day: int
    team_id: int
    action_type: str  # 'move', 'conquer', 'rest', 'thunder_god'
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    food_cost: int
    exp_gain: int
    resources_after: Dict  # 行动后的资源状态


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


# ========== 游戏核心逻辑（增强版） ==========
class GameCore:
    """无GUI的游戏核心逻辑，带详细记录"""

    def __init__(self, map_file='map_save.json'):
        with open(map_file, 'r') as f:
            self.map_data = json.load(f)

        self.action_history = []  # 记录所有行动
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.current_day = 1
        self.experience = 100
        self.level = 1
        self.food = 6800
        self.conquest_score = 1000
        self.thunder_god_items = 1
        self.treasures_conquered = set()
        self.has_treasure_buff = False
        self.conquered_tiles = set()

        # 多队伍支持
        self.teams = {
            1: {'pos': None, 'action_points': 6, 'active': True}
        }

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
            if tile.get('terrain_type') == 'START_POSITION':
                self.start_pos = pos
                self.conquered_tiles.add(pos)
                break

        if not self.start_pos:
            self.start_pos = (0, 0)

        self.teams[1]['pos'] = self.start_pos
        self.current_team = 1

        # 检查队伍解锁
        self.check_team_unlock()

    def check_team_unlock(self):
        """检查是否解锁新队伍"""
        level = self.experience // 100

        if level >= 20 and len(self.teams) == 1:
            # 解锁二队
            self.teams[2] = {
                'pos': self.teams[1]['pos'],  # 在当前位置生成
                'action_points': 6,
                'active': True
            }

        if level >= 60 and len(self.teams) == 2:
            # 解锁三队
            self.teams[3] = {
                'pos': self.teams[self.current_team]['pos'],
                'action_points': 6,
                'active': True
            }

    def step(self, action):
        """执行动作并记录详细信息"""
        reward = 0
        done = False
        old_exp = self.experience
        old_food = self.food

        team = self.teams[self.current_team]
        from_pos = team['pos']
        action_type = None
        to_pos = from_pos

        if action == 0:  # 结束回合
            action_type = 'rest'
            self.next_day()
            reward = -1

        elif 1 <= action <= 6:  # 移动
            neighbors = self.get_neighbors(team['pos'])
            if action - 1 < len(neighbors):
                target = neighbors[action - 1]
                cost = self.calculate_move_cost(target)
                if cost > 0 and cost <= self.food:
                    self.food -= cost
                    team['pos'] = target
                    to_pos = target
                    action_type = 'move'
                    reward = -cost / 1000

        elif action == 7:  # 征服
            if team['pos'] not in self.conquered_tiles:
                tile = self.hex_map[team['pos']]
                exp_gain = self.get_tile_exp(tile)
                food_cost = self.get_tile_food_cost(tile)

                if food_cost <= self.food and team['action_points'] > 0:
                    self.food -= food_cost
                    self.experience += exp_gain
                    self.conquered_tiles.add(team['pos'])
                    team['action_points'] -= 1
                    action_type = 'conquer'

                    reward = exp_gain / 100 - food_cost / 1000

                    # 检查秘宝
                    if self.is_treasure(tile):
                        treasure_id = self.get_treasure_id(tile)
                        self.treasures_conquered.add(treasure_id)
                        reward += 10

                        if len(self.treasures_conquered) == 8:
                            self.has_treasure_buff = True
                            reward += 100

        elif action == 8:  # 飞雷神
            if self.thunder_god_items > 0:
                # 简化：传送到最近的未征服秘宝
                target = self.find_nearest_treasure()
                if target:
                    team['pos'] = target
                    to_pos = target
                    self.thunder_god_items -= 1
                    action_type = 'thunder_god'
                    reward = 5  # 鼓励使用

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
                    'level': self.experience // 100
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
            self.current_team = (self.current_team % len(team_ids)) + 1
            if self.current_team in self.teams and self.teams[self.current_team]['action_points'] > 0:
                break

    def get_state(self):
        """获取状态向量（包含多队伍信息）"""
        state = np.zeros(150)  # 扩大状态空间

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

    def find_nearest_treasure(self):
        """找最近的未征服秘宝"""
        # 简化实现
        for pos, tile in self.hex_map.items():
            if self.is_treasure(tile) and pos not in self.conquered_tiles:
                return pos
        return None

    def get_treasure_id(self, tile):
        """获取秘宝ID"""
        terrain = tile.get('terrain_type', '')
        if 'TREASURE' in terrain:
            return int(terrain.split('_')[-1])
        return 0

    # ... 其他辅助方法保持不变 ...
    def get_neighbors(self, pos):
        if not pos: return []
        q, r = pos
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        return [(q + dq, r + dr) for dq, dr in directions]

    def next_day(self):
        self.current_day += 1
        self.food += self.get_daily_food()
        self.conquest_score += 1000
        for team in self.teams.values():
            team['action_points'] = min(team['action_points'] + 6, 18)

    def get_daily_food(self):
        level = self.experience // 100
        if level <= 4:
            return 800
        elif level <= 14:
            return 900
        elif level <= 29:
            return 1000
        else:
            return 1100

    def calculate_move_cost(self, pos):
        base_cost = 50
        return int(base_cost * 0.8) if self.has_treasure_buff else base_cost

    def get_tile_exp(self, tile):
        return tile.get('exp_gain', 0)

    def get_tile_food_cost(self, tile):
        cost = tile.get('food_cost', 0)
        return int(cost * 0.8) if self.has_treasure_buff else cost

    def is_treasure(self, tile):
        return 'TREASURE' in str(tile.get('terrain_type', ''))

    def get_valid_actions(self):
        actions = [0]  # 总是可以结束回合
        team = self.teams[self.current_team]

        if team['pos']:
            neighbors = self.get_neighbors(team['pos'])
            for i, pos in enumerate(neighbors[:6]):
                if pos not in self.conquered_tiles:
                    actions.append(i + 1)

            if team['pos'] not in self.conquered_tiles:
                actions.append(7)

        if self.thunder_god_items > 0:
            actions.append(8)

        return actions


# ========== 训练监视器 ==========
class TrainingMonitor:
    """训练过程可视化和记录"""

    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.exp_values = []
        self.best_episode = None

        # 创建图表
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('PPO Training Progress')

    def update(self, episode_num, reward, exp, game_core):
        """更新监视器"""
        self.rewards.append(reward)
        self.exp_values.append(exp)

        # 创建Episode记录
        episode = Episode(
            episode_id=episode_num,
            total_reward=reward,
            final_exp=exp,
            final_day=game_core.current_day,
            actions=game_core.action_history.copy(),
            conquered_tiles=list(game_core.conquered_tiles),
            treasure_order=list(game_core.treasures_conquered)
        )
        self.episodes.append(episode)

        # 更新最佳记录
        if self.best_episode is None or exp > self.best_episode.final_exp:
            self.best_episode = episode

        # 更新图表
        if episode_num % 10 == 0:
            self.plot_progress()

    def plot_progress(self):
        """绘制训练进度"""
        # 清空子图
        for ax in self.axes.flat:
            ax.clear()

        # 1. 奖励曲线
        self.axes[0, 0].plot(self.rewards)
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Total Reward')

        # 2. 经验值曲线
        self.axes[0, 1].plot(self.exp_values)
        self.axes[0, 1].set_title('Final Experience')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Experience')

        # 3. 移动平均
        if len(self.rewards) > 20:
            window = 20
            ma_rewards = np.convolve(self.rewards, np.ones(window) / window, 'valid')
            ma_exp = np.convolve(self.exp_values, np.ones(window) / window, 'valid')

            self.axes[1, 0].plot(ma_rewards)
            self.axes[1, 0].set_title(f'Rewards (MA-{window})')

            self.axes[1, 1].plot(ma_exp)
            self.axes[1, 1].set_title(f'Experience (MA-{window})')

        plt.tight_layout()
        plt.pause(0.01)

    def save_best_route(self, filename='best_route.pkl'):
        """保存最优路线"""
        if self.best_episode:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_episode, f)

            # 同时生成人类可读的报告
            self.generate_route_report(self.best_episode)

    def generate_route_report(self, episode):
        """生成详细的路线报告"""
        report = []
        report.append("=" * 60)
        report.append("最优路线报告")
        report.append("=" * 60)
        report.append(f"总经验值: {episode.final_exp}")
        report.append(f"完成天数: {episode.final_day}")
        report.append(f"征服地块数: {len(episode.conquered_tiles)}")
        report.append(f"收集秘宝顺序: {episode.treasure_order}")
        report.append("\n" + "=" * 60)
        report.append("详细行动记录")
        report.append("=" * 60)

        # 按天分组行动
        actions_by_day = defaultdict(list)
        for action in episode.actions:
            actions_by_day[action.day].append(action)

        for day in sorted(actions_by_day.keys()):
            report.append(f"\n第 {day} 天:")
            for action in actions_by_day[day]:
                if action.action_type == 'move':
                    report.append(
                        f"  队伍{action.team_id}: 移动 {action.from_pos} -> {action.to_pos} (消耗{action.food_cost}粮草)")
                elif action.action_type == 'conquer':
                    report.append(f"  队伍{action.team_id}: 征服 {action.to_pos} (获得{action.exp_gain}经验)")
                elif action.action_type == 'thunder_god':
                    report.append(f"  队伍{action.team_id}: 飞雷神传送到 {action.to_pos}")
                elif action.action_type == 'rest':
                    report.append(f"  队伍{action.team_id}: 休息")

            # 显示当天结束时的资源
            if actions_by_day[day]:
                last_action = actions_by_day[day][-1]
                res = last_action.resources_after
                report.append(f"  资源状态 - 粮草:{res['food']} 经验:{res['exp']} 等级:{res['level']}")

        # 保存报告
        with open('best_route_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print('\n'.join(report[:20]))  # 打印前20行
        print(f"...\n完整报告已保存到 best_route_report.txt")


# ========== PPO模型（与之前相同） ==========
class PPOModel(nn.Module):
    def __init__(self, state_dim=150, action_dim=9, hidden_dim=256):
        super(PPOModel, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

    def get_action(self, state, valid_actions=None):
        logits, value = self.forward(state)

        if valid_actions is not None:
            mask = torch.ones_like(logits) * -1e8
            mask[valid_actions] = 0
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value


# ========== PPO训练器（增强版） ==========
class PPOTrainer:
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=10):
        self.game = GameCore()
        self.model = PPOModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        # 监视器
        self.monitor = TrainingMonitor()

        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def train(self, total_episodes=500):
        """训练主循环"""
        print("开始训练...")
        best_exp = 0

        for episode in range(total_episodes):
            # 收集轨迹
            self.collect_trajectory()

            # PPO更新
            self.update()

            # 评估和记录
            if episode % 5 == 0:
                avg_reward, avg_exp = self.evaluate()
                self.monitor.update(episode, avg_reward, avg_exp, self.game)

                if avg_exp > best_exp:
                    best_exp = avg_exp
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    self.monitor.save_best_route()

                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                      f"Avg Exp={avg_exp:.0f}, Best Exp={best_exp:.0f}")

        print("\n训练完成！")
        print(f"最佳经验值: {best_exp}")
        print("最优路线已保存到 best_route_report.txt")

        # 显示最终图表
        plt.ioff()
        plt.show()

    def collect_trajectory(self, n_steps=2048):
        """收集轨迹数据"""
        self.game.reset()
        state = torch.FloatTensor(self.game.get_state())

        for _ in range(n_steps):
            valid_actions = self.game.get_valid_actions()
            action, log_prob, value = self.model.get_action(state, valid_actions)

            reward, done = self.game.step(action)

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(done)

            if done:
                self.game.reset()

            state = torch.FloatTensor(self.game.get_state())

    def compute_returns(self):
        """计算折扣回报"""
        returns = []
        discounted_reward = 0

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        """PPO更新"""
        returns = self.compute_returns()

        states = torch.stack(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach().squeeze()

        advantages = returns - old_values

        for _ in range(self.epochs):
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((returns - values.squeeze()) ** 2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def evaluate(self, n_eval=3):
        """评估模型性能"""
        total_rewards = []
        total_exp = []

        for _ in range(n_eval):
            self.game.reset()
            state = torch.FloatTensor(self.game.get_state())
            episode_reward = 0
            done = False

            while not done:
                valid_actions = self.game.get_valid_actions()
                with torch.no_grad():
                    action, _, _ = self.model.get_action(state, valid_actions)
                reward, done = self.game.step(action)
                episode_reward += reward
                state = torch.FloatTensor(self.game.get_state())

            total_rewards.append(episode_reward)
            total_exp.append(self.game.experience)

        return np.mean(total_rewards), np.mean(total_exp)


# ========== 主函数 ==========
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建训练器并开始训练
    trainer = PPOTrainer(
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        epochs=10
    )

    trainer.train(total_episodes=500)

    print("\n" + "=" * 60)
    print("训练完成！生成的文件：")
    print("1. best_model.pth - 最佳模型权重")
    print("2. best_route.pkl - 最优路线数据")
    print("3. best_route_report.txt - 人类可读的路线报告")
    print("=" * 60)