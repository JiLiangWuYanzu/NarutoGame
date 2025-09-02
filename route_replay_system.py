"""
AI最优路线回放系统
在游戏地图上展示训练得到的最优路线，包含完整的可视化和日志功能
"""
import pygame
import pickle
import os
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

# 导入游戏系统基础功能
from game_play_system import GamePlaySystem
from font_manager import get_font, get_small_font, get_medium_font, get_large_font


@dataclass
class DetailedAction:
    """详细的行动记录"""
    day: int
    step: int  # 当天内的步骤顺序
    team_id: int  # 0表示系统行动
    action_type: str  # move, conquer, thunder_god, claim_weekly_exp等
    from_pos: Optional[Tuple[int, int]]
    to_pos: Optional[Tuple[int, int]]
    food_cost: int
    exp_gain: int
    resources_after: Dict  # 行动后的资源状态
    extra_info: Dict = None  # 额外信息


@dataclass
class Episode:
    """完整的一局游戏记录"""
    episode_id: int
    total_reward: float
    final_exp: int
    final_day: int
    actions: List[DetailedAction]
    conquered_tiles: List[Tuple[int, int]]
    treasure_order: List[int]


class LogWindow:
    """操作日志窗口"""

    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.logs = []
        self.scroll_offset = 0
        self.line_height = 20
        self.max_visible_lines = (height - 40) // self.line_height

        # 字体
        self.title_font = get_medium_font()
        self.log_font = get_font(14)
        self.detail_font = get_small_font()

    def clear(self):
        """清空日志"""
        self.logs = []
        self.scroll_offset = 0

    def add_log(self, log_entry):
        """添加日志条目"""
        self.logs.append(log_entry)
        if len(self.logs) > self.max_visible_lines:
            self.scroll_offset = len(self.logs) - self.max_visible_lines

    def format_action_log(self, action):
        """格式化行动为日志条目"""
        timestamp = f"[步骤{action.step:02d}]"

        if action.action_type == 'move':
            main_text = f"队伍{action.team_id} 移动: {action.from_pos} → {action.to_pos}"
            detail = f"消耗粮草: {action.food_cost}" if action.food_cost > 0 else ""
            color = (200, 200, 255)

        elif action.action_type == 'conquer':
            main_text = f"队伍{action.team_id} 征服: {action.to_pos}"
            detail = f"获得 {action.exp_gain} 经验, 消耗 {action.food_cost} 粮草"
            color = (255, 200, 100)

            # 检查特殊地块
            if action.extra_info and 'terrain_type' in action.extra_info:
                terrain = action.extra_info['terrain_type']
                if 'TREASURE' in str(terrain):
                    detail += " [秘宝]"
                    color = (255, 100, 255)
                elif 'BOSS' in str(terrain):
                    detail += " [BOSS]"
                    color = (255, 100, 100)

        elif action.action_type == 'thunder_god':
            main_text = f"队伍{action.team_id} 飞雷神传送到 {action.to_pos}"
            detail = "消耗飞雷神道具 x1"
            color = (255, 150, 255)

        elif action.action_type == 'claim_weekly_exp':
            amount = action.extra_info.get('amount', 0) if action.extra_info else 0
            main_text = f"领取周经验 {amount}"
            detail = f"剩余额度: {action.resources_after.get('weekly_quota', 0)}"
            color = (255, 215, 0)

        elif action.action_type == 'auto_weekly_exp':
            amount = action.extra_info.get('amount', 0) if action.extra_info else 0
            main_text = f"周日自动发放 {amount} 经验"
            detail = "未领取额度自动发放"
            color = (150, 255, 150)

        elif action.action_type == 'weekly_refresh':
            main_text = "周一额度刷新"
            detail = "本周可领取500经验"
            color = (100, 200, 255)

        elif action.action_type == 'team_unlock':
            team_id = action.extra_info.get('unlocked_team', 0) if action.extra_info else 0
            main_text = f"解锁队伍{team_id}！"
            detail = f"等级达到{action.resources_after.get('level', 0)}级"
            color = (255, 255, 100)

        elif action.action_type == 'rest':
            main_text = "结束当天行动"
            detail = "所有队伍行动点+6"
            color = (150, 150, 150)

        else:
            main_text = f"行动: {action.action_type}"
            detail = ""
            color = (200, 200, 200)

        return {
            'timestamp': timestamp,
            'main_text': main_text,
            'detail': detail,
            'color': color,
            'action': action
        }

    def draw(self, screen):
        """绘制日志窗口"""
        # 背景
        window_surface = pygame.Surface((self.rect.width, self.rect.height))
        window_surface.set_alpha(240)
        window_surface.fill((20, 20, 30))
        screen.blit(window_surface, self.rect)

        # 边框
        pygame.draw.rect(screen, (100, 100, 110), self.rect, 2)

        # 标题栏
        title_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 30)
        pygame.draw.rect(screen, (40, 40, 50), title_rect)
        pygame.draw.rect(screen, (100, 100, 110), title_rect, 1)

        title_text = f"操作日志 (共{len(self.logs)}条)"
        title_surf = self.title_font.render(title_text, True, (255, 255, 255))
        screen.blit(title_surf, (self.rect.x + 10, self.rect.y + 5))

        # 滚动提示
        if len(self.logs) > self.max_visible_lines:
            scroll_text = f"↑↓ {self.scroll_offset + 1}-{min(self.scroll_offset + self.max_visible_lines, len(self.logs))}/{len(self.logs)}"
            scroll_surf = self.detail_font.render(scroll_text, True, (150, 150, 150))
            screen.blit(scroll_surf, (self.rect.x + self.rect.width - 80, self.rect.y + 8))

        # 日志内容
        content_y = self.rect.y + 35
        visible_logs = self.logs[self.scroll_offset:self.scroll_offset + self.max_visible_lines]

        for i, log in enumerate(visible_logs):
            y_pos = content_y + i * self.line_height

            # 时间戳
            time_surf = self.log_font.render(log['timestamp'], True, (150, 150, 150))
            screen.blit(time_surf, (self.rect.x + 10, y_pos))

            # 主文本
            main_surf = self.log_font.render(log['main_text'], True, log['color'])
            screen.blit(main_surf, (self.rect.x + 70, y_pos))

            # 详细信息
            if log['detail'] and i < self.max_visible_lines - 1:
                detail_surf = self.detail_font.render(f"  {log['detail']}", True, (120, 120, 120))
                screen.blit(detail_surf, (self.rect.x + 80, y_pos + 12))

        # 滚动条
        if len(self.logs) > self.max_visible_lines:
            self.draw_scrollbar(screen)

    def draw_scrollbar(self, screen):
        """绘制滚动条"""
        scrollbar_x = self.rect.x + self.rect.width - 10
        scrollbar_y = self.rect.y + 35
        scrollbar_height = self.rect.height - 40

        pygame.draw.rect(screen, (50, 50, 60),
                         (scrollbar_x, scrollbar_y, 8, scrollbar_height))

        if len(self.logs) > 0:
            thumb_height = max(20, scrollbar_height * self.max_visible_lines // len(self.logs))
            thumb_pos = scrollbar_y + (scrollbar_height - thumb_height) * self.scroll_offset // max(1,
                                                                                                    len(self.logs) - self.max_visible_lines)

            pygame.draw.rect(screen, (100, 100, 110),
                             (scrollbar_x, thumb_pos, 8, thumb_height))

    def handle_scroll(self, direction):
        """处理滚动"""
        if direction == 'up' and self.scroll_offset > 0:
            self.scroll_offset -= 1
        elif direction == 'down' and self.scroll_offset < len(self.logs) - self.max_visible_lines:
            self.scroll_offset += 1


class RouteReplaySystem:
    """路线回放系统"""

    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height

        # 基础游戏系统（用于地图渲染）
        self.base_game = GamePlaySystem(screen, width, height)

        # 字体
        self.font = get_medium_font()
        self.small_font = get_small_font()
        self.large_font = get_large_font()

        # 回放相关
        self.best_episode = None
        self.current_day = 1
        self.max_day = 91
        self.is_playing = False
        self.play_timer = 0
        self.play_speed = 1000

        # 逐步回放
        self.show_step_by_step = False
        self.current_step = 0

        # 当天数据
        self.today_actions = []
        self.conquered_today = set()
        self.team_paths = {1: [], 2: [], 3: []}

        # 日志窗口
        self.log_window = LogWindow(
            x=width - 420,
            y=100,
            width=400,
            height=500
        )
        self.show_log = True

        # 加载最优路线
        self.load_best_route()

    def load_best_route(self):
        """加载训练得到的最优路线"""
        route_file = 'best_route.pkl'
        if os.path.exists(route_file):
            try:
                with open(route_file, 'rb') as f:
                    self.best_episode = pickle.load(f)
                    if hasattr(self.best_episode, 'final_day'):
                        self.max_day = self.best_episode.final_day
                    print(f"成功加载最优路线，总天数：{self.max_day}")
                    self.update_current_day()
            except Exception as e:
                print(f"加载路线失败：{e}")
                self.best_episode = None
        else:
            print("未找到最优路线文件，请先进行AI训练")

    def update_current_day(self):
        """更新当天的显示数据"""
        if not self.best_episode:
            return

        # 清空当天数据
        self.today_actions = []
        self.conquered_today.clear()
        for team_id in self.team_paths:
            self.team_paths[team_id].clear()

        # 更新日志
        self.log_window.clear()

        # 获取日期信息
        date_str = self.base_game.get_date_string(self.current_day)
        day_of_week = self.base_game.get_day_of_week(self.current_day)
        weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]

        self.log_window.add_log({
            'timestamp': '[00:00]',
            'main_text': f'=== {date_str} {weekdays[day_of_week]} ===',
            'detail': '',
            'color': (255, 255, 255)
        })

        # 收集历史征服地块
        all_conquered = set()
        for action in self.best_episode.actions:
            if action.day < self.current_day:
                if action.action_type == 'conquer':
                    all_conquered.add(action.to_pos)
            elif action.day == self.current_day:
                self.today_actions.append(action)

                # 记录移动路径
                if action.action_type == 'move' and action.team_id in self.team_paths:
                    self.team_paths[action.team_id].append((action.from_pos, action.to_pos))

                # 记录征服
                elif action.action_type == 'conquer':
                    self.conquered_today.add(action.to_pos)
                    all_conquered.add(action.to_pos)

                # 添加到日志
                log_entry = self.log_window.format_action_log(action)
                self.log_window.add_log(log_entry)

        # 更新基础游戏的征服状态
        self.base_game.conquered_tiles = all_conquered

        # 添加当天结束总结
        if self.today_actions:
            last_action = self.today_actions[-1]
            if hasattr(last_action, 'resources_after'):
                res = last_action.resources_after
                self.log_window.add_log({
                    'timestamp': '[23:59]',
                    'main_text': '=== 当天结束 ===',
                    'detail': f"等级{res.get('level', 1)} | 经验{res.get('exp', 0)} | 粮草{res.get('food', 0)}",
                    'color': (200, 200, 200)
                })

    def draw(self):
        """绘制回放界面"""
        self.screen.fill((30, 30, 40))

        # 绘制地图
        self.base_game.draw_map()

        # 绘制路径和标记
        if self.show_step_by_step:
            self.draw_actions_up_to_step()
        else:
            self.draw_day_paths()
            self.draw_conquest_marks()

        # 绘制UI
        self.draw_info_panel()
        self.draw_control_panel()

        # 绘制日志窗口
        if self.show_log:
            self.log_window.draw(self.screen)

        # 日志开关按钮
        self.draw_log_toggle_button()

    def draw_day_paths(self):
        """绘制当天的移动路径"""
        team_colors = {
            1: (255, 100, 100),
            2: (100, 255, 100),
            3: (100, 100, 255)
        }

        # 为每个队伍绘制路径
        for team_id in [1, 2, 3]:
            team_actions = [a for a in self.today_actions if a.team_id == team_id]
            if not team_actions:
                continue

            color = team_colors[team_id]

            for i, action in enumerate(team_actions):
                if action.action_type == 'move':
                    start = self.base_game.hex_to_pixel(*action.from_pos)
                    end = self.base_game.hex_to_pixel(*action.to_pos)

                    # 绘制实线
                    pygame.draw.line(self.screen, color, start, end, 3)
                    self.draw_arrow(start, end, color)

                    # 步骤编号
                    mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                    step_text = str(i + 1)
                    step_surf = self.small_font.render(step_text, True, (255, 255, 255))
                    step_rect = step_surf.get_rect(center=mid_point)

                    pygame.draw.circle(self.screen, color, mid_point, 12)
                    pygame.draw.circle(self.screen, (0, 0, 0), mid_point, 10)
                    self.screen.blit(step_surf, step_rect)

                elif action.action_type == 'thunder_god':
                    # 飞雷神虚线
                    if action.from_pos:
                        start = self.base_game.hex_to_pixel(*action.from_pos)
                    else:
                        # 从上一个位置
                        prev_actions = team_actions[:i]
                        if prev_actions:
                            start = self.base_game.hex_to_pixel(*prev_actions[-1].to_pos)
                        else:
                            continue

                    end = self.base_game.hex_to_pixel(*action.to_pos)
                    self.draw_dashed_line(start, end, color)
                    self.draw_teleport_effect(start, color, "出发")
                    self.draw_teleport_effect(end, color, "到达")

    def draw_arrow(self, start, end, color):
        """绘制箭头"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            return

        dx /= length
        dy /= length

        arrow_pos = (end[0] - dx * 10, end[1] - dy * 10)
        wing_length = 15
        angle = 0.5

        wing1 = (
            arrow_pos[0] - wing_length * (dx * math.cos(angle) - dy * math.sin(angle)),
            arrow_pos[1] - wing_length * (dx * math.sin(angle) + dy * math.cos(angle))
        )

        wing2 = (
            arrow_pos[0] - wing_length * (dx * math.cos(-angle) - dy * math.sin(-angle)),
            arrow_pos[1] - wing_length * (dx * math.sin(-angle) + dy * math.cos(-angle))
        )

        pygame.draw.polygon(self.screen, color, [end, wing1, wing2])

    def draw_dashed_line(self, start, end, color):
        """绘制虚线"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance == 0:
            return

        dash_length = 10
        gap_length = 5
        total_segment = dash_length + gap_length
        segments = int(distance / total_segment)

        for i in range(segments):
            t1 = (i * total_segment) / distance
            t2 = min((i * total_segment + dash_length) / distance, 1.0)

            x1 = start[0] + dx * t1
            y1 = start[1] + dy * t1
            x2 = start[0] + dx * t2
            y2 = start[1] + dy * t2

            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)

    def draw_teleport_effect(self, pos, color, label=""):
        """绘制传送门效果"""
        pygame.draw.circle(self.screen, (200, 100, 255), pos, 20, 3)
        pygame.draw.circle(self.screen, (255, 150, 255), pos, 15, 2)
        pygame.draw.circle(self.screen, color, pos, 8)

        if label:
            label_surf = self.small_font.render(label, True, (255, 255, 255))
            label_rect = label_surf.get_rect(center=(pos[0], pos[1] - 30))
            self.screen.blit(label_surf, label_rect)

    def draw_conquest_marks(self):
        """绘制征服标记"""
        for pos in self.conquered_today:
            pixel_pos = self.base_game.hex_to_pixel(*pos)

            pygame.draw.circle(self.screen, (255, 215, 0), pixel_pos, 12, 3)
            pygame.draw.circle(self.screen, (255, 255, 100), pixel_pos, 4)

    def draw_actions_up_to_step(self):
        """逐步显示行动"""
        for i, action in enumerate(self.today_actions[:self.current_step + 1]):
            if action.action_type == 'move':
                color = {1: (255, 100, 100), 2: (100, 255, 100), 3: (100, 100, 255)}.get(action.team_id,
                                                                                         (255, 255, 255))
                start = self.base_game.hex_to_pixel(*action.from_pos)
                end = self.base_game.hex_to_pixel(*action.to_pos)

                pygame.draw.line(self.screen, color, start, end, 3)
                self.draw_arrow(start, end, color)

            elif action.action_type == 'conquer':
                pixel_pos = self.base_game.hex_to_pixel(*action.to_pos)
                pygame.draw.circle(self.screen, (255, 215, 0), pixel_pos, 12, 3)

        # 高亮当前步骤
        if self.current_step < len(self.today_actions):
            current_action = self.today_actions[self.current_step]
            if current_action.to_pos:
                pos = self.base_game.hex_to_pixel(*current_action.to_pos)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 20, 2)

    def draw_info_panel(self):
        """绘制信息面板"""
        panel_width = 350
        panel_height = 250
        panel_x = 10
        panel_y = 10

        # 背景
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(230)
        panel_surface.fill((30, 30, 40))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (100, 100, 110),
                         (panel_x, panel_y, panel_width, panel_height), 2)

        # 标题
        title = f"第 {self.current_day} 天"
        title_surf = self.large_font.render(title, True, (255, 215, 0))
        self.screen.blit(title_surf, (panel_x + 10, panel_y + 10))

        # 日期信息
        date_str = self.base_game.get_date_string(self.current_day)
        day_of_week = self.base_game.get_day_of_week(self.current_day)
        weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
        date_text = f"{date_str} {weekdays[day_of_week]}"
        date_surf = self.font.render(date_text, True, (200, 200, 200))
        self.screen.blit(date_surf, (panel_x + 10, panel_y + 40))

        # 统计信息
        y_offset = 70
        stats = [
            f"今日行动数: {len(self.today_actions)}",
            f"征服地块数: {len(self.conquered_today)}",
        ]

        # 获取资源信息
        if self.today_actions:
            last_action = self.today_actions[-1]
            if hasattr(last_action, 'resources_after'):
                res = last_action.resources_after
                stats.append(f"等级: {res.get('level', 1)}")
                stats.append(f"经验: {res.get('exp', 0)}")
                stats.append(f"粮草: {res.get('food', 0)}")

        for stat in stats:
            stat_surf = self.font.render(stat, True, (180, 180, 180))
            self.screen.blit(stat_surf, (panel_x + 10, panel_y + y_offset))
            y_offset += 25

        # 逐步回放提示
        if self.show_step_by_step:
            step_text = f"步骤: {self.current_step + 1}/{len(self.today_actions)}"
            step_surf = self.font.render(step_text, True, (255, 255, 100))
            self.screen.blit(step_surf, (panel_x + 10, panel_y + panel_height - 30))

    def draw_control_panel(self):
        """绘制控制面板"""
        panel_height = 100
        panel_y = self.height - panel_height

        pygame.draw.rect(self.screen, (40, 40, 50), (0, panel_y, self.width, panel_height))
        pygame.draw.rect(self.screen, (80, 80, 90), (0, panel_y, self.width, panel_height), 2)

        # 进度条
        bar_width = 600
        bar_height = 20
        bar_x = (self.width - bar_width) // 2
        bar_y = panel_y + 20

        pygame.draw.rect(self.screen, (60, 60, 70), (bar_x, bar_y, bar_width, bar_height))

        progress = (self.current_day - 1) / (self.max_day - 1) if self.max_day > 1 else 0
        fill_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, (100, 200, 100), (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, (100, 100, 110), (bar_x, bar_y, bar_width, bar_height), 2)

        day_text = f"天数: {self.current_day} / {self.max_day}"
        day_surf = self.font.render(day_text, True, (255, 255, 255))
        day_rect = day_surf.get_rect(center=(self.width // 2, bar_y + bar_height // 2))
        self.screen.blit(day_surf, day_rect)

        # 控制提示
        controls = [
            ("←/→", "切换天数"),
            ("↑/↓", "逐步回放"),
            ("空格", "自动播放"),
            ("Tab", "逐步模式"),
            ("L", "显示日志"),
            ("ESC", "返回")
        ]

        x_offset = 50
        y_offset = panel_y + 60

        for key, desc in controls:
            key_surf = self.font.render(key, True, (255, 215, 0))
            self.screen.blit(key_surf, (x_offset, y_offset))

            desc_surf = self.small_font.render(desc, True, (180, 180, 180))
            self.screen.blit(desc_surf, (x_offset + 60, y_offset + 3))

            x_offset += 180
            if x_offset > self.width - 200:
                x_offset = 50
                y_offset += 25

    def draw_log_toggle_button(self):
        """绘制日志开关按钮"""
        button_rect = pygame.Rect(self.width - 440, 70, 100, 25)

        if self.show_log:
            color = (100, 150, 100)
            text = "隐藏日志"
        else:
            color = (150, 100, 100)
            text = "显示日志"

        pygame.draw.rect(self.screen, color, button_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), button_rect, 1)

        text_surf = self.small_font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def handle_keyboard(self, key):
        """处理键盘输入"""
        if key == pygame.K_LEFT:
            if self.current_day > 1:
                self.current_day -= 1
                self.update_current_day()
                self.current_step = 0

        elif key == pygame.K_RIGHT:
            if self.current_day < self.max_day:
                self.current_day += 1
                self.update_current_day()
                self.current_step = 0

        elif key == pygame.K_UP and self.show_step_by_step:
            if self.current_step > 0:
                self.current_step -= 1

        elif key == pygame.K_DOWN and self.show_step_by_step:
            if self.current_step < len(self.today_actions) - 1:
                self.current_step += 1

        elif key == pygame.K_HOME:
            self.current_day = 1
            self.update_current_day()
            self.current_step = 0

        elif key == pygame.K_END:
            self.current_day = self.max_day
            self.update_current_day()
            self.current_step = 0

        elif key == pygame.K_SPACE:
            self.is_playing = not self.is_playing

        elif key == pygame.K_TAB:
            self.show_step_by_step = not self.show_step_by_step
            self.current_step = 0

        elif key == pygame.K_l:
            self.show_log = not self.show_log

    def handle_mouse_click(self, pos):
        """处理鼠标点击"""
        # 日志开关按钮
        button_rect = pygame.Rect(self.width - 440, 70, 100, 25)
        if button_rect.collidepoint(pos):
            self.show_log = not self.show_log

    def handle_mouse_wheel(self, event):
        """处理鼠标滚轮"""
        if self.show_log and self.log_window.rect.collidepoint(pygame.mouse.get_pos()):
            if event.button == 4:
                self.log_window.handle_scroll('up')
            elif event.button == 5:
                self.log_window.handle_scroll('down')

    def update(self, dt):
        """更新回放状态"""
        if self.is_playing:
            self.play_timer += dt * 1000
            if self.play_timer >= self.play_speed:
                self.play_timer = 0
                if self.current_day < self.max_day:
                    self.current_day += 1
                    self.update_current_day()
                else:
                    self.is_playing = False

    def run(self):
        """运行回放系统"""
        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'menu'
                    else:
                        self.handle_keyboard(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_mouse_click(event.pos)
                    elif event.button in [4, 5]:
                        self.handle_mouse_wheel(event)

            self.update(dt)
            self.draw()
            pygame.display.flip()

        return 'menu'