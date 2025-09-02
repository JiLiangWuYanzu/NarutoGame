"""
六边形地图策略游戏 - 游戏玩法系统（中文版）
实现队伍管理、移动、征服、资源管理等核心玩法
更新版本：实现新规则系统
"""
import pygame
import json
import os
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque

# 导入字体管理器
from font_manager import get_font_manager, get_font, get_small_font, get_medium_font, get_large_font

# 导入地图相关模块
# 导入地图相关模块
from map_style_config import StyleConfig, TerrainType
from hex_map_editor_chinese import HexTile


@dataclass
class Team:
    """队伍类"""
    id: int
    position: Tuple[int, int]  # (q, r) 坐标
    action_points: int = 6  # 行动点数
    max_action_points: int = 18  # 最大行动点数
    is_active: bool = True


class GameState(Enum):
    """游戏状态"""
    PLAYING = "playing"
    SELECTING_ACTION = "selecting_action"
    MOVING = "moving"
    CONQUERING = "conquering"
    DAY_END = "day_end"
    GAME_OVER = "game_over"
    WEEKLY_EXP_CLAIM = "weekly_exp_claim"  # 新增：周经验领取界面


class GamePlaySystem:
    """游戏玩法系统主类"""

    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.style = StyleConfig()

        # 初始化字体管理器
        self.font_manager = get_font_manager()

        # 字体设置 - 使用中文字体
        self.font = get_medium_font()
        self.small_font = get_small_font()
        self.large_font = get_large_font()

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.current_day = 1
        self.max_days = 91  # 修改：6.13-9.11共91天

        # 玩家数据 - 根据新规则调整
        self.experience = 100  # 初始1级 = 100经验
        self.level = 1
        self.food = 6800  # 修改：初始粮草6800
        self.conquest_score = 1000  # 修改：初始征服积分1000

        # 飞雷神系统
        self.thunder_god_items = 1  # 新增：初始1个飞雷神道具

        # 秘宝系统
        self.treasures_conquered = set()  # 已征服的秘宝
        self.has_treasure_buff = False  # 是否有秘宝buff（20%减免）

        # 周经验系统
        self.weekly_exp_quota = 500  # 本周可领取额度
        self.weekly_exp_claimed = 0  # 本周已领取
        self.weekly_claim_count = 0  # 本周领取次数
        self.current_week = 1  # 当前周数

        # 队伍管理
        self.teams: List[Team] = []
        self.current_team_index = 0
        self.max_teams = 1  # 当前最大队伍数

        # 地图数据
        self.hex_map: Dict[Tuple[int, int], HexTile] = {}
        self.conquered_tiles: Set[Tuple[int, int]] = set()  # 已征服的地块
        self.load_map_data()

        # 初始化第一个队伍
        self.init_first_team()

        # 相机和视图
        self.camera_x = width // 2
        self.camera_y = height // 2
        self.zoom = 1.0
        self.hex_size = 30

        # UI状态
        self.selected_tile: Optional[Tuple[int, int]] = None
        self.path_preview: List[Tuple[int, int]] = []  # 路径预览
        self.message = ""
        self.message_timer = 0

        # 飞雷神选择模式
        self.thunder_god_mode = False
        self.valid_thunder_targets = set()  # 有效的飞雷神目标

        # 拖拽控制
        self.is_dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_camera = (0, 0)

        # 时钟
        self.clock = pygame.time.Clock()

        # 动画相关
        self.animation_timer = 0
        self.team_animation_offset = 0

    def get_day_of_week(self, day):
        """获取星期几（6.13是周五）
        返回值：0=周日，1=周一，2=周二...6=周六
        """
        # 6月13日是周五(5)
        base_day = 5  # 周五
        days_passed = day - 1
        return (base_day + days_passed) % 7

    def get_week_number(self, day):
        """获取当前是第几周"""
        # 第一周：6.13(周五)-6.16(周日)
        # 第二周：6.17(周一)-6.23(周日)
        # ...
        day_of_week = self.get_day_of_week(day)
        # 计算从第一个周一开始的周数
        if day <= 3:  # 第一周的特殊处理（周五到周日）
            return 1
        else:
            days_from_first_monday = day - 4  # 6.17是第4天
            return 2 + (days_from_first_monday // 7)

    def reset_weekly_exp(self):
        """重置周经验额度（周一调用）"""
        self.weekly_exp_quota = 500
        self.weekly_exp_claimed = 0
        self.weekly_claim_count = 0
        self.add_message("新的一周开始，500经验额度已刷新！", "success")

    def auto_claim_weekly_exp(self):
        """周日自动发放未领取的周经验"""
        if self.weekly_exp_quota > 0:
            self.experience += self.weekly_exp_quota
            self.add_message(f"周日自动发放：+{self.weekly_exp_quota} 经验", "success")
            self.weekly_exp_quota = 0
            self.weekly_exp_claimed = 500

            # 检查升级
            old_level = self.level
            self.calculate_level()
            if self.level > old_level:
                self.add_message(f"升级了！现在是 {self.level} 级", "success")
                self.check_team_unlock()

    def claim_weekly_exp(self, amount):
        """手动领取周经验

        Args:
            amount: 领取数量（必须是100的倍数）
        Returns:
            bool: 是否领取成功
        """
        # 检查条件
        if self.weekly_claim_count >= 5:
            self.add_message("本周领取次数已达上限（5次）", "error")
            return False

        if amount % 100 != 0:
            self.add_message("领取数量必须是100的倍数", "error")
            return False

        if amount > self.weekly_exp_quota:
            self.add_message(f"额度不足，剩余 {self.weekly_exp_quota}", "error")
            return False

        if amount <= 0:
            self.add_message("领取数量必须大于0", "error")
            return False

        # 执行领取
        self.experience += amount
        self.weekly_exp_quota -= amount
        self.weekly_exp_claimed += amount
        self.weekly_claim_count += 1

        self.add_message(f"成功领取 {amount} 经验！剩余额度：{self.weekly_exp_quota}", "success")

        # 检查升级
        old_level = self.level
        self.calculate_level()
        if self.level > old_level:
            self.add_message(f"升级了！现在是 {self.level} 级", "success")
            self.check_team_unlock()

        return True

    def init_first_team(self):
        """初始化第一个队伍"""
        # 寻找初始位置地块
        start_pos = None
        for pos, tile in self.hex_map.items():
            if tile.terrain_type == TerrainType.START_POSITION:
                start_pos = pos
                break

        if not start_pos:
            # 如果没有找到初始位置，提示错误
            print("警告：地图中没有设置初始位置！")
            # 在中心创建一个初始位置
            start_pos = (0, 0)
            if start_pos not in self.hex_map:
                self.hex_map[start_pos] = HexTile(0, 0, TerrainType.START_POSITION)
            else:
                # 如果(0,0)位置已存在，强制改为初始位置
                self.hex_map[start_pos].terrain_type = TerrainType.START_POSITION

        # 创建第一个队伍
        team = Team(id=1, position=start_pos, action_points=6)
        self.teams.append(team)

        # 将起始位置标记为已征服
        self.conquered_tiles.add(start_pos)

    def load_map_data(self):
        """加载地图数据"""
        if os.path.exists("map_save.json"):
            try:
                with open("map_save.json", 'r') as f:
                    save_data = json.load(f)

                # 重建地图
                for tile_data in save_data['tiles']:
                    tile = HexTile.from_dict(tile_data)
                    self.hex_map[(tile.q, tile.r)] = tile

                print(f"加载地图：{len(self.hex_map)} 个地块")
            except Exception as e:
                print(f"加载地图失败: {e}")
                self.create_default_map()
        else:
            self.create_default_map()

    def create_default_map(self):
        """创建默认地图"""
        # 创建一个小型默认地图用于测试
        has_start = False
        for q in range(-10, 11):
            for r in range(-10, 11):
                if abs(q + r) <= 10:  # 六边形范围
                    # 中心位置设为初始位置
                    if q == 0 and r == 0 and not has_start:
                        terrain = TerrainType.START_POSITION
                        has_start = True
                    else:
                        # 随机分配地形
                        import random
                        terrain_choices = [
                            TerrainType.WALL,  # 墙壁作为主要地形
                            TerrainType.NORMAL_LV1,
                            TerrainType.NORMAL_LV2,
                            TerrainType.DUMMY_LV1,
                            TerrainType.TREASURE_1,
                            TerrainType.TENT,
                            TerrainType.BLACK_MARKET,
                            TerrainType.BOSS_ZETSU,  # 有飞雷神
                            TerrainType.BOSS_KUSHINA,  # 有飞雷神
                        ]
                        # 让墙壁有更高概率
                        weights = [3, 1, 1, 1, 1, 1, 1, 0.5, 0.5]
                        terrain = random.choices(terrain_choices, weights=weights)[0]
                    self.hex_map[(q, r)] = HexTile(q, r, terrain)

    def calculate_level(self):
        """计算当前等级"""
        self.level = self.experience // 100
        if self.level < 1:
            self.level = 1

    def check_team_unlock(self):
        """检查是否解锁新队伍 - 立即在当前位置生成"""
        # 20级解锁二队
        if self.level >= 20 and self.max_teams == 1:
            self.max_teams = 2
            if len(self.teams) == 1:
                # 在一队当前位置生成二队
                pos = self.teams[0].position
                new_team = Team(id=2, position=pos, action_points=6)
                self.teams.append(new_team)
                self.add_message("解锁了队伍2！", "success")

        # 60级解锁三队
        if self.level >= 60 and self.max_teams == 2:
            self.max_teams = 3
            if len(self.teams) == 2:
                # 在当前活动队伍位置生成三队
                pos = self.teams[self.current_team_index].position
                new_team = Team(id=3, position=pos, action_points=6)
                self.teams.append(new_team)
                self.add_message("解锁了队伍3！", "success")

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

    def check_treasure_buff(self):
        """检查是否集齐8个远征秘宝"""
        treasure_types = [
            TerrainType.TREASURE_1, TerrainType.TREASURE_2,
            TerrainType.TREASURE_3, TerrainType.TREASURE_4,
            TerrainType.TREASURE_5, TerrainType.TREASURE_6,
            TerrainType.TREASURE_7, TerrainType.TREASURE_8
        ]

        if not self.has_treasure_buff:
            all_treasures = all(t in self.treasures_conquered for t in treasure_types)
            if all_treasures:
                self.has_treasure_buff = True
                self.add_message("集齐8个远征秘宝！所有消耗减少20%！", "success")

    def apply_cost_reduction(self, cost):
        """应用秘宝buff减免"""
        if self.has_treasure_buff:
            return int(cost * 0.8)
        return cost

    def get_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """获取六边形的邻居"""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in directions:
            neighbor = (q + dq, r + dr)
            if neighbor in self.hex_map:
                neighbors.append(neighbor)
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int],
                  conquered_only: bool = False) -> Optional[List[Tuple[int, int]]]:
        """使用A*算法寻找路径"""
        if start == end:
            return [start]

        # A*算法
        from heapq import heappush, heappop

        def heuristic(a, b):
            # 六边形距离
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

            for neighbor in self.get_neighbors(*current):
                # 检查是否可通过
                tile = self.hex_map[neighbor]

                # 如果要求只通过已征服地块
                if conquered_only and neighbor not in self.conquered_tiles:
                    continue

                # 检查是否是墙壁
                if tile.terrain_type == TerrainType.WALL:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # 没有找到路径

    def calculate_move_cost(self, team: Team, target: Tuple[int, int]) -> int:
        """计算移动成本"""
        # 新规则：不能停留在已征服地块
        if target in self.conquered_tiles:
            return -1  # 不允许

        # 移动到未征服地块
        # 检查是否通过已征服路径
        path = self.find_path(team.position, target, conquered_only=True)
        if path:
            steps = len(path)  # 包括起点
            cost = 30 + 10 * steps
        else:
            # 直接相邻的未征服地块
            if target in self.get_neighbors(*team.position):
                cost = 50
            else:
                return -1  # 无法到达

        # 应用秘宝buff
        return self.apply_cost_reduction(cost)

    def move_team(self, team: Team, target: Tuple[int, int]) -> bool:
        """移动队伍"""
        # 不能移动到已征服地块（新规则）
        if target in self.conquered_tiles:
            self.add_message("不能停留在已征服地块！", "error")
            return False

        cost = self.calculate_move_cost(team, target)
        if cost < 0:
            self.add_message("无法到达目标！", "error")
            return False

        if cost > self.food:
            self.add_message(f"粮草不足！需要 {cost}", "error")
            return False

        # 执行移动
        self.food -= cost
        team.position = target

        self.add_message(f"移动成功。粮草 -{cost}", "info")
        return True

    def use_thunder_god(self, team: Team, target: Tuple[int, int]) -> bool:
        """使用飞雷神道具"""
        if self.thunder_god_items <= 0:
            self.add_message("没有飞雷神道具！", "error")
            return False

        # 检查目标是否有效（必须是未征服且与已征服地块相邻）
        if target in self.conquered_tiles:
            self.add_message("目标已被征服！", "error")
            return False

        # 检查是否与已征服地块相邻
        neighbors = self.get_neighbors(*target)
        has_conquered_neighbor = any(n in self.conquered_tiles for n in neighbors)

        if not has_conquered_neighbor:
            self.add_message("目标必须与已征服地块相邻！", "error")
            return False

        # 执行传送
        team.position = target
        self.thunder_god_items -= 1
        self.add_message(f"使用飞雷神传送！剩余 {self.thunder_god_items} 个", "success")
        return True

    def get_valid_thunder_targets(self) -> Set[Tuple[int, int]]:
        """获取所有有效的飞雷神目标"""
        valid_targets = set()

        for pos, tile in self.hex_map.items():
            # 必须是未征服地块
            if pos in self.conquered_tiles:
                continue

            # 必须与已征服地块相邻
            neighbors = self.get_neighbors(*pos)
            if any(n in self.conquered_tiles for n in neighbors):
                # 不能是墙壁
                if tile.terrain_type != TerrainType.WALL:
                    valid_targets.add(pos)

        return valid_targets

    def conquer_tile(self, team: Team) -> bool:
        """征服当前地块"""
        pos = team.position
        if pos in self.conquered_tiles:
            self.add_message("已经征服过了！", "warning")
            return False

        tile = self.hex_map.get(pos)
        if not tile:
            return False

        # 获取地块属性
        props = self.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})
        food_cost = props.get('food_cost', 0)
        exp_gain = props.get('exp_gain', 0)
        score_cost = props.get('score_cost', 0)

        # 检查是否可征服
        if food_cost < 0:  # 不可征服的地块（如墙壁）
            self.add_message("无法征服此地块！", "error")
            return False

        # 应用秘宝buff到成本
        food_cost = self.apply_cost_reduction(food_cost)

        # 检查资源
        if food_cost > self.food:
            self.add_message(f"粮草不足！需要 {food_cost}", "error")
            return False

        if score_cost > self.conquest_score:
            self.add_message(f"积分不足！需要 {score_cost}", "error")
            return False

        # 帐篷不消耗行动点，其他地块消耗1点
        if tile.terrain_type != TerrainType.TENT:
            if team.action_points <= 0:
                self.add_message("没有行动点数！", "error")
                return False
            team.action_points -= 1

        # 执行征服
        self.food -= food_cost
        self.conquest_score -= score_cost
        self.experience += exp_gain

        # 特殊地块效果
        if tile.terrain_type == TerrainType.TENT:
            # 帐篷效果
            food_gain = self.style.get_tent_food_gain(self.current_day)
            self.food += food_gain
            team.action_points = min(team.action_points + 1, team.max_action_points)
            self.add_message(f"征服帐篷！+{food_gain} 粮草，+1 行动", "success")

        elif tile.terrain_type in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
            # BOSS掉落飞雷神
            self.thunder_god_items += 1
            self.add_message(f"征服BOSS！+{exp_gain} 经验，获得飞雷神道具！", "success")

        elif tile.terrain_type in [TerrainType.TREASURE_1, TerrainType.TREASURE_2,
                                  TerrainType.TREASURE_3, TerrainType.TREASURE_4,
                                  TerrainType.TREASURE_5, TerrainType.TREASURE_6,
                                  TerrainType.TREASURE_7, TerrainType.TREASURE_8]:
            # 记录秘宝征服
            self.treasures_conquered.add(tile.terrain_type)
            self.check_treasure_buff()
            self.add_message(f"征服秘宝！+{exp_gain} 经验，-{food_cost} 粮草", "success")
        else:
            self.add_message(f"征服成功！+{exp_gain} 经验，-{food_cost} 粮草", "success")

        # 标记为已征服
        self.conquered_tiles.add(pos)
        tile.conquered = True

        # 检查等级和队伍解锁
        old_level = self.level
        self.calculate_level()
        if self.level > old_level:
            self.add_message(f"升级了！现在是 {self.level} 级", "success")
            self.check_team_unlock()

        return True

    def next_day(self):
        """进入下一天"""
        if self.current_day >= self.max_days:
            self.game_state = GameState.GAME_OVER
            return

        self.current_day += 1

        # 获取星期几
        day_of_week = self.get_day_of_week(self.current_day)

        # 发放每日粮草
        daily_food = self.get_daily_food()
        self.food += daily_food

        # 发放每日征服积分
        self.conquest_score += 1000

        # 恢复所有队伍的行动点数
        for team in self.teams:
            team.action_points = min(team.action_points + 6, team.max_action_points)

        # 周一刷新周经验额度
        if day_of_week == 1:  # 周一
            self.reset_weekly_exp()
        # 周日自动发放未领取的周经验
        elif day_of_week == 0:  # 周日
            self.auto_claim_weekly_exp()

        self.add_message(f"第 {self.current_day} 天：+{daily_food} 粮草，+1000 积分，+6 行动", "info")

    def add_message(self, text: str, msg_type: str = "info"):
        """添加消息"""
        self.message = text
        self.message_timer = 3000  # 3秒
        print(f"[{msg_type.upper()}] {text}")

    def hex_to_pixel(self, q: int, r: int) -> Tuple[float, float]:
        """六边形坐标转像素坐标"""
        x = self.hex_size * 3 / 2 * q * self.zoom
        y = self.hex_size * math.sqrt(3) * (r + q / 2) * self.zoom
        return x + self.camera_x, y + self.camera_y

    def pixel_to_hex(self, x: float, y: float) -> Tuple[int, int]:
        """像素坐标转六边形坐标"""
        x = (x - self.camera_x) / (self.hex_size * self.zoom)
        y = (y - self.camera_y) / (self.hex_size * self.zoom)

        q = 2 / 3 * x
        r = -1 / 3 * x + math.sqrt(3) / 3 * y

        return self.hex_round(q, r)

    def hex_round(self, q: float, r: float) -> Tuple[int, int]:
        """六边形坐标舍入"""
        s = -q - r
        rq = round(q)
        rr = round(r)
        rs = round(s)

        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)

        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs

        return int(rq), int(rr)

    def draw_hexagon(self, center: Tuple[float, float], color: Tuple[int, int, int],
                     border_color: Optional[Tuple[int, int, int]] = None, border_width: int = 2):
        """绘制六边形"""
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            x = center[0] + self.hex_size * self.zoom * math.cos(angle)
            y = center[1] + self.hex_size * self.zoom * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(self.screen, color, points)
        if border_color:
            pygame.draw.polygon(self.screen, border_color, points, max(1, int(border_width * self.zoom)))

    def draw_map(self):
        """绘制地图"""
        # 获取可见范围
        visible_tiles = self.get_visible_tiles()

        # 创建一个Surface用于所有已征服地块的半透明覆盖（性能优化）
        overlay_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)

        # 绘制所有地块
        for pos in visible_tiles:
            if pos not in self.hex_map:
                continue

            tile = self.hex_map[pos]
            center = self.hex_to_pixel(*pos)

            # 获取地块颜色（不再对已征服地块做颜色调整）
            color = self.style.TERRAIN_COLORS[tile.terrain_type]

            # 飞雷神模式下高亮有效目标
            if self.thunder_god_mode and pos in self.valid_thunder_targets:
                color = tuple(min(255, int(c * 1.5)) for c in color)

            # 绘制六边形
            border_color = (80, 80, 90)  # 默认边框
            if pos == self.selected_tile:
                border_color = (255, 215, 0)  # 金色边框
            elif pos in self.conquered_tiles:
                border_color = (50, 50, 60)  # 已征服地块使用更暗的边框
            elif self.thunder_god_mode and pos in self.valid_thunder_targets:
                border_color = (255, 100, 255)  # 紫色边框

            self.draw_hexagon(center, color, border_color)

            # 收集已征服地块的多边形点，稍后统一绘制覆盖层
            if pos in self.conquered_tiles:
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    x = center[0] + self.hex_size * self.zoom * math.cos(angle)
                    y = center[1] + self.hex_size * self.zoom * math.sin(angle)
                    points.append((x, y))

                # 绘制到覆盖层Surface上（半透明黑色）
                pygame.draw.polygon(overlay_surface, (0, 0, 0, 120), points)  # 黑色，alpha=120

            # 绘制地块名称（缩放时调整）
            if self.zoom >= 0.5 and tile.terrain_type != TerrainType.WALL:
                name = self.style.get_terrain_name_chinese(tile.terrain_type)
                font_size = max(8, int(16 * self.zoom))
                font = get_font(font_size)

                # 已征服地块的文字颜色调整为更亮，以便在黑色覆盖上仍然可见
                if pos in self.conquered_tiles:
                    text_color = (180, 180, 180)  # 灰色文字
                else:
                    text_color = (255, 255, 255)  # 白色文字

                text = font.render(name[:8], True, text_color)
                text_rect = text.get_rect(center=(int(center[0]), int(center[1])))
                self.screen.blit(text, text_rect)

        # 一次性绘制所有已征服地块的覆盖层
        self.screen.blit(overlay_surface, (0, 0))

        # 绘制路径预览
        if self.path_preview:
            for i in range(len(self.path_preview) - 1):
                start_pos = self.hex_to_pixel(*self.path_preview[i])
                end_pos = self.hex_to_pixel(*self.path_preview[i + 1])
                pygame.draw.line(self.screen, (255, 255, 0), start_pos, end_pos, 3)

        # 绘制队伍
        self.draw_teams()

    def draw_teams(self):
        """绘制所有队伍"""
        for i, team in enumerate(self.teams):
            center = self.hex_to_pixel(*team.position)

            # 动画效果
            offset = math.sin(self.animation_timer + i * math.pi / 2) * 2

            # 队伍颜色
            if i == self.current_team_index:
                color = (255, 215, 0)  # 当前队伍 - 金色
            else:
                color = (200, 200, 200)  # 其他队伍 - 灰色

            # 绘制队伍标记（三角形）
            size = 15 * self.zoom
            points = [
                (center[0], center[1] - size + offset),
                (center[0] - size * 0.866, center[1] + size * 0.5 + offset),
                (center[0] + size * 0.866, center[1] + size * 0.5 + offset)
            ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)

            # 绘制队伍编号
            text = self.small_font.render(str(team.id), True, (0, 0, 0))
            text_rect = text.get_rect(center=(int(center[0]), int(center[1] + offset)))
            self.screen.blit(text, text_rect)

    def get_visible_tiles(self) -> Set[Tuple[int, int]]:
        """获取可见的地块"""
        visible = set()

        # 计算屏幕边界对应的六边形坐标
        corners = [
            (0, 0),
            (self.width, 0),
            (0, self.height - 180),  # 留出UI空间
            (self.width, self.height - 180)
        ]

        min_q, max_q = float('inf'), float('-inf')
        min_r, max_r = float('inf'), float('-inf')

        for corner in corners:
            q, r = self.pixel_to_hex(*corner)
            min_q = min(min_q, q - 2)
            max_q = max(max_q, q + 2)
            min_r = min(min_r, r - 2)
            max_r = max(max_r, r + 2)

        # 添加范围内的所有地块
        for q in range(int(min_q), int(max_q) + 1):
            for r in range(int(min_r), int(max_r) + 1):
                if (q, r) in self.hex_map:
                    visible.add((q, r))

        return visible

    def draw_ui(self):
        """绘制UI界面"""
        # UI背景
        ui_height = 180  # 增加高度以容纳更多信息
        ui_rect = pygame.Rect(0, self.height - ui_height, self.width, ui_height)
        pygame.draw.rect(self.screen, (40, 40, 50), ui_rect)
        pygame.draw.rect(self.screen, (80, 80, 90), ui_rect, 2)

        # 左侧 - 游戏状态
        y_offset = self.height - ui_height + 10

        # 第一行：天数和等级
        day_text = f"第 {self.current_day}/{self.max_days} 天"
        level_text = f"等级 {self.level} ({self.experience} 经验)"

        day_surface = self.font.render(day_text, True, (255, 255, 255))
        level_surface = self.font.render(level_text, True, (255, 215, 0))

        self.screen.blit(day_surface, (10, y_offset))
        self.screen.blit(level_surface, (10, y_offset + 25))

        # 第二行：资源
        food_text = f"粮草: {self.food}"
        score_text = f"积分: {self.conquest_score}"

        # 显示buff状态
        if self.has_treasure_buff:
            food_text += " (消耗-20%)"

        food_color = (255, 100, 100) if self.food < 100 else (255, 255, 255)
        food_surface = self.font.render(food_text, True, food_color)
        score_surface = self.font.render(score_text, True, (255, 255, 255))

        self.screen.blit(food_surface, (10, y_offset + 50))
        self.screen.blit(score_surface, (10, y_offset + 75))

        # 第三行：特殊道具和周经验
        thunder_text = f"飞雷神: {self.thunder_god_items}"
        week_exp_text = f"周经验: {self.weekly_exp_claimed}/500 (余{self.weekly_exp_quota})"

        thunder_surface = self.font.render(thunder_text, True, (255, 200, 255))
        week_surface = self.font.render(week_exp_text, True, (200, 255, 200))

        self.screen.blit(thunder_surface, (10, y_offset + 100))
        self.screen.blit(week_surface, (10, y_offset + 125))

        # 中间 - 队伍信息
        x_offset = 350
        for i, team in enumerate(self.teams):
            team_x = x_offset + i * 200

            # 队伍框
            team_rect = pygame.Rect(team_x, y_offset, 180, 120)
            if i == self.current_team_index:
                pygame.draw.rect(self.screen, (80, 120, 160), team_rect, 2)
            else:
                pygame.draw.rect(self.screen, (60, 60, 70), team_rect, 1)

            # 队伍信息
            team_title = f"队伍 {team.id}"
            if i == self.current_team_index:
                team_title += " (当前)"

            title_surface = self.font.render(team_title, True, (255, 255, 255))
            self.screen.blit(title_surface, (team_x + 10, y_offset + 5))

            pos_text = f"位置: ({team.position[0]}, {team.position[1]})"
            action_text = f"行动: {team.action_points}/{team.max_action_points}"

            pos_surface = self.small_font.render(pos_text, True, (200, 200, 200))
            action_surface = self.small_font.render(action_text, True, (200, 255, 200))

            self.screen.blit(pos_surface, (team_x + 10, y_offset + 30))
            self.screen.blit(action_surface, (team_x + 10, y_offset + 50))

        # 右侧 - 操作提示
        hints_x = self.width - 300
        hints = [
            "Tab: 切换队伍",
            "点击: 选择/移动",
            "C: 征服地块",
            "N: 下一天",
            "W: 领取周经验",
            "T: 飞雷神模式",
            "ESC: 返回菜单"
        ]

        for i, hint in enumerate(hints):
            hint_surface = self.small_font.render(hint, True, (150, 150, 150))
            self.screen.blit(hint_surface, (hints_x, y_offset + i * 18))

    def draw_weekly_exp_ui(self):
        """绘制周经验领取界面"""
        # 半透明背景
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # 窗口
        window_width = 500
        window_height = 400
        window_x = (self.width - window_width) // 2
        window_y = (self.height - window_height) // 2

        window_rect = pygame.Rect(window_x, window_y, window_width, window_height)
        pygame.draw.rect(self.screen, (50, 50, 60), window_rect)
        pygame.draw.rect(self.screen, (100, 100, 110), window_rect, 3)

        # 标题
        title = "领取周经验"
        title_surface = self.large_font.render(title, True, (255, 215, 0))
        title_rect = title_surface.get_rect(centerx=window_x + window_width // 2, y=window_y + 20)
        self.screen.blit(title_surface, title_rect)

        # 信息
        y = window_y + 80
        info_lines = [
            f"本周剩余额度: {self.weekly_exp_quota}/500",
            f"已领取次数: {self.weekly_claim_count}/5",
            f"当前经验: {self.experience} (等级 {self.level})"
        ]

        for line in info_lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(centerx=window_x + window_width // 2, y=y)
            self.screen.blit(text_surface, text_rect)
            y += 35

        # 领取按钮
        button_y = window_y + 220
        amounts = [100, 200, 300, 400, 500]
        button_width = 80
        button_height = 40
        button_spacing = 10
        total_width = len(amounts) * button_width + (len(amounts) - 1) * button_spacing
        start_x = window_x + (window_width - total_width) // 2

        for i, amount in enumerate(amounts):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            # 检查是否可领取
            can_claim = (amount <= self.weekly_exp_quota and
                        self.weekly_claim_count < 5)

            if can_claim:
                pygame.draw.rect(self.screen, (80, 120, 160), button_rect)
                text_color = (255, 255, 255)
            else:
                pygame.draw.rect(self.screen, (60, 60, 70), button_rect)
                text_color = (150, 150, 150)

            pygame.draw.rect(self.screen, (100, 100, 110), button_rect, 2)

            text_surface = self.font.render(str(amount), True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

        # 关闭提示
        close_text = "按 ESC 或 W 关闭"
        close_surface = self.small_font.render(close_text, True, (150, 150, 150))
        close_rect = close_surface.get_rect(centerx=window_x + window_width // 2,
                                           y=window_y + window_height - 40)
        self.screen.blit(close_surface, close_rect)

    def draw_info_panel(self):
        """绘制信息面板"""
        if self.selected_tile and self.selected_tile in self.hex_map:
            tile = self.hex_map[self.selected_tile]
            props = self.style.TERRAIN_PROPERTIES.get(tile.terrain_type, {})

            # 面板背景
            panel_width = 280
            panel_height = 150
            panel_x = 10
            panel_y = 60

            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            panel_surface = pygame.Surface((panel_width, panel_height))
            panel_surface.set_alpha(230)
            panel_surface.fill((30, 30, 40))
            self.screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(self.screen, (100, 100, 110), panel_rect, 2)

            # 地块信息
            y = panel_y + 10

            name = self.style.get_terrain_name_chinese(tile.terrain_type)
            name_surface = self.font.render(name, True, (255, 255, 255))
            self.screen.blit(name_surface, (panel_x + 10, y))

            y += 25
            status = "已征服" if self.selected_tile in self.conquered_tiles else "未征服"
            status_color = (100, 255, 100) if status == "已征服" else (255, 150, 150)
            status_surface = self.small_font.render(f"状态: {status}", True, status_color)
            self.screen.blit(status_surface, (panel_x + 10, y))

            y += 20
            food_cost = props.get('food_cost', 0)
            if food_cost >= 0:
                # 应用buff
                display_cost = self.apply_cost_reduction(food_cost)
                cost_text = f"征服消耗: {display_cost} 粮草"
                if self.has_treasure_buff and food_cost > 0:
                    cost_text += f" (原{food_cost})"
            else:
                cost_text = "无法征服"
            cost_surface = self.small_font.render(cost_text, True, (200, 200, 200))
            self.screen.blit(cost_surface, (panel_x + 10, y))

            y += 20
            exp_gain = props.get('exp_gain', 0)
            exp_surface = self.small_font.render(f"经验值: +{exp_gain}", True, (200, 200, 200))
            self.screen.blit(exp_surface, (panel_x + 10, y))

            # 特殊说明
            y += 20
            if tile.terrain_type in [TerrainType.BOSS_ZETSU, TerrainType.BOSS_KUSHINA]:
                special_surface = self.small_font.render("掉落: 飞雷神道具", True, (255, 200, 255))
                self.screen.blit(special_surface, (panel_x + 10, y))
            elif tile.terrain_type == TerrainType.TENT:
                special_surface = self.small_font.render("效果: +粮草 +行动点", True, (200, 255, 200))
                self.screen.blit(special_surface, (panel_x + 10, y))

            # 移动成本（如果选中了队伍）
            if self.teams:
                current_team = self.teams[self.current_team_index]
                if current_team.position != self.selected_tile:
                    y += 20
                    move_cost = self.calculate_move_cost(current_team, self.selected_tile)
                    if move_cost > 0:
                        move_text = f"移动消耗: {move_cost} 粮草"
                        move_color = (255, 100, 100) if move_cost > self.food else (200, 200, 200)
                    else:
                        move_text = "无法到达"
                        move_color = (255, 100, 100)
                    move_surface = self.small_font.render(move_text, True, move_color)
                    self.screen.blit(move_surface, (panel_x + 10, y))

    def draw_message(self):
        """绘制消息"""
        if self.message and self.message_timer > 0:
            # 消息背景
            text_surface = self.large_font.render(self.message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.width // 2, 40))

            # 背景
            bg_rect = text_rect.inflate(40, 20)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.set_alpha(200)
            bg_surface.fill((50, 50, 60))
            self.screen.blit(bg_surface, bg_rect)

            # 文字
            self.screen.blit(text_surface, text_rect)

    def handle_mouse_click(self, pos):
        """处理鼠标点击"""
        x, y = pos

        # 周经验界面点击处理
        if self.game_state == GameState.WEEKLY_EXP_CLAIM:
            # 检查领取按钮
            window_width = 500
            window_height = 400
            window_x = (self.width - window_width) // 2
            window_y = (self.height - window_height) // 2
            button_y = window_y + 220

            amounts = [100, 200, 300, 400, 500]
            button_width = 80
            button_height = 40
            button_spacing = 10
            total_width = len(amounts) * button_width + (len(amounts) - 1) * button_spacing
            start_x = window_x + (window_width - total_width) // 2

            for i, amount in enumerate(amounts):
                button_x = start_x + i * (button_width + button_spacing)
                button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

                if button_rect.collidepoint(pos):
                    if self.claim_weekly_exp(amount):
                        # 成功领取后可以选择关闭界面
                        if self.weekly_exp_quota == 0:
                            self.game_state = GameState.PLAYING
            return

        # 检查是否点击UI区域
        if y > self.height - 180:
            return

        # 转换为六边形坐标
        hex_pos = self.pixel_to_hex(x, y)

        if hex_pos in self.hex_map:
            # 飞雷神模式
            if self.thunder_god_mode:
                if hex_pos in self.valid_thunder_targets:
                    current_team = self.teams[self.current_team_index]
                    if self.use_thunder_god(current_team, hex_pos):
                        self.thunder_god_mode = False
                        self.valid_thunder_targets.clear()
                return

            # 正常模式
            if self.teams:
                current_team = self.teams[self.current_team_index]

                # 如果点击当前位置，尝试征服
                if hex_pos == current_team.position:
                    if hex_pos not in self.conquered_tiles:
                        self.conquer_tile(current_team)
                # 否则尝试移动
                else:
                    self.move_team(current_team, hex_pos)

            self.selected_tile = hex_pos

    def handle_keyboard(self, key):
        """处理键盘输入"""
        if self.game_state == GameState.WEEKLY_EXP_CLAIM:
            if key == pygame.K_ESCAPE or key == pygame.K_w:
                self.game_state = GameState.PLAYING
            # 数字键快捷领取
            elif pygame.K_1 <= key <= pygame.K_5:
                amount = (key - pygame.K_1 + 1) * 100
                self.claim_weekly_exp(amount)
            return

        if key == pygame.K_TAB:
            # 切换队伍
            if len(self.teams) > 1:
                self.current_team_index = (self.current_team_index + 1) % len(self.teams)
                team = self.teams[self.current_team_index]
                self.add_message(f"切换到队伍 {team.id}", "info")

                # 将相机移动到当前队伍
                target_x, target_y = self.hex_to_pixel(*team.position)
                self.camera_x = self.width // 2 - (target_x - self.camera_x)
                self.camera_y = self.height // 2 - (target_y - self.camera_y)

        elif key == pygame.K_c:
            # 征服当前地块
            if self.teams:
                current_team = self.teams[self.current_team_index]
                self.conquer_tile(current_team)

        elif key == pygame.K_n:
            # 下一天
            self.next_day()

        elif key == pygame.K_w:
            # 打开周经验界面
            if self.game_state == GameState.PLAYING:
                self.game_state = GameState.WEEKLY_EXP_CLAIM
            else:
                self.game_state = GameState.PLAYING

        elif key == pygame.K_t:
            # 切换飞雷神模式
            if self.thunder_god_items > 0:
                self.thunder_god_mode = not self.thunder_god_mode
                if self.thunder_god_mode:
                    self.valid_thunder_targets = self.get_valid_thunder_targets()
                    self.add_message(f"飞雷神模式开启！选择目标地块", "info")
                else:
                    self.valid_thunder_targets.clear()
                    self.add_message("飞雷神模式关闭", "info")

        elif key == pygame.K_SPACE:
            # 聚焦到当前队伍
            if self.teams:
                team = self.teams[self.current_team_index]
                target_x, target_y = self.hex_to_pixel(*team.position)
                self.camera_x = self.width // 2 - (target_x - self.camera_x)
                self.camera_y = self.height // 2 - (target_y - self.camera_y)

    def update(self, dt):
        """更新游戏状态"""
        # 更新动画
        self.animation_timer += dt * 2

        # 更新消息计时器
        if self.message_timer > 0:
            self.message_timer -= dt * 1000

        # 检查游戏结束
        if self.current_day >= self.max_days:
            self.game_state = GameState.GAME_OVER

    def draw(self):
        """绘制游戏画面"""
        # 清屏
        self.screen.fill((30, 30, 40))

        if self.game_state == GameState.GAME_OVER:
            self.draw_game_over()
        elif self.game_state == GameState.WEEKLY_EXP_CLAIM:
            # 先绘制游戏场景
            self.draw_map()
            self.draw_ui()
            # 再绘制周经验界面
            self.draw_weekly_exp_ui()
        else:
            # 绘制游戏元素
            self.draw_map()
            self.draw_ui()
            self.draw_info_panel()
            self.draw_message()

    def draw_game_over(self):
        """绘制游戏结束画面"""
        # 标题
        title = "游戏结束"
        title_surface = self.large_font.render(title, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 200))
        self.screen.blit(title_surface, title_rect)

        # 统计信息
        stats = [
            f"最终等级: {self.level}",
            f"总经验值: {self.experience}",
            f"征服地块: {len(self.conquered_tiles)}",
            f"解锁队伍: {len(self.teams)}",
            f"剩余粮草: {self.food}",
            f"剩余积分: {self.conquest_score}",
            "",
            "按 ESC 返回主菜单"
        ]

        y = 300
        for stat in stats:
            if stat:
                stat_surface = self.font.render(stat, True, (255, 255, 255))
                stat_rect = stat_surface.get_rect(center=(self.width // 2, y))
                self.screen.blit(stat_surface, stat_rect)
            y += 40

    # 在 game_play_system.py 的 run 方法中，将鼠标事件处理部分替换为以下代码：

    def run(self):
        """游戏主循环"""
        running = True

        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.game_state == GameState.WEEKLY_EXP_CLAIM:
                            self.game_state = GameState.PLAYING
                        else:
                            return 'menu'
                    else:
                        self.handle_keyboard(event.key)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键 - 点击或开始拖动
                        # 检查是否点击UI区域
                        if event.pos[1] > self.height - 180:
                            # UI区域点击，不处理拖动
                            continue

                        # 记录拖动开始状态
                        self.is_dragging = True
                        self.drag_start_pos = event.pos
                        self.drag_start_camera = (self.camera_x, self.camera_y)

                        # 同时处理点击
                        self.handle_mouse_click(event.pos)

                    elif event.button == 3:  # 右键 - 专用拖动（保持原有功能）
                        self.is_dragging = True
                        self.drag_start_pos = event.pos
                        self.drag_start_camera = (self.camera_x, self.camera_y)

                    elif event.button == 4:  # 滚轮上 - 放大
                        old_zoom = self.zoom
                        self.zoom = min(3.0, self.zoom * 1.1)
                        # 向鼠标位置缩放
                        if self.zoom != old_zoom:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            zoom_factor = self.zoom / old_zoom
                            self.camera_x = mouse_x - (mouse_x - self.camera_x) * zoom_factor
                            self.camera_y = mouse_y - (mouse_y - self.camera_y) * zoom_factor

                    elif event.button == 5:  # 滚轮下 - 缩小
                        old_zoom = self.zoom
                        self.zoom = max(0.3, self.zoom / 1.1)
                        # 从鼠标位置缩放
                        if self.zoom != old_zoom:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            zoom_factor = self.zoom / old_zoom
                            self.camera_x = mouse_x - (mouse_x - self.camera_x) * zoom_factor
                            self.camera_y = mouse_y - (mouse_y - self.camera_y) * zoom_factor

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 or event.button == 3:  # 左键或右键释放
                        self.is_dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        # 基于拖拽更新相机位置
                        dx = event.pos[0] - self.drag_start_pos[0]
                        dy = event.pos[1] - self.drag_start_pos[1]
                        self.camera_x = self.drag_start_camera[0] + dx
                        self.camera_y = self.drag_start_camera[1] + dy

            # 更新和绘制
            self.update(dt)
            self.draw()
            pygame.display.flip()

        return 'menu'