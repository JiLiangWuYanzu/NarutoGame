"""
六边形地图编辑器 - 游戏地块系统版本（中文版）- 带裁剪功能（修复版）
特点：平顶六边形（尖顶在两侧）
使用轴向坐标系统作为六边形网格
增强功能：分类地块选择系统，支持41种地块类型，地图裁剪功能
"""
import pygame
import math
import json
import os
from enum import Enum
from typing import Dict, Tuple, List, Optional
import sys
from datetime import datetime

# 导入样式配置
from map_style_config import StyleConfig, TerrainType
# 导入字体管理器
from font_manager import get_font_manager, get_font, get_small_font, get_medium_font, get_large_font


class LogMessage:
    """日志消息类"""

    def __init__(self, text: str, msg_type: str = "info", timestamp: bool = True):
        self.text = text
        self.msg_type = msg_type  # info, success, warning, error
        self.timestamp = datetime.now().strftime("%H:%M:%S") if timestamp else None
        self.alpha = 255  # 用于淡出效果

    def get_color(self):
        """根据消息类型返回颜色"""
        colors = {
            "info": (200, 200, 200),  # 浅灰色
            "success": (100, 255, 100),  # 绿色
            "warning": (255, 200, 100),  # 橙色
            "error": (255, 100, 100),  # 红色
        }
        return colors.get(self.msg_type, (200, 200, 200))


class TerrainCategory(Enum):
    """地块分类"""
    BASIC = "基础"
    TRAINING = "训练"
    WATCHTOWER = "瞭望塔"
    SPECIAL = "特殊"
    TREASURE = "宝藏"
    BOSS = "BOSS"
    OBSTACLE = "障碍"


class HexTile:
    """六边形瓦片类 - 平顶六边形，使用轴向坐标"""

    def __init__(
            self,
            q: int,
            r: int,
            terrain_type: TerrainType = TerrainType.WALL
    ):
        self.q = q  # 列坐标（水平）
        self.r = r  # 行坐标（对角线）
        self.terrain_type = terrain_type
        self.attributes = {}  # 可以存储额外属性
        self.conquered = False  # 是否已征服

    def to_dict(self):
        """转换为可序列化的字典"""
        return {
            'q': self.q,
            'r': self.r,
            'terrain_type': self.terrain_type.value,
            'attributes': self.attributes,
            'conquered': self.conquered
        }

    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        tile = cls(
            data['q'],
            data['r'],
            TerrainType(data['terrain_type'])
        )
        tile.attributes = data.get('attributes', {})
        tile.conquered = data.get('conquered', False)
        return tile


class HexMapEditor:
    """六边形地图编辑器主类 - 游戏地块系统版本"""

    def __init__(
            self,
            width: int = 1200,
            height: int = 800,
            hex_size: int = 20
    ):
        # 首先存储基本参数
        self.width = width
        self.height = height
        self.hex_size = hex_size
        self.style = StyleConfig()

        # 地图数据
        self.hex_map: Dict[Tuple[int, int], HexTile] = {}
        self.map_width = 100  # 地图宽度（列数）
        self.map_height = 100  # 地图高度（行数）

        # 地图实际边界（新增）
        self.actual_min_q = -self.map_width // 2
        self.actual_max_q = self.map_width // 2
        self.actual_min_r = -self.map_height // 2
        self.actual_max_r = self.map_height // 2

        # 在已知地图大小后初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"六边形地图编辑器 - 游戏地形系统")

        # 初始化字体管理器
        self.font_manager = get_font_manager()

        # 使用中文字体
        self.font = get_medium_font()
        self.small_font = get_small_font()
        self.log_font = get_small_font()

        # 当前选中的地形类型
        self.current_terrain = TerrainType.NORMAL_LV1

        # 相机偏移
        self.camera_x = width // 2
        self.camera_y = height // 2
        self.start_position_placed = None  # 追踪初始位置坐标，确保唯一性

        # 鼠标拖拽
        self.is_dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_camera = (0, 0)

        # 缩放级别
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 3.0

        # 为大地图自动调整初始缩放
        if self.map_width > 50 or self.map_height > 50:
            self.zoom = 0.5
        if self.map_width > 80 or self.map_height > 80:
            self.zoom = 0.3

        # 性能优化 - 可见瓦片缓存
        self.visible_tiles = set()

        # UI相关
        self.ui_height = 180  # 增加UI高度以容纳更多地块

        # 地块分类系统 - 包含帐篷
        self.terrain_categories = {
            TerrainCategory.BASIC: [
                TerrainType.START_POSITION,  # 添加初始位置
                TerrainType.NORMAL_LV1, TerrainType.NORMAL_LV2, TerrainType.NORMAL_LV3,
                TerrainType.NORMAL_LV4, TerrainType.NORMAL_LV5, TerrainType.NORMAL_LV6
            ],
            TerrainCategory.TRAINING: [
                TerrainType.DUMMY_LV1, TerrainType.DUMMY_LV2, TerrainType.DUMMY_LV3,
                TerrainType.DUMMY_LV4, TerrainType.DUMMY_LV5, TerrainType.DUMMY_LV6
            ],
            TerrainCategory.WATCHTOWER: [
                TerrainType.WATCHTOWER_LV1, TerrainType.WATCHTOWER_LV2, TerrainType.WATCHTOWER_LV3,
                TerrainType.WATCHTOWER_LV4, TerrainType.WATCHTOWER_LV5, TerrainType.WATCHTOWER_LV6
            ],
            TerrainCategory.SPECIAL: [
                TerrainType.TRAINING_GROUND, TerrainType.BLACK_MARKET, TerrainType.RELIC_STONE,
                TerrainType.TENT  # 添加帐篷到特殊地块分类
            ],
            TerrainCategory.TREASURE: [
                TerrainType.TREASURE_1, TerrainType.TREASURE_2, TerrainType.TREASURE_3,
                TerrainType.TREASURE_4, TerrainType.TREASURE_5, TerrainType.TREASURE_6,
                TerrainType.TREASURE_7, TerrainType.TREASURE_8,
                # 添加新的特殊秘宝
                TerrainType.AKATSUKI_TREASURE, TerrainType.KONOHA_TREASURE_1,
                TerrainType.KONOHA_TREASURE_2
            ],
            TerrainCategory.BOSS: [
                TerrainType.BOSS_GAARA, TerrainType.BOSS_ZETSU, TerrainType.BOSS_DART,
                TerrainType.BOSS_SHIRA, TerrainType.BOSS_KUSHINA, TerrainType.BOSS_KISAME,
                TerrainType.BOSS_HANA
            ],
            TerrainCategory.OBSTACLE: [
                TerrainType.WALL,
                TerrainType.BOUNDARY
            ]
        }

        self.current_category = TerrainCategory.BASIC
        self.category_scroll = {cat: 0 for cat in TerrainCategory}  # 每个分类的滚动位置

        # ========== 裁剪模式相关变量（优化版本）==========
        self.crop_mode = False  # 是否处于裁剪模式
        self.crop_preview = {
            'left': 0,  # 左边裁剪列数
            'right': 0,  # 右边裁剪列数
            'top': 0,  # 上边裁剪行数
            'bottom': 0  # 下边裁剪行数
        }
        # 性能优化：缓存相关变量
        self.tiles_to_remove_cache = 0  # 缓存要删除的地块数量
        self.crop_preview_changed = True  # 标记裁剪预览是否改变

        # 日志系统
        self.log_messages: List[LogMessage] = []
        self.max_log_messages = 6  # 最多显示的日志条数（减少以保持清爽）
        self.log_width = 350
        self.log_height = 150  # 调整高度
        self.log_x = self.width - self.log_width - 10
        self.log_y = self.height - self.ui_height - self.log_height - 10

        # 添加初始欢迎消息（中文）
        self.add_log("地图编辑器已启动", "success")
        self.add_log(f"地图大小: {self.map_width}x{self.map_height}", "info")
        self.add_log("已加载41种地形类型", "info")
        self.add_log("Ctrl+C 进入裁剪模式", "info")

        # 初始化地图
        self.init_map()

        # 时钟
        self.clock = pygame.time.Clock()

        # 消息提示系统（保留用于屏幕中央的重要提示）
        self.message = ""
        self.message_timer = 0
        self.message_duration = 3000  # 消息显示3秒

    def add_log(self, text: str, msg_type: str = "info"):
        """添加日志消息"""
        # 同时输出到控制台
        print(f"[{msg_type.upper()}] {text}")

        # 添加到日志列表
        self.log_messages.append(LogMessage(text, msg_type))

        # 限制日志数量
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop(0)

    def toggle_crop_mode(self):
        """切换裁剪模式"""
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.add_log("进入裁剪模式 - 使用方向键调整裁剪范围", "info")
            self.message = "裁剪模式: 方向键调整, Enter确认, ESC取消"
            self.message_timer = 5000
            # 重置裁剪预览
            self.crop_preview = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
            self.crop_preview_changed = True  # 标记需要重新计算
            self.tiles_to_remove_cache = 0
        else:
            self.add_log("退出裁剪模式", "info")
            self.crop_preview = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
            self.crop_preview_changed = True

    def get_map_bounds(self):
        """获取当前地图中非空地块的边界"""
        min_q, max_q = float('inf'), float('-inf')
        min_r, max_r = float('inf'), float('-inf')

        for (q, r), tile in self.hex_map.items():
            if tile.terrain_type != TerrainType.WALL:
                min_q = min(min_q, q)
                max_q = max(max_q, q)
                min_r = min(min_r, r)
                max_r = max(max_r, r)

        if min_q == float('inf'):  # 地图全是墙
            return None

        return {
            'min_q': min_q,
            'max_q': max_q,
            'min_r': min_r,
            'max_r': max_r
        }

    def calculate_crop_bounds(self):
        """计算裁剪后的地图边界"""
        # 使用实际地图边界而不是假设的对称边界
        current_min_q = self.actual_min_q
        current_max_q = self.actual_max_q
        current_min_r = self.actual_min_r
        current_max_r = self.actual_max_r

        # 应用裁剪
        new_min_q = current_min_q + self.crop_preview['left']
        new_max_q = current_max_q - self.crop_preview['right']
        new_min_r = current_min_r + self.crop_preview['top']
        new_max_r = current_max_r - self.crop_preview['bottom']

        return new_min_q, new_max_q, new_min_r, new_max_r

    def count_tiles_to_remove(self):
        """统计将要删除的非空地块数量 - 带缓存"""
        # 只在裁剪预览改变时重新计算
        if not self.crop_preview_changed:
            return self.tiles_to_remove_cache

        new_min_q, new_max_q, new_min_r, new_max_r = self.calculate_crop_bounds()
        count = 0

        # 只检查非空地块，而不是遍历整个地图
        for (q, r), tile in self.hex_map.items():
            if tile.terrain_type != TerrainType.WALL:
                q_offset = q // 2
                if (q < new_min_q or q > new_max_q or
                        r < new_min_r - q_offset or r > new_max_r - q_offset):
                    count += 1

        self.tiles_to_remove_cache = count
        self.crop_preview_changed = False
        return count

    def apply_crop(self):
        """应用裁剪操作 - 修复版本"""
        if sum(self.crop_preview.values()) == 0:
            self.add_log("没有选择裁剪区域", "warning")
            return

        # 计算新的地图尺寸
        new_width = self.map_width - self.crop_preview['left'] - self.crop_preview['right']
        new_height = self.map_height - self.crop_preview['top'] - self.crop_preview['bottom']

        if new_width < 10 or new_height < 10:
            self.add_log("裁剪后地图太小（最小10x10）", "error")
            return

        # 计算新边界
        new_min_q, new_max_q, new_min_r, new_max_r = self.calculate_crop_bounds()

        # 保存要保留的地块
        tiles_to_keep = {}
        removed_count = 0

        for (q, r), tile in self.hex_map.items():
            q_offset = q // 2
            if (new_min_q <= q <= new_max_q and
                new_min_r - q_offset <= r <= new_max_r - q_offset):
                tiles_to_keep[(q, r)] = tile
            elif tile.terrain_type != TerrainType.WALL:
                removed_count += 1

        # 更新地图尺寸和实际边界
        self.map_width = new_width
        self.map_height = new_height
        self.actual_min_q = new_min_q
        self.actual_max_q = new_max_q
        self.actual_min_r = new_min_r
        self.actual_max_r = new_max_r

        # 重新初始化地图
        self.hex_map.clear()
        self.init_map()  # 现在会使用更新后的 actual_min/max 值

        # 恢复保留的地块
        for (q, r), tile in tiles_to_keep.items():
            if (q, r) in self.hex_map:
                self.hex_map[(q, r)] = tile

        # 更新起始位置坐标（如果存在）
        if self.start_position_placed:
            old_q, old_r = self.start_position_placed
            q_offset = old_q // 2
            if (new_min_q <= old_q <= new_max_q and
                new_min_r - q_offset <= old_r <= new_max_r - q_offset):
                self.start_position_placed = (old_q, old_r)
            else:
                self.start_position_placed = None
                self.add_log("起始位置被裁剪掉了", "warning")

        # 更新窗口标题
        pygame.display.set_caption(f"六边形地图编辑器 - {self.map_width}x{self.map_height} 地图")

        # 记录日志
        self.add_log(f"地图裁剪完成: {self.map_width}x{self.map_height}", "success")
        if removed_count > 0:
            self.add_log(f"删除了 {removed_count} 个地块", "warning")

        self.message = f"裁剪成功! 新尺寸: {self.map_width}x{self.map_height}"
        self.message_timer = self.message_duration

        # 退出裁剪模式
        self.crop_mode = False
        self.crop_preview = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}

    def draw_crop_preview(self):
        """绘制裁剪预览 - 优化版本"""
        if not self.crop_mode:
            return

        # 计算裁剪边界
        new_min_q, new_max_q, new_min_r, new_max_r = self.calculate_crop_bounds()

        # 创建一个单独的半透明表面用于所有要删除的六边形
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)

        # 批量绘制所有要删除的六边形到同一个surface上
        for (q, r) in self.visible_tiles:
            q_offset = q // 2
            if (q < new_min_q or q > new_max_q or
                    r < new_min_r - q_offset or r > new_max_r - q_offset):
                center = self.hex_to_pixel(q, r)
                # 绘制红色半透明遮罩
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    x = center[0] + self.hex_size * self.zoom * math.cos(angle)
                    y = center[1] + self.hex_size * self.zoom * math.sin(angle)
                    points.append((x, y))

                # 绘制到同一个overlay surface上
                pygame.draw.polygon(overlay, (255, 0, 0, 100), points)

        # 一次性绘制整个overlay
        self.screen.blit(overlay, (0, 0))

    def draw_crop_ui(self):
        """绘制裁剪模式UI"""
        if not self.crop_mode:
            return

        # 绘制裁剪信息面板
        panel_width = 400
        panel_height = 200
        panel_x = (self.width - panel_width) // 2
        panel_y = 50

        # 背景
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (30, 30, 40), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 120), panel_rect, 2)

        # 标题
        title_text = "地图裁剪模式"
        title_surface = self.font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(centerx=panel_x + panel_width // 2, y=panel_y + 10)
        self.screen.blit(title_surface, title_rect)

        # 当前裁剪设置
        info_y = panel_y + 40
        info_lines = [
            f"左侧裁剪: {self.crop_preview['left']} 列",
            f"右侧裁剪: {self.crop_preview['right']} 列",
            f"顶部裁剪: {self.crop_preview['top']} 行",
            f"底部裁剪: {self.crop_preview['bottom']} 行",
            "",
            f"新尺寸: {self.map_width - self.crop_preview['left'] - self.crop_preview['right']}x"
            f"{self.map_height - self.crop_preview['top'] - self.crop_preview['bottom']}",
            f"将删除 {self.count_tiles_to_remove()} 个非空地块"
        ]

        for line in info_lines:
            if line:
                text_surface = self.small_font.render(line, True, (200, 200, 200))
                self.screen.blit(text_surface, (panel_x + 20, info_y))
            info_y += 20

        # 操作提示
        hint_text = "方向键:调整 | Shift+方向键:快速调整 | Enter:确认 | ESC:取消"
        hint_surface = self.small_font.render(hint_text, True, (150, 200, 150))
        hint_rect = hint_surface.get_rect(centerx=panel_x + panel_width // 2, y=panel_y + panel_height - 25)
        self.screen.blit(hint_surface, hint_rect)

    def handle_crop_input(self, event):
        """处理裁剪模式下的输入"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # 取消裁剪
                self.toggle_crop_mode()
                return True

            elif event.key == pygame.K_RETURN:
                # 确认裁剪
                tiles_to_remove = self.count_tiles_to_remove()
                if tiles_to_remove > 0:
                    self.message = f"确认裁剪? 将删除 {tiles_to_remove} 个地块 (再次按Enter确认)"
                    self.message_timer = 5000
                    self.apply_crop()
                else:
                    self.apply_crop()
                return True

            # 方向键调整裁剪范围
            shift_pressed = pygame.key.get_mods() & pygame.KMOD_SHIFT
            adjust = 5 if shift_pressed else 1

            old_preview = self.crop_preview.copy()

            if event.key == pygame.K_LEFT:
                self.crop_preview['left'] = max(0, self.crop_preview['left'] + adjust)
            elif event.key == pygame.K_RIGHT:
                self.crop_preview['right'] = max(0, self.crop_preview['right'] + adjust)
            elif event.key == pygame.K_UP:
                self.crop_preview['top'] = max(0, self.crop_preview['top'] + adjust)
            elif event.key == pygame.K_DOWN:
                self.crop_preview['bottom'] = max(0, self.crop_preview['bottom'] + adjust)

            # 如果裁剪预览改变了，设置标记
            if old_preview != self.crop_preview:
                self.crop_preview_changed = True

            # 确保裁剪后地图不会太小
            new_width = self.map_width - self.crop_preview['left'] - self.crop_preview['right']
            new_height = self.map_height - self.crop_preview['top'] - self.crop_preview['bottom']

            if new_width < 10:
                self.crop_preview['left'] = max(0, self.crop_preview['left'] - 1)
                self.crop_preview['right'] = max(0, self.crop_preview['right'] - 1)
                self.crop_preview_changed = True
            if new_height < 10:
                self.crop_preview['top'] = max(0, self.crop_preview['top'] - 1)
                self.crop_preview['bottom'] = max(0, self.crop_preview['bottom'] - 1)
                self.crop_preview_changed = True

            # 只在改变时显示日志
            if self.crop_preview_changed:
                if event.key == pygame.K_LEFT:
                    self.add_log(f"左侧裁剪: {self.crop_preview['left']} 列", "info")
                elif event.key == pygame.K_RIGHT:
                    self.add_log(f"右侧裁剪: {self.crop_preview['right']} 列", "info")
                elif event.key == pygame.K_UP:
                    self.add_log(f"顶部裁剪: {self.crop_preview['top']} 行", "info")
                elif event.key == pygame.K_DOWN:
                    self.add_log(f"底部裁剪: {self.crop_preview['bottom']} 行", "info")

            return True

        return False

    def draw_log_panel(self):
        """绘制日志面板"""
        # 绘制日志背景
        log_rect = pygame.Rect(self.log_x, self.log_y, self.log_width, self.log_height)

        # 半透明背景
        log_surface = pygame.Surface((self.log_width, self.log_height))
        log_surface.set_alpha(230)
        log_surface.fill((25, 25, 35))
        self.screen.blit(log_surface, (self.log_x, self.log_y))

        # 绘制边框
        pygame.draw.rect(self.screen, (80, 80, 90), log_rect, 2)

        # 绘制标题（中文）
        title_surface = self.log_font.render("活动日志", True, (255, 255, 255))
        title_rect = title_surface.get_rect(centerx=self.log_x + self.log_width // 2, y=self.log_y + 5)
        self.screen.blit(title_surface, title_rect)

        # 绘制分割线
        pygame.draw.line(
            self.screen,
            (80, 80, 90),
            (self.log_x + 10, self.log_y + 25),
            (self.log_x + self.log_width - 10, self.log_y + 25),
            1
        )

        # 绘制日志消息
        y_offset = 35
        for i, log_msg in enumerate(self.log_messages):
            # 获取消息颜色
            color = log_msg.get_color()

            # 构建完整的消息文本
            if log_msg.timestamp:
                full_text = f"[{log_msg.timestamp}] {log_msg.text}"
            else:
                full_text = log_msg.text

            # 处理长文本换行
            max_width = self.log_width - 20

            # 绘制文本（简化处理，直接显示）
            if y_offset < self.log_height - 10:
                # 如果文本太长，截断
                text_surface = self.log_font.render(full_text[:40], True, color)
                self.screen.blit(text_surface, (self.log_x + 10, self.log_y + y_offset))
                y_offset += 18

    def init_map(self):
        """为平顶六边形初始化空地图，使用矩形布局"""
        # 创建矩形地图（使用实际边界）
        for q in range(self.actual_min_q, self.actual_max_q + 1):
            q_offset = q // 2
            for r in range(self.actual_min_r - q_offset, self.actual_max_r - q_offset + 1):
                self.hex_map[(q, r)] = HexTile(q, r, TerrainType.WALL)

    def hex_to_pixel(self, q: int, r: int) -> Tuple[float, float]:
        """将六边形坐标转换为平顶六边形的像素坐标"""
        x = self.hex_size * 3 / 2 * q * self.zoom
        y = self.hex_size * math.sqrt(3) * (r + q / 2) * self.zoom
        return x + self.camera_x, y + self.camera_y

    def pixel_to_hex(self, x: float, y: float) -> Tuple[int, int]:
        """将像素坐标转换为平顶六边形的六边形坐标"""
        # 调整相机偏移和缩放
        x = (x - self.camera_x) / (self.hex_size * self.zoom)
        y = (y - self.camera_y) / (self.hex_size * self.zoom)

        # 转换为平顶的轴向坐标
        q = 2 / 3 * x
        r = -1 / 3 * x + math.sqrt(3) / 3 * y

        return self.hex_round(q, r)

    def hex_round(self, q: float, r: float) -> Tuple[int, int]:
        """将浮点六边形坐标舍入为整数"""
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

    def draw_hexagon(
            self,
            center: Tuple[float, float],
            color: Tuple[int, int, int],
            border_color: Optional[Tuple[int, int, int]] = None
    ):
        """绘制支持缩放的平顶六边形"""
        points = []
        for i in range(6):
            # 平顶六边形的角度（不需要偏移）
            angle = math.pi / 3 * i
            x = center[0] + self.hex_size * self.zoom * math.cos(angle)
            y = center[1] + self.hex_size * self.zoom * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(self.screen, color, points)
        if border_color:
            # 缩小时边框变细
            border_width = max(1, int(2 * self.zoom))
            pygame.draw.polygon(self.screen, border_color, points, border_width)

    def get_visible_tiles(self):
        """计算屏幕上可见的瓦片以提高性能"""
        visible = set()

        # 计算六边形坐标中的屏幕边界
        margin = 2  # 额外的瓦片以确保平滑滚动

        # 获取角落位置
        corners = [
            (0, 0),
            (self.width, 0),
            (0, self.height - self.ui_height),
            (self.width, self.height - self.ui_height)
        ]

        # 找到最小/最大的q和r值
        min_q, max_q = float('inf'), float('-inf')
        min_r, max_r = float('inf'), float('-inf')

        for corner in corners:
            q, r = self.pixel_to_hex(*corner)
            min_q = min(min_q, q - margin)
            max_q = max(max_q, q + margin)
            min_r = min(min_r, r - margin)
            max_r = max(max_r, r + margin)

        # 添加可见范围内的所有瓦片
        for q in range(int(min_q), int(max_q) + 1):
            for r in range(int(min_r), int(max_r) + 1):
                if (q, r) in self.hex_map:
                    visible.add((q, r))

        return visible

    def draw_map(self):
        """绘制平顶六边形地图 - 为大地图优化"""
        # 更新可见瓦片
        self.visible_tiles = self.get_visible_tiles()

        # 只绘制可见瓦片
        for (q, r) in self.visible_tiles:
            tile = self.hex_map[(q, r)]
            center = self.hex_to_pixel(q, r)
            color = self.style.TERRAIN_COLORS[tile.terrain_type]

            # 如果地块已征服，颜色变暗
            if tile.conquered:
                color = tuple(int(c * 0.7) for c in color)

            # 使用微妙的边框以获得更好的视觉连接
            border_color = (45, 45, 55)  # 深色微妙边框
            self.draw_hexagon(center, color, border_color)

            # 为特殊地块添加标记
            if tile.terrain_type in [TerrainType.BOSS_GAARA, TerrainType.BOSS_ZETSU,
                                     TerrainType.BOSS_DART, TerrainType.BOSS_SHIRA,
                                     TerrainType.BOSS_KUSHINA, TerrainType.BOSS_KISAME,
                                     TerrainType.BOSS_HANA]:
                # 为BOSS地块添加特殊标记
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (int(center[0]), int(center[1])),
                                   int(5 * self.zoom), 2)

            # 为帐篷地块添加特殊标记
            elif tile.terrain_type == TerrainType.TENT:
                # 绘制一个小三角形表示帐篷
                tent_size = int(5 * self.zoom)
                tent_points = [
                    (int(center[0]), int(center[1] - tent_size)),  # 顶点
                    (int(center[0] - tent_size), int(center[1] + tent_size)),  # 左下
                    (int(center[0] + tent_size), int(center[1] + tent_size))   # 右下
                ]
                pygame.draw.polygon(self.screen, (255, 255, 255), tent_points, 2)

                # 在 hex_map_editor_chinese.py 的 draw_map 方法中
                # 找到这段代码（大约在第600-650行之间）：

                # 在 hex_map_editor_chinese.py 的 draw_map 方法中
                # 找到这段代码（大约在第600-650行之间）：

                # 在六边形中显示地块名称（非空地块）
            if tile.terrain_type != TerrainType.WALL and tile.terrain_type != TerrainType.BOUNDARY:
                # 使用中文名称
                name = self.style.get_terrain_name_chinese(tile.terrain_type)

                # 根据缩放级别调整字体大小和显示策略
                if self.zoom >= 0.6:  # 小于0.6不显示文字
                    # 根据缩放级别决定显示字数和字体大小
                    if self.zoom < 0.7:
                        font_size = 8
                        display_name = name[:1] if len(name) > 0 else ""
                    elif self.zoom < 0.9:
                        font_size = 10
                        display_name = name[:2] if len(name) > 1 else name
                    elif self.zoom < 1.2:
                        font_size = 12
                        display_name = name[:3] if len(name) > 2 else name
                    elif self.zoom < 1.5:
                        font_size = 14
                        display_name = name[:4] if len(name) > 3 else name
                    else:  # zoom >= 1.5 就显示完整名称
                        font_size = 16
                        display_name = name  # 显示完整名称

                    # 确保字体大小在合理范围内
                    font_size = max(8, min(font_size, 24))

                    try:
                        text_font = get_font(font_size)
                        # 渲染文字
                        text_surface = text_font.render(display_name, True, (255, 255, 255))

                        # 计算六边形的实际像素大小
                        hex_pixel_size = self.hex_size * self.zoom

                        # 检查文字宽度
                        text_width = text_surface.get_width()
                        max_text_width = hex_pixel_size * 1.6

                        # 如果文字太宽，调整策略
                        if text_width > max_text_width:
                            if self.zoom < 1.5:
                                # 小缩放时，尝试缩短文字
                                while len(display_name) > 1 and text_width > max_text_width:
                                    display_name = display_name[:-1]
                                    text_surface = text_font.render(display_name, True, (255, 255, 255))
                                    text_width = text_surface.get_width()
                            else:
                                # 大缩放时（>=1.5），缩小字体
                                scale_factor = max_text_width / text_width
                                new_font_size = max(12, int(font_size * scale_factor))
                                text_font = get_font(new_font_size)
                                text_surface = text_font.render(display_name, True, (255, 255, 255))

                        # 计算文字位置（居中）
                        text_rect = text_surface.get_rect(center=(int(center[0]), int(center[1])))

                        # 根据字体大小决定是否添加背景
                        if font_size >= 12 and self.zoom >= 1.0:
                            # 为较大字体添加半透明背景
                            padding = 1
                            bg_rect = text_rect.inflate(padding * 2, padding)
                            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
                            bg_surface.set_alpha(120)
                            bg_surface.fill((20, 20, 30))
                            self.screen.blit(bg_surface, bg_rect)

                        # 绘制文字
                        self.screen.blit(text_surface, text_rect)

                    except Exception as e:
                        # 如果出现任何错误，跳过文字绘制
                        print(f"文字绘制错误: {e}")
                        pass

    def draw_ui(self):
        """绘制用户界面 - 分类选择系统（中文）"""
        # 绘制UI背景
        ui_rect = pygame.Rect(0, self.height - self.ui_height, self.width, self.ui_height)
        pygame.draw.rect(self.screen, self.style.UI_BG_COLOR, ui_rect)
        pygame.draw.rect(self.screen, self.style.BORDER_COLOR, ui_rect, 1)

        # 绘制分类标签
        tab_width = 100
        tab_height = 30
        tab_y = self.height - self.ui_height + 5

        for i, category in enumerate(TerrainCategory):
            tab_x = 10 + i * (tab_width + 5)
            tab_rect = pygame.Rect(tab_x, tab_y, tab_width, tab_height)

            # 高亮当前选中的分类
            if category == self.current_category:
                pygame.draw.rect(self.screen, self.style.SELECTED_COLOR, tab_rect)
            else:
                pygame.draw.rect(self.screen, (70, 70, 80), tab_rect)

            pygame.draw.rect(self.screen, self.style.BORDER_COLOR, tab_rect, 1)

            # 绘制分类名称（中文）
            text = self.small_font.render(category.value, True, self.style.TEXT_COLOR)
            text_rect = text.get_rect(center=tab_rect.center)
            self.screen.blit(text, text_rect)

        # 绘制当前分类的地块按钮
        terrains = self.terrain_categories[self.current_category]
        button_width = 120
        button_height = 50
        buttons_per_row = 7
        button_start_y = self.height - self.ui_height + 45

        for i, terrain in enumerate(terrains):
            row = i // buttons_per_row
            col = i % buttons_per_row
            button_x = 10 + col * (button_width + 5)
            button_y = button_start_y + row * (button_height + 5)

            # 检查按钮是否在UI区域内
            if button_y + button_height > self.height:
                continue

            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            # 高亮当前选中的地形
            if terrain == self.current_terrain:
                pygame.draw.rect(self.screen, self.style.SELECTED_COLOR, button_rect)

            # 绘制地形颜色背景
            color = self.style.TERRAIN_COLORS[terrain]
            inner_rect = button_rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)
            pygame.draw.rect(self.screen, self.style.BORDER_COLOR, button_rect, 2)

            # 绘制地形名称（中文）
            name = self.style.get_terrain_name_chinese(terrain)

            # 缩短过长的名称
            if len(name) > 6:
                name = name[:6]

            text = self.small_font.render(name, True, self.style.TEXT_COLOR)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)

        # 绘制当前地块信息（中文）
        if self.current_terrain:
            props = self.style.TERRAIN_PROPERTIES[self.current_terrain]
            info_x = self.width - 300
            info_y = self.height - self.ui_height + 10

            # 绘制信息背景
            info_rect = pygame.Rect(info_x - 5, info_y - 5, 290, 100)
            pygame.draw.rect(self.screen, (40, 40, 50), info_rect)
            pygame.draw.rect(self.screen, self.style.BORDER_COLOR, info_rect, 1)

            # 特殊处理帐篷的显示
            if self.current_terrain == TerrainType.TENT:
                info_lines = [
                    f"当前: 帐篷",
                    f"粮草消耗: 0 | 行动增加: +1",
                    f"按游戏天数获得粮草:",
                    f"  1-10天: 300 | 11-20天: 250",
                    f"  21-35天: 200 | 36-50天: 150",
                    f"  51-90天: 100"
                ]
            else:
                # 显示地块属性
                name = self.style.get_terrain_name_chinese(self.current_terrain)
                info_lines = [
                    f"当前: {name}",
                    f"粮草消耗: {props['food_cost'] if props['food_cost'] >= 0 else '不可征服'}",
                    f"积分消耗: {props.get('score_cost', 0)}",
                    f"经验获得: {props['exp_gain']}",
                    f"可通过: {'是' if props['passable'] else '否'}",
                    f"特殊道具: {'是' if props.get('has_item', False) else '否'}"
                ]

            for i, line in enumerate(info_lines):
                text = self.small_font.render(line, True, self.style.TEXT_COLOR)
                self.screen.blit(text, (info_x, info_y + i * 16))

        # 操作说明（中文）
        instructions = [
            "Tab:切换分类 | 双击:放置地块 | Ctrl+S:保存 | Ctrl+L:加载 | Ctrl+C:裁剪 | R:重置视图"
        ]
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.style.TEXT_COLOR)
            self.screen.blit(text, (10, self.height - 20))

        # 绘制缩放和地图信息（中文）
        center_q, center_r = self.pixel_to_hex(self.width // 2, self.height // 2)
        info_text = f"缩放: {self.zoom:.1f}x | 地图: {self.map_width}x{self.map_height} | 中心: ({center_q}, {center_r})"
        info_surface = self.small_font.render(info_text, True, self.style.TEXT_COLOR)
        info_bg = info_surface.get_rect(topright=(self.width - 10, 10))
        pygame.draw.rect(self.screen, (30, 30, 40), info_bg.inflate(10, 5))
        self.screen.blit(info_surface, info_bg)

    def draw_minimap(self):
        """为大地图导航绘制小地图 - 修复版，基于实际边界"""
        if self.map_width <= 30 and self.map_height <= 30:
            return  # 小地图不显示小地图

        # 小地图设置
        minimap_size = 150
        minimap_x = self.width - minimap_size - 10
        minimap_y = 40

        # 绘制小地图背景
        minimap_rect = pygame.Rect(minimap_x, minimap_y, minimap_size, minimap_size)
        pygame.draw.rect(self.screen, (20, 20, 30), minimap_rect)
        pygame.draw.rect(self.screen, (80, 80, 90), minimap_rect, 2)

        # 使用实际边界计算地图的真实范围
        actual_width = self.actual_max_q - self.actual_min_q
        actual_height = self.actual_max_r - self.actual_min_r

        # 计算比例（基于实际边界）
        scale_x = minimap_size / (actual_width * 1.5)
        scale_y = minimap_size / (actual_height * math.sqrt(3))
        scale = min(scale_x, scale_y) * 0.8

        # 计算中心偏移（将实际地图中心映射到小地图中心）
        center_q = (self.actual_min_q + self.actual_max_q) / 2
        center_r = (self.actual_min_r + self.actual_max_r) / 2

        # 在小地图上绘制瓦片（为性能采样）
        sample_rate = max(1, max(actual_width, actual_height) // 50)  # 基于实际尺寸采样

        for (q, r), tile in self.hex_map.items():
            if (q % sample_rate == 0 or r % sample_rate == 0) and tile.terrain_type != TerrainType.WALL:
                # 相对于实际地图中心的偏移
                rel_q = q - center_q
                rel_r = r - center_r

                # 将六边形转换为小地图像素
                x = minimap_x + minimap_size / 2 + rel_q * scale * 1.5
                y = minimap_y + minimap_size / 2 + (rel_r + rel_q / 2) * scale * math.sqrt(3)

                if minimap_x <= x <= minimap_x + minimap_size and minimap_y <= y <= minimap_y + minimap_size:
                    color = self.style.TERRAIN_COLORS[tile.terrain_type]
                    # 根据缩放调整点的大小
                    dot_size = max(1, int(scale * 0.8))
                    pygame.draw.circle(self.screen, color, (int(x), int(y)), dot_size)

        # 绘制视口指示器
        view_center_q, view_center_r = self.pixel_to_hex(self.width // 2, (self.height - self.ui_height) // 2)

        # 视口中心相对于地图中心的偏移
        rel_view_q = view_center_q - center_q
        rel_view_r = view_center_r - center_r

        viewport_x = minimap_x + minimap_size / 2 + rel_view_q * scale * 1.5
        viewport_y = minimap_y + minimap_size / 2 + (rel_view_r + rel_view_q / 2) * scale * math.sqrt(3)

        # 绘制视口矩形
        view_width = (self.width / (self.hex_size * self.zoom * 1.5)) * scale * 1.5
        view_height = ((self.height - self.ui_height) / (self.hex_size * self.zoom * math.sqrt(3))) * scale * math.sqrt(
            3)

        viewport_rect = pygame.Rect(
            viewport_x - view_width / 2,
            viewport_y - view_height / 2,
            view_width,
            view_height
        )

        # 裁剪到小地图边界
        viewport_rect = viewport_rect.clip(minimap_rect)
        if viewport_rect.width > 0 and viewport_rect.height > 0:
            pygame.draw.rect(self.screen, (255, 215, 0), viewport_rect, 2)

        # 在小地图上显示实际尺寸信息（可选）
        size_text = f"{actual_width}x{actual_height}"
        size_surface = self.small_font.render(size_text, True, (150, 150, 150))
        size_rect = size_surface.get_rect(center=(minimap_x + minimap_size // 2, minimap_y + minimap_size + 10))
        self.screen.blit(size_surface, size_rect)

    def handle_minimap_click(self, pos: Tuple[int, int]) -> bool:
        """处理小地图点击以跳转到位置 - 修复版，基于实际边界"""
        if self.map_width <= 30 and self.map_height <= 30:
            return False

        minimap_size = 150
        minimap_x = self.width - minimap_size - 10
        minimap_y = 40
        minimap_rect = pygame.Rect(minimap_x, minimap_y, minimap_size, minimap_size)

        if minimap_rect.collidepoint(pos):
            # 使用实际边界计算
            actual_width = self.actual_max_q - self.actual_min_q
            actual_height = self.actual_max_r - self.actual_min_r

            # 计算比例（基于实际边界）
            scale_x = minimap_size / (actual_width * 1.5)
            scale_y = minimap_size / (actual_height * math.sqrt(3))
            scale = min(scale_x, scale_y) * 0.8

            # 计算地图中心
            center_q = (self.actual_min_q + self.actual_max_q) / 2
            center_r = (self.actual_min_r + self.actual_max_r) / 2

            # 将点击位置转换为相对于小地图中心的偏移
            rel_x = (pos[0] - minimap_x - minimap_size / 2) / (scale * 1.5)
            rel_y = (pos[1] - minimap_y - minimap_size / 2) / (scale * math.sqrt(3))

            # 转换为世界坐标
            target_q = center_q + rel_x
            target_r = center_r + rel_y - rel_x / 2

            # 将相机中心定位到此位置
            target_x, target_y = self.hex_to_pixel(int(target_q), int(target_r))
            self.camera_x = self.width // 2 - (target_x - self.camera_x)
            self.camera_y = self.height // 2 - (target_y - self.camera_y)
            return True

        return False

    def draw_message(self):
        """绘制消息提示（保留用于重要的中央提示）"""
        if self.message and self.message_timer > 0:
            # 创建消息背景
            font = get_large_font()
            text = font.render(self.message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, 60))

            # 绘制半透明背景
            bg_rect = text_rect.inflate(40, 20)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.set_alpha(200)
            bg_surface.fill((50, 50, 60))
            self.screen.blit(bg_surface, bg_rect)

            # 绘制边框
            pygame.draw.rect(self.screen, (100, 150, 200), bg_rect, 2)

            # 绘制文本
            self.screen.blit(text, text_rect)

    def handle_mouse_click(self, pos: Tuple[int, int], double_click: bool = False):
        """处理鼠标点击"""
        x, y = pos

        # 检查是否点击了日志区域（避免误操作）
        log_rect = pygame.Rect(self.log_x, self.log_y, self.log_width, self.log_height)
        if log_rect.collidepoint(pos):
            return

        # 检查是否点击了UI区域
        if y > self.height - self.ui_height:
            # 检查分类标签点击
            tab_width = 100
            tab_height = 30
            tab_y = self.height - self.ui_height + 5

            for i, category in enumerate(TerrainCategory):
                tab_x = 10 + i * (tab_width + 5)
                tab_rect = pygame.Rect(tab_x, tab_y, tab_width, tab_height)

                if tab_rect.collidepoint(pos):
                    self.current_category = category
                    self.add_log(f"切换到: {category.value}", "info")
                    return

            # 检查地块按钮点击
            terrains = self.terrain_categories[self.current_category]
            button_width = 120
            button_height = 50
            buttons_per_row = 7
            button_start_y = self.height - self.ui_height + 45

            for i, terrain in enumerate(terrains):
                row = i // buttons_per_row
                col = i % buttons_per_row
                button_x = 10 + col * (button_width + 5)
                button_y = button_start_y + row * (button_height + 5)

                if button_y + button_height > self.height:
                    continue

                button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

                if button_rect.collidepoint(pos):
                    self.current_terrain = terrain
                    name = self.style.get_terrain_name_chinese(terrain)
                    self.add_log(f"选中: {name}", "info")
                    return
        else:
            if double_click:
                q, r = self.pixel_to_hex(x, y)
                if (q, r) in self.hex_map:
                    # 特殊处理初始位置
                    if self.current_terrain == TerrainType.START_POSITION:
                        # 如果已有初始位置，先移除
                        if self.start_position_placed:
                            old_q, old_r = self.start_position_placed
                            if (old_q, old_r) in self.hex_map:
                                self.hex_map[(old_q, old_r)].terrain_type = TerrainType.WALL
                                self.add_log(f"移除旧初始位置 ({old_q}, {old_r})", "info")

                        # 设置新初始位置
                        self.hex_map[(q, r)].terrain_type = self.current_terrain
                        self.start_position_placed = (q, r)
                        self.add_log(f"设置初始位置于 ({q}, {r})", "success")
                    else:
                        # 普通地块放置
                        old_terrain = self.hex_map[(q, r)].terrain_type
                        # 如果覆盖的是初始位置，清除记录
                        if old_terrain == TerrainType.START_POSITION:
                            self.start_position_placed = None
                            self.add_log("初始位置被覆盖", "warning")

                        if old_terrain != self.current_terrain:
                            self.hex_map[(q, r)].terrain_type = self.current_terrain
                            name = self.style.get_terrain_name_chinese(self.current_terrain)
                            self.add_log(f"在 ({q}, {r}) 放置了 {name}", "success")

    def resize_map(self, new_width: int, new_height: int):
        """调整地图大小，保留中心区域的地块"""
        old_width = self.map_width
        old_height = self.map_height

        # 更新实际边界
        self.actual_min_q = -new_width // 2
        self.actual_max_q = new_width // 2
        self.actual_min_r = -new_height // 2
        self.actual_max_r = new_height // 2

        # 计算保留区域
        min_q = self.actual_min_q
        max_q = self.actual_max_q
        min_r = self.actual_min_r
        max_r = self.actual_max_r

        # 保存现有的非空地块
        tiles_to_keep = {}
        removed_count = 0

        for (q, r), tile in self.hex_map.items():
            if tile.terrain_type != TerrainType.WALL:
                q_offset = q // 2
                if min_q <= q <= max_q and min_r - q_offset <= r <= max_r - q_offset:
                    tiles_to_keep[(q, r)] = tile
                else:
                    removed_count += 1

        # 更新地图尺寸
        self.map_width = new_width
        self.map_height = new_height

        # 重新初始化地图
        self.hex_map.clear()
        self.init_map()

        # 恢复保留的地块
        for (q, r), tile in tiles_to_keep.items():
            if (q, r) in self.hex_map:
                self.hex_map[(q, r)] = tile

        # 更新窗口标题
        pygame.display.set_caption(f"六边形地图编辑器 - {self.map_width}x{self.map_height} 地图")

        # 记录日志
        self.add_log(f"地图大小调整为 {new_width}x{new_height}", "success")
        if removed_count > 0:
            self.add_log(f"移除了 {removed_count} 个超出边界的地块", "warning")

        self.message = f"地图大小调整为 {new_width}x{new_height}!"
        self.message_timer = self.message_duration

        # 调整缩放
        if new_width > 50 or new_height > 50:
            self.zoom = 0.5
        elif new_width > 80 or new_height > 80:
            self.zoom = 0.3
        else:
            self.zoom = 1.0

    def save_map(self, filename: str = "map_save.json"):
        """保存地图"""
        try:
            save_data = {
                'map_width': self.map_width,
                'map_height': self.map_height,
                'actual_min_q': self.actual_min_q,  # 新增
                'actual_max_q': self.actual_max_q,  # 新增
                'actual_min_r': self.actual_min_r,  # 新增
                'actual_max_r': self.actual_max_r,  # 新增
                'start_position': self.start_position_placed,
                'tiles': [
                    tile.to_dict() for tile in self.hex_map.values()
                    if tile.terrain_type != TerrainType.WALL
                ]  # 只保存非空瓦片
            }

            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)

            self.add_log(f"地图已保存: {len(save_data['tiles'])} 个地块", "success")
            self.message = f"地图保存成功!"
            self.message_timer = self.message_duration

        except Exception as e:
            self.add_log(f"保存失败: {str(e)}", "error")
            self.message = "保存失败!"
            self.message_timer = self.message_duration

    def load_map(self, filename: str = "map_save.json"):
        """加载地图"""
        if not os.path.exists(filename):
            self.add_log(f"文件未找到: {filename}", "error")
            return

        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)

            self.map_width = save_data['map_width']
            self.map_height = save_data['map_height']

            # 加载实际边界（如果存在，为了兼容旧版本）
            if 'actual_min_q' in save_data:
                self.actual_min_q = save_data['actual_min_q']
                self.actual_max_q = save_data['actual_max_q']
                self.actual_min_r = save_data['actual_min_r']
                self.actual_max_r = save_data['actual_max_r']
            else:
                # 兼容旧版本保存文件
                self.actual_min_q = -self.map_width // 2
                self.actual_max_q = self.map_width // 2
                self.actual_min_r = -self.map_height // 2
                self.actual_max_r = self.map_height // 2

            self.start_position_placed = save_data.get('start_position')  # 加载初始位置

            # 清除并重新初始化地图
            self.hex_map.clear()
            self.init_map()  # 用墙壁初始化

            # 加载保存的地块
            for tile_data in save_data['tiles']:
                tile = HexTile.from_dict(tile_data)
                self.hex_map[(tile.q, tile.r)] = tile

            self.add_log(f"地图已加载: {len(save_data['tiles'])} 个地块", "success")
            self.message = f"地图加载成功!"
            self.message_timer = self.message_duration

        except Exception as e:
            self.add_log(f"加载失败: {str(e)}", "error")
            self.message = "加载失败!"
            self.message_timer = self.message_duration

    def run(self, standalone=True):
        """支持鼠标拖拽和缩放的主循环

        Args:
            standalone: 是否独立运行。如果为False，则可以通过ESC返回主菜单
        """
        running = True
        double_click_timer = 0
        last_click_time = 0

        while running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS

            # 更新消息计时器
            if self.message_timer > 0:
                self.message_timer -= self.clock.get_time()

            for event in pygame.event.get():
                # 如果在裁剪模式，优先处理裁剪输入
                if self.crop_mode and event.type == pygame.KEYDOWN:
                    if self.handle_crop_input(event):
                        continue  # 如果裁剪模式处理了输入，跳过其他处理

                if event.type == pygame.QUIT:
                    if standalone:
                        running = False
                    else:
                        return 'quit'  # 返回退出信号

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        # 首先检查小地图点击
                        if self.handle_minimap_click(event.pos):
                            continue

                        current_time = pygame.time.get_ticks()
                        if current_time - last_click_time < 300:  # 300毫秒内双击
                            self.handle_mouse_click(event.pos, double_click=True)
                        else:
                            # 开始拖拽
                            self.is_dragging = True
                            self.drag_start_pos = event.pos
                            self.drag_start_camera = (self.camera_x, self.camera_y)
                            self.handle_mouse_click(event.pos, double_click=False)
                        last_click_time = current_time

                    elif event.button == 4:  # 鼠标滚轮向上 - 放大
                        old_zoom = self.zoom
                        self.zoom = min(self.max_zoom, self.zoom * 1.1)
                        # 向鼠标位置缩放
                        if self.zoom != old_zoom:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            zoom_factor = self.zoom / old_zoom
                            self.camera_x = mouse_x - (mouse_x - self.camera_x) * zoom_factor
                            self.camera_y = mouse_y - (mouse_y - self.camera_y) * zoom_factor

                    elif event.button == 5:  # 鼠标滚轮向下 - 缩小
                        old_zoom = self.zoom
                        self.zoom = max(self.min_zoom, self.zoom / 1.1)
                        # 从鼠标位置缩放
                        if self.zoom != old_zoom:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            zoom_factor = self.zoom / old_zoom
                            self.camera_x = mouse_x - (mouse_x - self.camera_x) * zoom_factor
                            self.camera_y = mouse_y - (mouse_y - self.camera_y) * zoom_factor

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # 左键释放
                        self.is_dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        # 基于拖拽更新相机位置
                        dx = event.pos[0] - self.drag_start_pos[0]
                        dy = event.pos[1] - self.drag_start_pos[1]
                        self.camera_x = self.drag_start_camera[0] + dx
                        self.camera_y = self.drag_start_camera[1] + dy

                elif event.type == pygame.KEYDOWN:
                    # ESC键返回主菜单（非独立运行时）
                    if event.key == pygame.K_ESCAPE and not standalone:
                        return 'menu'  # 返回主菜单信号
                    # 保存和加载
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.save_map()
                    elif event.key == pygame.K_l and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.load_map()
                    # 裁剪模式
                    elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.toggle_crop_mode()
                    # 重置视图
                    elif event.key == pygame.K_r:
                        self.camera_x = self.width // 2
                        self.camera_y = self.height // 2
                        self.zoom = 1.0
                        self.message = "视图已重置"
                        self.message_timer = self.message_duration
                    # 切换分类
                    elif event.key == pygame.K_TAB:
                        categories = list(TerrainCategory)
                        current_index = categories.index(self.current_category)
                        self.current_category = categories[(current_index + 1) % len(categories)]
                        self.add_log(f"切换到: {self.current_category.value}", "info")
                    # 快速缩放
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.zoom = min(self.max_zoom, self.zoom * 1.2)
                    elif event.key == pygame.K_MINUS:
                        self.zoom = max(self.min_zoom, self.zoom / 1.2)

            # 绘制
            self.screen.fill(self.style.BG_COLOR)
            self.draw_map()

            # 如果在裁剪模式，绘制裁剪预览
            if self.crop_mode:
                self.draw_crop_preview()

            # 高亮鼠标下的六边形（微妙效果）- 仅当不拖拽时
            mouse_pos = pygame.mouse.get_pos()
            if not self.is_dragging and mouse_pos[1] < self.height - self.ui_height:
                q, r = self.pixel_to_hex(*mouse_pos)
                if (q, r) in self.hex_map:
                    center = self.hex_to_pixel(q, r)
                    # 用较粗的边框绘制高亮
                    points = []
                    for i in range(6):
                        angle = math.pi / 3 * i
                        x = center[0] + self.hex_size * self.zoom * math.cos(angle)
                        y = center[1] + self.hex_size * self.zoom * math.sin(angle)
                        points.append((x, y))
                    pygame.draw.polygon(
                        self.screen,
                        (255, 215, 0),
                        points,
                        max(1, int(2 * self.zoom))
                    )  # 金色高亮

            self.draw_ui()
            self.draw_minimap()
            self.draw_log_panel()  # 绘制日志面板

            # 如果在裁剪模式，绘制裁剪UI
            if self.crop_mode:
                self.draw_crop_ui()

            self.draw_message()  # 绘制消息提示

            # 如果非独立运行，显示返回提示（中文）
            if not standalone:
                hint_text = "按 ESC 返回主菜单"
                hint_surface = self.small_font.render(hint_text, True, (150, 150, 150))
                hint_rect = hint_surface.get_rect(topleft=(10, self.height - self.ui_height - 30))
                # 背景
                pygame.draw.rect(self.screen, (30, 30, 40), hint_rect.inflate(10, 5))
                self.screen.blit(hint_surface, hint_rect)

            # 在当前鼠标位置显示六边形坐标和地块信息（中文）
            if mouse_pos[1] < self.height - self.ui_height:
                q, r = self.pixel_to_hex(*mouse_pos)
                if (q, r) in self.hex_map:
                    tile = self.hex_map[(q, r)]
                    name = self.style.get_terrain_name_chinese(tile.terrain_type)
                    info_font = get_font(20)
                    coord_text = info_font.render(
                        f"({q}, {r}) - {name}",
                        True,
                        self.style.TEXT_COLOR
                    )
                    # 绘制背景以提高可读性
                    text_rect = coord_text.get_rect(topleft=(10, 10))
                    pygame.draw.rect(self.screen, (30, 30, 40), text_rect.inflate(10, 5))
                    self.screen.blit(coord_text, (10, 10))

                    # 显示可见瓦片数以进行性能监控（中文）
                    if len(self.visible_tiles) > 0:
                        perf_text = info_font.render(
                            f"可见: {len(self.visible_tiles)}/{len(self.hex_map)}",
                            True,
                            self.style.TEXT_COLOR
                        )
                        perf_rect = perf_text.get_rect(topleft=(10, 35))
                        pygame.draw.rect(self.screen, (30, 30, 40), perf_rect.inflate(10, 5))
                        self.screen.blit(perf_text, (10, 35))

            pygame.display.flip()

        if standalone:
            pygame.quit()
            sys.exit()
        else:
            return 'normal'  # 正常返回


if __name__ == "__main__":
    editor = HexMapEditor()
    editor.run()