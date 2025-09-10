"""
六边形地图策略游戏 - 主程序（简化版）
只包含主界面、地图编辑器、游戏模式
"""
import pygame
import sys
from enum import Enum

# 导入字体管理器
from font_manager import get_font_manager, get_font, get_small_font, get_medium_font, get_large_font, get_title_font, get_huge_font

# 导入地图编辑器
from hex_map_editor_chinese import HexMapEditor
from map_style_config import StyleConfig, TerrainType
# 导入游戏系统
from game_play_system import GamePlaySystem


class GameState(Enum):
    """游戏状态枚举"""
    MAIN_MENU = "main_menu"
    MAP_EDITOR = "map_editor"
    GAME_PLAY = "game_play"


class MainMenu:
    """主菜单类"""

    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.style = StyleConfig()

        # 初始化字体管理器
        self.font_manager = get_font_manager()

        # 字体设置 - 使用中文字体
        self.title_font = get_huge_font()
        self.button_font = get_title_font()
        self.small_font = get_small_font()

        # 按钮设置
        self.button_width = 300
        self.button_height = 70
        self.button_spacing = 25

        # 创建按钮（中文）
        self.buttons = {
            'start_game': {
                'text': '开始游戏',
                'rect': None,
                'hovered': False
            },
            'map_editor': {
                'text': '地图编辑器',
                'rect': None,
                'hovered': False
            },
            'exit': {
                'text': '退出游戏',
                'rect': None,
                'hovered': False
            }
        }

        # 计算按钮位置
        self._calculate_button_positions()

        # 背景效果
        self.hex_decorations = []
        self._generate_background_hexagons()

    def _calculate_button_positions(self):
        """计算按钮位置"""
        total_height = len(self.buttons) * self.button_height + (len(self.buttons) - 1) * self.button_spacing
        start_y = (self.height - total_height) // 2 + 30

        button_order = ['start_game', 'map_editor', 'exit']
        for i, button_key in enumerate(button_order):
            button_x = (self.width - self.button_width) // 2
            button_y = start_y + i * (self.button_height + self.button_spacing)
            self.buttons[button_key]['rect'] = pygame.Rect(button_x, button_y, self.button_width, self.button_height)

    def _generate_background_hexagons(self):
        """生成背景装饰六边形"""
        import random
        import math

        for _ in range(20):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.randint(20, 60)
            color = (
                random.randint(40, 60),
                random.randint(40, 60),
                random.randint(50, 70)
            )
            alpha = random.randint(30, 100)
            rotation_speed = random.uniform(-0.5, 0.5)

            self.hex_decorations.append({
                'x': x,
                'y': y,
                'size': size,
                'color': color,
                'alpha': alpha,
                'rotation': 0,
                'rotation_speed': rotation_speed
            })

    def draw_hexagon(self, center, size, color, alpha, rotation=0):
        """绘制六边形装饰"""
        import math

        # 创建临时surface用于透明度
        hex_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)

        points = []
        for i in range(6):
            angle = math.pi / 3 * i + rotation
            x = size + size * math.cos(angle)
            y = size + size * math.sin(angle)
            points.append((x, y))

        # 在临时surface上绘制
        color_with_alpha = (*color, alpha)
        pygame.draw.polygon(hex_surface, color_with_alpha, points)

        # 绘制到主屏幕
        self.screen.blit(hex_surface, (center[0] - size, center[1] - size))

    def update(self, dt):
        """更新背景动画"""
        import math

        for hex_dec in self.hex_decorations:
            hex_dec['rotation'] += hex_dec['rotation_speed'] * dt
            # 缓慢移动
            hex_dec['y'] += 10 * dt
            if hex_dec['y'] > self.height + hex_dec['size']:
                hex_dec['y'] = -hex_dec['size']

    def draw(self):
        """绘制主菜单"""
        # 绘制背景
        self.screen.fill((20, 20, 30))

        # 绘制装饰六边形
        for hex_dec in self.hex_decorations:
            self.draw_hexagon(
                (hex_dec['x'], hex_dec['y']),
                hex_dec['size'],
                hex_dec['color'],
                hex_dec['alpha'],
                hex_dec['rotation']
            )

        # 绘制标题（中文）
        title_text = "远征模拟器风华自制1.0"
        title_surface = self.title_font.render(title_text, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 120))

        # 标题阴影
        shadow_surface = self.title_font.render(title_text, True, (50, 50, 60))
        shadow_rect = shadow_surface.get_rect(center=(self.width // 2 + 3, 123))
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(title_surface, title_rect)

        # 绘制按钮
        for button_key, button in self.buttons.items():
            if button['rect']:
                # 按钮颜色处理
                if button['hovered']:
                    color = (80, 120, 160)
                    border_color = (255, 215, 0)
                    border_width = 3
                else:
                    color = (50, 70, 90)
                    border_color = (100, 120, 140)
                    border_width = 2

                pygame.draw.rect(self.screen, color, button['rect'])
                pygame.draw.rect(self.screen, border_color, button['rect'], border_width)

                # 按钮文字
                text_color = (255, 255, 255) if button['hovered'] else (200, 200, 200)
                text_surface = self.button_font.render(button['text'], True, text_color)
                text_rect = text_surface.get_rect(center=button['rect'].center)
                self.screen.blit(text_surface, text_rect)

        # 版本信息（中文）
        version_text = "V1.0 阿丑是风厘的狗"
        version_surface = self.small_font.render(version_text, True, (100, 100, 100))
        version_rect = version_surface.get_rect(bottomright=(self.width - 10, self.height - 10))
        self.screen.blit(version_surface, version_rect)

    def handle_mouse_motion(self, pos):
        """处理鼠标移动"""
        for button in self.buttons.values():
            if button['rect']:
                button['hovered'] = button['rect'].collidepoint(pos)

    def handle_click(self, pos):
        """处理点击事件，返回要切换的状态"""
        for button_key, button in self.buttons.items():
            if button['rect'] and button['rect'].collidepoint(pos):
                if button_key == 'map_editor':
                    return GameState.MAP_EDITOR
                elif button_key == 'start_game':
                    return GameState.GAME_PLAY
                elif button_key == 'exit':
                    return 'exit'
        return None


class MainGame:
    """主游戏类"""

    def __init__(self):
        pygame.init()

        # 窗口设置
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("火影远征模拟器")

        # 游戏状态
        self.current_state = GameState.MAIN_MENU
        self.running = True

        # 时钟
        self.clock = pygame.time.Clock()

        # 创建各个界面
        self.main_menu = MainMenu(self.screen, self.width, self.height)
        self.map_editor = None
        self.game_play = None

    def run(self):
        """主游戏循环"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS

            # 处理不同状态
            if self.current_state == GameState.MAIN_MENU:
                self._run_main_menu(dt)
            elif self.current_state == GameState.MAP_EDITOR:
                self._run_map_editor()
            elif self.current_state == GameState.GAME_PLAY:
                self._run_game_play()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _run_main_menu(self, dt):
        """运行主菜单"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.main_menu.handle_mouse_motion(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    result = self.main_menu.handle_click(event.pos)
                    if result == 'exit':
                        self.running = False
                    elif result == GameState.MAP_EDITOR:
                        self.map_editor = HexMapEditor(self.width, self.height)
                        self.current_state = GameState.MAP_EDITOR
                    elif result == GameState.GAME_PLAY:
                        self.game_play = GamePlaySystem(self.screen, self.width, self.height)
                        self.current_state = GameState.GAME_PLAY

        # 更新和绘制
        self.main_menu.update(dt)
        self.main_menu.draw()

    def _run_map_editor(self):
        """运行地图编辑器"""
        if not self.map_editor:
            self.map_editor = HexMapEditor(self.width, self.height)

        result = self.map_editor.run(standalone=False)

        if result == 'quit':
            self.running = False
        elif result == 'menu':
            self.current_state = GameState.MAIN_MENU
            self.map_editor = None
        else:
            self.current_state = GameState.MAIN_MENU
            self.map_editor = None

    def _run_game_play(self):
        """运行游戏模式"""
        if not self.game_play:
            self.game_play = GamePlaySystem(self.screen, self.width, self.height)

        result = self.game_play.run()

        if result == 'quit':
            self.running = False
        elif result == 'menu':
            self.current_state = GameState.MAIN_MENU
            self.game_play = None


if __name__ == "__main__":
    game = MainGame()
    game.run()