"""
六边形地图策略游戏 - 主程序（中文版）
包含主界面、地图编辑器、游戏模式、AI训练和路线回放
"""
import pygame
import sys
import os
import json
from enum import Enum
from typing import Optional

# 导入字体管理器
from font_manager import get_font_manager, get_font, get_small_font, get_medium_font, get_large_font, get_title_font, get_huge_font

# 导入地图编辑器
from hex_map_editor_chinese import HexMapEditor
from map_style_config import StyleConfig, TerrainType
# 导入游戏系统
from game_play_system import GamePlaySystem


class MessageBox:
    """消息弹窗"""
    def __init__(self, screen, message, title="提示"):
        self.screen = screen
        self.message = message
        self.title = title
        self.width = 400
        self.height = 200

        # 字体
        self.title_font = get_medium_font()
        self.message_font = get_small_font()

        # 位置（居中）
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        self.x = (screen_width - self.width) // 2
        self.y = (screen_height - self.height) // 2

        # 按钮
        self.ok_button = pygame.Rect(
            self.x + (self.width - 100) // 2,
            self.y + self.height - 50,
            100, 35
        )

    def draw(self):
        """绘制消息框"""
        # 背景遮罩
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # 弹窗背景
        window_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.screen, (50, 50, 60), window_rect)
        pygame.draw.rect(self.screen, (100, 100, 110), window_rect, 3)

        # 标题
        title_surf = self.title_font.render(self.title, True, (255, 215, 0))
        title_rect = title_surf.get_rect(center=(self.x + self.width // 2, self.y + 30))
        self.screen.blit(title_surf, title_rect)

        # 消息
        lines = self.message.split('\n')
        y_offset = self.y + 70
        for line in lines:
            if line:  # 跳过空行
                msg_surf = self.message_font.render(line, True, (200, 200, 200))
                msg_rect = msg_surf.get_rect(center=(self.x + self.width // 2, y_offset))
                self.screen.blit(msg_surf, msg_rect)
            y_offset += 30

        # 确定按钮
        pygame.draw.rect(self.screen, (80, 120, 160), self.ok_button)
        pygame.draw.rect(self.screen, (120, 160, 200), self.ok_button, 2)

        ok_text = self.message_font.render("确定", True, (255, 255, 255))
        ok_rect = ok_text.get_rect(center=self.ok_button.center)
        self.screen.blit(ok_text, ok_rect)

    def handle_click(self, pos):
        """处理点击"""
        if self.ok_button.collidepoint(pos):
            return True  # 关闭弹窗
        return False


class GameState(Enum):
    """游戏状态枚举"""
    MAIN_MENU = "main_menu"
    MAP_EDITOR = "map_editor"
    GAME_PLAY = "game_play"
    GAME_SETTINGS = "game_settings"
    AI_TRAINING = "ai_training"
    ROUTE_REPLAY = "route_replay"


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
            'ai_training': {
                'text': 'AI训练',
                'rect': None,
                'hovered': False
            },
            'route_replay': {
                'text': '查看最优路线',
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

        button_order = ['start_game', 'map_editor', 'ai_training', 'route_replay', 'exit']
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
        title_text = "拳打飞图脚踢萧瑟"
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
                # 特殊颜色处理
                if button_key == 'ai_training':
                    # AI训练按钮 - 紫色
                    if button['hovered']:
                        color = (140, 100, 180)
                        border_color = (255, 215, 0)
                        border_width = 3
                    else:
                        color = (100, 70, 130)
                        border_color = (150, 120, 180)
                        border_width = 2
                elif button_key == 'route_replay':
                    # 路线回放按钮 - 青色
                    if button['hovered']:
                        color = (80, 160, 140)
                        border_color = (255, 215, 0)
                        border_width = 3
                    else:
                        color = (60, 120, 100)
                        border_color = (100, 160, 140)
                        border_width = 2
                else:
                    # 普通按钮
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

                # 添加小标签
                if button_key == 'ai_training':
                    label_text = "PPO强化学习"
                    label_surface = self.small_font.render(label_text, True, (180, 180, 200))
                    label_rect = label_surface.get_rect(center=(button['rect'].centerx, button['rect'].bottom + 12))
                    self.screen.blit(label_surface, label_rect)
                elif button_key == 'route_replay':
                    label_text = "回放AI路线"
                    label_surface = self.small_font.render(label_text, True, (180, 200, 180))
                    label_rect = label_surface.get_rect(center=(button['rect'].centerx, button['rect'].bottom + 12))
                    self.screen.blit(label_surface, label_rect)

        # 版本信息（中文）
        version_text = "V1.0 阿丑是风华的狗"
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
                elif button_key == 'ai_training':
                    return GameState.AI_TRAINING
                elif button_key == 'route_replay':
                    return GameState.ROUTE_REPLAY
                elif button_key == 'exit':
                    return 'exit'
        return None


class AITrainingInterface:
    """AI训练界面"""

    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height

        # 字体
        self.title_font = get_large_font()
        self.font = get_medium_font()
        self.small_font = get_small_font()

        # 训练状态
        self.is_training = False
        self.training_progress = 0
        self.current_episode = 0
        self.total_episodes = 500
        self.best_exp = 0
        self.current_exp = 0

        # 训练器实例
        self.trainer = None

        # 按钮区域
        self.start_button_rect = pygame.Rect((width - 200) // 2, 300, 200, 60)

    def draw(self):
        """绘制训练界面"""
        self.screen.fill((30, 30, 40))

        # 标题
        title = "AI训练中心"
        title_surface = self.title_font.render(title, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_surface, title_rect)

        # 训练状态
        if self.is_training:
            # 进度条
            progress_width = 600
            progress_height = 30
            progress_x = (self.width - progress_width) // 2
            progress_y = 150

            # 背景
            pygame.draw.rect(self.screen, (50, 50, 60),
                           (progress_x, progress_y, progress_width, progress_height))
            # 进度
            filled_width = int(progress_width * self.training_progress)
            pygame.draw.rect(self.screen, (100, 200, 100),
                           (progress_x, progress_y, filled_width, progress_height))
            # 边框
            pygame.draw.rect(self.screen, (100, 100, 110),
                           (progress_x, progress_y, progress_width, progress_height), 2)

            # 进度文字
            progress_text = f"Episode {self.current_episode}/{self.total_episodes}"
            progress_surface = self.font.render(progress_text, True, (255, 255, 255))
            progress_rect = progress_surface.get_rect(center=(self.width // 2, progress_y + progress_height // 2))
            self.screen.blit(progress_surface, progress_rect)

            # 统计信息
            stats = [
                f"当前经验值: {self.current_exp}",
                f"最佳经验值: {self.best_exp}",
                f"训练进度: {self.training_progress*100:.1f}%"
            ]

            y = 250
            for stat in stats:
                stat_surface = self.font.render(stat, True, (200, 200, 200))
                stat_rect = stat_surface.get_rect(center=(self.width // 2, y))
                self.screen.blit(stat_surface, stat_rect)
                y += 40

            # 训练提示
            tip_text = "训练中... 请稍候"
            tip_surface = self.small_font.render(tip_text, True, (150, 150, 150))
            tip_rect = tip_surface.get_rect(center=(self.width // 2, 400))
            self.screen.blit(tip_surface, tip_rect)

        else:
            # 开始训练按钮
            pygame.draw.rect(self.screen, (80, 160, 80), self.start_button_rect)
            pygame.draw.rect(self.screen, (120, 200, 120), self.start_button_rect, 2)

            start_text = "开始训练"
            start_surface = self.font.render(start_text, True, (255, 255, 255))
            start_rect = start_surface.get_rect(center=self.start_button_rect.center)
            self.screen.blit(start_surface, start_rect)

            # 说明文字
            info_texts = [
                "点击开始训练将启动PPO强化学习算法",
                "训练将寻找获得最高经验值的路线",
                "训练完成后会生成最优路线报告",
                "",
                "训练完成后可在主菜单选择'查看最优路线'回放"
            ]

            y = 450
            for text in info_texts:
                if text:
                    text_surface = self.small_font.render(text, True, (150, 150, 150))
                    text_rect = text_surface.get_rect(center=(self.width // 2, y))
                    self.screen.blit(text_surface, text_rect)
                y += 30

        # 返回按钮
        back_text = "按 ESC 返回主菜单"
        back_surface = self.small_font.render(back_text, True, (100, 100, 100))
        back_rect = back_surface.get_rect(center=(self.width // 2, self.height - 30))
        self.screen.blit(back_surface, back_rect)

    def start_training(self):
        """启动训练"""
        print("启动PPO训练...")

        try:
            from rl_trainer_enhanced import PPOTrainer, setup_device
            import threading

            self.is_training = True

            def train_thread():
                try:
                    # 设置设备
                    device = setup_device()

                    # 创建训练器，传入device参数
                    self.trainer = PPOTrainer(
                        device=device,
                        lr=1e-4,
                        gamma=0.99,
                        eps_clip=0.2,
                        epochs=10,
                        batch_size=64,
                        n_workers=4,
                        use_amp=True
                    )

                    # 开始训练
                    self.trainer.train(total_episodes=self.total_episodes)

                except Exception as e:
                    print(f"训练过程中出错: {e}")
                    import traceback
                    traceback.print_exc()  # 打印完整错误信息便于调试
                finally:
                    self.is_training = False
                    print("训练结束")

            thread = threading.Thread(target=train_thread)
            thread.daemon = True
            thread.start()

        except ImportError as e:
            print(f"导入训练模块失败: {e}")
            self.is_training = False
            return
        except Exception as e:
            print(f"启动训练失败: {e}")
            self.is_training = False
            return

    def handle_click(self, pos):
        """处理点击"""
        if not self.is_training:
            if self.start_button_rect.collidepoint(pos):
                self.start_training()


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
        self.ai_training = None
        self.route_replay = None

        # 消息框
        self.message_box = None

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
            elif self.current_state == GameState.AI_TRAINING:
                self._run_ai_training()
            elif self.current_state == GameState.ROUTE_REPLAY:
                self._run_route_replay()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _run_main_menu(self, dt):
        """运行主菜单"""
        # 如果有消息框，优先处理
        if self.message_box:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.message_box.handle_click(event.pos):
                            self.message_box = None
                            return  # 立即返回，避免调用已经为None的对象

            # 绘制主菜单和消息框
            self.main_menu.update(dt)
            self.main_menu.draw()
            if self.message_box:  # 再次检查以确保安全
                self.message_box.draw()
            return

        # 正常的主菜单事件处理
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
                    elif result == GameState.AI_TRAINING:
                        self.ai_training = AITrainingInterface(self.screen, self.width, self.height)
                        self.current_state = GameState.AI_TRAINING
                    elif result == GameState.ROUTE_REPLAY:
                        # 检查文件是否存在
                        if os.path.exists('best_route.pkl'):
                            try:
                                from route_replay_system import RouteReplaySystem
                                self.route_replay = RouteReplaySystem(self.screen, self.width, self.height)
                                self.current_state = GameState.ROUTE_REPLAY
                            except ImportError:
                                self.message_box = MessageBox(
                                    self.screen,
                                    "无法导入回放系统模块！\n\n请确保 route_replay_system.py 文件存在",
                                    "错误"
                                )
                        else:
                            # 显示消息框
                            self.message_box = MessageBox(
                                self.screen,
                                "未找到最优路线文件！\n\n请先进行AI训练以生成路线文件",
                                "提示"
                            )

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

    def _run_ai_training(self):
        """运行AI训练界面"""
        if not self.ai_training:
            self.ai_training = AITrainingInterface(self.screen, self.width, self.height)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.current_state = GameState.MAIN_MENU
                    self.ai_training = None
                    return  # 立即返回，避免调用已经为None的对象
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.ai_training.handle_click(event.pos)

        if self.ai_training:  # 确保对象存在再调用draw
            self.ai_training.draw()

    def _run_route_replay(self):
        """运行路线回放系统"""
        if not self.route_replay:
            try:
                from route_replay_system import RouteReplaySystem
                self.route_replay = RouteReplaySystem(self.screen, self.width, self.height)
            except ImportError:
                self.message_box = MessageBox(
                    self.screen,
                    "无法导入回放系统模块！\n\n请确保 route_replay_system.py 文件存在",
                    "错误"
                )
                self.current_state = GameState.MAIN_MENU
                self.route_replay = None  # 确保设置为None
                return
            except Exception as e:
                self.message_box = MessageBox(
                    self.screen,
                    f"加载回放系统时出错！\n\n{str(e)[:100]}",
                    "错误"
                )
                self.current_state = GameState.MAIN_MENU
                self.route_replay = None  # 确保设置为None
                return

        if self.route_replay:  # 确保对象存在
            result = self.route_replay.run()

            if result == 'quit':
                self.running = False
            elif result == 'menu':
                self.current_state = GameState.MAIN_MENU
                self.route_replay = None


if __name__ == "__main__":
    game = MainGame()
    game.run()