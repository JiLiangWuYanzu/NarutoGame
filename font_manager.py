"""
中文字体管理器
处理pygame中的中文显示问题
"""
import pygame
import os
import platform


class FontManager:
    """字体管理类"""

    def __init__(self):
        """初始化字体管理器"""
        self.fonts = {}
        self.chinese_font_path = None
        self.load_chinese_font()

    def load_chinese_font(self):
        """加载中文字体"""
        # 根据操作系统选择合适的中文字体
        system = platform.system()

        font_paths = []

        if system == "Windows":
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/simkai.ttf",  # 楷体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # 苹方
                "/Library/Fonts/Songti.ttc",  # 宋体
                "/System/Library/Fonts/STHeiti Light.ttc",  # 黑体
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
            ]

        # 尝试加载字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                self.chinese_font_path = font_path
                print(f"成功加载中文字体: {font_path}")
                break

        if not self.chinese_font_path:
            print("警告：未找到系统中文字体，将使用pygame默认字体")
            print("建议安装中文字体以获得最佳显示效果")

    def get_font(self, size=24, bold=False):
        """获取指定大小的字体"""
        key = (size, bold)

        if key not in self.fonts:
            if self.chinese_font_path:
                try:
                    self.fonts[key] = pygame.font.Font(self.chinese_font_path, size)
                    if bold:
                        self.fonts[key].set_bold(True)
                except:
                    print(f"加载字体失败，使用默认字体")
                    self.fonts[key] = pygame.font.Font(None, size)
            else:
                # 使用pygame默认字体
                self.fonts[key] = pygame.font.Font(None, size)

        return self.fonts[key]

    def get_small_font(self):
        """获取小号字体"""
        return self.get_font(18)

    def get_medium_font(self):
        """获取中号字体"""
        return self.get_font(24)

    def get_large_font(self):
        """获取大号字体"""
        return self.get_font(32)

    def get_title_font(self):
        """获取标题字体"""
        return self.get_font(48, bold=True)

    def get_huge_font(self):
        """获取超大字体"""
        return self.get_font(72, bold=True)


# 全局字体管理器实例
_font_manager = None


def get_font_manager():
    """获取全局字体管理器实例"""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager


# 便捷函数
def get_font(size=24, bold=False):
    """便捷函数：获取字体"""
    return get_font_manager().get_font(size, bold)


def get_small_font():
    """便捷函数：获取小号字体"""
    return get_font_manager().get_small_font()


def get_medium_font():
    """便捷函数：获取中号字体"""
    return get_font_manager().get_medium_font()


def get_large_font():
    """便捷函数：获取大号字体"""
    return get_font_manager().get_large_font()


def get_title_font():
    """便捷函数：获取标题字体"""
    return get_font_manager().get_title_font()


def get_huge_font():
    """便捷函数：获取超大字体"""
    return get_font_manager().get_huge_font()