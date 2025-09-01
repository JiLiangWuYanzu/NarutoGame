from enum import Enum
from typing import Dict, Tuple, Optional


class TerrainType(Enum):
    """地形类型枚举 - 基于游戏地块系统"""
    # 基础地块
    START_POSITION = 0  # 玩家初始位置（全场唯一）
    NORMAL_LV1 = 1  # 普通地块LV1
    NORMAL_LV2 = 2  # 普通地块LV2
    NORMAL_LV3 = 3  # 普通地块LV3
    NORMAL_LV4 = 4  # 普通地块LV4
    NORMAL_LV5 = 5  # 普通地块LV5
    NORMAL_LV6 = 6  # 普通地块LV6

    # 训练地块
    DUMMY_LV1 = 7  # 木人桩LV1
    DUMMY_LV2 = 8  # 木人桩LV2
    DUMMY_LV3 = 9  # 木人桩LV3
    DUMMY_LV4 = 10  # 木人桩LV4
    DUMMY_LV5 = 11  # 木人桩LV5
    DUMMY_LV6 = 12  # 木人桩LV6

    # 特殊地块
    TRAINING_GROUND = 13  # 历练之地
    WATCHTOWER_LV1 = 14  # 瞭望塔LV1
    WATCHTOWER_LV2 = 15  # 瞭望塔LV2
    WATCHTOWER_LV3 = 16  # 瞭望塔LV3
    WATCHTOWER_LV4 = 17  # 瞭望塔LV4
    WATCHTOWER_LV5 = 18  # 瞭望塔LV5
    WATCHTOWER_LV6 = 19  # 瞭望塔LV6

    # 功能地块
    BLACK_MARKET = 20  # 神秘黑商
    RELIC_STONE = 21  # 遗迹石板

    # 宝藏地块
    TREASURE_1 = 22  # 远征秘宝1号
    TREASURE_2 = 23  # 远征秘宝2号
    TREASURE_3 = 24  # 远征秘宝3号
    TREASURE_4 = 25  # 远征秘宝4号
    TREASURE_5 = 26  # 远征秘宝5号
    TREASURE_6 = 27  # 远征秘宝6号
    TREASURE_7 = 28  # 远征秘宝7号
    TREASURE_8 = 29  # 远征秘宝8号

    # 障碍物
    WALL = 30  # 墙壁（默认地形）

    # BOSS地块
    BOSS_GAARA = 31  # BOSS 我爱罗
    BOSS_ZETSU = 32  # BOSS 绝
    BOSS_DART = 33  # BOSS 飞镖人
    BOSS_SHIRA = 34  # BOSS 紫罗
    BOSS_KUSHINA = 35  # BOSS 玖辛奈
    BOSS_KISAME = 36  # BOSS 鬼鲛
    BOSS_HANA = 37  # BOSS 犬冢花

    # 资源地块
    TENT = 38  # 帐篷

    # 特殊秘宝地块（新增）
    AKATSUKI_TREASURE = 39  # 晓秘宝
    KONOHA_TREASURE_1 = 40  # 木叶秘宝1
    KONOHA_TREASURE_2 = 41  # 木叶秘宝2


class StyleConfig:
    """样式配置类"""
    # 背景颜色
    BG_COLOR = (30, 30, 40)

    # UI颜色
    UI_BG_COLOR = (50, 50, 60)
    SELECTED_COLOR = (100, 150, 200)
    BORDER_COLOR = (80, 80, 90)
    TEXT_COLOR = (255, 255, 255)

    # 地形颜色配置 - 根据地块类型分级着色
    TERRAIN_COLORS: Dict[TerrainType, Tuple[int, int, int]] = {
        TerrainType.START_POSITION: (255, 255, 100),  # 亮黄色 - 起始位置

        # 普通地块 - 绿色系，等级越高越深
        TerrainType.NORMAL_LV1: (144, 238, 144),  # 浅绿色
        TerrainType.NORMAL_LV2: (124, 205, 124),  # 淡绿色
        TerrainType.NORMAL_LV3: (104, 175, 104),  # 中绿色
        TerrainType.NORMAL_LV4: (84, 145, 84),  # 深绿色
        TerrainType.NORMAL_LV5: (64, 115, 64),  # 更深绿色
        TerrainType.NORMAL_LV6: (44, 85, 44),  # 暗绿色

        # 木人桩 - 棕色系，等级越高越深
        TerrainType.DUMMY_LV1: (210, 180, 140),  # 浅棕色
        TerrainType.DUMMY_LV2: (188, 143, 143),  # 玫瑰棕
        TerrainType.DUMMY_LV3: (160, 82, 45),  # 赭色
        TerrainType.DUMMY_LV4: (139, 69, 19),  # 马鞍棕
        TerrainType.DUMMY_LV5: (101, 67, 33),  # 深棕色
        TerrainType.DUMMY_LV6: (61, 43, 31),  # 暗棕色

        # 瞭望塔 - 蓝色系，等级越高越深
        TerrainType.WATCHTOWER_LV1: (173, 216, 230),  # 浅蓝色
        TerrainType.WATCHTOWER_LV2: (135, 206, 235),  # 天蓝色
        TerrainType.WATCHTOWER_LV3: (100, 149, 237),  # 矢车菊蓝
        TerrainType.WATCHTOWER_LV4: (65, 105, 225),  # 皇家蓝
        TerrainType.WATCHTOWER_LV5: (0, 71, 171),  # 钴蓝色
        TerrainType.WATCHTOWER_LV6: (25, 25, 112),  # 午夜蓝

        # 特殊功能地块
        TerrainType.TRAINING_GROUND: (148, 0, 211),  # 紫罗兰色 - 历练之地
        TerrainType.BLACK_MARKET: (47, 79, 79),  # 暗灰色 - 神秘黑商
        TerrainType.RELIC_STONE: (112, 128, 144),  # 石板灰 - 遗迹石板

        # 宝藏地块 - 金色/黄色系
        TerrainType.TREASURE_1: (255, 215, 0),  # 金色
        TerrainType.TREASURE_2: (255, 193, 37),  # 深金色
        TerrainType.TREASURE_3: (255, 165, 0),  # 橙色
        TerrainType.TREASURE_4: (238, 173, 14),  # 金黄色
        TerrainType.TREASURE_5: (255, 228, 181),  # 桃色金
        TerrainType.TREASURE_6: (255, 236, 139),  # 浅金色
        TerrainType.TREASURE_7: (250, 250, 210),  # 亮金色
        TerrainType.TREASURE_8: (255, 248, 220),  # 米金色

        # 障碍物
        TerrainType.WALL: (105, 105, 105),  # 深灰色 - 墙壁

        # BOSS地块 - 红色/紫色系，威胁等级用颜色深浅表示
        TerrainType.BOSS_GAARA: (220, 20, 60),  # 深红色 - 我爱罗
        TerrainType.BOSS_ZETSU: (139, 0, 0),  # 暗红色 - 绝（最强BOSS）
        TerrainType.BOSS_DART: (205, 92, 92),  # 印第安红 - 飞镖人
        TerrainType.BOSS_SHIRA: (199, 21, 133),  # 中紫红 - 紫罗
        TerrainType.BOSS_KUSHINA: (148, 0, 211),  # 暗紫罗兰 - 玖辛奈（强BOSS）
        TerrainType.BOSS_KISAME: (75, 0, 130),  # 靛青色 - 鬼鲛
        TerrainType.BOSS_HANA: (255, 105, 180),  # 热粉色 - 犬冢花

        # 资源地块
        TerrainType.TENT: (34, 139, 34),  # 森林绿 - 帐篷
        TerrainType.AKATSUKI_TREASURE: (128, 0, 128),  # 紫色 - 晓组织主题
        TerrainType.KONOHA_TREASURE_1: (34, 139, 34),  # 森林绿 - 木叶主题
        TerrainType.KONOHA_TREASURE_2: (60, 179, 113),  # 中海绿 - 木叶主题
    }

    # 地形属性配置 - 基于游戏数据
    TERRAIN_PROPERTIES = {
        TerrainType.START_POSITION: {
            "food_cost": 0,  # 起始位置无需征服
            "exp_gain": 0,
            "status": 1,  # 默认已征服
            "has_item": False,
            "score_cost": 0,
            "food_gain": 0,
            "action_gain": 0,
            "passable": True,
            "name": "Start Position"
        },

        # 普通地块
        TerrainType.NORMAL_LV1: {
            "food_cost": 100, "exp_gain": 35, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV1"
        },
        TerrainType.NORMAL_LV2: {
            "food_cost": 110, "exp_gain": 40, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV2"
        },
        TerrainType.NORMAL_LV3: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV3"
        },
        TerrainType.NORMAL_LV4: {
            "food_cost": 130, "exp_gain": 50, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV4"
        },
        TerrainType.NORMAL_LV5: {
            "food_cost": 140, "exp_gain": 55, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV5"
        },
        TerrainType.NORMAL_LV6: {
            "food_cost": 150, "exp_gain": 60, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Normal LV6"
        },

        # 木人桩
        TerrainType.DUMMY_LV1: {
            "food_cost": 100, "exp_gain": 35, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV1"
        },
        TerrainType.DUMMY_LV2: {
            "food_cost": 100, "exp_gain": 40, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV2"
        },
        TerrainType.DUMMY_LV3: {
            "food_cost": 100, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV3"
        },
        TerrainType.DUMMY_LV4: {
            "food_cost": 100, "exp_gain": 50, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV4"
        },
        TerrainType.DUMMY_LV5: {
            "food_cost": 100, "exp_gain": 55, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV5"
        },
        TerrainType.DUMMY_LV6: {
            "food_cost": 100, "exp_gain": 60, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Dummy LV6"
        },

        # 特殊地块
        TerrainType.TRAINING_GROUND: {
            "food_cost": 0, "exp_gain": 520, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Training Ground"
        },

        # 瞭望塔
        TerrainType.WATCHTOWER_LV1: {
            "food_cost": 100, "exp_gain": 35, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV1"
        },
        TerrainType.WATCHTOWER_LV2: {
            "food_cost": 100, "exp_gain": 40, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV2"
        },
        TerrainType.WATCHTOWER_LV3: {
            "food_cost": 100, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV3"
        },
        TerrainType.WATCHTOWER_LV4: {
            "food_cost": 100, "exp_gain": 50, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV4"
        },
        TerrainType.WATCHTOWER_LV5: {
            "food_cost": 100, "exp_gain": 55, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV5"
        },
        TerrainType.WATCHTOWER_LV6: {
            "food_cost": 100, "exp_gain": 60, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Tower LV6"
        },

        # 功能地块
        TerrainType.BLACK_MARKET: {
            "food_cost": 0, "exp_gain": 30, "status": 0,
            "has_item": False, "score_cost": 1000, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Black Market"
        },
        TerrainType.RELIC_STONE: {
            "food_cost": 100, "exp_gain": 40, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Relic Stone"
        },

        # 宝藏地块
        TerrainType.TREASURE_1: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 1"
        },
        TerrainType.TREASURE_2: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 2"
        },
        TerrainType.TREASURE_3: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 3"
        },
        TerrainType.TREASURE_4: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 4"
        },
        TerrainType.TREASURE_5: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 5"
        },
        TerrainType.TREASURE_6: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 6"
        },
        TerrainType.TREASURE_7: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 7"
        },
        TerrainType.TREASURE_8: {
            "food_cost": 120, "exp_gain": 45, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Treasure 8"
        },

        # 特殊秘宝地块（新增）
        TerrainType.AKATSUKI_TREASURE: {
            "food_cost": 130, "exp_gain": 50, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Akatsuki Treasure"
        },
        TerrainType.KONOHA_TREASURE_1: {
            "food_cost": 110, "exp_gain": 40, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Konoha Treasure 1"
        },
        TerrainType.KONOHA_TREASURE_2: {
            "food_cost": 100, "exp_gain": 35, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "Konoha Treasure 2"
        },

        # 障碍物
        TerrainType.WALL: {
            "food_cost": -1, "exp_gain": 0, "status": 2,  # 不可征服
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": False, "name": "Wall"
        },

        # BOSS地块
        TerrainType.BOSS_GAARA: {
            "food_cost": 200, "exp_gain": 500, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Gaara"
        },
        TerrainType.BOSS_ZETSU: {
            "food_cost": 200, "exp_gain": 1000, "status": 0,
            "has_item": True, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Zetsu"  # 有飞雷神道具
        },
        TerrainType.BOSS_DART: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Darts"
        },
        TerrainType.BOSS_SHIRA: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Shira"
        },
        TerrainType.BOSS_KUSHINA: {
            "food_cost": 200, "exp_gain": 1000, "status": 0,
            "has_item": True, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Kushina"  # 有飞雷神道具
        },
        TerrainType.BOSS_KISAME: {
            "food_cost": 200, "exp_gain": 500, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Kisame"
        },
        TerrainType.BOSS_HANA: {
            "food_cost": 200, "exp_gain": 300, "status": 0,
            "has_item": False, "score_cost": 0, "food_gain": 0,
            "action_gain": 0, "passable": True, "name": "BOSS Hana"
        },

        # 资源地块
        TerrainType.TENT: {
            "food_cost": 0, "exp_gain": 0, "status": 0,
            "has_item": False, "score_cost": 0,
            "food_gain_by_day": {  # 根据游戏天数给予不同粮草
                (1, 10): 300,  # 游戏1-10天
                (11, 20): 250,  # 游戏11-20天
                (21, 35): 200,  # 游戏21-35天
                (36, 50): 150,  # 游戏36-50天
                (51, 90): 100,  # 游戏51-90天
            },
            "action_gain": 1,  # 征服后增加1个行动次数
            "passable": True,
            "name": "Tent"
        },
    }

    # 视觉效果配置
    HIGHLIGHT_COLOR = (255, 255, 0, 128)  # 高亮颜色（带透明度）
    PATH_COLOR = (255, 0, 0)  # 路径颜色
    GRID_COLOR = (100, 100, 100)  # 网格颜色

    # 动画配置
    ANIMATION_SPEED = 0.5  # 动画速度
    FADE_DURATION = 0.3  # 淡入淡出时间

    # 字体配置
    FONT_SIZE_SMALL = 12
    FONT_SIZE_MEDIUM = 16
    FONT_SIZE_LARGE = 24

    @classmethod
    def get_terrain_property(cls, terrain_type: TerrainType, property_name: str):
        """获取地形属性"""
        return cls.TERRAIN_PROPERTIES.get(terrain_type, {}).get(property_name, 0)

    @classmethod
    def get_tent_food_gain(cls, game_day: int) -> int:
        """根据游戏天数获取帐篷的粮草奖励"""
        tent_props = cls.TERRAIN_PROPERTIES[TerrainType.TENT]
        food_gain_by_day = tent_props.get("food_gain_by_day", {})

        for (min_day, max_day), food_gain in food_gain_by_day.items():
            if min_day <= game_day <= max_day:
                return food_gain

        # 超过90天后默认给100粮草
        return 100 if game_day > 90 else 0

    @classmethod
    def set_terrain_color(cls, terrain_type: TerrainType, color: Tuple[int, int, int]):
        """设置地形颜色"""
        cls.TERRAIN_COLORS[terrain_type] = color

    @classmethod
    def set_terrain_property(cls, terrain_type: TerrainType, property_name: str, value):
        """设置地形属性"""
        if terrain_type in cls.TERRAIN_PROPERTIES:
            cls.TERRAIN_PROPERTIES[terrain_type][property_name] = value

    @classmethod
    def get_terrain_by_level(cls, base_type: str, level: int) -> Optional[TerrainType]:
        """根据基础类型和等级获取地形类型"""
        type_map = {
            "normal": [TerrainType.NORMAL_LV1, TerrainType.NORMAL_LV2, TerrainType.NORMAL_LV3,
                       TerrainType.NORMAL_LV4, TerrainType.NORMAL_LV5, TerrainType.NORMAL_LV6],
            "dummy": [TerrainType.DUMMY_LV1, TerrainType.DUMMY_LV2, TerrainType.DUMMY_LV3,
                      TerrainType.DUMMY_LV4, TerrainType.DUMMY_LV5, TerrainType.DUMMY_LV6],
            "watchtower": [TerrainType.WATCHTOWER_LV1, TerrainType.WATCHTOWER_LV2, TerrainType.WATCHTOWER_LV3,
                           TerrainType.WATCHTOWER_LV4, TerrainType.WATCHTOWER_LV5, TerrainType.WATCHTOWER_LV6],
        }

        if base_type in type_map and 1 <= level <= 6:
            return type_map[base_type][level - 1]
        return None

    @classmethod
    def get_terrain_name(cls, terrain_type: TerrainType) -> str:
        """获取地形的显示名称"""
        return cls.TERRAIN_PROPERTIES.get(terrain_type, {}).get("name", terrain_type.name)


# 额外的样式预设
class MapTheme(Enum):
    """地图主题预设"""
    DEFAULT = "default"  # 默认主题
    DARK = "dark"  # 暗色主题
    LIGHT = "light"  # 亮色主题
    NATURE = "nature"  # 自然主题
    NINJA = "ninja"  # 忍者主题（适合火影忍者风格）


class ThemePresets:
    """主题预设配置"""
    THEMES = {
        MapTheme.DEFAULT: {
            "bg_color": (30, 30, 40),
            "ui_bg_color": (50, 50, 60),
            "border_color": (80, 80, 90),
        },
        MapTheme.DARK: {
            "bg_color": (10, 10, 10),
            "ui_bg_color": (20, 20, 20),
            "border_color": (40, 40, 40),
        },
        MapTheme.LIGHT: {
            "bg_color": (240, 240, 240),
            "ui_bg_color": (220, 220, 220),
            "border_color": (180, 180, 180),
        },
        MapTheme.NATURE: {
            "bg_color": (34, 49, 39),
            "ui_bg_color": (46, 64, 52),
            "border_color": (76, 106, 86),
        },
        MapTheme.NINJA: {
            "bg_color": (25, 20, 35),  # 深紫夜色
            "ui_bg_color": (45, 35, 55),  # 紫灰色
            "border_color": (255, 69, 0),  # 橙红色（火影标志色）
        },
    }

    @classmethod
    def apply_theme(cls, style_config: StyleConfig, theme: MapTheme):
        """应用主题到样式配置"""
        if theme in cls.THEMES:
            theme_data = cls.THEMES[theme]
            style_config.BG_COLOR = theme_data["bg_color"]
            style_config.UI_BG_COLOR = theme_data["ui_bg_color"]
            style_config.BORDER_COLOR = theme_data["border_color"]


# 为StyleConfig类添加中文名称方法
@classmethod
def get_terrain_name_chinese(cls, terrain_type: TerrainType) -> str:
    """获取地形的中文显示名称"""
    chinese_names = {
        TerrainType.START_POSITION: "初始位置",

        # 基础地块
        TerrainType.NORMAL_LV1: "普通 LV1",
        TerrainType.NORMAL_LV2: "普通 LV2",
        TerrainType.NORMAL_LV3: "普通 LV3",
        TerrainType.NORMAL_LV4: "普通 LV4",
        TerrainType.NORMAL_LV5: "普通 LV5",
        TerrainType.NORMAL_LV6: "普通 LV6",

        # 训练地块
        TerrainType.DUMMY_LV1: "木人桩 LV1",
        TerrainType.DUMMY_LV2: "木人桩 LV2",
        TerrainType.DUMMY_LV3: "木人桩 LV3",
        TerrainType.DUMMY_LV4: "木人桩 LV4",
        TerrainType.DUMMY_LV5: "木人桩 LV5",
        TerrainType.DUMMY_LV6: "木人桩 LV6",

        # 瞭望塔
        TerrainType.WATCHTOWER_LV1: "瞭望塔 LV1",
        TerrainType.WATCHTOWER_LV2: "瞭望塔 LV2",
        TerrainType.WATCHTOWER_LV3: "瞭望塔 LV3",
        TerrainType.WATCHTOWER_LV4: "瞭望塔 LV4",
        TerrainType.WATCHTOWER_LV5: "瞭望塔 LV5",
        TerrainType.WATCHTOWER_LV6: "瞭望塔 LV6",

        # 特殊地块
        TerrainType.TRAINING_GROUND: "历练之地",
        TerrainType.BLACK_MARKET: "神秘黑商",
        TerrainType.RELIC_STONE: "遗迹石板",
        TerrainType.TENT: "帐篷",

        # 宝藏地块
        TerrainType.TREASURE_1: "远征秘宝1",
        TerrainType.TREASURE_2: "远征秘宝2",
        TerrainType.TREASURE_3: "远征秘宝3",
        TerrainType.TREASURE_4: "远征秘宝4",
        TerrainType.TREASURE_5: "远征秘宝5",
        TerrainType.TREASURE_6: "远征秘宝6",
        TerrainType.TREASURE_7: "远征秘宝7",
        TerrainType.TREASURE_8: "远征秘宝8",

        # 特殊秘宝
        TerrainType.AKATSUKI_TREASURE: "晓组织秘宝",
        TerrainType.KONOHA_TREASURE_1: "木叶秘宝1",
        TerrainType.KONOHA_TREASURE_2: "木叶秘宝2",

        # BOSS地块
        TerrainType.BOSS_GAARA: "BOSS 我爱罗",
        TerrainType.BOSS_ZETSU: "BOSS 绝",
        TerrainType.BOSS_DART: "BOSS 飞镖人",
        TerrainType.BOSS_SHIRA: "BOSS 紫罗",
        TerrainType.BOSS_KUSHINA: "BOSS 玖辛奈",
        TerrainType.BOSS_KISAME: "BOSS 鬼鲛",
        TerrainType.BOSS_HANA: "BOSS 犬冢花",

        # 障碍物
        TerrainType.WALL: "墙壁",
    }

    return chinese_names.get(terrain_type, terrain_type.name)


# 将方法添加到StyleConfig类
StyleConfig.get_terrain_name_chinese = get_terrain_name_chinese