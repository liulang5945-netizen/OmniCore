"""
stock_brain 插件配置
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# 配置文件路径
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "data_store")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")


@dataclass
class RiskProfile:
    """风险偏好配置"""
    risk_tolerance: str = "稳健"        # 保守 / 稳健 / 积极 / 激进
    target_annual_return: float = 0.15   # 目标年化收益率 (0.15 = 15%)
    max_drawdown: float = 0.10           # 最大可接受回撤
    max_single_position: float = 0.30    # 单股最大仓位
    stop_loss: float = 0.05              # 止损线
    preferred_hold_period: str = "中期"  # 短线 / 中期 / 长线
    max_positions: int = 5               # 最大持仓数
    cash_reserve: float = 0.20           # 现金保留比例
    stock_preference: str = "白马+周期"  # 大盘蓝筹 / 白马+周期 / 成长股 / 题材热点


@dataclass
class CapitalConfig:
    """资本适配配置"""
    initial_capital: float = 300000.0    # 初始本金
    current_capital: float = 300000.0    # 当前本金
    strategy_mode: str = "均衡成长"      # 集中突破(<5万) / 均衡成长(5-50万) / 稳健配置(>50万)


@dataclass
class LearningConfig:
    """学习配置"""
    # 自学习循环
    max_iterations: int = 5              # 最大迭代轮数
    early_stop_patience: int = 2         # 连续无改善则停止
    target_sharpe: float = 1.5           # 目标夏普比率
    target_win_rate: float = 0.60        # 目标胜率
    target_annual_return: float = 0.15   # 目标年化收益
    
    # 模型参数
    model_type: str = "lstm"             # lstm / transformer / hybrid
    lookback_days: int = 60              # 回看天数
    forecast_days: int = 5               # 预测天数
    train_years: int = 10                # 训练数据年数
    
    # 特征参数
    technical_indicators: List[str] = field(default_factory=lambda: [
        "MA5", "MA10", "MA20", "MA60",
        "EMA12", "EMA26",
        "MACD", "MACD_signal", "MACD_hist",
        "RSI_14",
        "KDJ_K", "KDJ_D", "KDJ_J",
        "BOLL_upper", "BOLL_middle", "BOLL_lower",
        "ATR_14",
        "OBV",
        "VOL_ratio",
    ])
    use_event_features: bool = True      # 是否使用事件特征
    use_time_features: bool = True       # 是否使用时间特征


@dataclass
class DailyCycleConfig:
    """每日循环配置"""
    pre_market_time: str = "09:00"       # 盘前预测时间
    intraday_interval: int = 30          # 盘中更新间隔(分钟)
    post_market_time: str = "15:30"      # 收盘复盘时间
    auto_adjust: bool = True             # 是否自动调整参数
    consecutive_error_threshold: int = 3 # 连续错误天数阈值（触发模型微调）


@dataclass
class StockBrainConfig:
    """stock_brain 插件总配置"""
    risk: RiskProfile = field(default_factory=RiskProfile)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    daily: DailyCycleConfig = field(default_factory=DailyCycleConfig)
    
    # 关注的股票池
    watch_list: List[str] = field(default_factory=lambda: [
        "600519",  # 贵州茅台
        "300750",  # 宁德时代
        "601318",  # 中国平安
        "000858",  # 五粮液
        "600036",  # 招商银行
    ])
    
    # 插件版本
    version: str = "1.0.0"


# ======================== 风险偏好映射表 ========================

RISK_PROFILES = {
    "保守": {
        "max_drawdown": 0.05,
        "max_single_position": 0.20,
        "stop_loss": 0.03,
        "max_positions": 8,
        "cash_reserve": 0.30,
        "stock_preference": "大盘蓝筹",
        "preferred_hold_period": "长线",
    },
    "稳健": {
        "max_drawdown": 0.10,
        "max_single_position": 0.30,
        "stop_loss": 0.05,
        "max_positions": 5,
        "cash_reserve": 0.20,
        "stock_preference": "白马+周期",
        "preferred_hold_period": "中期",
    },
    "积极": {
        "max_drawdown": 0.20,
        "max_single_position": 0.40,
        "stop_loss": 0.08,
        "max_positions": 4,
        "cash_reserve": 0.10,
        "stock_preference": "成长股",
        "preferred_hold_period": "中短线",
    },
    "激进": {
        "max_drawdown": 0.30,
        "max_single_position": 0.50,
        "stop_loss": 0.12,
        "max_positions": 3,
        "cash_reserve": 0.05,
        "stock_preference": "题材热点",
        "preferred_hold_period": "短线",
    },
}

# 资本适配策略映射
CAPITAL_STRATEGIES = {
    "集中突破": {   # < 5万
        "max_positions": 2,
        "preferred_hold_period": "短线",
        "stop_loss": 0.03,
        "min_gain_threshold": 0.08,
        "position_ratio": (0.8, 1.0),
        "description": "资金集中，短线高收益，严格止损",
    },
    "均衡成长": {   # 5万 - 50万
        "max_positions": 4,
        "preferred_hold_period": "中期",
        "stop_loss": 0.05,
        "min_gain_threshold": 0.05,
        "position_ratio": (0.6, 0.8),
        "description": "均衡配置，中短线结合，稳健增长",
    },
    "稳健配置": {   # > 50万
        "max_positions": 8,
        "preferred_hold_period": "长线",
        "stop_loss": 0.07,
        "min_gain_threshold": 0.03,
        "position_ratio": (0.4, 0.7),
        "description": "分散风险，行业轮动，长线为主",
    },
}


def get_capital_strategy(capital: float) -> str:
    """根据本金自动选择策略"""
    if capital < 50000:
        return "集中突破"
    elif capital < 500000:
        return "均衡成长"
    else:
        return "稳健配置"


def apply_risk_profile(config: StockBrainConfig, risk_level: str) -> StockBrainConfig:
    """应用风险偏好到配置"""
    profile = RISK_PROFILES.get(risk_level, RISK_PROFILES["稳健"])
    config.risk.risk_tolerance = risk_level
    for key, value in profile.items():
        if hasattr(config.risk, key):
            setattr(config.risk, key, value)
    return config


def apply_capital_strategy(config: StockBrainConfig, capital: float) -> StockBrainConfig:
    """根据本金应用资本适配策略"""
    config.capital.current_capital = capital
    strategy_name = get_capital_strategy(capital)
    config.capital.strategy_mode = strategy_name
    strategy = CAPITAL_STRATEGIES.get(strategy_name, CAPITAL_STRATEGIES["均衡成长"])
    # 资本适配可以覆盖部分风控参数
    config.risk.max_positions = strategy["max_positions"]
    config.risk.stop_loss = strategy["stop_loss"]
    return config


def load_config() -> StockBrainConfig:
    """从文件加载配置"""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = StockBrainConfig()
            # 递归加载子配置
            if "risk" in data:
                for k, v in data["risk"].items():
                    if hasattr(config.risk, k):
                        setattr(config.risk, k, v)
            if "capital" in data:
                for k, v in data["capital"].items():
                    if hasattr(config.capital, k):
                        setattr(config.capital, k, v)
            if "learning" in data:
                for k, v in data["learning"].items():
                    if hasattr(config.learning, k):
                        setattr(config.learning, k, v)
            if "daily" in data:
                for k, v in data["daily"].items():
                    if hasattr(config.daily, k):
                        setattr(config.daily, k, v)
            if "watch_list" in data:
                config.watch_list = data["watch_list"]
            return config
        except Exception:
            pass
    return StockBrainConfig()


def save_config(config: StockBrainConfig):
    """保存配置到文件"""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    data = {
        "risk": asdict(config.risk),
        "capital": asdict(config.capital),
        "learning": asdict(config.learning),
        "daily": asdict(config.daily),
        "watch_list": config.watch_list,
        "version": config.version,
    }
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)