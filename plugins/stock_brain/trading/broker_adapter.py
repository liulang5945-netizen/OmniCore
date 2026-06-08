"""
券商适配器统一接口
==================
定义交易执行的标准接口，支持同花顺、东方财富、模拟交易。
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("StockBrain.Broker")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"       # 市价单
    LIMIT = "limit"         # 限价单
    STOP_LOSS = "stop_loss" # 止损单
    TAKE_PROFIT = "take_profit"  # 止盈单


class OrderStatus(Enum):
    PENDING = "pending"         # 待执行
    SUBMITTED = "submitted"     # 已提交
    FILLED = "filled"           # 已成交
    PARTIAL = "partial"         # 部分成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 已拒绝
    FAILED = "failed"           # 失败


@dataclass
class Order:
    """交易订单"""
    order_id: str = ""
    symbol: str = ""
    side: str = "buy"           # buy / sell
    order_type: str = "limit"   # market / limit
    quantity: int = 0            # 股数（A股最小100股）
    price: float = 0.0           # 委托价格
    status: str = "pending"
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    note: str = ""


@dataclass
class Position:
    """持仓"""
    symbol: str = ""
    name: str = ""
    quantity: int = 0
    available_quantity: int = 0   # 可卖数量（T+1限制）
    cost_price: float = 0.0       # 成本价
    current_price: float = 0.0
    market_value: float = 0.0     # 市值
    profit: float = 0.0           # 盈亏
    profit_pct: float = 0.0       # 盈亏比例


@dataclass
class AccountInfo:
    """账户信息"""
    total_assets: float = 0.0      # 总资产
    available_cash: float = 0.0    # 可用资金
    market_value: float = 0.0      # 持仓市值
    frozen_cash: float = 0.0       # 冻结资金
    total_profit: float = 0.0      # 总盈亏
    positions: List[Position] = field(default_factory=list)


class BrokerAdapter(ABC):
    """
    券商适配器抽象基类
    所有券商实现都必须继承此类。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """券商名称"""
        ...

    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """连接券商"""
        ...

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """获取账户信息"""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: int,
                    price: float = 0, order_type: str = "limit") -> Order:
        """
        下单

        Args:
            symbol: 股票代码 (如 "600519")
            side: "buy" 或 "sell"
            quantity: 股数（A股最少100股，100的整数倍）
            price: 委托价格（市价单可为0）
            order_type: "market" 或 "limit"

        Returns:
            Order 对象
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        ...

    @abstractmethod
    def get_today_orders(self) -> List[Order]:
        """获取今日委托"""
        ...

    def buy(self, symbol: str, quantity: int, price: float = 0,
            order_type: str = "limit") -> Order:
        """买入"""
        return self.place_order(symbol, "buy", quantity, price, order_type)

    def sell(self, symbol: str, quantity: int, price: float = 0,
             order_type: str = "limit") -> Order:
        """卖出"""
        return self.place_order(symbol, "sell", quantity, price, order_type)

    def buy_amount(self, symbol: str, amount: float, price: float) -> Order:
        """按金额买入（自动计算整百股数）"""
        if price <= 0:
            logger.error("按金额买入需要指定价格")
            return Order(status="failed", note="需要指定价格")
        quantity = int(amount / price / 100) * 100
        if quantity < 100:
            logger.warning(f"金额不足买入1手(100股): 需要 ¥{price * 100:.0f}")
            return Order(status="failed", note="金额不足买入1手")
        return self.buy(symbol, quantity, price)


# ======================== 工厂方法 ========================

_broker_registry: Dict[str, type] = {}


def register_broker(name: str, cls: type):
    """注册券商实现"""
    _broker_registry[name] = cls


def create_broker(broker_type: str, **kwargs) -> BrokerAdapter:
    """
    创建券商实例

    Args:
        broker_type: "ths"(同花顺) / "em"(东方财富) / "sim"(模拟)
        **kwargs: 传递给券商构造函数的参数

    Returns:
        BrokerAdapter 实例
    """
    cls = _broker_registry.get(broker_type)
    if cls is None:
        available = list(_broker_registry.keys())
        raise ValueError(f"不支持的券商类型: {broker_type}。可选: {available}")
    return cls(**kwargs)