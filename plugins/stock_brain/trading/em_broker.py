"""
东方财富交易适配器
==================
通过 easytrader 控制东方财富客户端自动下单。
需要：1) 东方财富客户端运行并登录  2) pip install easytrader
"""
import logging
from typing import List

from plugins.stock_brain.trading.broker_adapter import (
    BrokerAdapter, Order, Position, AccountInfo,
    register_broker,
)

logger = logging.getLogger("StockBrain.EMBroker")


class EMBroker(BrokerAdapter):
    """东方财富交易适配器（easytrader）"""

    def __init__(self, exe_path: str = ""):
        self._user = None
        self._connected = False
        self._exe_path = exe_path

    @property
    def name(self) -> str:
        return "东方财富"

    def connect(self, **kwargs) -> bool:
        try:
            import easytrader
            self._user = easytrader.use("em")
            if self._exe_path:
                self._user.connect(self._exe_path)
            else:
                self._user.connect(r"C:\东方财富\xiadan.exe")
            self._connected = True
            logger.info("东方财富连接成功")
            return True
        except ImportError:
            logger.error("easytrader 未安装，请执行: pip install easytrader")
            return False
        except Exception as e:
            logger.error(f"东方财富连接失败: {e}")
            return False

    def disconnect(self):
        self._user = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._user is not None

    def get_account(self) -> AccountInfo:
        self._check_connected()
        try:
            balance = self._user.balance
            positions = self.get_positions()
            market_value = sum(p.market_value for p in positions)
            return AccountInfo(
                total_assets=balance.get("总资产", 0),
                available_cash=balance.get("可用金额", 0),
                market_value=market_value,
                frozen_cash=balance.get("冻结金额", 0),
                total_profit=balance.get("盈亏", 0),
                positions=positions,
            )
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return AccountInfo()

    def get_positions(self) -> List[Position]:
        self._check_connected()
        try:
            raw_positions = self._user.position
            positions = []
            for p in raw_positions:
                positions.append(Position(
                    symbol=str(p.get("证券代码", "")),
                    name=p.get("证券名称", ""),
                    quantity=int(p.get("股票余额", 0)),
                    available_quantity=int(p.get("可用余额", 0)),
                    cost_price=float(p.get("成本价", 0)),
                    current_price=float(p.get("市价", 0)),
                    market_value=float(p.get("市值", 0)),
                    profit=float(p.get("盈亏", 0)),
                    profit_pct=float(p.get("盈亏比例(%)", 0)) / 100,
                ))
            return positions
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return []

    def place_order(self, symbol: str, side: str, quantity: int,
                    price: float = 0, order_type: str = "limit") -> Order:
        self._check_connected()
        import datetime
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            created_at=datetime.datetime.now().isoformat(),
        )
        try:
            if side == "buy":
                result = self._user.buy(symbol, price=price, amount=quantity)
            else:
                result = self._user.sell(symbol, price=price, amount=quantity)
            order.order_id = str(result.get("entrust_no", ""))
            order.status = "submitted"
            order.note = f"委托已提交: {symbol} {side} {quantity}股 @ ¥{price}"
            logger.info(order.note)
        except Exception as e:
            order.status = "failed"
            order.note = f"下单失败: {e}"
            logger.error(order.note)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._check_connected()
        try:
            self._user.cancel_entrust(order_id)
            return True
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        self._check_connected()
        try:
            orders = self._user.today_entrusts
            for o in orders:
                if str(o.get("entrust_no", "")) == order_id:
                    status_map = {
                        "未报": "pending", "已报": "submitted",
                        "已成": "filled", "部成": "partial",
                        "已撤": "cancelled", "废单": "rejected",
                    }
                    return Order(
                        order_id=order_id,
                        symbol=str(o.get("证券代码", "")),
                        side="buy" if o.get("买卖标志") == "买入" else "sell",
                        quantity=int(o.get("委托数量", 0)),
                        price=float(o.get("委托价格", 0)),
                        status=status_map.get(o.get("委托状态", ""), "pending"),
                        filled_quantity=int(o.get("成交数量", 0)),
                        filled_price=float(o.get("成交价格", 0)),
                    )
            return Order(order_id=order_id, status="pending", note="未找到")
        except Exception as e:
            return Order(order_id=order_id, status="failed", note=str(e))

    def get_today_orders(self) -> List[Order]:
        self._check_connected()
        try:
            orders = self._user.today_entrusts
            result = []
            for o in orders:
                result.append(Order(
                    order_id=str(o.get("entrust_no", "")),
                    symbol=str(o.get("证券代码", "")),
                    side="buy" if o.get("买卖标志") == "买入" else "sell",
                    quantity=int(o.get("委托数量", 0)),
                    price=float(o.get("委托价格", 0)),
                    status=o.get("委托状态", ""),
                    filled_quantity=int(o.get("成交数量", 0)),
                ))
            return result
        except Exception as e:
            logger.error(f"获取今日委托失败: {e}")
            return []

    def _check_connected(self):
        if not self.is_connected():
            raise RuntimeError("东方财富未连接，请先调用 connect()")


# 注册到工厂
register_broker("em", EMBroker)