"""
模拟交易券商
============
用虚拟资金模拟撮合，用于测试策略。
无需连接真实券商客户端。
"""
import datetime
import json
import logging
import os
import uuid
from typing import Dict, List, Optional

from plugins.stock_brain.trading.broker_adapter import (
    BrokerAdapter, Order, Position, AccountInfo,
    register_broker,
)

logger = logging.getLogger("StockBrain.SimBroker")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "simulation")


class SimBroker(BrokerAdapter):
    """模拟交易券商"""

    def __init__(self, initial_cash: float = 100000, commission_rate: float = 0.0003,
                 stamp_tax: float = 0.001, min_commission: float = 5.0):
        self._connected = False
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._commission_rate = commission_rate   # 佣金率（万三）
        self._stamp_tax = stamp_tax               # 印花税（千一，仅卖出）
        self._min_commission = min_commission      # 最低佣金
        self._positions: Dict[str, dict] = {}     # symbol -> {name, qty, cost_price, available_qty}
        self._orders: List[dict] = []
        self._trade_log: List[dict] = []
        os.makedirs(_DATA_DIR, exist_ok=True)

    @property
    def name(self) -> str:
        return "模拟交易"

    def connect(self, **kwargs) -> bool:
        self._connected = True
        self._load_state()
        logger.info(f"模拟交易已连接 | 初始资金: ¥{self._initial_cash:,.0f} | 当前资金: ¥{self._cash:,.0f}")
        return True

    def disconnect(self):
        self._save_state()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> AccountInfo:
        positions = self.get_positions()
        market_value = sum(p.market_value for p in positions)
        total_profit = sum(p.profit for p in positions)
        return AccountInfo(
            total_assets=self._cash + market_value,
            available_cash=self._cash,
            market_value=market_value,
            frozen_cash=0,
            total_profit=total_profit,
            positions=positions,
        )

    def get_positions(self) -> List[Position]:
        from plugins.stock_brain.data.market_data import get_realtime_quote
        positions = []
        for symbol, pos in self._positions.items():
            if pos["qty"] <= 0:
                continue
            quote = get_realtime_quote(symbol)
            current_price = quote["price"] if quote else pos["cost_price"]
            market_value = current_price * pos["qty"]
            cost_value = pos["cost_price"] * pos["qty"]
            profit = market_value - cost_value
            profit_pct = profit / (cost_value + 1e-10)
            positions.append(Position(
                symbol=symbol,
                name=pos.get("name", ""),
                quantity=pos["qty"],
                available_quantity=pos.get("available_qty", pos["qty"]),
                cost_price=pos["cost_price"],
                current_price=current_price,
                market_value=market_value,
                profit=profit,
                profit_pct=profit_pct,
            ))
        return positions

    def place_order(self, symbol: str, side: str, quantity: int,
                    price: float = 0, order_type: str = "limit") -> Order:
        order_id = f"SIM_{uuid.uuid4().hex[:8]}"
        now = datetime.datetime.now().isoformat()

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="submitted",
            created_at=now,
            updated_at=now,
        )

        if quantity < 100 or quantity % 100 != 0:
            order.status = "rejected"
            order.note = "数量必须为100的整数倍"
            self._orders.append(self._order_to_dict(order))
            return order

        # 获取当前价格（模拟成交用）
        from plugins.stock_brain.data.market_data import get_realtime_quote
        quote = get_realtime_quote(symbol)
        if quote is None:
            order.status = "failed"
            order.note = f"无法获取 {symbol} 行情"
            self._orders.append(self._order_to_dict(order))
            return order

        current_price = quote["price"]
        name = quote.get("name", "")

        if price <= 0:
            fill_price = current_price
        else:
            if side == "buy" and price >= current_price:
                fill_price = current_price
            elif side == "sell" and price <= current_price:
                fill_price = current_price
            else:
                fill_price = price  # 限价单假设成交

        # 计算费用
        commission = max(fill_price * quantity * self._commission_rate, self._min_commission)
        stamp = fill_price * quantity * self._stamp_tax if side == "sell" else 0
        total_cost = commission + stamp

        if side == "buy":
            total_needed = fill_price * quantity + total_cost
            if total_needed > self._cash:
                order.status = "rejected"
                order.note = f"资金不足: 需要 ¥{total_needed:,.2f}，可用 ¥{self._cash:,.2f}"
                self._orders.append(self._order_to_dict(order))
                return order

            self._cash -= fill_price * quantity + total_cost
            if symbol in self._positions:
                old = self._positions[symbol]
                old_qty = old["qty"]
                old_cost = old["cost_price"] * old_qty
                new_qty = old_qty + quantity
                old["cost_price"] = (old_cost + fill_price * quantity) / new_qty
                old["qty"] = new_qty
                old["name"] = name
            else:
                self._positions[symbol] = {
                    "name": name,
                    "qty": quantity,
                    "cost_price": fill_price,
                    "available_qty": 0,  # T+1，当日买入不可卖
                }
            order.note = f"买入 {name}({symbol}) {quantity}股 @ ¥{fill_price:.2f}"

        elif side == "sell":
            pos = self._positions.get(symbol)
            if not pos or pos.get("available_qty", 0) < quantity:
                order.status = "rejected"
                available = pos["available_qty"] if pos else 0
                order.note = f"可卖数量不足: 可卖{available}股，委托{quantity}股"
                self._orders.append(self._order_to_dict(order))
                return order

            self._cash += fill_price * quantity - total_cost
            pos["qty"] -= quantity
            pos["available_qty"] -= quantity
            if pos["qty"] <= 0:
                del self._positions[symbol]
            order.note = f"卖出 {name}({symbol}) {quantity}股 @ ¥{fill_price:.2f}"

        order.status = "filled"
        order.filled_quantity = quantity
        order.filled_price = fill_price
        order.commission = total_cost

        self._orders.append(self._order_to_dict(order))
        self._trade_log.append({
            "order_id": order_id,
            "symbol": symbol,
            "name": name,
            "side": side,
            "quantity": quantity,
            "price": fill_price,
            "commission": total_cost,
            "timestamp": now,
        })
        self._save_state()
        logger.info(f"模拟成交: {order.note}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o["order_id"] == order_id and o["status"] == "submitted":
                o["status"] = "cancelled"
                return True
        return False

    def get_order_status(self, order_id: str) -> Order:
        for o in self._orders:
            if o["order_id"] == order_id:
                return self._dict_to_order(o)
        return Order(order_id=order_id, status="failed", note="订单不存在")

    def get_today_orders(self) -> List[Order]:
        today = datetime.date.today().isoformat()
        return [self._dict_to_order(o) for o in self._orders
                if o.get("created_at", "").startswith(today)]

    def get_trade_log(self) -> List[dict]:
        return self._trade_log

    def reset(self, initial_cash: float = None):
        """重置模拟账户"""
        if initial_cash:
            self._initial_cash = initial_cash
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._trade_log.clear()
        self._save_state()
        logger.info(f"模拟账户已重置: ¥{self._cash:,.0f}")

    def update_available_qty(self):
        """T+1 更新：每日开盘前调用，将持仓的可用数量设为全部"""
        for pos in self._positions.values():
            pos["available_qty"] = pos["qty"]

    def _save_state(self):
        state = {
            "initial_cash": self._initial_cash,
            "cash": self._cash,
            "positions": self._positions,
            "orders": self._orders[-200:],  # 只保留最近200条
            "trade_log": self._trade_log[-500:],
            "saved_at": datetime.datetime.now().isoformat(),
        }
        path = os.path.join(_DATA_DIR, "sim_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    def _load_state(self):
        path = os.path.join(_DATA_DIR, "sim_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._cash = state.get("cash", self._initial_cash)
            self._positions = state.get("positions", {})
            self._orders = state.get("orders", [])
            self._trade_log = state.get("trade_log", [])
            logger.info(f"已加载模拟账户状态: ¥{self._cash:,.0f}")
        except Exception as e:
            logger.warning(f"加载模拟账户失败: {e}")

    @staticmethod
    def _order_to_dict(order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "filled_price": order.filled_price,
            "commission": order.commission,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "note": order.note,
        }

    @staticmethod
    def _dict_to_order(d: dict) -> Order:
        return Order(**{k: v for k, v in d.items() if k in Order.__dataclass_fields__})


# 注册到工厂
register_broker("sim", SimBroker)