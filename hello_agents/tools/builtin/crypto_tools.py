"""加密货币专用工具 - 提供实时行情、技术指标、市场情绪等数据。

使用免费 API：
- CoinGecko: 价格、市值、成交量、涨跌幅
- Alternative.me: Fear & Greed 恐惧贪婪指数
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests

from ..base import Tool, ToolParameter

logger = logging.getLogger(__name__)

# ============================================================
# CoinGecko 工具：实时价格与市场数据
# ============================================================

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# 常见币种 ID 映射（用户可能用各种名称）
COIN_ALIASES: Dict[str, str] = {
    "btc": "bitcoin", "比特币": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "以太坊": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "索拉纳": "solana", "solana": "solana",
    "bnb": "binancecoin", "币安币": "binancecoin",
    "xrp": "ripple", "瑞波": "ripple", "ripple": "ripple",
    "doge": "dogecoin", "狗狗币": "dogecoin", "dogecoin": "dogecoin",
    "ada": "cardano", "卡尔达诺": "cardano", "cardano": "cardano",
    "avax": "avalanche-2", "雪崩": "avalanche-2",
    "dot": "polkadot", "波卡": "polkadot", "polkadot": "polkadot",
    "link": "chainlink", "chainlink": "chainlink",
    "matic": "matic-network", "polygon": "matic-network",
    "uni": "uniswap", "uniswap": "uniswap",
    "atom": "cosmos", "cosmos": "cosmos",
    "ltc": "litecoin", "莱特币": "litecoin", "litecoin": "litecoin",
    "trx": "tron", "波场": "tron", "tron": "tron",
}


def _resolve_coin_id(name: str) -> str:
    """将用户输入的币种名称解析为 CoinGecko ID"""
    key = name.strip().lower()
    return COIN_ALIASES.get(key, key)


class CryptoMarketTool(Tool):
    """加密货币实时行情工具（基于 CoinGecko 免费 API）"""

    def __init__(self) -> None:
        super().__init__(
            name="crypto_price",
            description=(
                "查询加密货币实时价格、市值、24h成交量和涨跌幅。"
                "支持的币种：BTC/ETH/SOL/BNB/XRP/DOGE/ADA/DOT/LINK/UNI 等主流币种。"
                "输入币种名称或代码即可查询，多个币种用逗号分隔。"
            ),
        )

    def run(self, parameters: Dict[str, Any]) -> str:  # type: ignore[override]
        query = (parameters.get("input") or parameters.get("query") or "").strip()
        if not query:
            return "错误：请输入要查询的币种名称（如 BTC、ETH、bitcoin）"

        # 解析币种
        raw_names = [s.strip() for s in query.replace("，", ",").split(",") if s.strip()]
        coin_ids = list(dict.fromkeys(_resolve_coin_id(n) for n in raw_names))  # 去重保序

        try:
            # 批量查询价格
            ids_str = ",".join(coin_ids)
            resp = requests.get(
                f"{COINGECKO_BASE}/simple/price",
                params={
                    "ids": ids_str,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_24hr_vol": "true",
                    "include_market_cap": "true",
                    "include_last_updated_at": "true",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"❌ CoinGecko API 请求失败: {exc}"

        if not data:
            return f"❌ 未找到币种: {query}。请检查名称是否正确。"

        # 格式化输出
        lines = ["📊 加密货币实时行情（数据来源：CoinGecko）\n"]
        for coin_id in coin_ids:
            info = data.get(coin_id)
            if not info:
                lines.append(f"❌ 未找到: {coin_id}")
                continue

            price = info.get("usd", "N/A")
            change_24h = info.get("usd_24h_change", 0)
            vol_24h = info.get("usd_24h_vol", 0)
            market_cap = info.get("usd_market_cap", 0)

            change_emoji = "🟢" if change_24h >= 0 else "🔴"
            lines.append(f"**{coin_id.upper()}**")
            lines.append(f"  💰 价格: ${price:,.2f}" if isinstance(price, (int, float)) else f"  💰 价格: {price}")
            lines.append(f"  {change_emoji} 24h涨跌: {change_24h:+.2f}%")
            lines.append(f"  📈 24h成交量: ${vol_24h:,.0f}")
            lines.append(f"  🏦 市值: ${market_cap:,.0f}")
            lines.append("")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="币种名称或代码，多个用逗号分隔（如 BTC,ETH 或 比特币,以太坊）",
                required=True,
            )
        ]


# ============================================================
# Fear & Greed 指数工具
# ============================================================

class FearGreedTool(Tool):
    """加密货币恐惧贪婪指数工具（基于 Alternative.me 免费 API）"""

    def __init__(self) -> None:
        super().__init__(
            name="fear_greed",
            description=(
                "查询加密货币市场恐惧与贪婪指数（Fear & Greed Index）。"
                "指数范围 0-100：0-24 极度恐惧，25-49 恐惧，50 中性，51-74 贪婪，75-100 极度贪婪。"
                "可查最近 1-30 天的历史数据。输入天数即可（默认 7 天）。"
            ),
        )

    def run(self, parameters: Dict[str, Any]) -> str:  # type: ignore[override]
        query = (parameters.get("input") or parameters.get("query") or "").strip()

        # 解析天数
        days = 7
        if query:
            try:
                days = int(query)
                days = max(1, min(30, days))
            except ValueError:
                days = 7

        try:
            resp = requests.get(
                "https://api.alternative.me/fng/",
                params={"limit": days},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            return f"❌ Fear & Greed API 请求失败: {exc}"

        data_list = result.get("data", [])
        if not data_list:
            return "❌ 未获取到恐惧贪婪指数数据"

        lines = ["😱📊 加密货币恐惧与贪婪指数（数据来源：Alternative.me）\n"]

        # 当前值（最新一条）
        latest = data_list[0]
        value = int(latest.get("value", 0))
        classification = latest.get("value_classification", "")
        emoji = self._get_emoji(value)

        lines.append(f"**当前指数: {value} — {classification}** {emoji}")
        lines.append(self._get_description(value))
        lines.append("")

        # 历史趋势
        if len(data_list) > 1:
            lines.append(f"📅 最近 {len(data_list)} 天趋势：")
            from datetime import datetime
            for item in data_list:
                ts = int(item.get("timestamp", 0))
                date_str = datetime.fromtimestamp(ts).strftime("%m-%d") if ts else "N/A"
                v = item.get("value", "?")
                cls = item.get("value_classification", "")
                lines.append(f"  {date_str}: {v} ({cls})")

        return "\n".join(lines)

    @staticmethod
    def _get_emoji(value: int) -> str:
        if value <= 24:
            return "😱"
        elif value <= 49:
            return "😰"
        elif value == 50:
            return "😐"
        elif value <= 74:
            return "😊"
        else:
            return "🤑"

    @staticmethod
    def _get_description(value: int) -> str:
        if value <= 24:
            return "市场处于极度恐惧状态，投资者信心极低，可能是逆向买入的机会。"
        elif value <= 49:
            return "市场偏向恐惧，投资者较为谨慎，市场可能处于回调或盘整中。"
        elif value == 50:
            return "市场情绪中性，多空平衡，方向不明。"
        elif value <= 74:
            return "市场偏向贪婪，投资者情绪乐观，需注意追高风险。"
        else:
            return "市场处于极度贪婪状态，往往是风险较高的时期，需警惕回调。"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="查询天数（1-30），默认 7 天",
                required=False,
            )
        ]


# ============================================================
# 便捷函数
# ============================================================

# ============================================================
# 技术指标工具（Binance K线 + pandas-ta 本地计算）
# ============================================================

# Binance 交易对映射
SYMBOL_MAP: Dict[str, str] = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT", "比特币": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT", "以太坊": "ETHUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT", "索拉纳": "SOLUSDT",
    "bnb": "BNBUSDT", "doge": "DOGEUSDT", "狗狗币": "DOGEUSDT",
    "xrp": "XRPUSDT", "ada": "ADAUSDT", "dot": "DOTUSDT",
    "link": "LINKUSDT", "avax": "AVAXUSDT", "matic": "MATICUSDT",
    "uni": "UNIUSDT", "atom": "ATOMUSDT", "ltc": "LTCUSDT",
    "trx": "TRXUSDT", "near": "NEARUSDT", "apt": "APTUSDT",
}

INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "小时": "1h", "小时线": "1h",
    "4h": "4h", "4小时": "4h",
    "1d": "1d", "日线": "1d", "日": "1d",
    "1w": "1w", "周线": "1w", "周": "1w",
}


def _resolve_symbol(name: str) -> str:
    key = name.strip().lower()
    return SYMBOL_MAP.get(key, key.upper() + "USDT")


def _resolve_interval(text: str) -> str:
    key = text.strip().lower()
    return INTERVAL_MAP.get(key, "1h")


class TechnicalIndicatorTool(Tool):
    """加密货币技术指标工具（Binance K线 + pandas-ta 本地计算）

    自动计算 RSI、MACD、布林带、EMA、支撑阻力位等指标，数据精确可靠。
    """

    def __init__(self) -> None:
        super().__init__(
            name="technical",
            description=(
                "查询加密货币技术指标（RSI、MACD、布林带、EMA、支撑阻力位）。"
                "基于 Binance 实时K线数据本地计算，数据精确。"
                "输入格式：币种 周期（如 BTC 1h、ETH 4h、SOL 日线）。默认 BTC 1h。"
            ),
        )

    def run(self, parameters: Dict[str, Any]) -> str:  # type: ignore[override]
        query = (parameters.get("input") or parameters.get("query") or "BTC 1h").strip()

        # 解析输入：币种 + 周期
        parts = query.replace(",", " ").replace("，", " ").split()
        coin = parts[0] if parts else "BTC"
        interval_raw = parts[1] if len(parts) > 1 else "1h"

        symbol = _resolve_symbol(coin)
        interval = _resolve_interval(interval_raw)

        try:
            import pandas as pd
            import pandas_ta as ta
        except ImportError:
            return "❌ 需要安装 pandas 和 pandas-ta：pip install pandas pandas-ta"

        # 获取 K线数据
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": 100},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            return f"❌ Binance K线数据获取失败: {exc}"

        if not raw:
            return f"❌ 未找到 {symbol} 的K线数据，请检查币种名称。"

        # 构建 DataFrame
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # 计算技术指标
        close = df["close"]

        # RSI (14)
        rsi_series = ta.rsi(close, length=14)
        rsi = rsi_series.iloc[-1] if rsi_series is not None and len(rsi_series) > 0 else None

        # MACD (12, 26, 9)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_val = macd_hist = macd_signal = None
        if macd_df is not None and len(macd_df) > 0:
            macd_val = macd_df.iloc[-1, 0]
            macd_signal = macd_df.iloc[-1, 1]
            macd_hist = macd_df.iloc[-1, 2]

        # 布林带 (20, 2)
        bbands = ta.bbands(close, length=20, std=2)
        bb_upper = bb_mid = bb_lower = None
        if bbands is not None and len(bbands) > 0:
            bb_lower = bbands.iloc[-1, 0]
            bb_mid = bbands.iloc[-1, 1]
            bb_upper = bbands.iloc[-1, 2]

        # EMA
        ema7 = ta.ema(close, length=7)
        ema25 = ta.ema(close, length=25)
        ema99 = ta.ema(close, length=99)

        # 支撑阻力：近期高低点
        recent = df.tail(20)
        support = recent["low"].min()
        resistance = recent["high"].max()

        # 当前价格
        current_price = close.iloc[-1]

        # 格式化输出
        lines = [
            f"📐 **{symbol} {interval} 技术指标**（基于最近 100 根K线，Binance 实时数据）\n",
            f"**当前价格**: ${current_price:,.2f}",
            "",
            "**📊 RSI (14)**",
        ]

        if rsi is not None:
            rsi_status = "超卖 🟢" if rsi < 30 else ("超买 🔴" if rsi > 70 else "中性 ⚪")
            lines.append(f"  RSI = {rsi:.1f} — {rsi_status}")
        else:
            lines.append("  RSI = N/A")

        lines.append("")
        lines.append("**📈 MACD (12, 26, 9)**")
        if macd_val is not None:
            macd_trend = "金叉（看多）🟢" if macd_hist > 0 else "死叉（看空）🔴"
            lines.append(f"  MACD = {macd_val:.2f}, Signal = {macd_signal:.2f}, Hist = {macd_hist:.2f}")
            lines.append(f"  状态: {macd_trend}")
        else:
            lines.append("  MACD = N/A")

        lines.append("")
        lines.append("**🎯 布林带 (20, 2)**")
        if bb_upper is not None:
            bb_pos = "上轨附近（可能超买）" if current_price > bb_upper * 0.98 else (
                "下轨附近（可能超卖）" if current_price < bb_lower * 1.02 else "中轨附近"
            )
            lines.append(f"  上轨: ${bb_upper:,.2f} | 中轨: ${bb_mid:,.2f} | 下轨: ${bb_lower:,.2f}")
            lines.append(f"  当前位置: {bb_pos}")

        lines.append("")
        lines.append("**📉 均线 EMA**")
        ema_parts = []
        if ema7 is not None and len(ema7) > 0:
            ema_parts.append(f"EMA7=${ema7.iloc[-1]:,.2f}")
        if ema25 is not None and len(ema25) > 0:
            ema_parts.append(f"EMA25=${ema25.iloc[-1]:,.2f}")
        if ema99 is not None and len(ema99) > 0:
            ema_parts.append(f"EMA99=${ema99.iloc[-1]:,.2f}")
        lines.append(f"  {' | '.join(ema_parts)}")

        # EMA 多空排列
        if ema7 is not None and ema25 is not None and len(ema7) > 0 and len(ema25) > 0:
            if ema7.iloc[-1] > ema25.iloc[-1]:
                lines.append("  排列: 短期均线在上（偏多）🟢")
            else:
                lines.append("  排列: 短期均线在下（偏空）🔴")

        lines.append("")
        lines.append("**🛡️ 近期支撑阻力（20根K线）**")
        lines.append(f"  支撑位: ${support:,.2f}")
        lines.append(f"  阻力位: ${resistance:,.2f}")
        lines.append(f"  当前距支撑: {((current_price - support) / support * 100):+.1f}%")
        lines.append(f"  当前距阻力: {((current_price - resistance) / resistance * 100):+.1f}%")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="币种+周期（如 BTC 1h、ETH 4h、SOL 日线），默认 BTC 1h",
                required=False,
            )
        ]


# ============================================================
# 合约数据工具（资金费率 + 持仓量 + 多空比）
# ============================================================

FUTURES_SYMBOL_MAP: Dict[str, str] = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT", "比特币": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT", "以太坊": "ETHUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "bnb": "BNBUSDT", "doge": "DOGEUSDT", "xrp": "XRPUSDT",
    "ada": "ADAUSDT", "dot": "DOTUSDT", "link": "LINKUSDT",
    "avax": "AVAXUSDT", "uni": "UNIUSDT", "ltc": "LTCUSDT",
}


def _resolve_futures_symbol(name: str) -> str:
    key = name.strip().lower()
    return FUTURES_SYMBOL_MAP.get(key, key.upper() + "USDT")


class FuturesDataTool(Tool):
    """加密货币合约数据工具（Binance Futures 免费 API）

    查询资金费率、持仓量、多空比等合约市场数据，用于判断市场杠杆情绪。
    """

    def __init__(self) -> None:
        super().__init__(
            name="futures_data",
            description=(
                "查询加密货币合约市场数据：资金费率、持仓量(OI)、多空比。"
                "资金费率极高→多头过热可能回调；极低/负值→空头过热可能反弹。"
                "输入币种名称即可（如 BTC、ETH）。默认 BTC。"
            ),
        )

    def run(self, parameters: Dict[str, Any]) -> str:  # type: ignore[override]
        query = (parameters.get("input") or parameters.get("query") or "BTC").strip()
        symbol = _resolve_futures_symbol(query.split(",")[0].split()[0])

        lines = [f"📋 **{symbol} 合约数据**（数据来源：Binance Futures）\n"]

        # 1. 资金费率（最近 5 期）
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 5},
                timeout=10,
            )
            resp.raise_for_status()
            funding_data = resp.json()

            lines.append("**💸 资金费率（最近 5 期）**")
            from datetime import datetime
            for item in funding_data:
                rate = float(item["fundingRate"])
                ts = int(item["fundingTime"]) / 1000
                time_str = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
                rate_pct = rate * 100
                emoji = "🟢" if rate > 0 else "🔴"
                lines.append(f"  {time_str}: {emoji} {rate_pct:+.4f}%")

            latest_rate = float(funding_data[-1]["fundingRate"])
            if latest_rate > 0.0005:
                lines.append("  📊 判读: 费率偏高，多头需支付空头，多头情绪过热 ⚠️")
            elif latest_rate < -0.0005:
                lines.append("  📊 判读: 费率为负，空头需支付多头，空头情绪过热，可能反弹 💡")
            else:
                lines.append("  📊 判读: 费率正常范围，多空平衡")
            lines.append("")
        except Exception as exc:
            lines.append(f"  ❌ 资金费率获取失败: {exc}\n")

        # 2. 持仓量
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": symbol},
                timeout=10,
            )
            resp.raise_for_status()
            oi_data = resp.json()
            oi = float(oi_data["openInterest"])

            lines.append("**📊 未平仓合约量 (Open Interest)**")
            lines.append(f"  OI = {oi:,.2f} {symbol.replace('USDT', '')}")

            # 获取当前价格估算美元价值
            try:
                price_resp = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": symbol},
                    timeout=5,
                )
                price = float(price_resp.json()["price"])
                oi_usd = oi * price
                lines.append(f"  OI (USD) ≈ ${oi_usd:,.0f}")
            except Exception:
                pass
            lines.append("")
        except Exception as exc:
            lines.append(f"  ❌ 持仓量获取失败: {exc}\n")

        # 3. 多空比（最近 5 期，1h 粒度）
        try:
            resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "1h", "limit": 5},
                timeout=10,
            )
            resp.raise_for_status()
            ls_data = resp.json()

            lines.append("**⚖️ 多空账户比（最近 5 小时）**")
            from datetime import datetime as _dt
            for item in ls_data:
                ts = int(item["timestamp"]) / 1000
                time_str = _dt.fromtimestamp(ts).strftime("%m-%d %H:%M")
                long_pct = float(item["longAccount"]) * 100
                short_pct = float(item["shortAccount"]) * 100
                ratio = float(item["longShortRatio"])
                lines.append(f"  {time_str}: 多 {long_pct:.1f}% | 空 {short_pct:.1f}% | 比值 {ratio:.2f}")

            latest_ratio = float(ls_data[-1]["longShortRatio"])
            if latest_ratio > 2.0:
                lines.append("  📊 判读: 多头占比过高，需警惕多杀多 ⚠️")
            elif latest_ratio < 0.8:
                lines.append("  📊 判读: 空头占优，但可能引发空头回补反弹 💡")
            else:
                lines.append("  📊 判读: 多空比正常范围")
        except Exception as exc:
            lines.append(f"  ❌ 多空比获取失败: {exc}")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="币种名称或代码（如 BTC、ETH），默认 BTC",
                required=False,
            )
        ]


# ============================================================
# 便捷函数
# ============================================================

def get_crypto_price(query: str) -> str:
    """查询加密货币价格，如 get_crypto_price('BTC,ETH')"""
    tool = CryptoMarketTool()
    return tool.run({"input": query})  # type: ignore[return-value]


def get_fear_greed(days: str = "7") -> str:
    """查询恐惧贪婪指数，如 get_fear_greed('7')"""
    tool = FearGreedTool()
    return tool.run({"input": days})  # type: ignore[return-value]


def get_technical(query: str = "BTC 1h") -> str:
    """查询技术指标，如 get_technical('BTC 1h') 或 get_technical('ETH 4h')"""
    tool = TechnicalIndicatorTool()
    return tool.run({"input": query})  # type: ignore[return-value]


def get_futures_data(query: str = "BTC") -> str:
    """查询合约数据（资金费率/持仓量/多空比），如 get_futures_data('BTC')"""
    tool = FuturesDataTool()
    return tool.run({"input": query})  # type: ignore[return-value]


def get_crypto_analysis(query: str = "BTC 1h") -> str:
    """【快捷】一次并行获取价格+技术指标+恐惧贪婪+合约数据，大幅减少等待时间。
    输入格式：币种 周期，如 crypto_analysis[BTC 1h]、crypto_analysis[ETH 4h]、crypto_analysis[SOL]
    周期缺省时默认 1h。"""
    parts = (query or "BTC 1h").strip().split()
    coin = parts[0] if parts else "BTC"
    interval_raw = parts[1] if len(parts) > 1 else "1h"

    def _price():
        return get_crypto_price(coin)

    def _technical():
        return get_technical(f"{coin} {interval_raw}")

    def _fear():
        return get_fear_greed("7")

    def _futures():
        return get_futures_data(coin)

    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(_price): "price",
            ex.submit(_technical): "technical",
            ex.submit(_fear): "fear_greed",
            ex.submit(_futures): "futures",
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                results[key] = f"❌ {key} 获取失败: {e}"

    sections = [
        results.get("price", ""),
        results.get("technical", ""),
        results.get("fear_greed", ""),
        results.get("futures", ""),
    ]
    return "\n\n---\n\n".join(s for s in sections if s)
