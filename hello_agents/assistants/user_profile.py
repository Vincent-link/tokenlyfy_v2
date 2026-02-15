"""用户画像 - 持久化用户投研偏好，供报告生成时注入"""

import json
import os
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List

logger = logging.getLogger(__name__)

# 默认存储目录（与 session 同基目录）
def _profile_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
    return os.path.abspath(os.path.join(base, "memory_data", "profiles"))


@dataclass
class UserProfile:
    """用户投研画像"""
    user_id: str
    coins: List[str] = field(default_factory=list)       # 主要关注币种，如 ["BTC", "ETH"]
    timeframe: str = ""                                   # 短线/中线/长线
    risk_preference: str = ""                             # 保守/中性/激进
    notes: str = ""                                       # 其他备注

    def to_summary(self, max_len: int = 400) -> str:
        """生成可注入 prompt 的摘要"""
        parts = []
        if self.coins:
            parts.append(f"主要关注币种：{', '.join(self.coins)}")
        if self.timeframe:
            parts.append(f"偏好周期：{self.timeframe}")
        if self.risk_preference:
            parts.append(f"风险偏好：{self.risk_preference}")
        if self.notes:
            parts.append(f"备注：{self.notes}")
        if not parts:
            return ""
        s = "；".join(parts)
        return s[:max_len] + "…" if len(s) > max_len else s


class UserProfileStore:
    """按 user_id 持久化/读取用户画像（文件存储）"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or _profile_dir()
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe = re.sub(r"[^\w\-]", "_", user_id)
        return os.path.join(self.base_dir, f"{safe}.json")

    def get(self, user_id: str) -> Optional[UserProfile]:
        try:
            p = self._path(user_id)
            if not os.path.exists(p):
                return None
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return UserProfile(
                user_id=d.get("user_id", user_id),
                coins=d.get("coins", []),
                timeframe=d.get("timeframe", ""),
                risk_preference=d.get("risk_preference", ""),
                notes=d.get("notes", ""),
            )
        except Exception as e:
            logger.debug(f"读取用户画像失败 {user_id}: {e}")
            return None

    def set(self, profile: UserProfile) -> None:
        try:
            p = self._path(profile.user_id)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入用户画像失败 {profile.user_id}: {e}")

    def update(self, user_id: str, **kwargs) -> UserProfile:
        """部分更新，未传字段保持原值"""
        profile = self.get(user_id) or UserProfile(user_id=user_id)
        for k, v in kwargs.items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        self.set(profile)
        return profile

    def update_from_memory_content(self, user_id: str, content: str) -> bool:
        """从记忆存储内容中简单抽取偏好并更新画像（启发式）"""
        if not content or len(content) < 2:
            return False
        content_lower = content.lower().strip()
        new_coins: List[str] = []
        for c, sym in [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL"), ("sui", "SUI"), ("bnb", "BNB")]:
            if c in content_lower or sym in content:
                new_coins.append(sym)
        timeframe = ""
        if "短线" in content:
            timeframe = "短线"
        elif "中线" in content or "中长线" in content:
            timeframe = "中线"
        elif "长线" in content:
            timeframe = "长线"
        risk = ""
        if "保守" in content or "稳健" in content:
            risk = "保守"
        elif "激进" in content:
            risk = "激进"
        elif "中性" in content:
            risk = "中性"
        if not new_coins and not timeframe and not risk:
            return False
        profile = self.get(user_id) or UserProfile(user_id=user_id)
        if new_coins:
            profile.coins = list(dict.fromkeys((profile.coins or []) + new_coins))[:10]
        if timeframe:
            profile.timeframe = timeframe
        if risk:
            profile.risk_preference = risk
        self.set(profile)
        return True
