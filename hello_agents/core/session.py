"""轻量会话模块 - 为 MVP 提供匿名用户 ID，支持 session / 匿名 ID 模式。"""

import uuid
from pathlib import Path

_SESSION_FILE = "session_id"
# 使用包目录相对路径，避免 cwd 变化导致 session 丢失
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STORAGE_DIR = str(_PACKAGE_ROOT.parent / "memory_data")
_PREFIX = "anon_"


def get_anonymous_user_id(persist: bool = True, storage_dir: str = None) -> str:
    """获取匿名用户 ID。

    两种模式：
    - ephemeral (persist=False): 每次调用生成新 ID，不持久化，适用于测试、演示
    - persisted (persist=True): 首次生成并写入 storage_dir/session_id；后续调用读取已有 ID，同一目录 = 同一用户

    Args:
        persist: 是否持久化到本地文件
        storage_dir: 持久化目录，默认 memory_data

    Returns:
        匿名 ID，格式 anon_<uuid12>，如 anon_a1b2c3d4e5f6
    """
    if not persist:
        return _generate_id()

    base = Path(storage_dir) if storage_dir else Path(_DEFAULT_STORAGE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    path = base / _SESSION_FILE

    try:
        if path.exists():
            content = path.read_text().strip()
            if content and content.startswith(_PREFIX):
                return content
    except (OSError, IOError):
        pass

    # 首次或文件损坏/为空：生成并写入
    user_id = _generate_id()
    try:
        path.write_text(user_id, encoding="utf-8")
    except (OSError, IOError):
        pass
    return user_id


def reset_session(storage_dir: str = None) -> None:
    """清除持久化的匿名 ID，下次 get_anonymous_user_id(persist=True) 将生成新 ID。

    用于「切换用户」等场景。

    Args:
        storage_dir: 持久化目录，默认 memory_data
    """
    base = Path(storage_dir) if storage_dir else Path(_DEFAULT_STORAGE_DIR)
    path = base / _SESSION_FILE
    try:
        if path.exists():
            path.unlink()
    except (OSError, IOError):
        pass


def _generate_id() -> str:
    """生成 anon_<uuid12> 格式的匿名 ID。"""
    return _PREFIX + uuid.uuid4().hex[:12]
