"""Đảm bảo `src` import được khi chạy pytest từ bất kỳ thư mục nào (giống scripts/)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
