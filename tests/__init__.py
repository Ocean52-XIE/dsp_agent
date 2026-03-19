# -*- coding: utf-8 -*-
"""测试包初始化

确保在导入任何测试模块前设置好 sys.path。
"""
import sys
from pathlib import Path

# 设置 src 路径
_src_path = Path(__file__).resolve().parents[1] / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
