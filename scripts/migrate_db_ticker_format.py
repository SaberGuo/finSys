#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移脚本：将 zz500_data.db 中的股票代码从 baostock 格式转换为配置文件格式

旧格式: sh.600000, sz.000001
新格式: 600000.SH, 000001.SZ

用法:
    python scripts/migrate_db_ticker_format.py
"""

import sqlite3
from pathlib import Path


def convert_code(old_code: str) -> str:
    """将 baostock 格式转换为配置文件格式

    sh.600000 -> 600000.SH
    sz.000001 -> 000001.SZ
    """
    if '.' not in old_code:
        return old_code

    parts = old_code.split('.')
    if len(parts) != 2:
        return old_code

    exchange, stock_num = parts
    exchange_upper = exchange.upper()

    # 如果已经是新格式，直接返回
    if stock_num.isdigit() and len(stock_num) == 6:
        return f"{stock_num}.{exchange_upper}"

    # 如果是旧格式 (sh.600000)
    if exchange.lower() in ['sh', 'sz'] and len(parts[1]) == 6:
        return f"{parts[1]}.{exchange_upper}"

    return old_code


def migrate_database(db_path: Path):
    """迁移数据库中的股票代码格式"""
    if not db_path.exists():
        print(f"数据库不存在: {db_path}")
        return

    # 使用 WAL 模式和更长的超时时间
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    print("开始迁移数据库股票代码格式...")
    print("=" * 60)

    # 1. 迁移 minute_data 表
    print("\n处理 minute_data 表...")
    cursor.execute("SELECT DISTINCT code FROM minute_data WHERE code LIKE '%.%'")
    minute_codes = [row[0] for row in cursor.fetchall()]

    minute_updated = 0
    for old_code in minute_codes:
        new_code = convert_code(old_code)
        if new_code != old_code:
            # 检查新代码是否已存在
            cursor.execute(
                "SELECT COUNT(*) FROM minute_data WHERE code = ?",
                (new_code,)
            )
            if cursor.fetchone()[0] > 0:
                print(f"  跳过 {old_code} -> {new_code} (新代码已存在)")
                continue

            cursor.execute(
                "UPDATE minute_data SET code = ? WHERE code = ?",
                (new_code, old_code)
            )
            minute_updated += cursor.rowcount
            print(f"  更新 {old_code} -> {new_code} ({cursor.rowcount} 条记录)")

    # 2. 迁移 daily_data 表
    print("\n处理 daily_data 表...")
    cursor.execute("SELECT DISTINCT code FROM daily_data WHERE code LIKE '%.%'")
    daily_codes = [row[0] for row in cursor.fetchall()]

    daily_updated = 0
    for old_code in daily_codes:
        new_code = convert_code(old_code)
        if new_code != old_code:
            # 检查新代码是否已存在
            cursor.execute(
                "SELECT COUNT(*) FROM daily_data WHERE code = ?",
                (new_code,)
            )
            if cursor.fetchone()[0] > 0:
                print(f"  跳过 {old_code} -> {new_code} (新代码已存在)")
                continue

            cursor.execute(
                "UPDATE daily_data SET code = ? WHERE code = ?",
                (new_code, old_code)
            )
            daily_updated += cursor.rowcount
            print(f"  更新 {old_code} -> {new_code} ({cursor.rowcount} 条记录)")

    # 3. 迁移 metadata 表
    print("\n处理 metadata 表...")
    cursor.execute("SELECT DISTINCT code FROM metadata WHERE code LIKE '%.%'")
    meta_codes = [row[0] for row in cursor.fetchall()]

    meta_updated = 0
    for old_code in meta_codes:
        new_code = convert_code(old_code)
        if new_code != old_code:
            # 检查新代码是否已存在
            cursor.execute(
                "SELECT COUNT(*) FROM metadata WHERE code = ?",
                (new_code,)
            )
            if cursor.fetchone()[0] > 0:
                print(f"  跳过 {old_code} -> {new_code} (新代码已存在)")
                continue

            cursor.execute(
                "UPDATE metadata SET code = ? WHERE code = ?",
                (new_code, old_code)
            )
            meta_updated += cursor.rowcount
            print(f"  更新 {old_code} -> {new_code}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("迁移完成!")
    print(f"minute_data: {minute_updated} 条记录")
    print(f"daily_data: {daily_updated} 条记录")
    print(f"metadata: {meta_updated} 条记录")
    print("=" * 60)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "zz500_data.db"

    migrate_database(DB_PATH)
