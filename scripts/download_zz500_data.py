#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中证500成分股数据下载脚本
支持5分钟和日线数据下载，存储到SQLite数据库，支持增量更新

用法:
    python download_zz500_data.py --init --start 2020-01-01 --end 2026-01-01
    python download_zz500_data.py --update
    python download_zz500_data.py --status
    python download_zz500_data.py --validate
"""

import baostock as bs
import pandas as pd
import sqlite3
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "zz500_data.db"

def convert_to_baostock_format(code: str) -> str:
    """
    将配置文件格式（000001.SZ）转换为baostock格式（sz.000001）

    Args:
        code: 配置文件格式的股票代码，如 "000001.SZ" 或 "600000.SH"

    Returns:
        baostock格式的股票代码，如 "sz.000001" 或 "sh.600000"
    """
    if '.' not in code:
        return code

    stock_num, exchange = code.split('.')
    exchange_lower = exchange.lower()
    return f"{exchange_lower}.{stock_num}"


def convert_from_baostock_format(code: str) -> str:
    """
    将baostock格式（sz.000001）转换为配置文件格式（000001.SZ）

    Args:
        code: baostock格式的股票代码，如 "sz.000001" 或 "sh.600000"

    Returns:
        配置文件格式的股票代码，如 "000001.SZ" 或 "600000.SH"
    """
    if '.' not in code:
        return code

    exchange, stock_num = code.split('.')
    exchange_upper = exchange.upper()
    return f"{stock_num}.{exchange_upper}"


# 导入股票列表（从配置文件格式）
def get_zz500_stocks():
    """获取中证500成分股 - 扩展版（355只典型成分股）

    返回配置文件格式的股票代码列表（如 "000001.SZ", "600000.SH"）
    """
    return [
        # 上海股票 - 金融/地产/消费
        "600000.SH", "600009.SH", "600016.SH", "600028.SH", "600030.SH",
        "600031.SH", "600036.SH", "600048.SH", "600050.SH", "600104.SH",
        "600115.SH", "600132.SH", "600143.SH", "600153.SH", "600157.SH",
        "600170.SH", "600177.SH", "600183.SH", "600188.SH", "600196.SH",
        "600208.SH", "600219.SH", "600221.SH", "600233.SH", "600271.SH",
        "600276.SH", "600297.SH", "600299.SH", "600307.SH", "600309.SH",
        "600315.SH", "600316.SH", "600320.SH", "600332.SH", "600340.SH",
        "600346.SH", "600352.SH", "600362.SH", "600369.SH", "600372.SH",
        "600373.SH", "600376.SH", "600383.SH", "600390.SH", "600395.SH",
        "600398.SH", "600406.SH", "600415.SH", "600436.SH", "600438.SH",
        "600446.SH", "600460.SH", "600466.SH", "600482.SH", "600489.SH",
        "600498.SH", "600499.SH", "600507.SH", "600516.SH", "600517.SH",
        "600521.SH", "600522.SH", "600528.SH", "600529.SH", "600535.SH",
        "600547.SH", "600549.SH", "600566.SH", "600570.SH", "600572.SH",
        "600582.SH", "600583.SH", "600584.SH", "600585.SH", "600588.SH",
        "600596.SH", "600597.SH", "600598.SH", "600600.SH", "600606.SH",
        "600612.SH", "600616.SH", "600619.SH", "600623.SH", "600633.SH",
        "600635.SH", "600637.SH", "600639.SH", "600641.SH", "600645.SH",
        "600648.SH", "600649.SH", "600655.SH", "600657.SH", "600660.SH",
        "600663.SH", "600666.SH", "600667.SH", "600673.SH", "600674.SH",
        "600675.SH", "600676.SH", "600682.SH", "600684.SH", "600685.SH",
        "600688.SH", "600690.SH", "600694.SH", "600699.SH", "600702.SH",
        "600703.SH", "600704.SH", "600705.SH", "600706.SH", "600711.SH",
        "600712.SH", "600718.SH", "600720.SH", "600723.SH", "600724.SH",
        "600729.SH", "600731.SH", "600733.SH", "600739.SH", "600740.SH",
        "600741.SH", "600742.SH", "600743.SH", "600745.SH", "600748.SH",
        "600750.SH", "600754.SH", "600755.SH", "600756.SH", "600757.SH",
        "600758.SH", "600759.SH", "600760.SH", "600761.SH", "600763.SH",
        "600764.SH", "600765.SH", "600766.SH", "600767.SH", "600768.SH",
        "600770.SH", "600771.SH", "600773.SH", "600774.SH", "600775.SH",
        "600776.SH", "600777.SH", "600778.SH", "600779.SH", "600780.SH",
        "600781.SH", "600782.SH", "600783.SH", "600784.SH", "600785.SH",
        "600787.SH", "600788.SH", "600789.SH", "600790.SH", "600791.SH",
        "600792.SH", "600793.SH", "600794.SH", "600795.SH", "600796.SH",
        "600797.SH", "600798.SH", "600800.SH", "600801.SH", "600802.SH",
        "600803.SH", "600804.SH", "600805.SH", "600806.SH", "600807.SH",
        "600808.SH", "600809.SH", "600810.SH", "600811.SH", "600812.SH",
        "600813.SH", "600814.SH", "600815.SH", "600816.SH", "600817.SH",
        "600818.SH", "600819.SH", "600820.SH", "600821.SH", "600822.SH",
        "600823.SH", "600824.SH", "600825.SH", "600826.SH", "600827.SH",
        "600828.SH", "600829.SH", "600830.SH", "600831.SH", "600832.SH",
        "600833.SH", "600834.SH", "600835.SH", "600836.SH", "600837.SH",
        "600838.SH", "600839.SH", "600841.SH", "600843.SH", "600844.SH",
        "600845.SH", "600846.SH", "600847.SH", "600848.SH", "600850.SH",
        "600851.SH", "600853.SH", "600855.SH", "600856.SH", "600857.SH",

        # 深圳主板 - 制造业/消费/医药
        "000001.SZ", "000002.SZ", "000063.SZ", "000100.SZ", "000333.SZ",
        "000538.SZ", "000568.SZ", "000651.SZ", "000725.SZ", "000768.SZ",
        "000858.SZ", "000895.SZ", "000921.SZ", "000938.SZ", "000963.SZ",
        "000977.SZ", "001289.SZ", "001979.SZ",

        # 中小板（002开头）
        "002001.SZ", "002007.SZ", "002008.SZ", "002027.SZ", "002044.SZ",
        "002049.SZ", "002081.SZ", "002120.SZ", "002129.SZ", "002142.SZ",
        "002146.SZ", "002179.SZ", "002180.SZ", "002202.SZ", "002230.SZ",
        "002236.SZ", "002252.SZ", "002271.SZ", "002304.SZ", "002311.SZ",
        "002340.SZ", "002352.SZ", "002371.SZ", "002384.SZ", "002385.SZ",
        "002410.SZ", "002414.SZ", "002415.SZ", "002422.SZ", "002456.SZ",
        "002460.SZ", "002463.SZ", "002466.SZ", "002468.SZ", "002475.SZ",
        "002493.SZ", "002508.SZ", "002555.SZ", "002568.SZ", "002572.SZ",
        "002594.SZ", "002600.SZ", "002601.SZ", "002602.SZ", "002607.SZ",
        "002624.SZ", "002648.SZ", "002673.SZ", "002714.SZ", "002736.SZ",
        "002812.SZ", "002821.SZ", "002841.SZ", "002916.SZ", "002938.SZ",
        "002939.SZ", "002958.SZ", "002959.SZ",

        # 创业板（300开头）
        "300003.SZ", "300014.SZ", "300015.SZ", "300033.SZ", "300058.SZ",
        "300059.SZ", "300072.SZ", "300122.SZ", "300124.SZ", "300136.SZ",
        "300142.SZ", "300144.SZ", "300207.SZ", "300223.SZ", "300251.SZ",
        "300253.SZ", "300274.SZ", "300296.SZ", "300316.SZ", "300357.SZ",
        "300408.SZ", "300413.SZ", "300418.SZ", "300433.SZ", "300442.SZ",
        "300454.SZ", "300496.SZ", "300498.SZ", "300529.SZ", "300558.SZ",
        "300595.SZ", "300601.SZ", "300628.SZ", "300676.SZ", "300682.SZ",
        "300750.SZ", "300751.SZ", "300760.SZ", "300769.SZ", "300782.SZ",
        "300832.SZ", "300866.SZ", "300888.SZ", "300896.SZ", "300919.SZ",
        "300999.SZ"
    ]


# ========== 数据库操作 ==========

def init_database(db_path):
    """初始化数据库，创建数据表和索引"""
    # 确保目录存在
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 5分钟数据表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS minute_data (
        code TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        amount REAL,
        PRIMARY KEY (code, date, time)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_minute_code_date ON minute_data(code, date)")

    # 日线数据表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_data (
        code TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        amount REAL,
        PRIMARY KEY (code, date)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_code_date ON daily_data(code, date)")

    # 元数据表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        code TEXT PRIMARY KEY,
        last_update_5min TEXT,
        last_update_daily TEXT,
        total_records_5min INTEGER DEFAULT 0,
        total_records_daily INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()
    print(f"✓ 数据库初始化完成: {db_path}")


def get_last_update_date(code, db_path, freq='5min'):
    """获取某股票最后更新日期"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    column = 'last_update_5min' if freq == '5min' else 'last_update_daily'
    cursor.execute(f"SELECT {column} FROM metadata WHERE code = ?", (code,))
    result = cursor.fetchone()

    conn.close()

    if result and result[0]:
        return result[0]
    return None


def update_metadata(code, db_path, last_date, freq='5min', record_count=None):
    """更新元数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查是否存在记录
    cursor.execute("SELECT code FROM metadata WHERE code = ?", (code,))
    exists = cursor.fetchone()

    if freq == '5min':
        if exists:
            cursor.execute(
                "UPDATE metadata SET last_update_5min = ?, total_records_5min = ? WHERE code = ?",
                (last_date, record_count, code)
            )
        else:
            cursor.execute(
                "INSERT INTO metadata (code, last_update_5min, total_records_5min) VALUES (?, ?, ?)",
                (code, last_date, record_count)
            )
    else:
        if exists:
            cursor.execute(
                "UPDATE metadata SET last_update_daily = ?, total_records_daily = ? WHERE code = ?",
                (last_date, record_count, code)
            )
        else:
            cursor.execute(
                "INSERT INTO metadata (code, last_update_daily, total_records_daily) VALUES (?, ?, ?)",
                (code, last_date, record_count)
            )

    conn.commit()
    conn.close()


def save_to_database(df, db_path, table_name):
    """将DataFrame保存到数据库（支持增量更新，分批插入避免SQL变量限制）"""
    if df.empty:
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQLite 默认参数限制为999，分批插入（每批100行比较安全）
    BATCH_SIZE = 100
    total_inserted = 0

    try:
        if table_name == 'minute_data':
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                for _, row in batch.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO minute_data (code, date, time, open, high, low, close, volume, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (row['code'], row['date'], row['time'], row['open'], row['high'],
                              row['low'], row['close'], row['volume'], row['amount']))
                        total_inserted += 1
                    except Exception:
                        pass
                conn.commit()
        else:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                for _, row in batch.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO daily_data (code, date, open, high, low, close, volume, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (row['code'], row['date'], row['open'], row['high'],
                              row['low'], row['close'], row['volume'], row['amount']))
                        total_inserted += 1
                    except Exception:
                        pass
                conn.commit()
    except Exception as e:
        print(f"  数据库插入错误: {e}")
    finally:
        conn.close()

    return total_inserted


def decode_baostock_msg(msg):
    """解码baostock的错误消息（处理GBK编码）"""
    if msg is None:
        return ''
    try:
        # 如果已经是字符串，直接返回
        if isinstance(msg, str):
            return msg
        # 如果是bytes，尝试GBK解码
        if isinstance(msg, bytes):
            return msg.decode('gbk', errors='ignore')
        return str(msg)
    except Exception:
        # 最后的fallback
        return str(msg)


# ========== 数据下载 ==========

def download_5min_data(symbol, start_date, end_date):
    """从baostock下载5分钟数据

    Args:
        symbol: 配置文件格式的股票代码（如 "000001.SZ"）
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with code in config format (e.g., "000001.SZ")
    """
    try:
        # 转换为baostock格式
        baostock_code = convert_to_baostock_format(symbol)

        rs = bs.query_history_k_data_plus(
            baostock_code,
            "date,time,code,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="3"
        )

        # 处理编码问题 - baostock 错误信息可能为 GBK 编码
        error_code = rs.error_code

        if error_code != '0':
            error_msg = decode_baostock_msg(getattr(rs, 'error_msg', ''))
            if error_msg:
                print(f"  API错误 [{symbol}]: {error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if len(data_list) == 0:
            return None

        df = pd.DataFrame(data_list, columns=['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        # 转换code为配置文件格式
        df['code'] = df['code'].apply(convert_from_baostock_format)

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna()

    except UnicodeDecodeError as e:
        print(f"  编码错误 [{symbol}]: {e}")
        return None
    except Exception as e:
        print(f"  下载5分钟数据失败 [{symbol}]: {e}")
        return None


def download_daily_data(symbol, start_date, end_date):
    """从baostock下载日线数据

    Args:
        symbol: 配置文件格式的股票代码（如 "000001.SZ"）
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with code in config format (e.g., "000001.SZ")
    """
    try:
        # 转换为baostock格式
        baostock_code = convert_to_baostock_format(symbol)

        rs = bs.query_history_k_data_plus(
            baostock_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )

        # 处理编码问题 - baostock 错误信息可能为 GBK 编码
        error_code = rs.error_code

        if error_code != '0':
            error_msg = decode_baostock_msg(getattr(rs, 'error_msg', ''))
            if error_msg:
                print(f"  API错误 [{symbol}]: {error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if len(data_list) == 0:
            return None

        df = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        # 使用配置文件格式的代码
        df['code'] = symbol

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna()

    except UnicodeDecodeError as e:
        print(f"  编码错误 [{symbol}]: {e}")
        return None
    except Exception as e:
        print(f"  下载日线数据失败 [{symbol}]: {e}")
        return None


def download_stock_data(code, db_path, end_date=None):
    """下载单只股票数据（增量更新）"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    results = {'code': code, 'status': 'success', '5min_records': 0, 'daily_records': 0}

    # 下载5分钟数据
    last_update_5min = get_last_update_date(code, db_path, '5min')
    if last_update_5min:
        start_5min = (datetime.strptime(last_update_5min, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_5min = '2020-01-01'

    if start_5min <= end_date:
        df_5min = download_5min_data(code, start_5min, end_date)
        if df_5min is not None and not df_5min.empty:
            records = save_to_database(df_5min, db_path, 'minute_data')
            results['5min_records'] = records
            last_date = df_5min['date'].max()
            update_metadata(code, db_path, last_date, '5min', records)
        else:
            results['status'] = 'no_data'

    # 下载日线数据
    last_update_daily = get_last_update_date(code, db_path, 'daily')
    if last_update_daily:
        start_daily = (datetime.strptime(last_update_daily, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_daily = '2020-01-01'

    if start_daily <= end_date:
        df_daily = download_daily_data(code, start_daily, end_date)
        if df_daily is not None and not df_daily.empty:
            records = save_to_database(df_daily, db_path, 'daily_data')
            results['daily_records'] = records
            last_date = df_daily['date'].max()
            update_metadata(code, db_path, last_date, 'daily', records)

    return results


def download_all_stocks(stock_list, db_path, end_date=None, max_workers=4):
    """批量下载所有股票数据"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"开始下载 {len(stock_list)} 只股票数据...")
    print(f"截止日期: {end_date}")
    print(f"并发数: {max_workers}")
    print("-" * 60)

    success_count = 0
    error_count = 0
    total_5min = 0
    total_daily = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {
            executor.submit(download_stock_data, code, db_path, end_date): code
            for code in stock_list
        }

        with tqdm(total=len(stock_list), desc="下载进度") as pbar:
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        success_count += 1
                        total_5min += result['5min_records']
                        total_daily += result['daily_records']
                except Exception as e:
                    print(f"\n  {code} 下载异常: {e}")
                    error_count += 1

                pbar.update(1)

    print("-" * 60)
    print(f"下载完成: 成功 {success_count}, 失败 {error_count}")
    print(f"新增5分钟记录: {total_5min}")
    print(f"新增日线记录: {total_daily}")


# ========== 数据验证和统计 ==========

def show_data_status(db_path):
    """显示数据状态"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("数据状态报告")
    print("=" * 60)

    # 统计5分钟数据
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM minute_data")
    result = cursor.fetchone()
    if result and result[0]:
        print(f"\n5分钟数据:")
        print(f"  总记录数: {result[0]:,}")
        print(f"  股票数量: {result[1]}")
        print(f"  日期范围: {result[2]} ~ {result[3]}")

    # 统计日线数据
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM daily_data")
    result = cursor.fetchone()
    if result and result[0]:
        print(f"\n日线数据:")
        print(f"  总记录数: {result[0]:,}")
        print(f"  股票数量: {result[1]}")
        print(f"  日期范围: {result[2]} ~ {result[3]}")

    # 元数据状态
    cursor.execute("SELECT COUNT(*) FROM metadata")
    meta_count = cursor.fetchone()[0]
    print(f"\n元数据: {meta_count} 只股票")

    # 显示每只股票的最新更新日期
    cursor.execute("""
        SELECT code, last_update_5min, last_update_daily,
               total_records_5min, total_records_daily
        FROM metadata
        ORDER BY code
        LIMIT 20
    """)
    rows = cursor.fetchall()

    print(f"\n前20只股票更新状态:")
    print("-" * 80)
    print(f"{'股票代码':<15} {'5分钟最后更新':<15} {'日线最后更新':<15} {'5分记录':<10} {'日线记录':<10}")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:<15} {str(row[1]):<15} {str(row[2]):<15} {row[3]:<10} {row[4]:<10}")

    if meta_count > 20:
        print(f"... 还有 {meta_count - 20} 只股票 ...")

    conn.close()
    print("=" * 60)


def validate_data(db_path):
    """验证数据完整性"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("数据完整性验证")
    print("=" * 60)

    issues = []

    # 检查价格异常
    cursor.execute("""
        SELECT code, date, time, open, high, low, close
        FROM minute_data
        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
        LIMIT 10
    """)
    zero_prices = cursor.fetchall()
    if zero_prices:
        issues.append(f"发现 {len(zero_prices)} 条价格<=0的记录")
        print("\n⚠ 价格异常记录:")
        for row in zero_prices:
            print(f"  {row}")

    # 检查OHLC逻辑异常
    cursor.execute("""
        SELECT code, date, time, open, high, low, close
        FROM minute_data
        WHERE high < open OR high < close OR low > open OR low > close
        LIMIT 10
    """)
    ohlc_issues = cursor.fetchall()
    if ohlc_issues:
        issues.append(f"发现 {len(ohlc_issues)} 条OHLC逻辑异常记录")
        print("\n⚠ OHLC逻辑异常记录:")
        for row in ohlc_issues:
            print(f"  {row}")

    # 检查缺失数据（简化检查：每只股票应该有相似的数据量）
    cursor.execute("""
        SELECT code, COUNT(*) as cnt
        FROM minute_data
        GROUP BY code
        ORDER BY cnt ASC
        LIMIT 5
    """)
    low_volume = cursor.fetchall()
    if low_volume:
        print("\n⚠ 数据量最少的5只股票:")
        for row in low_volume:
            print(f"  {row[0]}: {row[1]} 条记录")

    if not issues:
        print("\n✓ 数据验证通过，未发现明显异常")

    conn.close()
    print("=" * 60)


# ========== 主程序 ==========

def main():
    parser = argparse.ArgumentParser(description='中证500成分股数据下载工具')
    parser.add_argument('--db', type=str, default=str(DEFAULT_DB_PATH), help='数据库文件路径')
    parser.add_argument('--init', action='store_true', help='初始化数据库并下载全部数据')
    parser.add_argument('--update', action='store_true', help='增量更新数据')
    parser.add_argument('--status', action='store_true', help='查看数据状态')
    parser.add_argument('--validate', action='store_true', help='验证数据完整性')
    parser.add_argument('--start', type=str, default='20206-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=1, help='并发下载数')

    args = parser.parse_args()

    # 检查数据库文件路径
    db_path = args.db

    if args.init:
        # 初始化数据库
        init_database(db_path)

        # 登录baostock
        print("\n登录baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            error_msg = decode_baostock_msg(lg.error_msg)
            print(f"登录失败: {error_msg}")
            sys.exit(1)
        print("登录成功")

        try:
            # 获取股票列表
            stock_list = get_zz500_stocks()
            print(f"股票池: {len(stock_list)} 只")
            print(f"股票代码格式: 配置文件格式 (如 000001.SZ, 600000.SH)")

            # 批量下载
            end_date = args.end or datetime.now().strftime('%Y-%m-%d')
            download_all_stocks(stock_list, db_path, end_date, args.workers)

        finally:
            bs.logout()
            print("\nbaostock已退出")

    elif args.update:
        # 增量更新
        if not os.path.exists(db_path):
            print(f"数据库不存在: {db_path}")
            print("请先使用 --init 初始化数据库")
            sys.exit(1)

        print("登录baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            error_msg = decode_baostock_msg(lg.error_msg)
            print(f"登录失败: {error_msg}")
            sys.exit(1)

        try:
            stock_list = get_zz500_stocks()
            end_date = args.end or datetime.now().strftime('%Y-%m-%d')
            download_all_stocks(stock_list, db_path, end_date, args.workers)
        finally:
            bs.logout()
            print("\nbaostock已退出")

    elif args.status:
        if not os.path.exists(db_path):
            print(f"数据库不存在: {db_path}")
            sys.exit(1)
        show_data_status(db_path)

    elif args.validate:
        if not os.path.exists(db_path):
            print(f"数据库不存在: {db_path}")
            sys.exit(1)
        validate_data(db_path)

    else:
        parser.print_help()
        print("\n示例:")
        print("  python download_zz500_data.py --init --start 2020-01-01 --end 2026-01-01")
        print("  python download_zz500_data.py --update")
        print("  python download_zz500_data.py --status")
        print("  python download_zz500_data.py --validate")
        print(f"\n默认数据库路径: {DEFAULT_DB_PATH}")
        print("股票代码格式: 配置文件格式 (如 000001.SZ, 600000.SH)")


if __name__ == "__main__":
    main()
