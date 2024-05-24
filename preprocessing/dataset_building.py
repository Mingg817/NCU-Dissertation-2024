#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   dataset_building.py
@Time    :   2024/05/20
@Author  :   LI YIMING
@Version :   1.0
@Site    :   https://github.com/Mingg817
@Desc    :   数据集构建
"""

# 导入tushare
import tushare as ts
import pandas as pd
import numpy as np
import sqlite3
import json
import datasets
from functools import cache

ts.set_token("?")

con = sqlite3.connect("news.db")

window_size = 32
y_size = 3
skip_step = 3


@cache
def get_berted_news(date: str):
    return json.loads(
        con.execute(f"SELECT berted FROM news WHERE date='{date}'").fetchone()[0]
    )


def stock2parquet(ts_code: str, start_date: str, end_date: str, train: bool):
    df = ts.pro_bar(
        ts_code=ts_code, adj="hfq", start_date=start_date, end_date=end_date
    )
    # 判断数据是否为空
    assert len(df) > 0, "数据为空"
    # 按日期排序
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(by="trade_date")
    # 数据差分:先取log再差分
    df["close_log"] = df["close"].apply(lambda x: np.log(x))
    df["close_diff"] = df["close_log"].diff()
    # 去除空数据
    df = df.dropna()
    # 将数据归一化至[-1,1]
    df["close_norm"] = df["close_diff"] * 10
    # 滑动窗口截取训练数据
    start_date, x, y, y_avg, x_news, y_news = [], [], [], [], [], []
    for i in range(0, len(df) - window_size - y_size, skip_step):
        start_date.append(df["trade_date"].iloc[i].date().strftime("%Y-%m-%d"))
        # 精度控制在float16
        x.append(
            df["close_norm"]
            .iloc[i : i + window_size]
            .values.astype(np.float16)
            .tolist()
        )
        y.append(
            df["close_norm"]
            .iloc[i + window_size : i + window_size + y_size]
            .values.astype(np.float16)
            .tolist()
        )
        # 构建y_avg方便后续计算损失
        y_avg.append(
            df["close_norm"]
            .iloc[i + window_size : i + window_size + y_size]
            .mean()
            .tolist()
        )
        # 从数据库获取新闻数据
        news_line = []
        for date in df["trade_date"].iloc[i : i + window_size]:
            news_line.append(
                np.array(
                    get_berted_news(date.strftime("%Y-%m-%d"))
                    if get_berted_news(date.strftime("%Y-%m-%d"))
                    else [0] * 768
                )
            )
        x_news.append(np.array(news_line))
        news_line = []
        for date in df["trade_date"].iloc[i + window_size : i + window_size + y_size]:
            news_line.append(
                np.array(
                    get_berted_news(date.strftime("%Y-%m-%d"))
                    if get_berted_news(date.strftime("%Y-%m-%d"))
                    else [0] * 768
                )
            )
        y_news.append(np.array(news_line))

    # news=np.array(news).reshape(len(news),-1).tolist()
    ds = datasets.Dataset.from_dict(
        {
            "start_date": start_date,
            "ts_code": [ts_code] * len(start_date),
            "x": x,
            "y_hat": y,
            "y_avg": y_avg,
            "x_news": x_news,
            "y_news": y_news,
            "window_size": [window_size] * len(start_date),
        }
    )
    ds.to_parquet(f"../dataset/{ts_code}_{'train' if train else 'test'}_news.parquet")


if __name__ == "__main__":
    # 利用滑动窗口截取训练数据
    window_size = 32
    y_size = 3
    skip_step = 1
    train_start_date = "20191022"
    train_end_date = "202301101"
    eval_start_date = "20240101"
    eval_end_date = "20240501"

    stock2parquet("601988.SH", train_start_date, train_end_date, True)
    stock2parquet("601988.SH", eval_start_date, eval_end_date, False)
