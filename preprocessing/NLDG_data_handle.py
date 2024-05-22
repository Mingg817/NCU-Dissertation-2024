#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   NLDG_data_handle.py
@Time    :   2024/05/20
@Author  :   LI YIMING
@Version :   1.0
@Site    :   https://github.com/Mingg817
@Desc    :   处理新闻数据   
"""

import sqlite3
from typing import Tuple
from retry import retry
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd


def insert_data(con, data):
    """
        create table news
    (
        date      TEXT,
        text      TEXT primary key
    )
    """
    print(f"Insert data: {data}")
    assert len(data) == 2
    con.execute("INSERT INTO news VALUES(?, ?)", data)
    con.commit()


@retry(Exception, tries=3, delay=1)
def news_handel(model, d_news: Tuple[str, str]):
    messages = [
        {
            "role": "system",
            "content": "现在你是一个金融评论员,你可以通过新闻来分析股市走势.你的目标是结合以下新闻,提取**可能影响股价的信息**,写出**新闻总结**.",
        },
        {"role": "user", "content": "以下是新闻数据.\n------\n"},
    ]
    messages.extend(
        [
            {"role": "user", "content": f"标题`{title}`\n------\n{main_text}"}
            for title, main_text in d_news
        ]
    )

    input_ids_A = LLM_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )[:, :3584]

    messages = [{"role": "user", "content": "\n------\n请做出总结"}]

    input_ids_B = LLM_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )

    input_ids = torch.concat([input_ids_A, input_ids_B], dim=1).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    return LLM_tokenizer.decode(response, skip_special_tokens=True)


if __name__ == "__main__":
    # 利用数据库存储数据
    con = sqlite3.connect("news.db")

    # 加载模型
    LLM_model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"
    LLM_tokenizer = AutoTokenizer.from_pretrained(LLM_model_id)
    LLM_model = AutoModelForCausalLM.from_pretrained(
        LLM_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    BERT_model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    BERT_tokenizer = AutoTokenizer.from_pretrained(BERT_model_id)
    BERT_model = AutoModel.from_pretrained(BERT_model_id)

    # 读取数据,不使用SQL语句,直接使用pandas
    # 原始数据
    data = pd.read_csv("601998_news.csv")
    grouped = data.groupby(pd.Grouper(key="time", freq="D"))
    # 利用llm模型处理新闻
    for name, group in grouped:
        if con.execute(f"SELECT * FROM news WHERE date='{name.date()}'").fetchone():
            print(f"日期: {name.date()} 已经插入")
            continue
        print(f"日期: {name.date()}")

        insert_data(
            con,
            (
                name.date(),
                news_handel(LLM_model, group[["title", "main_text"]].values.tolist()),
            ),
        )
        print(f"日期: {name.date()} 插入成功")

    # 利用bert模型处理新闻
    for date in con.execute("SELECT date FROM news").fetchall():
        date = date[0]
        print(f"正在处理{date}")
        d_news = con.execute(f"SELECT text FROM news WHERE date='{date}'").fetchone()[0]
        sentiment = (
            BERT_model(
                BERT_tokenizer.encode(
                    d_news, return_tensors="pt", max_length=512, truncation=True
                )
            )
            .last_hidden_state[0, 0, :]
            .tolist()
        )
        con.execute(f"UPDATE news SET berted=? WHERE date='{date}'", (str(sentiment),))
        con.commit()
