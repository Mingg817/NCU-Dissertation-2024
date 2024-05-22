#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   crawler.py
@Time    :   2024/05/20
@Author  :   LI YIMING 
@Version :   1.0
@Site    :   https://github.com/Mingg817
@Desc    :   爬虫程序
'''


import logging

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='news.log',
                    filemode='a')

import sqlite3

con = sqlite3.connect("news.db")

# 创建页面对象
from DrissionPage import WebPage

page = WebPage()


def handel_sub_page(herf: str):
    logging.info(f"Handel sub page: {herf}")
    page.wait(0.5, 1)
    tab = page.new_tab()
    tab.get(herf)
    time_ele = tab.ele("""xpath://*[@id="newscontent"]/div[2]/div[2]/div/div[2]""")
    time = time_ele.text
    main_text_ls = []
    main_text_eles = tab.ele("""#newscontent""").eles('tag:p')
    for ele in main_text_eles:
        if ele.text.strip():
            main_text_ls.append(ele.text.strip())
    tab.close()
    main_text = '\n'.join(main_text_ls)

    logging.info(f"Time: {time}")
    logging.info(f"Main text: {main_text}")

    return time, main_text


def insert_data(data):
    logging.info(f"Insert data: {data}")
    con.execute("INSERT INTO news VALUES(?, ?, ?, ?, ?, ?)", data)
    con.commit()

def handel_main_page(path: str):
    # 访问网址
    page.get(path)
    page.change_mode()

    # 查找表格元素
    table = page.ele("""xpath://*[@id="mainlist"]/div/ul/li[1]/table/tbody""")
    logging.info(f"Handel table: {table}")

    # 查找行元素
    rows = table.eles('tag:tr')

    for r in rows:
        logging.info(f"Handel row: {r}")
        read_ele = r('.read')
        reply_ele = r('.reply')
        title_ele = r('.title')
        herf_ele = r('tag:a')
        # 检查是否已经存在
        check_herf_exist = con.execute(f"SELECT * FROM news WHERE herf = '{herf_ele.link}'").fetchone()
        if check_herf_exist:
            logging.info(f"Herf {herf_ele.link} exist")
            continue
        # 处理子页面,获得时间和正文
        time, main_text = handel_sub_page(herf_ele.link)
        # 将数据插入数据库
        insert_data((read_ele.text, reply_ele.text, title_ele.text, time, main_text, herf_ele.link))


path = [f'https://guba.eastmoney.com/list,601988,1,f_{i}.html' for i in range(2, 26)]

for p in path:
    logging.info(f"Handel main page: {p}")
    handel_main_page(p)
    page.wait(1, 2)

# handel_sub_page("https://guba.eastmoney.com/news,601988,1422679952.html")
