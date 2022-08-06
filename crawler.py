import math
import pandas as pd
import numpy as np
import requests
import time
from lxml import etree
import datetime
import re
import json
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import warnings
from tqdm import tqdm
from loguru import logger
from filter import df_reader, df_writer


def read_json(config_path="config.json"):
    """
    读取配置文件
    :param config_path: Any
        路径
    :return: json
        配置文件
    """
    with open(config_path, "r") as k:
        config_json = json.load(k)
    return config_json


def write_json(fig_dic):
    """
    写入参数
    :param fig_dic: dict
        参数
    :return: None
    """
    config_json = json.dumps(fig_dic)
    with open("config.json", "w") as f:
        f.write(config_json)


CONFIGS = read_json()
warnings.filterwarnings('ignore')


class BasicPrep:

    @staticmethod
    def url_get(s, url):
        req = s.get(url)
        req_text = req.text
        return req_text

    @staticmethod
    def init_file(init_path, cols):
        df_init = pd.DataFrame(columns=cols)
        df_init.to_csv(init_path, index=False)

    @staticmethod
    def adding_data(to_path, df_adding):
        df_adding.to_csv(to_path, index=False, mode='a', header=False)


class InitFundList(BasicPrep):

    def get_fund_list(self):
        path_out = CONFIGS["path_all_fund"]
        # 获取所有基金信息
        s = requests.Session()
        # TIPS: 此项目所用爬虫均遵守了网站的Robots.txt协议，且数据仅用于学术探讨，无商业用途！下同
        url = "http://fund.eastmoney.com/js/fundcode_search.js"
        req_text = self.url_get(s, url)
        req_text = eval(req_text[8:-1])  # del useless str
        df_raw = pd.DataFrame(req_text)
        cols = ["基金代码", "基金名称缩写", "基金名称", "基金类型", "基金全称"]
        df_raw.columns = cols
        df_raw.to_csv(path_out, index=False)
        return df_raw


class InitAssetValues(BasicPrep):
    def get_value(self, s, code, page, start_date="2000-01-01"):
        """
        获取单个基金数据
        :param s: requests.Session
        :param code: str
            基金代码
        :param page: int
            页数（调用中的参数）
        :param start_date: date
            数据开始日期
        :return: str
            网页原始数据
        """
        # 获取单个基金指定页数的历史数据
        end_date = "2030-01-01"
        url = "https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&" \
              "code={code}&page={page}&sdate={sdate}&edate={edate}&per=20" \
            .format(code=code, page=page, sdate=start_date, edate=end_date)
        req_text = self.url_get(s, url)
        return req_text

    def get_values(self, s, code, page):
        """
        获取基金的所有指定page的数据
        :param s: requests.Session
        :param code: str
            基金代码
        :param page: int
            请求页数
        :return: list
            基金净值数据
        """
        # 获取单个基金指定页数的历史数据并解析结果
        try:
            info = []
            req_text = self.get_value(s, code, page)
            time.sleep(0.1)
            res_html = re.findall('content:"(.*?)",records:', req_text)[0]
            html = etree.HTML(res_html)
            html_data = html.xpath('//tr/td')
            for j in html_data:
                info.append(j.text)
            return info
        except ConnectionResetError:
            logger.info("访问被拒, 代码为{code}, 页数为{page}", code=code, page=page)
            raise ConnectionResetError
        except Exception as e:
            logger.info("访问错误，代码{code}, 页数{page}, 错误类型{e}", code=code, page=page, e=repr(e))

    def get_code_values(self, code, last_all_num=1000):
        """
        获取单个基金的指定数量的历史数据
        :param code: str
            基金代码
        :param last_all_num: int
            指定获取的全量数据
        :return: (基金数据，基金类型)
        """
        s = requests.Session()
        # 首次访问，得到总记录数、总页数
        req_solo = self.get_value(s, code=code, page=1)
        # 查找记录数
        pages_num = int(re.findall('pages:(.*?),', req_solo)[0])  # 包含所有数据的页数
        # 遍历数据
        spc_page = math.ceil(last_all_num / 20)
        pages = spc_page if pages_num > spc_page else pages_num  # 如果全量数据比指定的数量少，则以全量数据为准
        page_num = list(range(1, int(pages) + 1))
        # 多线程爬取
        para = partial(self.get_values, s, code)
        with ThreadPool(2) as p:
            results = p.map(para, page_num)
        # 整理结果
        if len(results) != 0:
            if len(results[0]) == 140:  # 普通基金
                info_array = np.array(results)
                info_2darray = np.concatenate(info_array)
                info_ndarray = info_2darray.reshape(-1, 7)
                mutual_fund = pd.DataFrame(info_ndarray)
                col = ["净值日期", "单位净值", "累计净值", "日增长率", "申购状态", "赎回状态", "分红送配"]
                mutual_fund.columns = col
                mutual_fund["基金代码"] = code
                return mutual_fund, 1
            elif len(results[0]) == 120:  # 货币型基金
                info_array = np.array(results)
                info_2darray = np.concatenate(info_array)
                info_ndarray = info_2darray.reshape(-1, 6)
                money_fund = pd.DataFrame(info_ndarray)
                col = ["净值日期", "每万份收益", "7日年化收益率", "申购状态", "赎回状态", "分红送配"]
                money_fund.columns = col
                money_fund["基金代码"] = code
                return money_fund, 2
        else:
            return None, None

    def initial_asset_values(self, codes, break_point=0):
        # 【不推荐调用】首次获取单个基金的全量历史记录。
        mutual_fund_path = "./Data/mutual_fund_values.csv"
        money_fund_path = "./Data/money_fund_values.csv"
        if break_point == 0:
            # 混合型基金
            mutual_cols = ["净值日期", "单位净值", "累计净值", "日增长率", "申购状态", "赎回状态", "分红送配", "基金代码"]
            self.init_file(mutual_fund_path, mutual_cols)
            # 货币型基金
            money_cols = ["净值日期", "每万份收益", "7日年化收益率", "申购状态", "赎回状态", "分红送配", "基金代码"]
            self.init_file(money_fund_path, money_cols)
        # 爬取数据
        for i in tqdm(range(break_point, len(codes))):
            code = codes[i]
            df_i, data_sign = self.get_code_values(code)
            if data_sign == 1:  # 净值型基金
                self.adding_data(mutual_fund_path, df_i)
            elif data_sign == 2:  # 货币型基金
                self.adding_data(money_fund_path, df_i)
        logger.info("all 基金获取完毕")


class InitManagers(BasicPrep):

    def get_manager(self, code):
        s = requests.Session()
        url = 'http://fundf10.eastmoney.com/jjjl_{}.html'.format(code)
        req_text = self.url_get(s, url)
        try:
            start_date = re.findall('<strong>上任日期：</strong>(.*?)</p>', req_text)[0]
        except IndexError:
            start_date = ""
        except Exception as e:
            logger.info("error: {e}, code: {code}", e=repr(e), code=code)
            start_date = ""
        return start_date

    def initial_managers_info(self, codes, break_point=0):
        # 【不推荐调用】首次获取单个基金经理的任职数据
        path_out = CONFIGS["path_managers"]
        if break_point == 0:
            cols = ["基金代码", "上任日期"]
            self.init_file(path_out, cols)
        # 获取基金经理数据
        fund_codes_new = codes[break_point:]
        with ThreadPool(3) as p:
            results = p.map(self.get_manager, fund_codes_new)
            df_adding = pd.DataFrame()
            df_adding["基金代码"] = codes
            df_adding["上任日期"] = results
        self.adding_data(path_out, df_adding)


class UpdateAssetValues(InitAssetValues):

    def exec_asset_value(self, code, start_date, path_out):
        """
        更新全量基金净值数据，并保存到本地
        :param code: str
            基金代码，全拼
        :param start_date: datetime
            待更新数据的开始时间
        :param path_out: Any
            数据保存（添加）本地的路径
        :return: None
        """
        # 更新单个基金数据
        today = datetime.date.today()
        st = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        diff_num = (today - st).days  # 自然日大于工作日，获取后需去重
        try:
            df_update, data_sign = self.get_code_values(code, diff_num)
            if data_sign == 1:
                df_update.to_csv(path_out, index=False, mode='a', header=False)
        except TypeError:
            logger.info("基金代码：{code}, 无数据跳过", code=code)

    def update_asset_values(self, st_date, bk_point=0):
        """
        更新基金历史净值数据
        :param st_date: datetime
            需获取数据的开始时间（更新，避免每次获取全量数据）
        :param bk_point: int
            断点续传
        :return: None
            输出结果到本地文件
        """
        path_codes = CONFIGS["path_used_fund"]  # 基金代码数据路径
        path_update = CONFIGS["path_asset_values"]  # 基金净值数据路径
        # 根据历史更新的时间，更新基金数据到最新，并写入数据文档
        df_codes = df_reader(path_codes)
        code_lst = df_codes["基金代码"].to_list()
        code_lst = code_lst[bk_point:]
        for i in tqdm(code_lst):
            self.exec_asset_value(i, st_date, path_update)
        # 更新完后去重
        df_values = df_reader(path_update)
        df_values.drop_duplicates(inplace=True)
        df_writer(df_values, path_update)
        # 更新参数
        configs_inner = read_json()
        configs_inner["last_date"] = datetime.date.today().strftime('%Y-%m-%d')
        write_json(configs_inner)


if __name__ == '__main__':
    pass
