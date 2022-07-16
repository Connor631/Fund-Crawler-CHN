import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
import json
from loguru import logger

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
# 全局变量参数
with open("config.json", "r") as k:
    CONFIGS = json.load(k)


def df_reader(path):
    df = pd.read_csv(path, dtype={"基金代码": str})
    return df


def df_writer(df, path):
    df.to_csv(path, index=False)


class BasicIndex:
    @staticmethod
    def max_draw_down(data):
        i = np.argmax((np.maximum.accumulate(data) - data) / np.maximum.accumulate(data))
        if i > 0:
            j = np.argmax(data[:i])
            draw_down = (data[j] - data[i]) / data[j]
            return round(draw_down, 4)
        else:
            return None

    @staticmethod
    def VIX(data):
        # 标准差的定义区别于统计学种的标准差，具体可参见以下链接：
        # https://zhuanlan.zhihu.com/p/39289090
        log_rate = np.log(data)
        diff_rate = log_rate - log_rate.shift()
        vix = np.sqrt(250 / len(data)) * diff_rate.std()
        return round(vix, 4)

    def sharp_ratio(self, data):
        vix_v = self.VIX(data)
        # 假定无风险收益率为0%
        sr = ((1 - data.iloc[-1] - 0) * np.sqrt(250)) / vix_v
        return round(sr, 4)


class AggregateIndex(BasicIndex):
    @staticmethod
    def date_clean(df, delta_num=1):
        last_date = df.iloc[0, 0]
        date_spc = last_date - datetime.timedelta(days=delta_num)
        if df[df["净值日期"] <= date_spc].shape[0] > 0:  # 确保不是最后一条数据
            df_c = df[df["净值日期"] >= date_spc]
            rate = (df_c.iloc[0, -1] - df_c.iloc[-1, -1]) / df_c.iloc[-1, -1]
            return rate
        else:
            return None

    @staticmethod
    def growth_rate(code, df, index='raw'):
        df_ii = df[df["基金代码"] == code]
        # 没有值不返回数据
        if df_ii.shape[0] == 0:
            return None
        else:
            # 分类计算指标
            cum_value = df_ii[["净值日期", "累计净值"]]
            cum_value["净值日期"] = pd.to_datetime(cum_value["净值日期"])
            cum_value.reset_index(drop=True, inplace=True)
            try:
                if index == 'week':
                    rate = AggregateIndex.date_clean(cum_value, 7)
                    return rate
                elif index == 'month':
                    rate = AggregateIndex.date_clean(cum_value, 30)
                    return rate
                elif index == '3month':
                    rate = AggregateIndex.date_clean(cum_value, 90)
                    return rate
                elif index == '6month':
                    rate = AggregateIndex.date_clean(cum_value, 183)
                    return rate
                elif index == 'year':
                    rate = AggregateIndex.date_clean(cum_value, 365)
                    return rate
                elif index == 'raw':
                    return df_ii["累计净值"]
            except KeyError:
                logger.info(
                    "Error type: KeyError. Error code: {code}, length：{length}", code=code, length=len(cum_value)
                )
            except Exception as e:
                logger.info("Error code: {code}, error info: {e}", code=code, e=repr(e))

    def clac_indexes(self, code, df, index='VIX'):
        data_raw = self.growth_rate(code, df, 'raw')
        if data_raw is None:
            return None
        elif len(data_raw) > 0:
            data = data_raw.reindex(data_raw.index[::-1])  # 倒序
            data.reset_index(drop=True, inplace=True)
            if index == "drawdown":
                # 计算最大回撤率
                max_draw_down_v = self.max_draw_down(data)
                return max_draw_down_v
            elif index == "VIX":
                # 计算收益波动率
                VIX_v = self.VIX(data)
                return VIX_v
            elif index == "sharp_ratio":
                # 计算夏普比率
                sharp_ratio_v = self.sharp_ratio(data)
                return sharp_ratio_v
        else:
            return None


class FilterTypes(AggregateIndex):

    @staticmethod
    def fund_type_filter(vis=False):
        path_in = CONFIGS["path_all_fund"]
        path_out = CONFIGS["path_used_fund"]
        df_raw = df_reader(path_in)
        # 保留下的基金种类
        stay_type = ['股票型', '混合型-平衡', '混合型-偏股']
        df = df_raw[df_raw["基金类型"].isin(stay_type)]
        df_writer(df, path_out)
        if vis:
            def func(pct, vals):
                absolute = int(np.round(pct / 100. * np.sum(vals)))
                return "{:.1f}%({:d})".format(pct, absolute)

            df_vis = df_raw[["基金代码", "基金类型"]]
            # 分组后统计
            df_vis_group = df_vis.groupby("基金类型").基金代码.agg('count')
            drop_type = ['FOF', 'Reits', '债券型-中短债', '债券型-混合债', '债券型-可转债', '债券型-长债',
                         '商品（不含QDII）', 'QDII', '混合型-偏债', '理财型', '货币型', '混合-绝对收益', '混合型-灵活']
            new_index = stay_type + drop_type
            df_vis_new = df_vis_group.reindex(new_index)
            # 设置颜色
            colors = ['mistyrose', 'salmon', 'tomato'] + ['grey'] * len(drop_type)
            plt.pie(df_vis_new, labels=df_vis_new.index, autopct=lambda pct: func(pct, df_vis_new),
                    pctdistance=0.8, colors=colors)
            plt.suptitle("各类型基金占比", fontsize=18)
            plt.title("首次筛选后的基金占比为{:.2f}%".format(100 * (df.shape[0] / df_raw.shape[0])), fontsize=10)
            plt.show()
        return df

    @staticmethod
    def manager_duration_filter():

        def diff_date(date_):
            today = datetime.date.today()
            start = datetime.datetime.strptime(date_, '%Y-%m-%d').date()
            return (today - start).days

        path_managers = CONFIGS["path_managers"]
        df_raw = CONFIGS["path_all_fund"]
        df_manager = df_reader(path_managers)
        df_m2 = df_manager[df_manager["上任日期"] != ""]
        df_m2.dropna(inplace=True)
        df_m2["就任时间"] = df_m2["上任日期"].map(diff_date)
        # 保留：第一基金经理任职时间在一年以上
        df_new = df_m2[df_m2["就任时间"] > 365]
        df_out = pd.merge(df_raw, df_new, on="基金代码", how='left')
        df_out.dropna(subset=["就任时间"], inplace=True)
        df_out.reset_index(drop=True, inplace=True)
        return df_out

    def performance_filter(self, online=False):
        # 加载全局变量的路径参数
        df_in_path = CONFIGS["path_used_fund"]
        raw_data_path = CONFIGS["path_asset_values"]
        performance_path = CONFIGS["path_indicators"]
        if not online:
            df_out = df_reader(performance_path)
        else:
            df_in = df_reader(df_in_path)
            df_raw = df_in.copy(deep=True)
            df_value = df_reader(raw_data_path)
            df_value.drop_duplicates(inplace=True)
            logger.info("基金周增长指标计算中......")
            df_raw["基金周增长"] = df_raw["基金代码"].map(lambda x: self.growth_rate(x, df_value, 'week'))
            logger.info("基金周增长指标计算完成。基金月增长指标计算中......")
            df_raw["基金月增长"] = df_raw["基金代码"].map(lambda x: self.growth_rate(x, df_value, 'month'))
            logger.info("基金月增长指标计算完成。基金3月增长指标计算中......")
            df_raw["基金3月增长"] = df_raw["基金代码"].map(lambda x: self.growth_rate(x, df_value, '3month'))
            logger.info("基金3月增长指标计算完成。基金6月增长指标计算中......")
            df_raw["基金6月增长"] = df_raw["基金代码"].map(lambda x: self.growth_rate(x, df_value, '6month'))
            logger.info("基金6月增长指标计算完成。基金年增长指标计算中......")
            df_raw["基金年增长"] = df_raw["基金代码"].map(lambda x: self.growth_rate(x, df_value, 'year'))
            logger.info("基金年增长指标计算完成。基金最大回撤指标计算中......")
            df_raw["最大回撤"] = df_raw["基金代码"].map(lambda x: self.clac_indexes(x, df_value, 'drawdown'))
            logger.info("基金最大回撤指标计算完成。基金波动率指标计算中......")
            df_raw["收益波动率"] = df_raw["基金代码"].map(lambda x: self.clac_indexes(x, df_value, 'VIX'))
            logger.info("基金波动率指标计算完成。基金夏普比率指标计算中......")
            df_raw["夏普比率"] = df_raw["基金代码"].map(lambda x: self.clac_indexes(x, df_value, 'sharp_ratio'))
            logger.info("基金夏普比率指标计算完成。")
            df_out = df_raw[
                ["基金代码", "基金周增长", "基金月增长", "基金3月增长", "基金6月增长", "基金年增长", "最大回撤", "收益波动率", "夏普比率"]
            ]
            df_out.dropna(subset=["基金月增长"], inplace=True)
            df_writer(df_out, path=performance_path)
        return df_out


class UpdateIndexes(FilterTypes):
    def perform_update(self):
        # 根据传入的指标，返回综合排序后的结果
        self.performance_filter(online=True)

    def sorting_data(self, indexes):
        # 更新指标数据
        self.perform_update()
        raw_data = CONFIGS["path_indicators"]
        performances = df_reader(raw_data)
        # 根据基金的表现筛选
        df_t = performances.dropna(how='any')  # 去除含有任意空值的数据
        # 排名，其中越大越好的指标有：基金增长、夏普比率；越小越好的指标有：最大回撤、波动率。
        if "基金周增长" in indexes:
            df_t["基金周增长排名"] = df_t["基金周增长"].rank(ascending=False)
        if "基金月增长" in indexes:
            df_t["基金月增长排名"] = df_t["基金月增长"].rank(ascending=False)
        if "基金3月增长" in indexes:
            df_t["基金3月增长排名"] = df_t["基金3月增长"].rank(ascending=False)
        if "基金6月增长" in indexes:
            df_t["基金6月增长排名"] = df_t["基金6月增长"].rank(ascending=False)
        if "基金年增长" in indexes:
            df_t["基金年增长排名"] = df_t["基金年增长"].rank(ascending=False)
        if "最大回撤" in indexes:
            df_t["最大回撤排名"] = df_t["最大回撤"].rank(ascending=True)
        if "夏普比率" in indexes:
            df_t["夏普比率排名"] = df_t["夏普比率"].rank(ascending=False)
        if "收益波动率" in indexes:
            df_t["收益波动率排名"] = df_t["收益波动率"].rank(ascending=True)
        # 根据选中指标排序
        cols = ["基金代码"] + [i + "排名" for i in indexes]
        df_t["总平均排名"] = df_t[cols].mean(axis=1)
        df_t.sort_values(by="总平均排名", inplace=True)
        df_t.reset_index(drop=True, inplace=True)
        # 输出
        fund_lst = df_t["基金代码"].to_list()[:10]
        logger.info("筛选出的前十大基金分别是{lst}", lst=fund_lst)
        return df_t


if __name__ == '__main__':
    pass
