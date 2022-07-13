from crawler import UpdateAssetValues
from crawler import read_json
from filter import UpdateIndexes


if __name__ == '__main__':
    configs = read_json()
    # 获取参数
    last_date = configs["last_date"]
    used_fund = configs["path_used_fund"]
    asset_values = configs["path_asset_values"]
    # 更新数据,大约需要20分钟
    up_values = UpdateAssetValues()
    up_values.update_asset_values(last_date)
    # 更新指标,大约需要15分钟
    # short_indexes = ["基金周增长", "基金月增长", "基金3月增长", "收益波动率"]  # 短期指标
    long_indexes = ["基金6月增长", "基金年增长", "最大回撤"]  # 长期指标
    up_index = UpdateIndexes()
    funds_rank = up_index.sorting_data(long_indexes)
    funds_rank.to_csv("output_info.csv", index=False, encoding='gbk')
