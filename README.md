# fund-filter-CHN
# 中国基金筛选

基于基金的历史表现，应用多个评价指标，得出综合的基金排名。

## 使用
生成基金排名之前需要基础数据支持，需将**Data.rar**压缩文件先解压后使用。

```python
# 自动更新数据，并输出排名
$ python main.py
```

## 结构
* crawler: 获取并更新基金历史数据
* config: 存放基础设置，包括数据日期和文件路径
* filter: 定义更新所用指标

### 支持指标
-[x] 周、月、3月、6月、年增长率
-[x] 最大回撤
-[x] 历史波动率
-[ ] 夏普比率
-[ ] 其它指标

