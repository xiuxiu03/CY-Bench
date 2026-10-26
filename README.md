# CY-Bench 中国玉米：多模态滞后注意力与风险管理

本仓库实现论文模型的可复现实验：以气象、NDVI、FPAR、土壤水分、静态土壤和空间位置预测省级玉米单产；以可学习高斯滞后核模拟“气象驱动→作物遥感响应”；可选用 OpenAI 兼容 API 生成离线农业知识嵌入；并输出产量风险分级和论文图表。

## 数据与研究范围

输入应指向 CY-Bench 的 `maize/CN` 目录。程序读取九个原始 CSV，不复制也不修改原始数据。现有数据中有 31 个具有产量标签的中国省级区域；气象等预测变量从 2003 年开始，故脚本自动用 **2003–2022** 的区域-年度样本建模。默认时间外划分：2003–2016 训练、2017–2019 验证、2020–2022 测试。该划分禁止未来年份进入标准化器和训练过程。

省份代码按 GADM 一级区代码解释：北京、天津、河北、山西、内蒙古、辽宁、吉林、黑龙江、上海、江苏、浙江、安徽、福建、江西、山东、河南、湖北、湖南、广东、广西、海南、重庆、四川、贵州、云南、西藏、陕西、甘肃、青海、宁夏、新疆。数据没有产量标签的台湾和香港不参与训练。

它横跨东北春玉米带（冷凉、单季、热量资源约束）、华北/黄淮海夏玉米带（多为冬小麦–夏玉米轮作，关键风险是花粒期高温干旱和收获期涝渍）、西北灌溉绿洲（干旱蒸散强，灌溉和水分平衡关键）、西南山地及南方丘陵（地形和物候异质性强）。因此空间模块只能表达省级邻近性，**不能**替代县/地块尺度的因果结论。

CY-Bench 资料说明及变量来源见 [CY-Bench 论文](https://essd.copernicus.org/articles/18/3997/2026/) 与 [官方代码库](https://github.com/WUR-AI/AgML-CY-Bench)。

## 安装与运行

```bash
conda create -n maize python=3.11 -y
conda activate maize
pip install -r requirements.txt
cp .env.example .env                 # 仅在启用 LLM 时填写
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment xgboost
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment lstm
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment transformer
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment proposed
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment proposed --use-llm
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment ablate_no_lag
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment ablate_no_spatial
python -m src.train --config configs/default.yaml --data-dir /data/cybench-data/maize/CN --experiment ablate_no_llm
python -m src.report --artifacts artifacts
pytest -q
```

`--use-llm` 会调用 API 一次/区域并将嵌入缓存到 `artifacts/llm_embeddings.npz`；训练阶段绝不在线请求 API。默认配置不调用 API，且 `ablate_no_llm` 强制关闭知识向量。API 发送的是区域、经纬度、作物历和训练期聚合统计，不上传产量标签、原始逐日序列或密钥。

## 实验矩阵

| 组别 | 目的 |
|---|---|
| `xgboost` | 树模型强基线（季节统计特征） |
| `lstm` | 单一时序深度学习基线 |
| `transformer` | 标准多模态 Transformer 基线 |
| `proposed` | 知识增强 + 跨模态滞后注意力 + 空间注意力 |
| `ablate_no_lag` | 量化滞后核贡献 |
| `ablate_no_spatial` | 量化空间传播贡献 |
| `ablate_no_llm` | 量化知识向量贡献 |

输出包括指标表、预测散点图、误差分布、年度表现、风险矩阵、注意力滞后图、空间风险图和消融图。风险阈值仅由训练集产量的 5% 与 10% 分位数估计；低于 5% 是高风险、5%–10% 是中风险、其余为低风险。

## 可复现性与审查要点

- 训练标准化器只在训练集拟合；测试集从不参与阈值、缩放和早停选择。
- 以年份顺序切分，避免随机切分造成的时间泄漏。
- 所有随机源固定；每个配置建议至少运行 5 个种子并报告均值±标准差。
- 若数据中存在跨年生育季，数据加载器已处理 `eos < sos` 的情况；当前 CN 作物历主要为同年季节。
- LLM 文本由 `prompts/agronomic_knowledge_prompt.txt` 完整定义，可在论文附录原样列出。
