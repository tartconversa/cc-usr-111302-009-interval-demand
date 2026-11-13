# 区间申报需求的非参数分布估计

本项目保存虚构数据共享及产品运营的合成表格与Python工具。`src/demand_reader.py` 只解析文件并显示记录数，不执行统计推断、优化、估计或校准；`src/` 下其余模块是新增的区间删失分布分析器。所有地区、机构、产品与人数均为合成，输入为虚构机构的自愿价格区间，不做个人定价。

## 字段与单位

kind取interval/left/right/point；interval为(L,R]，left为[0,R]，right为(L,+inf)，point读取point列。空端点表示未使用，不等于0。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 区间删失分布分析器

每个独立机构一次有效申报，潜在支付意愿非负。分析器把全部申报端点与点质量划分成互不重叠、覆盖模式不变的支持单元（端点单点、相邻端点开区间、最大端点外的无穷尾），在单元质量 `p>=0, sum p=1` 上最大化非参数似然：

    max sum_i log(sum_j A_ij p_j)      A_ij = 单元 j 是否完全落在申报 i 的域内

不把每条申报当均匀分布，也不预置正态/对数正态等连续参数族。EM（Turnbull 自洽迭代）与 SLSQP 凸优化共享同一似然核心（`src/npmle.py`），收敛判据为 KKT 残差 `max(d)-n`（`d_j = sum_i A_ij/s_i`，支撑上应等于 n，支撑外不超过 n），并输出零质量边界诊断（pinned_at_zero / zero_but_active）。

多个最优质量分配可能给出相同似然，单个 EM 停止点不代表唯一分布。指定价位的购买概率 P(W>price) 在最优似然容差集合 `{p: ll(p) >= ll* - tol}` 内求上下界（`src/bounds.py`）：容差为 0 时水平集即最优面 `{A p = s*}`，单次 LP 精确求解；容差为正时用 Kelley 切平面法给出认证外界，并以可行点内值夹出真值。价位落在某覆盖单元内部时，该单元质量在价位两侧的分布不可识别，上界把跨价位单元质量全计入、下界全不计——不用质量中点伪造唯一尾概率；无穷尾单元不封存到最大观测价。

机构层 bootstrap（`src/bootstrap.py`）每次重抽样都重建支持单元与 A 矩阵、重新拟合，并统计 EM 收敛失败次数；每个复制内的 [L,U] 是部分识别范围而非抽样区间，分位数只描述识别集合的抽样波动。

    python -m src.analyze fixtures/intervals.csv fixtures/demand.json

输出 JSON 报告：支持单元与覆盖关系、似然值与 KKT 残差、零质量边界、似然非唯一性度量、各价位识别上下界（含认证是否收紧）、bootstrap 汇总，以及申报抽样偏差与需求外推局限说明（报告 `caveats` 字段）：自愿申报存在自选择偏差，估计只描述申报子总体；[L,U] 是部分识别范围而非置信区间；超出最大观测端点的尾概率是支持集假设而非数据证据；静态申报意愿不能用于预测调价后的行为反应。

## 运行

读取器只需 Python 3.11 以上标准库；分析器与测试需要 numpy 与 scipy（已在 pyproject.toml 登记，容器内按 Dockerfile 固定版本安装）。

    python -m src.demand_reader fixtures/intervals.csv
    python -m src.analyze fixtures/intervals.csv fixtures/demand.json
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 固定容器

    docker build -t interval-demand:local .
    docker run --rm interval-demand:local

容器镜像以完整digest固定，默认运行全部测试（输入契约 + 分析器）。分析功能所需依赖由开发者明确登记：pyproject.toml 声明 numpy/scipy 版本范围，Dockerfile 固定具体版本。
