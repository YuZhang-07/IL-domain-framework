# Task 2 训练脚本：GitHub 增补包

本包包含七个分类器的训练脚本、配套 JSON 配置、运行辅助模块及说明。
**只新增 Task 2 相关文件，不覆盖 Task 1 的代码、配置、根目录说明或依赖文件。**
不包含真实训练 CSV、已训练模型、原始脚本副本或测试运行日志。

## 一、放在哪里

把解压后的 `task2_github_release/` **里面的内容**合并到现有仓库根目录。
不是把整个 `task2_github_release/` 文件夹塞进 `training/`。

```text
你的仓库/
├── scripts/training/
│   ├── task1/                    # 保留原文件
│   └── task2/                    # 新增：七个训练脚本及 _runtime.py
├── configs/
│   ├── task1/                    # 保留原文件
│   └── task2/                    # 新增：七份 JSON 参数配置
├── data/task2/
│   ├── .gitignore
│   └── README.md
└── docs/task2/                   # 新增：本说明、修改记录、验证记录及依赖说明
```

训练 `.py` 放 `scripts/training/task2/`，参数 `.json` 放 `configs/task2/`。
`_runtime.py` 必须和七个训练脚本放在一起；不要拿 Task 2 的版本替换 Task 1 的版本。
程序不依赖 Task 1 的运行模块，两个任务的配置及默认结果目录也分别独立。

根目录原有的 `README.md`、`README_zh-CN.md`、`.gitignore`、`.gitattributes`、
`requirements.txt` 和 `requirements-tested.txt` 不需要被本包替换。
本包在 `data/task2/` 内单独放置 `.gitignore`，避免覆盖你原有的忽略规则。

## 二、输入与运行

默认输入位置：

```text
data/task2/proid_1_2_feature_labels_standard.csv
```

文件名来自本次上传的七个 Task 2 原脚本。你的实际文件名字不同也可以使用，
通过 `--data` 指定即可，不需要为了路径修改训练代码。

CSV 的读取规则保持不变：第一列是分组 ID，中间全部列是已有标准化特征，
最后一列是原始标签 1 或 2。不新增标准化、插补或自动特征列识别。
不要把额外的 `frame_id` 或导出索引插在特征区域；它们会按原逻辑被当成特征。

在仓库根目录运行：

```bash
python scripts/training/task2/train_xgboost.py
```

它自动读取 `configs/task2/xgboost.json`。指定输入、输出和并行参数：

```bash
python scripts/training/task2/train_xgboost.py --config configs/task2/xgboost.json --data data/task2/proid_1_2_feature_labels_standard.csv --output-dir results/task2/xgboost/run_01 --n-jobs -1
```

CSV 仍放在当前终端目录时：

```bash
python scripts/training/task2/train_xgboost.py --data proid_1_2_feature_labels_standard.csv
```

| 模型 | `scripts/training/task2/` 中的脚本 | `configs/task2/` 中的配置 |
|---|---|---|
| XGBoost | `train_xgboost.py` | `xgboost.json` |
| Logistic Regression | `train_logistic_regression.py` | `logistic_regression.json` |
| Gaussian NB | `train_gaussian_nb.py` | `gaussian_nb.json` |
| KNN | `train_knn.py` | `knn.json` |
| Linear SVM | `train_linear_svm.py` | `linear_svm.json` |
| Decision Tree | `train_decision_tree.py` | `decision_tree.json` |
| Random Forest | `train_random_forest.py` | `random_forest.json` |

命令行支持 `--config`、`--data`、`--output-dir`、`--n-jobs`、`--n-trials`、
`--seed`、`--dry-run`。`--dry-run` 只核对配置与文件是否存在，不读取 CSV 内容、
不训练、不创建结果目录，不能替代数据内容检查。

`--n-jobs -1` 保持原有支持该参数的模型并行方式，不会把五折改为并行或多节点任务。
正整数同时限制数值线程池；无模型级 `n_jobs` 接口的分类器不会凭空获得该接口。
按上面的方式分别启动各脚本；未将同一解释器内同时导入两个任务设计为使用接口。

## 三、路径和输出

JSON 内的相对路径以仓库根目录为基准；命令行显式传入的相对路径以终端当前目录为基准。
命令行可接受本地绝对路径，但个人路径不需要写入公开的代码或默认配置。

不写 `--output-dir` 时自动输出到：

```text
results/task2/模型名称/年月日_时分秒_微秒/
```

显式指定的输出目录必须不存在或为空；已有文件时拒绝继续，避免覆盖旧结果。
原脚本的结果文件名和预测列名保留，另外增加：

| 文件 | 内容 |
|---|---|
| `run_config.json` | 本次实际参数 |
| `environment.json` | 本次实际 Python 与包版本 |
| `run_metadata.json` | 运行状态、时间、输入及代码 SHA-256 摘要 |

可分享的 JSON 中，仓库外的绝对目录显示为 `<external>/文件名`。
这类脱敏记录不是可以直接还原本地路径的运行配置。保留的控制台输出及异常回溯
仍可能显示本地输出路径，不应把原始运行日志当成已经全面脱敏的材料。

## 四、Task 2 标签方向不能弄反

```python
LABEL_MAP = {1: 0, 2: 1}
INVERSE_LABEL_MAP = {0: 1, 1: 2}
```

输入仍是原始标签 **1/2**，内部才映射为 **0/1**。
输出的 `true_label`、`pred_label` 恢复到 **1/2**。
**正类概率及 SVM 分数均指向原始标签 2，而不是原始标签 1。**
同一 protein_id 内不同帧可以有不同标签，程序不把它们压缩成一个蛋白标签。

XGBoost、NB、KNN 和 DT 的概率列仍是 `pred_proba`；
RF 和 LR 仍是 `pred_proba_class_2`；SVM 仍是 `decision_score`，不是概率。
未为了统一列名改变已有输出，避免影响后续统计脚本。

## 五、保留的训练协议及必要说明

保留原始五折 GroupKFold、每个外折内的一次组感知 train/validation 划分、
默认 50 组候选及每个模型原有候选顺序。内部使用 StratifiedGroupKFold 的首个划分，
并保留原回退与重试逻辑；不是对五个内折取均值。

参数根据 inner validation AUC 选择。用于 outer test 的模型仍只在 inner train
上拟合，没有擅自改成用完整 outer train 重训。XGBoost 保留原早停逻辑。
近常数特征筛选、全量最终模型训练、各折指标定义均按原脚本保留。
最终全量模型不替代交叉验证模型，不能用它的训练集分数报告 CV 性能。

原 XGBoost 中 `BASELINE_AUC = 0.7931` 已原样迁入配置，仍只用于历史参考比较。
**该值不是本次计算所得，来源未独立核实，不参与模型拟合和参数选择。**
该值可设为 JSON `null`，仅关闭历史比较输出，不改变训练。
历史性能承诺式注释已改成实际流程说明，未删除搜索中的 baseline 参数候选。

取消原脚本对全部警告的全局屏蔽，不改变拟合算法。LR 的 SHAP 功能保留，
仅在 `RUN_SHAP=true` 时导入；默认仍关闭。本次未测试开启 SHAP 的流程。

## 六、依赖与验证边界

所需直接依赖与上一版 Task 1 包相同，已有兼容环境无需另造一套环境。
单独部署时，可从仓库根目录执行：

```bash
python -m pip install -r docs/task2/requirements-tested.txt
```

该文件记录本次实际测试环境，不冒充论文原始训练环境，也不是完整传递依赖锁文件。
未锁版本的直接依赖清单在 `docs/task2/requirements.txt`；其他版本组合未在本次验证。

七个模型均在 20 个分组、240 行合成样本上完成原版/新版对照，每个组内含两种标签。
每个模型执行全部五个外折，每折 3 组候选；XGBoost 测试时采用 60 棵树上限、8 轮早停。
两边同样限制数值线程；发布配置仍为 50 组候选，XGBoost 3000 棵树上限、100 轮早停。
另逐折核对了默认 50 个候选的完整生成顺序。

所有比较的预测表、指标、参数及模型预测通过 `rtol=atol=1e-12` 的数值一致性检查。
没有收到真实 Task 2 数据，因此没有重跑或确认论文 AUC；没有验证上游标准化、
科学标签定义、全研究的数据泄漏状态或旧版本软件回退分支。

详见 `VERIFICATION.md` 和 `verification_results.json`。
