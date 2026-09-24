# Task 1 评估部分：GitHub 增补包

将 `task1_evaluation_github_release/` **里面的内容**合并到现有仓库根目录。
不要把整个外层包装文件夹再塞进 `training/` 或 `evaluation/`。
本包不覆盖已整理的 Task 1/Task 2 训练脚本、训练配置、根目录 README 或依赖文件。

## 本次包含什么

本次实际收到的 15 个文件对应：7 个分类器置信区间脚本、4 个消融数据删列脚本、4 个 XGBoost 消融训练脚本。
发布包保留 15 个一一对应的入口，另增加 1 个运行辅助模块与 11 份 JSON 配置。

**本次没有收到真正执行 XGBoost SHAP 的代码。**
消融脚本中“保存最终模型供后续 SHAP 使用”的注释不等于 SHAP 已执行。
`shap/README.md` 只是缺失说明，没有虚构一个替代脚本、背景样本或特征子集。
包内不包含真实数据、已训练模型、合成测试数据、原始脚本备份或运行日志。

## 放在哪里

```text
scripts/evaluation/task1/
├── _eval_runtime.py
├── bootstrap/                # 7 个置信区间计算脚本
├── ablation/                 # 4 个删列脚本 + 4 个消融训练脚本
└── shap/
    └── README.md             # 仅缺失说明，没有 SHAP 程序

configs/task1/evaluation/
├── bootstrap/                # 7 个分类器 JSON
└── ablation/                 # 4 个特征组 JSON，删列与训练共用

data/task1/ablation/          # 本地生成消融 CSV 的默认位置

docs/task1/evaluation/        # 本次说明、修改记录、验证记录和依赖
```

这部分按科学用途归入 **evaluation**。消融中虽然调用了模型拟合，但用途是特征组比较，因此删列与消融训练均归入 `ablation/`，不用再复制一份到 `training/`。
`_eval_runtime.py` 必须保留，不要用它替换此前训练目录里的 `_runtime.py`。

JSON 中的相对路径以**仓库根目录**为基准；命令行显式传入的相对路径以**当前终端目录**为基准。
允许运行时指定绝对路径，但公开代码和默认配置中不绑定个人路径。
`--dry-run` 只显示设置，不读取研究数据，也不创建结果目录。

## 一、置信区间

以下假设 XGBoost 某一次完整训练结果已经保存在 `results/task1/xgboost/run_01/`。
该路径只是示例，应换为你的真实运行目录，不能指向包含多次运行的上级目录。

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py --result-dir results/task1/xgboost/run_01
```

默认读取 `configs/task1/evaluation/bootstrap/xgboost.json`。
其余分类器使用对应脚本：

| 分类器 | 入口文件 | 预测分数列 |
| --- | --- | --- |
| XGBoost | `bootstrap_ci_xgboost.py` | `pred_proba` |
| DT | `bootstrap_ci_decision_tree.py` | `pred_proba` |
| NB | `bootstrap_ci_gaussian_nb.py` | `pred_proba` |
| KNN | `bootstrap_ci_knn.py` | `pred_proba` |
| SVM | `bootstrap_ci_linear_svm.py` | `decision_score` |
| LR | `bootstrap_ci_logistic_regression.py` | `pred_proba_class_1` |
| RF | `bootstrap_ci_random_forest.py` | `pred_proba_class_1` |

预测 CSV 的原始文件名不变，例如 `fold1_xgb_predictions.csv`、`fold1_random_forest_predictions.csv`。
六个非 XGBoost 脚本还要求对应的 `cv_*_results.csv`，且必须与预测文件来自同一次训练。
不要把 `decision_score` 改称概率，也不要从最终全量训练模型生成训练集预测来替代 outer-test 预测。

默认参数保留：2000 次 Bootstrap、随机种子 42、95% 置信水平、最大重抽设置 10000。
每个 outer fold 内按 `protein_id` 整簇有放回抽样，同一蛋白内部允许混合帧级 0/1 标签；每次先计算各折指标，再取折均值，最后计算百分位数区间。
没有改成逐帧独立抽样、合并 OOF AUC，也没有改成先逐蛋白计算 AUC 再平均。

**原脚本之间的差异没有被擅自统一：**
XGBoost 使用 `p >= 0.5` 重算类别，不一致时警告；其 CV 对照原本为可选/警告，不额外改成强制中止。
其余六个脚本仍保留原有严格检查。DT/NB/KNN 使用 `p >= 0.5`；RF/LR 使用 `p > 0.5`；SVM 使用 `score > 0`。
CV 核对容差仍为 KNN `1e-5`、LR `1e-7`、DT/NB/SVM/RF `1e-8`。
XGBoost 原来没有强制恰好发现五折，本次也没有新增该规则；论文分析请提供完整五折文件。

默认输出在输入运行目录的 `bootstrap/` 下，包含原来的五个统计文件：

```text
bootstrap_ci_summary.csv
bootstrap_distribution.csv
recomputed_fold_metrics.csv
protein_label_composition.csv
bootstrap_metadata.json
```

另增加 `run_config.json`、`environment.json`、`run_metadata.json`，记录成功运行的设置、环境与输入引用。
目录已经有文件时会停止，避免覆盖旧结果。可以显式指定新位置：

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_random_forest.py --result-dir results/task1/random_forest/run_01 --output-dir results/task1/evaluation/random_forest_ci_run_01 --n-bootstrap 2000 --seed 42
```

这里的 CI 反映的是**固定外层预测在蛋白体系重采样下的不确定性**，不包含重新调参、重新训练的不确定性。

## 二、XGBoost 消融

每组共用一份 JSON，例如：

```text
configs/task1/evaluation/ablation/remove_global.json
```

该配置同时定义 full_table、ablated_table、删列规则和消融训练参数。

| 新名称 | 对应原删列脚本 | 删除内容 | 默认生成文件 |
| --- | --- | --- | --- |
| `remove_global` | `remove_W_0_1.py` | 8 个整体标量特征 | `data/task1/ablation/removeW.csv` |
| `remove_interface` | `removeInter_0_1.py` | 5 个界面标量特征 | `data/task1/ablation/remove_Inter.csv` |
| `remove_contacts` | `remove_resn_0_1.py` | 匹配 `^[A-Z]{3}-[A-Z]{3}$` 的接触特征 | `data/task1/ablation/remove_resn.csv` |
| `remove_hbonds` | `remove_hbond_0_1.py` | 匹配 `^hbond[A-Z]{3}-[A-Z]{3}$` 的氢键特征 | `data/task1/ablation/remove_hbond.csv` |

8 个和 5 个特征的原始列名已逐项保留在配置中。`remove_global` 不是“只删除与水相关的特征”。
残基对匹配结果不是 210 列时仍然只警告；没有匹配列则仍报错。没有根据论文中的简称猜测或改写列名。

### 先生成删列数据

默认完整输入为：

```text
data/task1/proid_0_1_feature_labels_standard.csv
```

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py
python scripts/evaluation/task1/ablation/prepare_remove_interface.py
python scripts/evaluation/task1/ablation/prepare_remove_contacts.py
python scripts/evaluation/task1/ablation/prepare_remove_hbonds.py
```

只删除指定列，不改变行顺序、剩余列顺序、已有数值或标签。每个 CSV 旁新增 `.csv.metadata.json`，记录实际删列清单。

**明确修正了两处文件衔接：**
原来 `remove_W_0_1.py` 输出 `removeW_1_2_feature_labels_standard.csv`，但消融训练读取 `removeW.csv`；
`removeInter_0_1.py` 输出 `removeInter_1_2_feature_labels_standard.csv`，但训练读取 `remove_Inter.csv`。
新版统一为后两者。这里只改文件名与路径，**没有进行 1/2 → 0/1 标签转换**。

已有删列 CSV 可以直接用于下一步，无需重复生成。需要另存新的删列文件时：

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py --data data/task1/proid_0_1_feature_labels_standard.csv --output-file data/task1/ablation/removeW_v2.csv
```

### 再运行对应的消融训练

```bash
python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_interface.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_contacts.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_hbonds.py --n-jobs -1
```

训练入口的 `--data` 必须指向**已经删列的 CSV**，不会在内部再删除一次，不能指向全特征表。
默认输出为 `results/task1/ablation/<组名>/<运行时间>/`。
也可指定新目录：

```bash
python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py --data data/task1/ablation/removeW.csv --output-dir results/task1/ablation/remove_global/run_01 --n-jobs -1
```

五折 GroupKFold、每折一个组感知内部验证划分、50 组候选、3000 树上限、100 轮早停、类别加权、近常数处理、折内拟合范围、最终全量模型逻辑与输出列保持原来规则。
原实现是在每个消融条件下重新选参，没有擅自换成固定全特征模型参数的消融。
`--n-jobs -1` 不会自动把五折分配到五个节点。

原有输入约定仍是**首列 group ID、末列 label、中间全部为模型特征**。没有新增标准化、插补、标签合并或自动排除 `frame_id`。

原脚本的 `BASELINE_AUC = 0.7931` 已保留在 JSON，但仅为历史显示参考，不参与拟合和选参。
本次没有核实它是否为最终论文的 Task 1 全特征基线。需要关闭这一项显示，可将其设为 `null`；训练不受影响。

消融运行结束后，直接使用同一份 XGBoost CI 脚本：

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py --result-dir results/task1/ablation/remove_global/run_01
```

## 三、公开范围和验证边界

公开 JSON 会隐藏仓库外绝对路径的目录前缀，但保留文件名、科学特征名。辅助模块不自动记录主机名、环境变量、凭据或控制台日志。
控制台和异常回溯仍可能带本地路径，不能把任意日志直接视为已经脱敏。
对本次 15 个原脚本、16 个发布版 Python 文件与 11 个 JSON 配置的所列敏感模式扫描，未发现匹配项；不覆盖未提供的文件、数据内容和 Git 历史。

本地 `.gitignore` 不能阻止手动网页上传，也不会自动移除已经纳入版本控制的文件。此增补包本身没有打包真实数据。

依赖和实测版本记录在本目录下两份 `requirements` 文件中，不覆盖仓库根目录。
`requirements-tested.txt` 是这次测试环境，不是论文历史环境。

7 个 CI 脚本已完成 2000 次重复的原版/新版合成数据对照；4 组删列和消融也完成对照。
详见 `VERIFICATION.md`、`verification_results.json`、`CHANGES.md` 与 `file_mapping.json`。
**没有真实 Task 1 CSV/预测文件，因此没有验证论文 AUC、CI 或消融数值；SHAP 未执行。**
