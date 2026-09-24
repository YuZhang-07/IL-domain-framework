# IL-domain-framework

## 用于研究离子液体对双结构域蛋白影响的分子动力学与分组机器学习框架

[English](README.md) · [代码仓库](https://github.com/YuZhang-07/IL-domain-framework) · [数据发布页面](https://github.com/YuZhang-07/IL-domain-framework/releases)

本仓库提供研究离子液体对双结构域蛋白界面影响的科研代码、配置文件和部分分子动力学（MD）输入。机器学习流程使用预先计算的帧级描述符，包含两个二分类任务，并在分组模型评价中将不同蛋白体系分开。

仓库整合了 **Task 1 和 Task 2 的分类器训练、蛋白簇 Bootstrap 置信区间、XGBoost 特征组消融，以及供后续 SHAP 解释使用的最终模型重训练**。未经标准化的特征与标签数据集作为 Release 附件单独提供。MD 目录公开代表性输入和执行脚本，不包含原始模拟轨迹。

本文件是全项目使用说明。各模块的 README、修改记录与验证记录描述对应模块及其最初整理包的范围；其中“该代码包不含数据”的表述不代表通过 Release 单独提供的数据集不存在。

[数据下载](#data) · [输入格式](#inputs) · [运行环境](#environment) · [模型训练](#training) · [置信区间](#bootstrap) · [消融实验](#ablation) · [SHAP 模型准备](#shap) · [MD](#md) · [复现范围](#scope) · [许可与引用](#license)

## 1. 仓库结构

```text
IL-domain-framework/
├── README.md
├── README_zh-CN.md
├── requirements.txt
├── requirements-tested.txt
├── .gitignore
├── .gitattributes
├── metadata/                         # 数据说明及总特征表的本地放置位置
├── data/
│   ├── task1/                        # 已准备的 Task 1 输入及本地消融表
│   └── task2/                        # 已准备的 Task 2 输入及本地消融表
├── configs/
│   ├── task1/                        # 各分类器 JSON 配置
│   │   └── evaluation/               # bootstrap/、ablation/、shap/
│   └── task2/
│       └── evaluation/
├── scripts/
│   ├── training/
│   │   ├── task1/                    # 七类分类器及 _runtime.py
│   │   └── task2/                    # 七类分类器及 _runtime.py
│   └── evaluation/
│       ├── task1/                    # _eval_runtime.py；bootstrap/、ablation/、shap/
│       └── task2/                    # _eval_runtime.py；bootstrap/、ablation/、shap/
├── MD/
│   ├── scripts/run_md.sh
│   ├── input_templates/              # amber/、packmol/、tleap/
│   └── structures/Bmim.pdb
└── docs/                             # 各模块说明及验证记录
```

`results/` 由分析程序在本地生成，不表示源码下载中已经包含预先计算好的论文结果。大型文件 `feature_labels.csv` 需要单独下载，再放入下文指定的本地位置。

请保留目录层级。每个任务的训练脚本需要本目录下的 `_runtime.py`；Bootstrap 和消融需要该任务的 `_eval_runtime.py`。不要交换不同任务的辅助文件，也不要把单个脚本移出其配置目录树。SHAP 模型重训程序不依赖上述辅助文件，但仍需要对应 JSON 配置。

本文使用当前仓库的**大写 `MD/`** 路径。部分较早的模块说明使用小写 `md/`；在此仓库中执行这些示例时，应相应使用 `MD/`。

<a id="data"></a>
## 2. 下载特征与标签数据集

进入[数据发布页面](https://github.com/YuZhang-07/IL-domain-framework/releases)，打开包含数据集的 Release，在 **Assets** 中下载 **`feature_labels.csv`**。初始数据发布的标题为 **Workflow data v1.0.0**；记录版本时，应使用该页面实际显示的标签和链接。

| 项目 | 说明 |
|---|---|
| 数据文件名 | `feature_labels.csv` |
| 内容 | 未经标准化的完整特征数据及对应标签 |
| 格式 | CSV |
| 下载后的本地位置 | `metadata/feature_labels.csv` |
| 与训练输入的关系 | 总特征表，不是任一任务可直接使用的标准化训练输入 |

保持文件名不变，放到仓库根目录下的以下位置：

```text
metadata/
└── feature_labels.csv
```

该文件是**已经提取的特征数据表**，不是原始 MD 轨迹归档。“未标准化”表示特征的缩放状态，不表示未进行特征提取或其他上游处理。

数据附件需要与源码分别下载。GitHub 自动生成的 `Source code (zip)` 和 `Source code (tar.gz)` 包含对应的仓库快照，不会自动包含另外上传的 Release 附件。参见 [GitHub Release 官方说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

请使用配套的代码与数据版本。之后提交到 `main` 的更改不会自动进入先前的 Release 标签。仓库和附件能否访问取决于仓库的可见性与权限；README 本身不会将私有仓库变成公开仓库。

<a id="inputs"></a>
## 3. 任务输入与标签处理

模型训练入口需要**按对应任务准备好的特征表**，默认本地路径如下：

| 任务 | 默认输入路径 | 输入标签 | 内部编码 | 保存的预测分数所对应的正类 |
|---|---|---|---|---|
| Task 1 | `data/task1/proid_0_1_feature_labels_standard.csv` | `0`、`1` | `0 → 0`、`1 → 1` | 输入标签 `1` |
| Task 2 | `data/task2/proid_1_2_feature_labels_standard.csv` | `1`、`2` | `1 → 0`、`2 → 1` | 原始标签 `2` |

这些路径说明程序**需要的本地输入**，不表示数据 Release 一定同时提供了这两份文件。位于其他位置的输入可以通过 `--data` 指定。

### 总表与任务训练表的区别

训练入口不会从 `metadata/feature_labels.csv` 自动生成上述任务表。任务样本筛选、科学标签赋值及特征标准化属于上游步骤。精确复现需要原研究实际采用的任务构建规则与转换，或与之配套的已处理输入表；本说明不另行补写一套替代性的预处理方案。

**不要只把未标准化的总表改名成某个标准化训练文件名。** 程序按输入值直接训练，不在内部增加标准化器。Task 1 不会自动把第三类合并为标签 `1`；Task 2 只执行上述 `1/2 → 0/1` 编码转换。生物学响应定义与样本纳入范围应遵循研究实际标注，不能仅凭标签整数推断。

### CSV 的按位置读取规则

```text
protein_id,feature_1,feature_2,...,feature_N,label
```

**第一列**提供蛋白体系分组 ID，**最后一列**提供任务标签，**中间所有列**都被视为模型特征。复现运行时，应保留列名、特征顺序和原始样本顺序。不要把 `frame_id`、导出的 dataframe 索引或其他非预测元数据插入中间特征列。

标签属于帧级信息，同一 `protein_id` 内可以存在不同标签；流程不会把一个蛋白强制归为单一类别。Task 2 导出的 `true_label` 和 `pred_label` 仍保留原始 `1/2` 编码。

保留的训练行为允许 XGBoost 接收缺失特征值，其他六类分类器脚本会拒绝缺失值；无穷大特征值会被拒绝。程序不额外执行缺失值插补。脚本内部的分组划分，本身不能证明上游标准化是在哪个样本范围上拟合的。

<a id="environment"></a>
## 4. Python 运行环境

本文命令均从**仓库根目录**执行。多行命令使用 Bash 续行语法；在 Windows PowerShell 或命令提示符中，可以去掉行末 `\`，将整条命令合并为一行。

在选定的 Python 环境中安装共用依赖：

```bash
python -m pip install -r requirements.txt
```

也可以安装代码整理对照中记录的直接依赖版本：

```bash
python -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` 记录的是 Python 3.13.5 环境中的代码对照版本，**不代表原论文运行环境的重建结果**，不是包含全部传递依赖的完整锁定文件，也不证明全部结果已在各种平台上复现。需要精确重现历史运行且原研究环境可用时，应使用原环境。各模块文档目录中另有相应依赖记录。

SHAP 模型重训程序不计算 SHAP 值，因此这一步不需要 `shap` 包。逻辑回归中独立的可选 SHAP 分支默认关闭，不等同于 XGBoost 解释流程。MD 需要本地另行安装 AMBER；体系建模还需要 Packmol 和 `tleap`。这些可执行程序不会通过 Python 依赖文件安装。

<a id="training"></a>
## 5. 分类器训练

### 分类器与配置

以下文件名同时适用于 `scripts/training/task1/` 和 `scripts/training/task2/`。每个程序会自动读取 `configs/task1/` 或 `configs/task2/` 下对应的 JSON。

| 分类器 | 训练脚本 | 配置文件名 |
|---|---|---|
| XGBoost | `train_xgboost.py` | `xgboost.json` |
| 逻辑回归 | `train_logistic_regression.py` | `logistic_regression.json` |
| 高斯朴素贝叶斯 | `train_gaussian_nb.py` | `gaussian_nb.json` |
| K 近邻 | `train_knn.py` | `knn.json` |
| 线性支持向量机 | `train_linear_svm.py` | `linear_svm.json` |
| 决策树 | `train_decision_tree.py` | `decision_tree.json` |
| 随机森林 | `train_random_forest.py` | `random_forest.json` |

准备好任务输入后，可按以下命令运行 XGBoost。

**Task 1**

```bash
python scripts/training/task1/train_xgboost.py \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --output-dir results/task1/xgboost/run_01 \
  --n-jobs -1
```

**Task 2**

```bash
python scripts/training/task2/train_xgboost.py \
  --data data/task2/proid_1_2_feature_labels_standard.csv \
  --output-dir results/task2/xgboost/run_01 \
  --n-jobs -1
```

运行其他分类器时，使用表中的对应入口。`run_01` 是示例运行名；不同任务、分类器和运行批次应分别使用独立输出目录。省略 `--output-dir` 时，程序自动生成带时间戳的目录。

### 评价协议

默认训练配置采用以 `protein_id` 分组的 **5 折外层 `GroupKFold`**，随机种子为 **42**，每个外折请求 **50 组候选参数**。候选由各模型自己的抽样或枚举代码生成，不是通过 Optuna 搜索。

每个外层训练折内部使用**一个组感知训练／验证留出划分**，保留原来的 `StratifiedGroupKFold` / `GroupShuffleSplit` 回退和重试逻辑，不是对五个内层验证折取平均。超参数依据内层验证 AUC 选择；XGBoost 还使用该验证子集早停。外层测试子集不参与这些选择。

用于外层评价的模型在**内层训练样本**上拟合，不再用完整外层训练折重新拟合。近常数特征筛选仍依据外层训练折进行。另行保存的全量模型不替代留出评价；应从匹配的 CV 表报告外折指标，不能改用全量模型在训练数据上的表现。

### 路径与命令行参数

**JSON 内部**的相对路径以仓库根目录为基准；**命令行显式传入**的相对路径以终端当前工作目录为基准。本地绝对路径可以在运行时指定，不需要写入共享代码。

训练入口支持 `--config`、`--data`、`--output-dir`、`--n-jobs`、`--n-trials`、`--seed` 和 `--dry-run`。自定义配置必须是对应模型的完整配置。修改种子、候选数量、搜索空间或验证设置会改变实验。

```bash
python scripts/training/task1/train_xgboost.py --help
python scripts/training/task2/train_xgboost.py --dry-run
```

对于这些训练入口，`--dry-run` 只打印实际配置和输入文件是否存在，不读取数据集或拟合模型，因此不等于完成数据内容验证。`--n-jobs -1` 保留模型支持的并行方式，不会将外折自动分发到多个节点，也不会为没有对应接口的模型增加模型级并行。

<a id="bootstrap"></a>
## 6. Bootstrap 置信区间

Bootstrap 程序位于 `scripts/evaluation/task1/bootstrap/` 和 `scripts/evaluation/task2/bootstrap/`，配置位于对应的 `configs/task*/evaluation/bootstrap/`。

输入为**同一次完整训练保存的外层测试预测**：

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task1/xgboost/run_01

python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/xgboost/run_01
```

默认输出到 `<result-dir>/bootstrap/`。其他分类器入口为 `bootstrap_ci_logistic_regression.py`、`bootstrap_ci_gaussian_nb.py`、`bootstrap_ci_knn.py`、`bootstrap_ci_linear_svm.py`、`bootstrap_ci_decision_tree.py` 和 `bootstrap_ci_random_forest.py`。

### 预测文件中的列

所有输入均需 `protein_id`、`true_label` 及相应模型的分数列。六个非 XGBoost 程序还要求保存的 `pred_label` 和匹配的 `cv_*_results.csv`。XGBoost 保留其原有的可选预测标签和警告式 CV 对照行为。复现五折分析时，应提供全部五个外折，不要混用不同任务或运行批次。

| 模型 | CV 汇总表 | Task 1 分数列 | Task 2 分数列 |
|---|---|---|---|
| XGBoost | `cv_xgb_results.csv` | `pred_proba` | `pred_proba` |
| 逻辑回归 | `cv_logistic_regression_results.csv` | `pred_proba_class_1` | `pred_proba_class_2` |
| 高斯朴素贝叶斯 | `cv_gaussian_nb_results.csv` | `pred_proba` | `pred_proba` |
| KNN | `cv_knn_results.csv` | `pred_proba` | `pred_proba` |
| 线性 SVM | `cv_linear_svm_results.csv` | `decision_score` | `decision_score` |
| 决策树 | `cv_decision_tree_results.csv` | `pred_proba` | `pred_proba` |
| 随机森林 | `cv_random_forest_results.csv` | `pred_proba_class_1` | `pred_proba_class_2` |

SVM 决策分数不是概率。XGBoost、NB、KNN 和 DT 保留 `>= 0.5` 的概率阈值；LR 和 RF 保留以 `> 0.5` 判为正类的规则；SVM 使用决策分数 `> 0`。因此，精确处于阈值上的样本不一定在所有模型中归入相同类别。

### 统计解释

默认采用 **2,000 次蛋白簇 Bootstrap**、随机种子 **42** 和 **95% 百分位数置信区间**。在**每个外折内部**对蛋白体系有放回抽样，同一蛋白全部帧共享该蛋白的抽样重数；每次先计算各折指标，再对各折指标取平均。

这些区间反映**固定外层测试预测在蛋白体系重采样下的不确定性**，不是独立逐帧 Bootstrap、合并全部预测后的 pooled AUC 区间，也不包含重新训练模型和重新选择超参数的不确定性。

主要输出为 `bootstrap_ci_summary.csv`、`bootstrap_distribution.csv`、`recomputed_fold_metrics.csv`、`protein_label_composition.csv`、`bootstrap_metadata.json` 及运行记录。详见 [Task 1 评估说明](docs/task1/evaluation/README_zh-CN.md) 和 [Task 2 评估说明](docs/task2/evaluation/README_zh-CN.md)。

<a id="ablation"></a>
## 7. XGBoost 特征组消融

消融由**生成删列数据表，再重新训练 XGBoost**两步组成。两个任务分别使用自己的程序和配置。确切删列名称或匹配表达式保存在 `configs/task*/evaluation/ablation/`。

| 变体 | 删除的特征组 | 配置中的预期数量 | 生成的 CSV 文件名 |
|---|---|---:|---|
| `remove_global` | 整体蛋白标量描述符 | 8 | `removeW.csv` |
| `remove_interface` | 界面标量描述符 | 5 | `remove_Inter.csv` |
| `remove_contacts` | 界面残基对接触 | 210 | `remove_resn.csv` |
| `remove_hbonds` | 界面残基对氢键 | 210 | `remove_hbond.csv` |

以上是配置中的特征组规模，不表示已独立统计了每份下载数据的实际列数。生成的 CSV 放到 `data/task1/ablation/` 或 `data/task2/ablation/`。

**Task 1 示例：删除整体蛋白标量描述符**

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py

python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py \
  --data data/task1/ablation/removeW.csv \
  --output-dir results/task1/ablation/remove_global/run_01 \
  --n-jobs -1
```

**Task 2 示例：对应的消融**

```bash
python scripts/evaluation/task2/ablation/prepare_remove_global.py

python scripts/evaluation/task2/ablation/run_xgboost_remove_global.py \
  --data data/task2/ablation/removeW.csv \
  --output-dir results/task2/ablation/remove_global/run_01 \
  --n-jobs -1
```

运行其他特征组时，替换脚本名和输出目录中的变体名称，并使用表中对应的 CSV。准备入口读取**完整的已处理任务输入表**；训练入口读取**已经删列的表**，不会再次删列。已有消融输入时可直接指定，无需重复准备。

每个变体均保留独立的内层验证选参和外层分组评价，不是对一个已经训练的模型进行固定超参数消融。应与同一任务、同一样本集的全特征运行结果比较。配置中保留的历史显示用 `BASELINE_AUC` 不能替代实际的全特征对照结果。

消融结果的置信区间复用 XGBoost Bootstrap 程序，例如：

```bash
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/ablation/remove_global/run_01
```

<a id="shap"></a>
## 8. 为 SHAP 准备最终 XGBoost 模型

`scripts/evaluation/task*/shap/retrain_xgboost_for_shap.py` 的职责是**为后续解释准备最终模型，不是计算 SHAP 值或绘制 SHAP 图**。

程序读取匹配的已处理任务表及已有的 `cv_xgb_results.csv`，将各折 `best_params` 去重，再用全数据内部的组感知交叉验证比较候选。选中候选的 `best_iteration + 1` 中位数决定最终树数，随后关闭早停进行全量拟合。

这一模型准备步骤会评价全部内部折，与分类器训练中每个外折只使用一个内层留出划分不同。这里的内部 AUC 是选参记录，**不是新的泛化性能估计**。原外层 CV 文件只读、不被修改，仍是报告外折 AUC 的来源。

```bash
python scripts/evaluation/task1/shap/retrain_xgboost_for_shap.py \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --cv-results results/task1/xgboost/run_01/cv_xgb_results.csv \
  --output-dir results/task1/shap/final_model_for_shap/run_01 \
  --n-jobs -1

python scripts/evaluation/task2/shap/retrain_xgboost_for_shap.py \
  --data data/task2/proid_1_2_feature_labels_standard.csv \
  --cv-results results/task2/xgboost/run_01/cv_xgb_results.csv \
  --output-dir results/task2/shap/final_model_for_shap/run_01 \
  --n-jobs -1
```

每个任务读取 `configs/task*/evaluation/shap/refit_xgboost.json`。已有 CV 文件需要明确指定，程序不会自动选择最新运行。

主要输出为 `xgb_final_model.joblib`、`final_model_for_shap_meta.json`、`candidate_parameter_evaluation.csv`、`candidate_fold_details.csv` 及运行记录。`features_used_by_model` 记录树分裂使用的特征，不是 SHAP 重要性。程序未新增固定 Top17 或 Top20 截取。

仅有重训程序不等于完整公开了 SHAP 数值分析；背景样本、解释设置、SHAP 值计算和图形生成仍需要实际的解释程序。详见 [SHAP 模型准备说明](docs/shap_model_refit/README_zh-CN.md)。

<a id="md"></a>
## 9. 代表性 MD 输入与执行流程

[MD 目录](MD/) 包含部分输入模板和执行脚本：

| 位置 | 内容 |
|---|---|
| `MD/input_templates/amber/` | `min1.in`、`min2.in`、`min3.in`、`heat.in`、`eq1.in`、`eq2.in`、`eq3.in` |
| `MD/input_templates/packmol/packmol_example.inp` | 一个代表性 Packmol 建模输入 |
| `MD/input_templates/tleap/tleap_example.in` | 代表性拓扑准备输入 |
| `MD/structures/Bmim.pdb` | Bmim 分子坐标 |
| `MD/scripts/run_md.sh` | AMBER 执行顺序及重启文件衔接 |

**Packmol 文件只是示例，不是全部蛋白各自的实际建模设置集合。** 不同蛋白体系使用了不同盒子尺寸；示例中的装填边界、周期盒子和约束残基掩码不能直接当作所有蛋白的统一设置。

保留的原始顺序为：三步最小化、升温、连续四次使用 `eq3.in`、一次使用 `eq1.in`，再连续十次使用 `eq2.in`。

该目录不包含原始轨迹、全部蛋白起始结构、完整拓扑集合或完整 Bmim 参数化文件。建模示例引用 `protein.pdb`、`water.pdb`、`Bmim.mol2`、`Bmim.frcmod` 等本地文件；MD 执行脚本需要已准备好的有效拓扑和初始坐标。`Bmim.pdb` 不能替代力场参数。

```bash
bash MD/scripts/run_md.sh --help
```

示例工作目录内已有有效的 `cmp.prmtop`、`cmp.inpcrd` 时：

```bash
bash MD/scripts/run_md.sh \
  --work-dir MD/work/system_example \
  --output-dir MD/work/system_example/md_output
```

执行前请阅读 [MD 说明](MD/README_zh-CN.md) 中的输入限制，尤其是其中已记录的升温输入 `&wt` / `nmropt` 问题，而不是静默替换原协议。这些文件属于代表性工作流材料，不是能够独立运行全部体系的模拟归档。不在仓库中公开轨迹，不等于关闭本地 MD 运行时的轨迹输出。

## 10. 输出与运行记录

本地输出通常位于：

```text
results/task1/<classifier>/<run>/
results/task2/<classifier>/<run>/
results/task1/ablation/<variant>/<run>/
results/task2/ablation/<variant>/<run>/
results/task1/shap/final_model_for_shap/<run>/
results/task2/shap/final_model_for_shap/<run>/
```

Bootstrap 默认使用所选训练运行目录下的 `bootstrap/`，也可以指定其他输出位置。程序会拒绝已有内容的输出目录；消融准备会拒绝已经存在的输出数据表，不会为了新运行删除旧结果。

训练与评估程序写入 `run_config.json`、`environment.json` 和 `run_metadata.json`，用于标识实际配置、观察到的软件版本以及输入／代码引用。各模块记录字段存在差别，详见相应说明。这些记录不会使合成测试输出变成论文实际结果。

分享用 JSON 会将支持处理的外部目录引用隐藏为 `<external>/<filename>`；恢复本地路径前，这类记录不一定能直接作为运行配置。终端异常回溯及 MD 引擎日志仍可能包含本地路径。忽略文件属于本地 Git 规则，不保证任意上传文件已经脱敏。

<a id="scope"></a>
## 11. 复现范围与模块文档

代码支持从已准备的任务表训练分类器、从匹配的留出预测计算统计量、重新训练消融模型，以及为解释准备最终模型。Release 附件另行提供未标准化的总特征表。它们是不同输入和阶段，不能混为一谈。

已提供的训练入口不负责从轨迹重建总特征表，也不定义从总表生成原始标准化任务输入的完整流程。SHAP 重训程序不执行 SHAP 计算。选定 MD 输入不包含轨迹及外部体系依赖。因此，不应把当前发布理解为从原始模拟数据出发、一条命令即可端到端复现全部论文结果的完整归档。

精确复现还取决于匹配的数据版本、样本筛选、预处理、软件环境和实验配置。程序可以生成的输出，应与实际附在 Release 中的预计算结果区分；不能根据示例文件名或说明推断论文 AUC、置信区间、DOI 或发表状态。

| 模块 | 使用说明 | 已记录的验证范围 |
|---|---|---|
| Task 1 训练 | 本根目录说明及[输入说明](data/task1/README.md) | [训练验证记录](docs/VERIFICATION.md) |
| Task 2 训练 | [Task 2 说明](docs/task2/README_zh-CN.md) | [Task 2 验证](docs/task2/VERIFICATION.md) |
| Task 1 Bootstrap 与消融 | [Task 1 评估说明](docs/task1/evaluation/README_zh-CN.md) | [Task 1 评估验证](docs/task1/evaluation/VERIFICATION.md) |
| Task 2 Bootstrap 与消融 | [Task 2 评估说明](docs/task2/evaluation/README_zh-CN.md) | [Task 2 评估验证](docs/task2/evaluation/VERIFICATION.md) |
| SHAP 模型准备 | [模型重训说明](docs/shap_model_refit/README_zh-CN.md) | [模型重训验证](docs/shap_model_refit/VERIFICATION.md) |
| MD 示例输入 | [MD 说明](MD/README_zh-CN.md) | 同一文档中的检查与限制 |

这些代码对照采用相应文档中记录的有限测试条件，不是对完整科学数据集或论文结论的独立验证；MD 文件整理时的检查没有实际运行 AMBER 或 Packmol 模拟。

<a id="license"></a>
## 12. 许可与引用

### 复用条款

本 README **不为代码或数据指定许可证**。适用的使用条款应以明确的许可证文件及对应数据 Release 中的许可声明为准。某项材料未提供许可证时，不应仅因其能够查看或下载，就推断已经获得开源或开放数据授权。第三方软件和材料保留各自条款。参见 [GitHub 许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。

### 引用实际使用的版本

引用本工作流时，请标明 **IL-domain-framework**、仓库链接，以及实际使用的代码标签或提交版本。引用数据时，另外标明 **`feature_labels.csv`** 及其具体版本的 Release 链接。相关论文或软件／数据归档记录存在时，应使用其真实书目信息；本说明不在未经核实的情况下提供论文引用或归档 DOI。

[Releases 页面](https://github.com/YuZhang-07/IL-domain-framework/releases) 用于查找数据。论文中的数据引用应复制单个 Release 的独立链接，而不只是引用会变化的 `main` 分支或 `latest` 页面。Release 标题不一定与其 Git 标签相同。

## 13. 使用问题与实现反馈

可访问时，通过[仓库 Issues](https://github.com/YuZhang-07/IL-domain-framework/issues) 反馈共享实现的问题，并注明任务、脚本、代码版本、相关配置和错误信息。不要公开凭据、私人系统路径或不在预定公开范围内的数据。
