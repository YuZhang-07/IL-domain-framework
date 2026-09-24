# Task 2：置信区间与 XGBoost 消融

本目录说明由本次 15 个 Task 2 原始脚本整理的发布增补包：7 个分类器的
Bootstrap、4 个特征删列入口、4 个 XGBoost 消融训练入口。
**本次没有收到计算 XGBoost SHAP 的脚本；`scripts/evaluation/task2/shap/README.md`
只是缺项说明，不代表 SHAP 方法已经实现或公开。**

## 合并位置

把压缩包中 `task2_evaluation_github_release/` **里面的内容**合并到仓库根目录。
不要把整个外层文件夹放入 `evaluation/`，也不要用本包替换已有的整个 `scripts/`
或 `configs/` 文件夹。保留原有内容，在相应目录下增添以下文件：

```text
scripts/evaluation/task2/
    _eval_runtime.py
    bootstrap/    # 7 个 bootstrap_ci_*.py
    ablation/     # 4 个 prepare_*.py + 4 个 run_xgboost_*.py
    shap/README.md
configs/task2/evaluation/
    bootstrap/    # 7 份 JSON
    ablation/     # 4 份 JSON，每组供准备和训练两个入口共用
data/task2/ablation/
    README.md
    .gitignore
docs/task2/evaluation/
    README.md
    README_zh-CN.md
    CHANGES.md
    VERIFICATION.md
    verification_results.json
    file_mapping.json
    requirements.txt
    requirements-tested.txt
```

`_eval_runtime.py` 必须保留。它只处理参数、路径和运行记录，不实现 Bootstrap
统计或模型拟合。它不依赖 Task 1 的同名文件，也不替换训练目录里的 `_runtime.py`。

本包的文档已经放在 `docs/task2/evaluation/`，不需要再单独下载一遍，更不能覆盖
仓库根目录的项目总 README。`VERIFICATION.md` 记录这次整理的测试边界，不能作为
论文真实结果已经重新验证的证明。`CHANGES.md` 是整理记录，不是运行依赖。

## 运行环境

沿用现有能够运行原脚本的环境，或在独立环境内从仓库根目录执行：

```bash
python -m pip install -r docs/task2/evaluation/requirements.txt
```

`requirements-tested.txt` 记录本次实际测试的软件包版本；它不是论文原始训练环境，
也不是包含全部传递依赖的完整锁定文件。原论文软件版本未提供，不作推测。
本次测试的 Python 版本及运行结果见 `VERIFICATION.md`。

## 路径规则与输出保护

配置 JSON 内的相对路径以**仓库根目录**为基准；命令行显式指定的相对路径以
**当前终端工作目录**为基准。支持用 `--config` 选择同结构的配置文件。
配置中的科学参数会被程序实际读取，不是旁置的参数说明。

Bootstrap 不猜测“最近一次训练”，需要明确给出一次完成训练的 `--result-dir`，
或在 JSON 的 `input.result_dir` 中指定它。该目录不是容纳多个运行批次的上层目录。

输出目录必须不存在或为空；删列输出 CSV 和其 metadata 文件必须尚不存在。
程序拒绝覆盖旧文件。再次分析同一输入时，另选一个 `--output-dir` 或 `--output-file`。
原始研究输入不会被修改。

## 一、七类模型的 Bootstrap

以下 `run_01` 是示例，需替换为本地实际运行目录或时间戳目录。

```bash
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py --result-dir results/task2/xgboost/run_01
```

默认输出到指定训练目录下面的 `bootstrap/`。需要其他位置或参数时：

```bash
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/xgboost/run_01 \
  --output-dir results/task2/xgboost/run_01/bootstrap_repeat \
  --n-bootstrap 2000 --seed 42 --confidence 0.95
```

其余入口：

```text
bootstrap_ci_decision_tree.py
bootstrap_ci_gaussian_nb.py
bootstrap_ci_knn.py
bootstrap_ci_linear_svm.py
bootstrap_ci_logistic_regression.py
bootstrap_ci_random_forest.py
```

### 输入格式

预测文件必须保留原始标签 **1/2**；内部使用 `1 -> 0, 2 -> 1`。
**正类是原始标签 2。** 不要先把预测表转换成 Task 1 的 0/1 格式，也不要将
Task 1 的结果目录传入本包。

| 模型 | 预测文件后缀 | 分数列 | 原脚本判为原始标签 2 的规则 |
| --- | --- | --- | --- |
| XGBoost | `fold*_xgb_predictions.csv` | `pred_proba` | `>= 0.5` |
| Decision tree | `fold*_decision_tree_predictions.csv` | `pred_proba` | `>= 0.5` |
| Gaussian NB | `fold*_gaussian_nb_predictions.csv` | `pred_proba` | `>= 0.5` |
| KNN | `fold*_knn_predictions.csv` | `pred_proba` | `>= 0.5` |
| Linear SVM | `fold*_linear_svm_predictions.csv` | `decision_score` | `> 0` |
| Logistic regression | `fold*_logistic_regression_predictions.csv` | `pred_proba_class_2` | `> 0.5` |
| Random forest | `fold*_random_forest_predictions.csv` | `pred_proba_class_2` | `> 0.5` |

所有预测表还需 `protein_id`、`true_label`；六个非 XGBoost 脚本必须提供
`pred_label`，XGBoost 原本允许其缺失，此行为保留。
六个非 XGBoost 脚本还必须有同一次运行的 `cv_*_results.csv`，且要求五折齐全。
XGBoost 保留原有文件发现逻辑及警告式 CV 对照，不擅自改成另六个模型的严格策略；
复现论文五折结果时仍应提供完整五折，而不是把子集当成五折结果。

### 保留的统计设计

在每个 outer fold 内按 `protein_id` 整簇有放回抽样，同一蛋白内部的帧级标签
可以不同；每次先计算各折指标，再取折均值。默认 2000 次、seed 42、percentile
95% CI，最大单类重抽次数默认 10000。没有改成独立帧抽样、蛋白单一标签分层，
也没有改成合并预测后的 pooled AUC。

这些 CI 基于**固定的外层测试预测**，不包含重新训练、重新选参的不确定性。
XGBoost 保留原有 `pred_label` 不一致时警告并按概率重算的处理；其他模型继续
严格核对各自的标签规则。SVM 的分数不是概率。

默认 CV 数值核对容差取自各自 Task 2 原脚本：LR 为 `1e-7`；DT、NB、KNN、SVM、
RF 为 `1e-8`。特别是 Task 2 KNN 没有套用 Task 1 的 `1e-5` 设置。
XGBoost 原有警告式核对阈值 `1e-8` 保留。

### 输出

```text
bootstrap_ci_summary.csv
bootstrap_distribution.csv
recomputed_fold_metrics.csv
protein_label_composition.csv
bootstrap_metadata.json
run_config.json
environment.json
run_metadata.json
```

前五个文件沿用原命名；后面三份是成功运行记录。六个严格核对模型的
`recomputed_fold_metrics.csv` 继续包含原有的 CV 对照列。

## 二、四组 XGBoost 消融

默认完整特征输入：

```text
data/task2/proid_1_2_feature_labels_standard.csv
```

训练代码仍按位置读列：第一列为 `protein_id`，最后一列为标签，其余列全部是特征。
不额外进行标准化、插补或标签合并。不要把索引、`frame_id` 等不应训练的字段
额外插入中间特征列。

| 变体 | 删除规则 | 默认删列输出 |
| --- | --- | --- |
| `remove_global` | 原始脚本明确指定的 8 个整体标量列 | `data/task2/ablation/removeW.csv` |
| `remove_interface` | 原始脚本明确指定的 5 个界面标量列 | `data/task2/ablation/remove_Inter.csv` |
| `remove_contacts` | `^[A-Z]{3}-[A-Z]{3}$`，预期 210 列 | `data/task2/ablation/remove_resn.csv` |
| `remove_hbonds` | `^hbond[A-Z]{3}-[A-Z]{3}$`，预期 210 列 | `data/task2/ablation/remove_hbond.csv` |

保留原正则匹配规则及“数量不是 210 则警告、完全未匹配则报错”的行为。
明确列名写在各组 JSON 的 `preparation.columns` 中，不使用论文里的简写来猜测实际列名。

### 默认完整流程示例

```bash
python scripts/evaluation/task2/ablation/prepare_remove_global.py

python scripts/evaluation/task2/ablation/run_xgboost_remove_global.py \
  --output-dir results/task2/ablation/remove_global/run_01 --n-jobs -1

python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/ablation/remove_global/run_01
```

另三组替换为 `remove_interface`、`remove_contacts`、`remove_hbonds`。
每个变体在 `configs/task2/evaluation/ablation/` 下有一份配置，准备与训练两个入口共用。
不指定训练 `--output-dir` 时，输出位于 `results/task2/ablation/<变体>/<时间戳>/`。

已有删列 CSV 时，跳过准备，直接使用：

```bash
python scripts/evaluation/task2/ablation/run_xgboost_remove_global.py \
  --data data/task2/ablation/removeW.csv \
  --output-dir results/task2/ablation/remove_global/run_02 --n-jobs -1
```

**训练入口需要已经删列的 CSV，不会自动再删一次列。** 准备入口的 `--data`
才是完整特征输入；准备输出也可由 `--output-file` 指定。

两处输出文件名作了明确衔接调整：整体组原准备输出为
`removeW_1_2_feature_labels_standard.csv`，新版为后续训练本就读取的 `removeW.csv`；
界面组原准备输出为 `removeInter_1_2_feature_labels_standard.csv`，新版为
`remove_Inter.csv`。不改变标签和数值。已存在的旧命名 CSV 仍可通过 `--data` 使用。

每个删列输出旁有 `.csv.metadata.json`，记录实际删除的特征、保留行数与运行参数。
这些生成表默认被本地 `.gitignore` 排除；论文数据的正式归档与访问说明另行维护。

### 训练协议与历史参考值

保留默认 5 个外折、50 组候选、3000 棵树上限、100 轮早停、内部组感知验证、
原候选生成顺序、原类别加权与近常数特征处理。每个消融仍独立重新调参；没有改成
固定全特征模型参数的另一种实验。外层评价模型仍采用原来的 inner-train 拟合范围；
最终另存全量模型，不用其训练集成绩替代 CV。

`BASELINE_AUC = 0.7931` 是原代码中的历史显示参考，原值保留、来源未独立核实，
不参与拟合或选参，不能仅凭该值认定其为论文最终全特征基线。JSON 中设为 `null`
可关闭这项显示比较，而不改变模型拟合。

## 三、公开范围与隐私边界

代码使用相对路径或命令行参数。本包不包含研究 CSV、训练模型、原始脚本副本、
服务器日志、凭据或 Git 历史。对原脚本及发布代码/配置的敏感模式检查范围与结果
见 `VERIFICATION.md`；有限模式检查不是对整个工作目录的安全保证。

生成的分享用 JSON 对仓库外绝对路径隐藏目录前缀，保留文件名、特征名和科学信息。
**控制台输出、异常回溯及任意已有日志不在自动脱敏范围内。** 程序没有从环境变量中
收集密码或令牌，也不把整份环境变量或主机用户名写入运行记录。

`run_config.json` 是实际运行记录；直接运行时以 `configs/` 下的输入配置为准，
不要把带记录字段的 JSON 自动当成输入配置。

## 四、SHAP 缺项

本次 15 个文件中没有实际计算 XGBoost SHAP 的程序。消融脚本只是保存最终模型，
没有调用 SHAP explainer，也没有输出 SHAP 数值。
需要补入论文实际采用的解释代码后，才能将 SHAP 部分称为已公开。这里没有新建一个
“看起来相似”的 SHAP 方法来替代原方法。
