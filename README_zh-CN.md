# Task 1 分类器训练：GitHub 整理版

本包包含七个分类器的修改后训练脚本、实际由代码读取的 JSON 配置、依赖说明及修改记录。

## 文件放在哪里

训练执行代码：`scripts/training/task1/`。

参数配置：`configs/task1/`。

`_runtime.py` 是七个脚本共用的路径、命令行和运行记录模块，必须留在训练脚本的同一目录。
不能只拿走某一个 `.py` 而丢掉 `_runtime.py` 和对应配置文件。
将本包内的 `scripts/`、`configs/` 等内容合并到你的仓库根目录即可，不必在仓库内再嵌套一层同名发布包目录。

## 输入文件

默认位置：

```text
data/task1/proid_0_1_feature_labels_standard.csv
```

输入契约与原脚本一致：**第一列为 protein_id，最后一列为标签 0/1，中间所有列都会被当成特征。**
因此不要额外在中间插入 `frame_id`、导出的 dataframe 索引或其他不应进入模型的元数据。
程序不要求同一蛋白的所有帧标签一致，也不会把第三类自动合并到 1。
本次没有重新标准化或加入 StandardScaler；继续使用输入中已有的数值。

这些代码本身不能证明 label 1 的生物学含义，也不能判断上游标准化的拟合范围。
本次只保留已有训练行为，不据此宣称上游处理已经过验证。

## 怎么运行

在仓库根目录，直接运行任意一个模型：

```bash
python scripts/training/task1/train_xgboost.py
python scripts/training/task1/train_logistic_regression.py
python scripts/training/task1/train_gaussian_nb.py
python scripts/training/task1/train_knn.py
python scripts/training/task1/train_linear_svm.py
python scripts/training/task1/train_decision_tree.py
python scripts/training/task1/train_random_forest.py
```

这些是七个独立命令；上面不是要求你在同一节点同时启动七个无限线程的训练进程。
脚本自动读取对应 JSON。例如 `train_xgboost.py` 自动读取 `configs/task1/xgboost.json`。

指定输入、输出和资源的示例：

```bash
python scripts/training/task1/train_xgboost.py \
  --config configs/task1/xgboost.json \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --output-dir results/task1/xgboost/run_01 \
  --n-jobs -1
```

JSON 中的相对路径以**仓库根目录**为基准；命令行显式传入的相对路径以**当前终端工作目录**为基准。
`--config` 可以省略；传入自定义配置时，需要完整的对应模型 JSON，而不是只有几个字段的片段。

不指定 `--output-dir` 时，默认生成：

```text
results/task1/模型名/年月日_时分秒_微秒/
```

指定的输出目录必须不存在或为空；非空目录会报错停止，**不会覆盖原有结果**。
代码没有提供自动删除旧结果的选项。

`--n-jobs -1` 保持原有并行设置。RF/KNN/XGBoost 会把线程参数传给模型；正整数还会限制数值库线程。
对于 LinearSVC 等没有模型级 n_jobs 参数的模型，这个选项不等于强行把算法变成全核并行。

可查看接口或配置，但不启动训练：

```bash
python scripts/training/task1/train_xgboost.py --help
python scripts/training/task1/train_xgboost.py --dry-run
```

`--dry-run` 不读取 CSV 内容、不创建输出、不训练。未放入真实数据时显示输入文件不存在是正常情况。

## 保留了哪些训练设置

原始默认种子 42、外层五折、每折请求 50 个候选、各模型搜索空间、参数采样顺序、标签 0/1 映射均保留。
外层训练折内部仍然只取**一次**组感知训练/验证划分，不是计算内层五折平均 AUC。
最佳模型仍只在 inner train 拟合后评价 outer test，没有改成在 outer train 全量重训。
最终全数据模型仍是另外保存的一份，不能替代交叉验证性能。

代码中实现的是原有随机采样或枚举搜索，不是 Optuna。此次没有替换搜索器。
RF/LR 与其他模型在最终全量特征处理上的原有差别也保留，没有为了目录整洁强行统一。

SVM 输出仍为 `decision_score`，不是概率。
RF/LR 的概率列仍为 `pred_proba_class_1`；其他概率模型仍为 `pred_proba`。
旧有结果文件名和预测列没有统一改名，避免破坏已有后处理脚本。

## 新增的小型运行记录

每次运行另存 `run_config.json`、`environment.json`、`run_metadata.json`，用于记录有效参数、实际版本和输入/代码摘要。
外部输入路径在可分享的 JSON 中显示为 `<external>/文件名`，不写入本地绝对目录。
对于外部文件，重新使用配置快照时仍需通过 `--data` 等选项提供真实位置。
本地报错堆栈、外部日志及模型对象不属于通用脱敏的保证范围。

`.gitignore` 默认排除本地训练数据、结果、模型文件、凭据/私钥文件及缓存。
本包不包含这些内容；后续真实数据与结果的公开范围仍需按论文材料另行确定。

## 测试情况与边界

七个模型均进行了原版/新版小规模对照：20 个分组、240 行合成样本，每组同时含 0/1，保留五个外折，每折训练三个候选。
XGBoost 为缩短测试将树数上限和早停耐心分别临时设为 12 和 3；正式发布配置仍为 3000 和 100。
所有模型在该测试上的各折预测、指标、所选参数及最终模型预测一致（比较容差 1e-12）。
另外单独核对了正式默认配置及五个外折种子的 50 个候选顺序，均与原脚本一致。

原始论文环境未提供，因此 `requirements-tested.txt` 仅代表本次测试环境（Python 3.13.5），不是原论文的历史环境。
`requirements.txt` 是不固定版本的依赖清单；LR 默认关闭的 SHAP 分支未测试。

详细修改记录见 `docs/CHANGES.md`；测试记录见 `docs/VERIFICATION.md`。
