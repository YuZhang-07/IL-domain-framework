# MD 输入与运行脚本：限定公开版本

本目录只整理本次提供的 **11 个 MD 相关文件**，另附两份说明和两个 Git
规则文件。不包含轨迹、完整蛋白体系、拓扑数据集或新增力场文件，也不自动
下载或上传任何文件。它是输入模板和运行流程的公开子集，不是完整复现包。

## 放在哪里

将整个 `md/` 文件夹合并到现有仓库最外层，与 `scripts/`、`configs/`、`docs/`
并列。不放到机器学习的 `training/` 或 `evaluation/`，也不按 Task 1/Task 2 重复。

```text
md/
├── README.md
├── README_zh-CN.md
├── .gitignore
├── .gitattributes
├── scripts/
│   └── run_md.sh
├── input_templates/
│   ├── amber/
│   │   ├── min1.in
│   │   ├── min2.in
│   │   ├── min3.in
│   │   ├── heat.in
│   │   ├── eq1.in
│   │   ├── eq2.in
│   │   └── eq3.in
│   ├── packmol/
│   │   └── packmol_example.inp
│   └── tleap/
│       └── tleap_example.in
└── structures/
    └── Bmim.pdb
```

`.in` 和 `.inp` 本身就是模拟输入配置，不再额外拆成 JSON/YAML。
本包所有文件只写入 `md/`，不替换已有机器学习模块或仓库根目录文件。

## 修改内容

- 去掉上传文件名中的 `(1)` 后缀；`run(1).sh` 改名为 `scripts/run_md.sh`。
- 七个 AMBER 输入保留全部原有关键字及取值，包括约束掩码、步数、步长、
  `ig=-1`、温压控制和输出频率。删除行尾注释，统一文本格式。
- **明确的格式修补：** `eq2.in` 增加首行标题；`heat.in` 最后一个
  `&wt TYPE='END'` 补上结束符 `/`；`min1.in` 与 `min2.in` 的两行约束赋值
  末尾补逗号。没有把这些改动描述为原文件字节不变，也不据此保证 MD 数值复现。
- Packmol 的示例蛋白文件名从 `214_3.30.200.20_2.60.40.10.pdb` 改为
  `protein.pdb`；其余有效指令不变。固定放置的六个 `30.0` 也原样保留。
- tleap 所有有效指令与执行顺序保留；其文件引用继续基于本地运行目录。
- `Bmim.pdb` 按原始字节复制，不改变坐标、原子名称、残基名称或排列。
- `run_md.sh` 支持命令行路径和可选 GPU，保留全部 19 次调用、输出名称及
  重启文件衔接；增加缺失输入、命令失败和已有输出的停止保护。

## 已记录的边界，不在本次补写计算方案

**升温设置需要保留警告。** 原 `heat.in` 写有 `&wt TYPE='TEMP0'`，但
`&cntrl` 没有显式设置 `nmropt`。新版没有擅自添加 `nmropt`、`tempi` 或
`temp0`，因此不能声称原模拟已经按该段文字实现了 10 到 300 的温度调度。
AMBER 官方[升温示例](https://ambermd.org/tutorials/basic/tutorial5/index.php)
在这类输入中显式使用 `nmropt=1`；这里仅记录原文件遗漏，不将整理变成协议修改。
运行脚本会显示这条警告的阅读提示。

**示例数值不是全部体系的统一设置。** Packmol 保留 `0–65` 的三维装填范围，
tleap 保留 `{67.0, 67.0, 67.0}` 的盒子，两者没有强行统一。
`min1.in` 的 `:1-25` 和 `min2.in` 的 `:1-25@CA,C,N` 也按原值保留。
这些尺寸和残基选择都不能直接代表所有蛋白。
Packmol 的 5000 个水、100 个 Bmim 分子副本，以及 tleap 的原有加离子指令均不改。

**本包未包含的本地依赖：** Packmol 还引用 `protein.pdb`、`water.pdb`；
tleap 还引用 `Bmim.frcmod`、`Bmim.mol2` 和装填后的 `BMIMS_pro.pdb`。
MD 脚本从已经准备好的 `cmp.prmtop`、`cmp.inpcrd` 开始。
`Bmim.pdb` 是结构文件，不等于力场参数。这些依赖没有凭空生成，也没有要求你
为了当前限定公开版本继续上传；这里只说明读者运行时仍需要它们。
软件安装与版本不能仅从这些文件完整确认，本包没有虚构版本锁定文件。

## 有本地依赖时如何运行

发布这份代码不要求重新跑 MD。以下命令仅供已有有效本地输入的使用者参考。
建议把运行数据放在本地 `md/work/system_example/` 或仓库外，代码保留在上述位置。

建模时，将 Packmol/tleap 示例和 `structures/Bmim.pdb` 复制到与本地蛋白、水、
参数文件相同的运行目录，分别使用 `packmol.inp`、`tleap.in`、`Bmim.pdb` 作为
本地文件名；然后从该运行目录调用：

```bash
packmol < packmol.inp
tleap -f tleap.in
```

已有有效拓扑与初始坐标时，从仓库根目录运行：

```bash
bash md/scripts/run_md.sh \
  --work-dir md/work/system_example \
  --output-dir md/work/system_example/md_output
```

默认读取 `--work-dir` 中的 `cmp.prmtop`、`cmp.inpcrd`；默认 AMBER 输入目录由
脚本位置确定，不依赖个人绝对路径。默认输出为 `WORK_DIR/md_output`。
所有**显式传入的相对路径**均相对于调用命令时的终端目录。

`--pmemd` 指定本地 AMBER 可执行程序；默认使用 `PMEMD_CUDA` 环境变量或
`pmemd.cuda`。`--gpu` 显式指定 `CUDA_VISIBLE_DEVICES`。不传时保留已有的 GPU
分配，只有变量未设置时才使用原默认值 `0`，避免覆盖调度器的分配。
`--help` 列出输入路径和 `--dry-run` 等选项。

`--dry-run` 只打印命令，不执行 MD 或新建结果文件，但仍检查模板及拓扑/坐标文件
存在。它不验证这些文件的物理正确性。脚本用于从头运行，不提供断点续跑逻辑。

原流程保持：

```text
min1 → min2 → min3 → heat
→ eq3 连续四次（cmp_eq13 至 cmp_eq16）
→ eq1 一次（cmp_eq0）
→ eq2 连续十次（cmp_eq1 至 cmp_eq10）
```

MD/快照/目录包含从水和 BmimCl 条件下的分子动力学模拟中提取的选定 PDB 结构快照。这些结构用于可视化和结构检查，并不代表完整的 MD 轨迹。

## 验证与隐私范围

已核对七个 AMBER 输入的原有关键字/取值、Packmol/tleap 有效指令和 PDB 字节；
执行 `bash -n`，并通过模拟可执行程序比较原版与新版的 19 条 AMBER 调用及衔接。
同样用模拟文件验证了已有输出保护、命令失败停止和 GPU 变量处理。
**未安装或执行 AMBER、tleap、Packmol，没有进行真实 MD 或物理有效性验证。**

对选定原文件和发布代码/输入检查了常见凭据、令牌、私钥块、硬编码个人路径及
明显内部主机标识，未发现所检查模式。这不覆盖未提供文件或 Git 历史。
控制台、`--dry-run` 命令、模拟引擎日志及报错可能出现本地路径，不视为已自动脱敏。

`md/.gitignore` 仅允许本包的 15 个文件被普通 Git 暂存，其他 MD 本地文件默认忽略。
它不替换仓库根目录规则，也不删除已经跟踪的文件；不能把它当成网页上传过滤器。
网页上传时只选择本包干净的 `md/` 目录，保留其中两个隐藏 Git 文件。
