# reaction_template_tutorial

这个目录现在包含一套更适合教学复用的材料：

- `RXNMapper_RDChiral_教学示例.ipynb`
- `tutorial_utils.py`
- `build_notebook.py`
- `requirements.txt`
- `setup_env.sh`
- `launch_notebook.sh`

## 教学目标

围绕 `RXNMapper + RDChiral` 的核心主线，演示四个步骤：

1. 输入 **没有 atom mapping** 的 reaction SMILES
2. 使用 **RXNMapper** 自动补齐 atom mapping
3. 使用 **RDChiral** 从带映射反应中抽取 reaction template
4. 使用 **RDChiral** 将模板应用到目标产物上，观察回推出的候选前体

新版 notebook 额外强调了两件事：

- 用 `RDKit` 图形化展示原始反应、mapped reaction、template 和回推案例
- 用更清晰的中文教学文案串起“输入是什么、输出是什么、每一步为什么必要”

## 一键配置与启动

环境统一创建到：

```text
envs/reaction_template_tutorial_envs
```

执行：

```bash
bash teaching_demos/reaction_template_tutorial/setup_env.sh
bash teaching_demos/reaction_template_tutorial/launch_notebook.sh
```

其中：

- `setup_env.sh` 会创建虚拟环境、安装依赖、以 editable 方式安装本地 `rxnmapper` 和 `rdchiral`
- `launch_notebook.sh` 会直接用该环境启动 Jupyter Lab 并打开本 notebook

## 依赖来源

- `rdchiral` 来自 `source_repos/rdchiral`
- `rxnmapper` 来自 `source_repos/rxnmapper`
- `torch` 通过 CPU wheel 安装
- `rdkit / transformers / pandas / scipy / jupyterlab` 通过 `requirements.txt` 安装

## 当前验证案例

示例反应：

```text
CCOCC.C[Mg+].O=Cc1ccc(F)cc1Cl.[Br-]>>CC(O)c1ccc(F)cc1Cl
```

已验证主线输出包括：

- 映射后的 reaction SMILES
- 抽取出的 retrosynthesis template
- 原始目标分子的候选前体：`O=Cc1ccc(F)cc1Cl.[CH3][Mg+]`
- 一组相似芳香醇目标分子的模板泛化结果

## 维护说明

如果你修改了 notebook 结构，建议同时更新：

- `tutorial_utils.py`
- `build_notebook.py`

然后重新生成 notebook：

```bash
envs/reaction_template_tutorial_envs/bin/python teaching_demos/reaction_template_tutorial/build_notebook.py
```

## 备注

为兼容当前 `rxnmapper` 的导入链，环境里固定了 `setuptools<81`。
