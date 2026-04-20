# Chemical_Synthesis

这是一个面向学习与实践的化学合成相关仓库。仓库会持续补充不同主题的教学示例，方便学习者把代码拉到本地后，结合 notebook 和源码完成动手实践。

## 如何开始

先拉取仓库：

```bash
git clone --recurse-submodules <your-repo-url>
cd Chemical_Synthesis
```

如果仓库已经拉取过，但还没有初始化子模块，可以补执行：

```bash
git submodule update --init --recursive
```

如果你需要把当前仓库一键同步到书仓库 Graph-Neural-Networks-in-Modern-Drug-Discovery 的 Chapter7，可以先预演：

```bash
bash sync_to_chapter7.sh
```

确认输出无误后再正式执行：

```bash
bash sync_to_chapter7.sh --apply
```

这个脚本只保留在当前 Chemical_Synthesis 源仓库中，不会被同步到 Chapter7。

脚本会同步普通文件，并对齐 Chapter7/source_repos 下各子模块的提交；如果源仓库某个子模块里还有未提交改动，脚本会给出警告并同步工作区文件，但这类改动仍需要在对应子模块中单独提交，才能被版本化保存。

然后进入 `teaching_demos/`，选择你想实践的教程目录。

当前已经整理好的示例：

- `teaching_demos/reaction_template_tutorial/`

这个教程聚焦“逆合成模板抽取与应用”，可以直接运行：

```bash
bash teaching_demos/reaction_template_tutorial/setup_env.sh
bash teaching_demos/reaction_template_tutorial/launch_notebook.sh
```

如果你只想先阅读内容，也可以直接打开：

- `teaching_demos/reaction_template_tutorial/逆合成模板抽取与应用.ipynb`

## 建议的学习方式

推荐按下面的顺序实践：

1. 先阅读目标教程目录下的 `README.md`
2. 运行该教程自己的环境脚本，准备独立运行环境
3. 打开 notebook，按单元逐步执行和理解
4. 需要追溯实现时，再去 `source_repos/` 查看对应源码

## 仓库目录说明

- `teaching_demos/`
  存放教学材料。通常一个子文件夹就是一个独立教程。

- `source_repos/`
  存放教程中依赖或对照阅读的上游源码仓库。目前包含 `rdchiral/` 和 `rxnmapper/`。

- `envs/`
  存放本地运行环境。一般由各教程自己的脚本创建，不作为教学内容的一部分。

## 后续新增教程时的组织方式

这个仓库后续会继续增加新的教程。为了便于维护，建议统一采用下面的组织方式：

- 一个教程对应 `teaching_demos/` 下的一个子文件夹
- 每个教程目录尽量自带 `README.md`
- 每个教程目录尽量有独立的 notebook 入口
- 如果教程需要独立环境，环境脚本放在该教程目录内部
- 如果教程需要辅助函数或可视化代码，也放在该教程目录内部

也就是说，后续新增内容时，通常只需要在 `teaching_demos/` 下增加一个新的子目录，并在该目录中放入该主题自己的说明、notebook 和运行脚本。

## 当前状态

仓库内容还在持续撰写中。随着新教程加入，`teaching_demos/` 下会逐步出现更多主题子目录。

如果你是第一次接触本仓库，建议先从 `reaction_template_tutorial/` 开始。
