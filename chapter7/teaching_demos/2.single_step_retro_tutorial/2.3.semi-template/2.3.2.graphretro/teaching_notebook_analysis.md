# Teaching Notebook Structure & Style Guide

## PROJECT CONTEXT

**Project Root:** `/home/xiaoruiwang/backup_data/ubuntu_data/other_work/GNN_AIDD/Chemical_Synthesis`

**Relevant Directories:**
- `teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.1.g2gs/` - G2Gs reference template
- `teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.2.graphretro/` - GraphRetro target (notebooks are currently empty)
- `source_repos/` - Original source code repository
- `envs/` - Environment setup files

**Status:**
- ✅ G2Gs notebooks (1-3) are complete with 3_模型展示.ipynb as reference template
- ⚠️ GraphRetro notebooks 1 and 2 are empty stubs (0 bytes)
- ✅ GraphRetro demo_data.csv exists with 20 sample reactions

---

## DEMO DATA (demo_data.csv)

**File:** `teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.2.graphretro/data/demo_data.csv`

**Structure:**
- **Columns:** `id`, `class`, `reactants>reagents>production`
- **Rows:** 20 reactions
- **Format:** SMILES with atom atom mapping

**Sample Data:**

| id | class | reactants>reagents>production |
|----|-------|-------|
| US07928231B2 | 5 | `[C:12](=[O:13])([O:14][C:15]...)>>[CH3:1][C:2](=[O:3])[c:4]1...` |
| US20090192322A1 | 5 | `[C:13](=[O:14])([O:15][C:16]...)>>[CH3:1][c:2]1[cH:3][cH:4]...` |
| US20080146614A1 | 10 | `[Br:28][N:35]1[C:30]...>>[CH3:1][CH2:2][O:3]...` |

**Rows included:** 20 complete USPTO reactions with reaction class labels (3-10)

---

## G2GS REFERENCE NOTEBOOK (3_模型展示.ipynb)

**File:** `teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.1.g2gs/3_模型展示.ipynb`

**Total Cells:** 28 cells (alternating markdown explanation + code demonstration)

### 📋 COMPLETE CELL STRUCTURE

| Cell # | Type | Content Summary | Lines |
|--------|------|-----------------|-------|
| 0 | Markdown | **Main Title:** "# G2Gs 源码拆解与预测流程" (G2Gs Source Code Dissection & Prediction Flow). Introduces the notebook's purpose and core formula: P(R\|P) = Σ P(c\|P) · P(R\|S(c,P)) | - |
| 1 | Code | **Setup Cell:** Project root finder, path configuration, imports, helper functions. Defines `find_project_root()`, `PROJECT_ROOT`, `TUTORIAL_DIR`, `CODE_DIR`, `DATA_DIR`, and `show_source_block()` helper. Imports torch, pandas, IPython, g2gs_tutorial functions. Defines `SOURCE_BLOCKS` list with 7 major stages. | 95 lines |
| 2 | Markdown | ## 1. 先看源码地图 (First, see the source code map). Explains why beginners should first understand code structure before diving into implementation details. | - |
| 3 | Code | Displays `SOURCE_BLOCKS` as a pandas DataFrame showing 7 prediction stages with file paths, symbols, and line ranges | 1 line |
| 4 | Markdown | ## 2. RGCN：公共图编码骨干 (RGCN: Shared Graph Encoding Backbone). Explains that both stages use RGCN to encode molecule graphs into node_feature and graph_feature tensors. | - |
| 5 | Code | `show_source_block()` call to display RGCN source code from `source_repos/torchdrug/torchdrug/models/gcn.py:88-168` | 7 lines |
| 6 | Code | **Demo execution:** Loads example reaction, creates product graph, instantiates CenterIdentificationModel, runs encoder, displays tensor shapes (node_feature, encoder output shapes) | 27 lines |
| 7 | Markdown | Key insight: RGCN only encodes, doesn't output reactants directly. Explains encoder-vs-task-head separation. | - |
| 8 | Markdown | ## 3. 数据预处理：center 与 synthon 的定义来自哪里 (Data preprocessing: where center & synthon definitions come from). Emphasizes understanding data processing before model code. | - |
| 9 | Code | `show_source_block()` displaying USPTO50k dataset preprocessing functions (`_get_difference`, `_get_reaction_center`, `_get_synthon`) from line 97-220 | 7 lines |
| 10 | Code | Builds synthon dataset from example, displays DataFrame with pair_id, reaction_center, reactant_smiles, synthon_smiles | 15 lines |
| 11 | Markdown | Explains why two-stage design exists: Stage 1 predicts center (not reactants), Stage 2 works on synthons (not full product). Intermediate representations are core design. | - |
| 12 | Markdown | ## 4. 第一阶段：CenterIdentification 源码 (Stage 1: CenterIdentification source code). Lists 3 reading focus points about inputs, node/edge scoring, center→synthon conversion. | - |
| 13 | Code | `show_source_block()` displaying `CenterIdentification.predict()` code from line 118-154 | 7 lines |
| 14 | Code | `show_source_block()` displaying `CenterIdentification.predict_synthon()` code from line 168-233 | 7 lines |
| 15 | Code | **Demo execution:** Runs center_model on product_graph, ranks centers (topk=8), displays tensor shapes and ranked centers. Prints oracle reaction center. | 20 lines |
| 16 | Markdown | Summarizes stage 1 as 5-step process: encode product → concat conditions → score node/edge heads → rank top-k centers → convert to synthons. Calls it "reaction center proposer" not "reactant predictor". | - |
| 17 | Markdown | ## 5. 第二阶段：SynthonCompletion 的动作空间 (Stage 2: SynthonCompletion action space). Explains action decomposition: each action has node_in, node_out, bond_type, stop components. | - |
| 18 | Code | `show_source_block()` displaying `SynthonCompletion._topk_action()` code from line 701-788 | 7 lines |
| 19 | Markdown | Explains `_topk_action()` design: breaks one edit step into multiple scorable decisions combined together. Benefits: clear decisions, structured action space, compatible with beam search. | - |
| 20 | Code | `show_source_block()` displaying `SynthonCompletion._apply_action()` and `predict_reactant()` code from line 790-920 | 7 lines |
| 21 | Code | **Demo execution:** Uses first pair from synthon_dataset, instantiates SynthonCompletionModel, runs `_build_node_context()` and `score_actions()`, displays tensor shapes and top actions DataFrame | 38 lines |
| 22 | Markdown | Stage 2 summary: score edit actions first, then apply them to graph, continuously maintain top-k high-scoring candidate paths. If you understand why beam search is needed here, you've mastered the concept. | - |
| 23 | Markdown | ## 6. 顶层总装：Retrosynthesis.predict (Top-level assembly: Retrosynthesis.predict). Explains this stage does workflow organization: call stage 1 → call stage 2 → merge/rank/dedupe results. | - |
| 24 | Code | `show_source_block()` displaying `Retrosynthesis.predict()` code from line 1090-1158 | 7 lines |
| 25 | Markdown | Completes the G2Gs explanation: predict where to break product → step-by-step synthesize reactants → rank all candidate paths uniformly. Core concept: "graph encoding + intermediate structure prediction + conditional search" | - |
| 26 | Code | Creates and displays `pipeline_summary` DataFrame with 5 steps mapping source symbols to teaching equivalents | 36 lines |
| 27 | Markdown | ## 7. 建议的阅读顺序 (Recommended reading order). Suggests 3-pass learning: understand data preprocessing → understand stage 1 (node/edge scoring) → understand stages 2&3 (beam search). Emphasizes learning framework first, then local details. | - |

### 🎨 CODING STYLE

**Imports:**
```python
import os
import sys
from pathlib import Path
import torch
import pandas as pd
from IPython.display import Markdown, display
from g2gs_tutorial import (
    CenterIdentificationModel,
    SynthonCompletionModel,
    build_synthon_completion_dataset,
    draw_reaction_pair_image,
    identify_reaction_center,
    load_demo_reactions,
    molecule_to_graph_tensor,
)
```

**Path Handling:**
- Uses `pathlib.Path` for all file operations
- Dynamic project root detection via `find_project_root()` function
- All paths computed relative to `PROJECT_ROOT`
- Pattern: `PROJECT_ROOT / "relative/path/to/file"`
- Example: `TUTORIAL_DIR = PROJECT_ROOT / "teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.1.g2gs"`

**Variable Naming:**
- UPPERCASE for configuration: `PROJECT_ROOT`, `TUTORIAL_DIR`, `CODE_DIR`, `DATA_DIR`
- snake_case for functions: `find_project_root()`, `show_source_block()`
- PascalCase for class references: `CenterIdentificationModel`, `SynthonCompletionModel`
- Descriptive: `SOURCE_BLOCKS`, `ranked_centers`, `synthon_dataset`, `pipeline_summary`

**Comments Language:**
- All comments and docstrings in **Chinese**
- Error messages in Chinese: `"无法定位项目根目录"`
- DataFrame headers and printed output in Chinese
- This makes it entirely suitable for Chinese learners

**Code Organization:**
1. Configuration setup (paths, imports)
2. Helper function definitions (`show_source_block`)
3. Data structure definitions (`SOURCE_BLOCKS` list)
4. Main learning flow: explanation (markdown) + code (execution)
5. Heavy use of IPython display for formatted output

---

## TASK.MD (graphretro/task.md)

**File:** `teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.2.graphretro/task.md`

**Content (1076 bytes):**
```
以GLN的教学代码为模板，撰写graphretro的教学代码与材料，graphretro的仓库地址为https://github.com/vsomnath/graphretro.git，
这个教程的路径放在/home/xiaoruiwang/data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.2.graphretro，
源码放在/home/xiaoruiwang/data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/source_repos，
环境放在/home/xiaoruiwang/data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/envs下，
要注意，所有的路径都要以本项目的根路径计算相对路径,
你需要先分析其代码结构，按照其代码创建下列notebook作为教学演示代码：
1.环境配置，在这个notebook中，你需要适配最新的环境版本；
2. 数据处理，在这个notebook中，你需要展示最小化的graphretro的数据处理代码，需要包括完整的数据处理流程，外带中文的详细讲解；
3. 模型展示，在这个notebook中，你需要展示graphretro的推理原理，模型计算架构，
```

**Translation:**
> Using GLN teaching code as template, write GraphRetro teaching code and materials. GraphRetro repo: https://github.com/vsomnath/graphretro.git
> Tutorial path: teaching_demos/.../2.3.2.graphretro
> Source code: source_repos/
> Environment: envs/
> **Important:** All paths relative to project root
> Required 3 notebooks:
> 1. **Environment Setup** - adapt to latest environment versions
> 2. **Data Processing** - minimal GraphRetro data preprocessing code with complete pipeline & detailed Chinese explanations
> 3. **Model Demo** - show GraphRetro inference principles and model computational architecture

---

## PEDAGOGICAL FLOW PATTERNS

### Notebook 3 (Model Demo) Pattern

The G2Gs 3_模型展示.ipynb establishes a clear teaching pattern:

**Pattern Structure:**
1. **Conceptual Introduction** (Markdown)
   - Explain the "why" and high-level framework
   - Pose key questions to focus learning
   - Use analogies and structural breakdown

2. **Source Code Reference** (Code with `show_source_block()`)
   - Display actual source code from `source_repos/`
   - Show line numbers and file paths
   - Let students read real code (not simplified)

3. **Executable Demo** (Code with actual execution)
   - Load real data
   - Instantiate models
   - Show intermediate tensor shapes
   - Display results as pandas DataFrames

4. **Synthesis Summary** (Markdown)
   - Recap key insights from code reading
   - Connect source code to educational objective
   - Suggest next learning steps

**Repetition:** This pattern repeats for each major component (RGCN → Stage 1 → Stage 2 → Top-level assembly)

### Chinese Documentation Structure

**Key Characteristics:**
- Section titles use `##` markdown with clear numbering
- Explains conceptual challenges in Chinese ("最容易遇到的问题不是...")
- Uses metaphors and analogies (e.g., "反应物预测器" vs "反应中心提议器")
- Critical insights highlighted with **bold** Chinese text
- Poses questions to guide thinking: "建议始终抓住一条主线"
- References back to earlier sections to build conceptual coherence

**Educational Philosophy:**
- Build framework first, then details (不要追求一遍读完)
- Emphasize understanding "why" over memorizing code
- Use intermediate representations as learning anchors
- Help students establish mental models before implementation details

---

## KEY TECHNICAL PATTERNS FOR NOTEBOOK 3

### 1. Source Code Embedding Helper

```python
def show_source_block(relative_path, start, end, title):
    path = PROJECT_ROOT / relative_path
    lines = path.read_text(encoding="utf-8").splitlines()
    snippet = "\n".join(f"{line_no:4d} | {lines[line_no - 1]}" 
                       for line_no in range(start, end + 1))
    display(Markdown(f"### {title}\n`{relative_path}:{start}-{end}`"))
    display(Markdown(f"```python\n{snippet}\n```"))
```

**Usage:** This function reads source files, extracts line ranges, and displays them with formatting.

### 2. Data Structure Documentation

```python
SOURCE_BLOCKS = [
    {
        "stage": "阶段名称 (Chinese)",
        "source_file": "相对路径从PROJECT_ROOT",
        "symbol": "函数/类名称",
        "lines": "start-end",
    },
    # ... repeated
]
```

This creates a roadmap DataFrame that students can reference.

### 3. Demo Data Loading Pattern

```python
examples = load_demo_reactions()  # From tutorial code module
example = next(item for item in examples if item.reaction_id == "demo_01")
product_graph = molecule_to_graph_tensor(example.product_mol, ...)

# Display shapes and intermediate results
display(pd.DataFrame([
    {"tensor": "name", "shape": tuple(tensor.shape)},
    # ...
]))
```

### 4. Model Instantiation & Execution

```python
model = CenterIdentificationModel(
    node_input_dim=...,
    edge_input_dim=...,
    num_relation=4,
    num_reaction=10,
    hidden_dim=64,
    num_layers=3,
)
output = model(product_graph, reaction=example.reaction_class)
ranked = model.rank_centers(product_graph, reaction=example.reaction_class, topk=8)
```

### 5. Results Display

Use pandas DataFrames for all results:
```python
display(pd.DataFrame([
    {"field1": value, "field2": value},
    # ... rows
]))
```

---

## NOTEBOOK 1 (环境配置) TYPICAL STRUCTURE

**Likely structure based on G2Gs pattern:**
1. Markdown: Title & learning objectives
2. Code: Project root finder + path setup
3. Markdown: Environment requirements & dependencies
4. Code: Version checks & import tests
5. Markdown: Installation notes & troubleshooting
6. Code: Verify all critical imports work
7. Markdown: Summary & next steps

---

## NOTEBOOK 2 (数据处理) TYPICAL STRUCTURE

**Likely structure based on G2Gs pattern:**
1. Markdown: Title & data pipeline overview
2. Code: Load demo data (CSV)
3. Markdown: Data format explanation
4. Code: Show source_block() of data loading functions
5. Code: Parse SMILES, extract atom maps
6. Markdown: Explain reaction center, synthons concepts
7. Code: Run data preprocessing demo
8. Markdown: Summary of complete pipeline

---

## NOTEBOOK 3 (模型展示) STRUCTURE

**Based on G2Gs template, should follow:**

1. **Conceptual Framework** (Cells 0-2)
   - Introduce GraphRetro's approach
   - Pose key questions about graph-based retrosynthesis
   - Provide high-level formula or diagram

2. **Graph Representation** (Cells 3-5)
   - Show source code for graph encoding
   - Execute on demo molecule
   - Display tensor shapes

3. **Message Passing Architecture** (Cells 6-8)
   - Show source code for core GNN layers
   - Explain graph convolution operations
   - Demo on real molecule

4. **Reaction Prediction** (Cells 9-12)
   - Show source code for action/bond prediction
   - Run inference on demo reaction
   - Display top-k predictions

5. **Full Pipeline** (Cells 13-16)
   - Show top-level predict() code
   - Execute complete retrosynthesis for example
   - Display full reactant predictions with scores

6. **Summary & Reading Guide** (Cells 17-18)
   - Recap key insights
   - Suggest reading order
   - Point to next learning resources

---

## CRITICAL DESIGN PATTERNS TO REPLICATE

1. ✅ **Dynamic Project Root Detection**
   - Never hardcode paths
   - Use `find_project_root()` pattern
   - Makes notebooks portable

2. ✅ **Relative Path Everything**
   - All paths computed from `PROJECT_ROOT`
   - Makes notebooks work in any environment

3. ✅ **Chinese-Only Documentation**
   - All explanations in Chinese
   - Makes it cohesive for Chinese learner audience

4. ✅ **Interleave Theory & Code**
   - Markdown explanation
   - Show actual source code
   - Demo execution with real results
   - Synthesis summary

5. ✅ **Heavy Use of IPython Display**
   - Markdown for formatted text
   - DataFrames for tabular data
   - Images for molecule visualization

6. ✅ **Guided Discovery**
   - Pose questions before answers
   - Highlight key insights
   - Suggest next learning steps

7. ✅ **Seed Random States**
   - `torch.manual_seed(7)` for reproducibility

8. ✅ **Line-Numbered Source Display**
   - Always show source file paths with line ranges
   - e.g., `source_repos/torchdrug/torchdrug/models/gcn.py:88-168`

---

## FILE ORGANIZATION

```
teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/
├── 2.3.1.g2gs/                          ← Reference template
│   ├── 1_环境配置.ipynb                  (10 KB, complete)
│   ├── 2_数据处理.ipynb                  (15 KB, complete)
│   ├── 3_模型展示.ipynb                  (24 KB, complete - PRIMARY TEMPLATE)
│   ├── code/                            ← g2gs_tutorial module
│   │   ├── g2gs_tutorial.py
│   │   └── ...
│   └── data/
│       └── demo_reactions.pkl
│
└── 2.3.2.graphretro/                    ← Your target
    ├── 1_环境配置.ipynb                  (0 KB, EMPTY - needs writing)
    ├── 2_数据处理.ipynb                  (0 KB, EMPTY - needs writing)
    ├── 3_模型展示.ipynb                  (NOT YET CREATED)
    ├── code/                            ← Where you'll put graphretro_tutorial.py
    ├── data/
    │   └── demo_data.csv                (20 reactions, ready)
    └── task.md                          (instructions)
```

---

## SUMMARY FOR CREATING NOTEBOOK 3

**Title:** `3_模型展示.ipynb` (Model Demo)

**Objective:** Teach GraphRetro's inference pipeline through:
1. Source code reading (from source_repos/)
2. Live execution on demo_data.csv
3. Tensor flow visualization
4. Conceptual synthesis

**Length:** Aim for 20-30 cells (similar to G2Gs template)

**Pattern:** Markdown explanation → Source code block → Live demo → Synthesis

**Language:** Entirely in Chinese for consistency

**Key Sections to Cover:**
1. Graph representation in GraphRetro
2. Graph attention/message passing
3. Action prediction mechanism
4. Beam search for candidate generation
5. Top-level inference pipeline

**Data:** Use demo_data.csv with 20 reactions for all examples

