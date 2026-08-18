# 视觉感知效能评估平台

面向单台 Linux GPU 工作站的浏览器科研工作台。当前版本提供可运行的软件闭环，并将尚未完成的生成模型和真实检测权重与平台解耦。

## 已实现

- 浅色/深色主题和 1920×1080 全屏演示模式
- 概览、五步数据构建、数据集版本、模型版本、评测中心、效能模型库、任务中心和环境页面
- SQLite WAL 持久化及单机任务 Agent
- 标准 JSON 子进程 Adapter 协议
- 本地目录导入、基础图像生成，以及无人机航拍域加雾与 ID-Blau 运动模糊生成
- 数据集真值状态与不可变冻结
- DroneDets YOLOv8m VisDrone 真实检测与 COCO 指标计算
- 三 seed 参考评测、mAP/PR/时延/Pareto 展示
- 现有 conda 环境只读枚举、环境指纹和 GPU 预检查
- 工作区内独立 mamba 环境、包缓存和模型缓存

参考检测器会在界面中显著标记为“流程样例”，其指标不属于正式科研结论。

## 隔离安装

```bash
./scripts/bootstrap.sh
```

脚本只在以下位置创建内容：

- `.runtime/envs/platform`：平台 Python/Node 环境
- `.runtime/pkgs`：独立 conda 包缓存
- `.runtime/cache`：pip、Torch、Hugging Face、CUDA 等缓存
- `data`：SQLite、Artifact、任务目录和日志

脚本不会修改 Miniforge `base`、已有 conda 环境、`.condarc`、`.bashrc` 或显卡驱动。安装前后会运行 `environment_guard.py` 比较环境和配置文件摘要。

## 启动与停止

```bash
./scripts/start.sh
```

浏览器访问 [http://127.0.0.1:18080](http://127.0.0.1:18080)。平台默认不监听局域网地址。

```bash
./scripts/stop.sh
```

停止操作不会删除环境或实验数据。

## 开发检查

```bash
.runtime/envs/platform/bin/python -m pytest
PATH="${PWD}/.runtime/envs/platform/bin:${PATH}" .runtime/envs/platform/bin/npm --prefix frontend run build
```

API 文档位于 [http://127.0.0.1:18080/docs](http://127.0.0.1:18080/docs)。

## 接入实际生成模型

已接入同级目录下的 BaseGen `Z-Image-Turbo`。生成模型通过
`AdapterManifest + TaskRequest + TaskResult` 接入，不允许直接写平台数据库。平台使用
BaseGen 已有的 `gen` Conda 环境启动独立 Python 子进程，不会向该环境安装或升级依赖。

```text
/home/yons/miniforge3/envs/gen/bin/python -B adapters/basegen_generator.py ...
```

默认路径为：

```text
BaseGen 项目：/home/yons/ws/project_generator/BaseGen
Conda 环境： /home/yons/miniforge3/envs/gen
```

路径不同时，在启动平台前覆盖：

```bash
export BASEGEN_ROOT=/absolute/path/to/BaseGen
export BASEGEN_CONDA_PREFIX=/absolute/path/to/conda/env
./scripts/start.sh
```

在“数据构建”中选择 `Z-Image-Turbo` 后，页面会读取 BaseGen 当前的场景目录，并根据
场景域动态显示区域、相机高度、视角、环境、时间、天气、活动密度和关键元素等字段。
每个单选字段都可以选择一个固定值或“随机”；关键元素可以随机组合，也可以手动选择
最多四项。页面会禁用与当前环境不兼容的选项，改变环境时也会把不再兼容的固定项恢复
为随机。

配置输出数量、分辨率、起始 seed、推理步数和设备策略后，可先进入“组合预览”，点击
“预览 3 个随机场景”。预览只解析场景和最终 prompt，不加载生成模型、不生成图片。
确认后再提交任务。一个任务只加载一次模型；任务内图片使用从起始 seed 开始的连续
整数。同一份字段规则和起始 seed 会得到相同的场景组合。首个任务可能需要将模型权重
下载到 `.runtime/cache/huggingface`。

可以先在“数据构建”的 `Z-Image-Turbo` 来源卡片中执行接口测试，或调用：

```bash
curl -X POST http://127.0.0.1:18080/api/adapters/adapter_basegen/health-check
```

生成结果保存在：

```text
data/artifacts/generated/<job-id>/
data/task_workspaces/<job-id>/request.json
data/task_workspaces/<job-id>/result.json
data/logs/<job-id>.log
```

BaseGen 是纯文本条件生成器，不产生目标框或分割真值，因此生成的数据集状态为
`UNLABELED`，完成标注前不能冻结或进入正式评测。

也可以直接调用 API：

```bash
curl -X POST http://127.0.0.1:18080/api/acquisition-jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "低空无人机混合场景生成",
    "adapter_id": "adapter_basegen",
    "source_type": "GENERATIVE",
    "sample_count": 4,
    "seeds": [1001],
    "conditions": {
      "scene": {
        "domain": "low-altitude-uav",
        "domain_label": "低空无人机",
        "fields": {
          "region": {"mode": "random"},
          "camera_height": {"mode": "fixed", "value": "low"},
          "viewpoint": {"mode": "random"},
          "field_of_view": {"mode": "random"},
          "environment": {"mode": "random"},
          "time_of_day": {"mode": "fixed", "value": "day"},
          "weather": {"mode": "random"},
          "activity_level": {"mode": "random"},
          "elements": {"mode": "fixed", "values": ["small_vehicles"]}
        },
        "custom": "Blue delivery trucks are visible"
      },
      "sensor": {"resolution": "1024×1024"}
    },
    "model_parameters": {
      "steps": 9,
      "guidance_scale": 0,
      "device_policy": "cuda",
      "local_files_only": false
    },
    "category_template": "custom",
    "categories": [
      {"id": 1, "name": "car"},
      {"id": 2, "name": "pedestrian"}
    ]
  }'
```

页面使用的场景字段目录和无生成预览也可以直接调用：

```text
GET  /api/adapters/adapter_basegen/scene-schema
POST /api/adapters/adapter_basegen/preview
```

`random` 会在当前环境兼容的候选中采样；BaseGen 目录提供权重时按权重采样。所有固定
字段会先共同约束环境，若组合没有交集，预览或提交会返回“不兼容”错误，而不会静默
替换用户的固定选择。

使用返回的任务 ID 查询进度：

```bash
curl http://127.0.0.1:18080/api/runs/<job-id>
```

默认单任务超时为 7200 秒，可以在启动前通过
`PERCEPTION_EVAL_ADAPTER_TIMEOUT_SECONDS` 调整。如果 Adapter 需要新增依赖，应先克隆到
`.runtime/envs/adapters/<adapter-version>`，不得修改原环境。

## 非理想条件生成

### 无人机航拍域加雾

“数据构建 → 非理想条件生成”调用同级 `DiffusionDegrade` 项目及其只读 `.venv`，默认使用
`uav_fog_8gpu_3125_content15/checkpoints/model_2501.pkl`。任务处理所选无人机航拍数据集的
全部图像，输出恢复原图分辨率并保持文件名；源数据存在 COCO 标注时，标注会作为候选
真值继承到输出数据集。

页面中的“加雾强度（视觉混合）”范围为 `0～1`：`0` 输出原图，`1` 输出完整模型
加雾结果，中间值在二者之间进行线性混合。该参数表示视觉效果强度，不是物理大气
散射系数。

路径不同时可在启动平台前设置：

```bash
export DIFFUSION_DEGRADE_ROOT=/absolute/path/to/DiffusionDegrade
export DIFFUSION_DEGRADE_RUNTIME_PREFIX=/absolute/path/to/DiffusionDegrade/.venv
export DIFFUSION_DEGRADE_UAV_FOG_CHECKPOINT=/absolute/path/to/model.pkl
export DIFFUSION_DEGRADE_HF_HOME=/absolute/path/to/huggingface-cache
```

### 无人机运动模糊

运动模糊调用同级 `DiffusionBlur` 项目的 ID-Blau 条件扩散模型，默认使用只读 `blau`
Conda 环境和 `weights/ID_Blau.pth`。页面可选择飞行、升降、偏航、云台倾斜或复合振动
预设，并在 `0.01～0.35` 范围内设置归一化运动条件强度；平台固定使用 DDIM 20 步推理。
该强度不表示无人机速度、曝光时间或像素位移。

```bash
export DIFFUSION_BLUR_ROOT=/absolute/path/to/DiffusionBlur
export DIFFUSION_BLUR_RUNTIME_PREFIX=/absolute/path/to/conda/env
export DIFFUSION_BLUR_CHECKPOINT=/absolute/path/to/ID_Blau.pth
```

## 注册本地目标检测模型

在“模型版本”点击“注册本地检测模型”，配置模型项目目录、命令
工作目录、Python 环境、参数列表和权重。平台自动使用所选环境的 `bin/python`，只保存
路径、结构化命令并计算权重摘要，不复制模型工程或权重。默认允许浏览：

数据构建和模型注册均需选择目标检测类别。平台内置 COCO 2017、
VisDrone 和 Pascal VOC 类别模板；自定义类别可逐项填写，也可从 JSON 类别数组、
COCO JSON 的 `categories` 字段或每行 `id,name` 的 CSV/TXT 文件读取。
本地导入包含标注时，平台直接从 COCO `categories`、YOLO `data.yaml`/`.names`
或 VisDrone 标准模板读取类别，页面只展示确认；未提供标注时，必须选择模板或
手动配置计划检测的类别。
VisDrone 和 Pascal VOC 模板生成的 COCO 类别 ID 均从 0 开始；VisDrone 原始 TXT
中的 1–10 类别编号会在导入时自动转换为 0–9。
评测要求数据集与模型的类别名称集合完全一致，ID 可以不同；平台在运行时
自动将模型输出 ID 映射回数据集 COCO 类别 ID。

```text
模型库：    /home/yons/ws/project_generator
Conda 环境：/home/yons/miniforge3/envs
```

可以在启动前通过 `PERCEPTION_EVAL_MODEL_LIBRARY_ROOT` 和
`PERCEPTION_EVAL_MODEL_ENVIRONMENT_ROOT` 修改这两个只读浏览根目录。

本地数据导入直接读取服务器上的目录，不经过浏览器上传。默认从文件系统根目录
`/` 开始浏览；可通过
`PERCEPTION_EVAL_DATASET_LIBRARY_ROOT` 修改允许访问的数据根目录。导入任务仍会在
平台 Artifact 中创建图像符号链接并保存派生标注，不复制源图像，也不会修改源数据。
源图像被移动、改名或删除后，对应平台数据集将不可用。

命令参数在页面中每行填写一个，平台直接以参数数组启动进程，不使用 `shell=True`。
例如：

```text
tools/evaluate.py
--weights
{weight_path}
--images
{image_directory}
--annotations
{annotation_path}
--output
{predictions_path}
--device
{device}
```

常用占位符包括：

```text
{project_directory} {weight_path}       {image_directory}
{annotation_path}    {output_directory} {predictions_path}
{device}             {precision}        {model_id}
{dataset_id}         {request_path}     {result_path}
```

模型命令最低只需在 `{predictions_path}` 生成 COCO Detection Results 数组：

```json
[
  {
    "image_id": 1,
    "category_id": 4,
    "bbox": [120, 80, 64, 48],
    "score": 0.96
  }
]
```

`image_id` 和 `category_id` 必须与输入 COCO 标注一致，`bbox` 使用绝对像素
`[x,y,width,height]`。模型若不生成 `result.json`，平台使用进程总耗时计算平均时延；
若需要预处理、推理、后处理和显存等详细指标，可通过 `{result_path}` 额外输出完整运行
元数据。

注册完成后，模型会出现在评测中心的“模型版本”选择框中。同一个项目可以通过不同的
权重和参数列表注册多个模型版本。模型命令不应写平台数据库，也不应修改输入数据集。

## 接入 DroneDets 目标检测

平台已只读接入同级目录的 DroneDets，首个可用模型为
`DroneDets · YOLOv8m VisDrone`。它使用 DroneDets 自带的 `.venv` 和本机缓存权重执行
真实推理，平台不会向 DroneDets 项目或环境写入文件。

默认路径为：

```text
DroneDets 项目：/home/yons/ws/project_generator/DroneDets
Python 环境：   /home/yons/ws/project_generator/DroneDets/.venv
模型权重：      /mnt/data/cache/huggingface/hub/models--mshamrai--yolov8m-visdrone/snapshots/*/best.pt
```

路径不同时，可以在启动前覆盖：

```bash
export DRONEDETS_ROOT=/absolute/path/to/DroneDets
export DRONEDETS_RUNTIME_PREFIX=/absolute/path/to/DroneDets/.venv
export DRONEDETS_YOLOV8M_WEIGHT=/absolute/path/to/best.pt
./scripts/start.sh
```

使用步骤：

1. 在“数据集”页面完成目标检测标注，导出 COCO `instances.json`，然后冻结数据集。
2. 在“模型版本”页面对 `DroneDets · YOLOv8m VisDrone` 执行接口测试。
3. 在“评测中心”选择已冻结数据集和该真实模型；首版支持 FP16/FP32、batch size 1。
4. 启动评测，在“任务中心”查看推理进度，在“效能模型库”查看 mAP、AP50、AP75、PR、
   时延、FPS 和显存指标。

评测使用 `pycocotools 2.0.11`，预测文件和运行协议分别保存在：

```text
data/artifacts/evaluations/<run-id>/predictions.json
data/task_workspaces/<job-id>/runs/<run-id>/request.json
data/task_workspaces/<job-id>/runs/<run-id>/result.json
data/task_workspaces/<job-id>/runs/<run-id>/adapter.log
```

VisDrone 的 `motor` 类会映射到平台默认的 `motorcycle`。数据集内没有同名类别的模型
输出不会参与指标计算，并记录在运行配置的 `unmatched_labels` 中。当前接入成熟度为
`EXPERIMENTAL`，因此即使是真实推理也不会标记为正式基准结果。Ultralytics 代码采用
AGPL-3.0，闭源分发前需单独确认许可证适用性。

## 目标检测标注

未冻结的数据集可以在“数据集”页面点击“目标标注”，进入全屏人工标注工作区：

1. 在右侧选择目标类别；默认提供 `car`、`truck`、`bus`、`pedestrian`、`bicycle`
   和 `motorcycle`，也可以新增类别。
2. 在图片上拖拽创建矩形框；拖动矩形框可移动，拖动四角可改变大小，鼠标滚轮可缩放
   图片。
3. 每张图片完成后点击“完成并下一张”。没有目标的图片也需要勾选“本图已确认”。
4. 所有图片确认后，点击“完成标注并导出 COCO”。
5. 数据集进入 `CANDIDATE` 后可以校核并冻结；冻结后标注和类别均为只读。

创建、移动、缩放、删除目标框以及改变类别后都会自动保存。快捷键包括：

```text
A / D       上一张 / 下一张
Delete      删除选中的目标框
Ctrl+S      立即保存
```

标注编辑状态保存在 SQLite，提交后同时导出标准 COCO 检测标注：

```text
data/artifacts/<dataset-artifact>/annotations/instances.json
```

对应 API 为：

```text
GET  /api/datasets/<dataset-id>/annotations
PUT  /api/datasets/<dataset-id>/annotation-schema
GET  /api/datasets/<dataset-id>/samples/<sample-name>/annotations
PUT  /api/datasets/<dataset-id>/samples/<sample-name>/annotations
POST /api/datasets/<dataset-id>/annotations/complete
```

标注状态按 `UNLABELED → ANNOTATING → CANDIDATE → VERIFIED/FROZEN` 流转。修改一张已经
确认的图片时，该图片会自动恢复为待确认，防止修改后未经复核就提交。

任务中心通过 `DELETE /api/jobs/<job-id>` 删除终态任务记录及其工作区和日志，
不会删除任务生成的数据集或评测结果。

## 浏览数据集图片

数据集列表和详情中的 6 张图片是快速预览。点击“浏览全部”会打开分页浏览抽屉，每页
仅加载 50 张图片。数据集“查看”抽屉还会统计类别、分辨率和 COCO 尺度分布。对应 API 为：

```text
GET /api/datasets/<dataset-id>/samples?offset=0&limit=50
GET /api/datasets/<dataset-id>/statistics
```

## 删除数据集

“数据集版本”页面允许删除未冻结且未被评测方案或评测运行引用的数据集。删除采用可恢复
策略：数据库中的活动记录会删除，Artifact 目录及删除前的数据集记录会移动到：

```text
data/trash/datasets/<dataset-id>/artifact/
data/trash/datasets/<dataset-id>/dataset.json
```

冻结数据集和已有评测引用的数据集受保护，不能删除。也可以调用 API：

```bash
curl -X DELETE http://127.0.0.1:18080/api/datasets/<dataset-id>
```
