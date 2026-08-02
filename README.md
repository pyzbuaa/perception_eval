# 视觉感知效能评估平台

面向单台 Linux GPU 工作站的浏览器科研工作台。当前版本提供可运行的软件闭环，并将尚未完成的生成模型和真实检测权重与平台解耦。

## 已实现

- 浅色/深色主题和 1920×1080 全屏演示模式
- 概览、五步数据构建、数据集版本、模型版本、评测中心、效能探索、任务中心和环境页面
- SQLite WAL 持久化及单机任务 Agent
- 标准 JSON 子进程 Adapter 协议
- 回放生成器、本地目录导入，以及对 PNG/JPEG/WebP 的真实模糊、雾化和噪声处理
- 数据集真值状态与不可变冻结
- DroneDets YOLOv8m VisDrone 真实检测与 COCO 指标计算
- 三 seed 参考评测、mAP/PR/时延/Pareto 展示
- 现有 conda 环境只读枚举、环境指纹和 GPU 预检查
- 工作区内独立 mamba 环境、包缓存和模型缓存

回放生成器和参考检测器会在界面中显著标记为“流程样例”，其指标不属于正式科研结论。

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

可以先在“数据构建”的 `Z-Image-Turbo` 来源卡片中执行健康检查，或调用：

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
    }
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

## 注册本地目标检测模型

在“模型版本”点击“注册本地检测模型”，配置模型项目目录、命令
工作目录、Python 环境、参数列表和权重。平台自动使用所选环境的 `bin/python`，只保存
路径、结构化命令并计算权重摘要，不复制模型工程或权重。默认允许浏览：

```text
模型库：    /home/yons/ws/project_generator
Conda 环境：/home/yons/miniforge3/envs
```

可以在启动前通过 `PERCEPTION_EVAL_MODEL_LIBRARY_ROOT` 和
`PERCEPTION_EVAL_MODEL_ENVIRONMENT_ROOT` 修改这两个只读浏览根目录。

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
2. 在“模型版本”页面对 `DroneDets · YOLOv8m VisDrone` 执行健康检查。
3. 在“评测中心”选择已冻结数据集和该真实模型；首版支持 FP16/FP32、batch size 1。
4. 启动评测，在“任务中心”查看推理进度，在“效能探索”查看 mAP、AP50、AP75、PR、
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

## 浏览数据集图片

数据集列表和详情中的 6 张图片是快速预览。点击“浏览全部”会打开滚轮浏览抽屉，并在接近
底部时每次加载 48 张图片，适用于包含大量图片的数据集。分页 API 为：

```text
GET /api/datasets/<dataset-id>/samples?offset=0&limit=48
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
