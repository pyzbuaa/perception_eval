# 视觉感知效能评估平台

面向单台 Linux GPU 工作站的浏览器科研工作台。当前版本提供可运行的软件闭环，并将尚未完成的生成模型和真实检测权重与平台解耦。

## 已实现

- 浅色/深色主题和 1920×1080 全屏演示模式
- 概览、五步数据构建、数据集版本、模型/适配器、评测中心、效能探索、任务中心和环境页面
- SQLite WAL 持久化及单机任务 Agent
- 标准 JSON 子进程 Adapter 协议
- 回放生成器、本地目录导入，以及对 PNG/JPEG/WebP 的真实模糊、雾化和噪声处理
- 数据集真值状态与不可变冻结
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

在“数据构建”中选择 `Z-Image-Turbo`，配置场景域、天气、输出数量、分辨率、起始 seed、
推理步数和设备策略，然后提交。一个任务只加载一次模型；任务内图片使用从起始 seed
开始的连续整数。首个任务可能需要将模型权重下载到 `.runtime/cache/huggingface`。
可以先在“模型 / Adapter”页面执行健康检查，或调用：

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
    "name": "无人机薄雾生成",
    "adapter_id": "adapter_basegen",
    "source_type": "GENERATIVE",
    "sample_count": 4,
    "seeds": [1001],
    "conditions": {
      "scene": {"domain": "无人机航拍", "weather": "雾"},
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

使用返回的任务 ID 查询进度：

```bash
curl http://127.0.0.1:18080/api/runs/<job-id>
```

默认单任务超时为 7200 秒，可以在启动前通过
`PERCEPTION_EVAL_ADAPTER_TIMEOUT_SECONDS` 调整。如果 Adapter 需要新增依赖，应先克隆到
`.runtime/envs/adapters/<adapter-version>`，不得修改原环境。

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
