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

生成模型通过 `AdapterManifest + TaskRequest + TaskResult` 接入，不允许直接写平台数据库。已有环境以绝对路径注册为 `conda_external/read_only`，执行方式为：

```text
conda run --prefix <existing-env> python -B <adapter-script>
```

如果 Adapter 需要新增依赖，应先克隆到 `.runtime/envs/adapters/<adapter-version>`，不得修改原环境。参考协议实现见 `adapters/replay_generator.py`。
