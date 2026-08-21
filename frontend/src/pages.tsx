import { cloneElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  Descriptions,
  Divider,
  Drawer,
  Empty,
  Form,
  Image,
  Input,
  InputNumber,
  List,
  Modal,
  Pagination,
  Popconfirm,
  Progress,
  Radio,
  Result,
  Row,
  Segmented,
  Select,
  Slider,
  Spin,
  Space,
  Statistic,
  Steps,
  Table,
  Tag,
  Timeline,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  ArrowRightOutlined,
  CheckCircleOutlined,
  CloudOutlined,
  CodeOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  ExperimentOutlined,
  EyeOutlined,
  FileImageOutlined,
  ImportOutlined,
  LeftOutlined,
  LockOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
  ReloadOutlined,
  RightOutlined,
  RobotOutlined,
  SaveOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { api, formatBytes, percent, post } from './api'
import { parseCategoryFile } from './categoryFiles'
import { DemoTag, Gallery, JobProgress, ParetoChart, PRChart, StatusTag } from './components'
import type { Adapter, AnnotationCategory, AnnotationSession, BaseGenSceneField, BaseGenSceneOption, BaseGenSceneSchema, CategoryDefinition, CategoryTemplate, Dataset, DatasetAnnotationCategories, DatasetSamplePage, DatasetStatistics, DetectionBox, EnvironmentStatus, EvaluationVisualization, Job, ModelVersion, Overview, ResultGroup, ResultResponse, ResultRun, SampleAnnotation } from './types'

type RouteKey = 'overview' | 'builder' | 'datasets' | 'registry' | 'evaluation' | 'explorer' | 'tasks' | 'environment'
interface PageProps { dark: boolean; navigate: (route: RouteKey) => void; refresh: () => void }

function useResource<T>(path: string, initial: T) {
  const [data, setData] = useState(initial)
  const [loading, setLoading] = useState(true)
  const load = useCallback(async () => {
    setLoading(true)
    try { setData(await api<T>(path)) } finally { setLoading(false) }
  }, [path])
  useEffect(() => { load().catch((error) => message.error(error.message)) }, [load])
  return { data, loading, reload: load }
}

type CategoryScope = 'dataset' | 'model'

function categoriesFromSelection(
  templates: CategoryTemplate[],
  templateId: string,
  scope: CategoryScope,
  customCategories: CategoryDefinition[],
) {
  if (templateId === 'custom') return customCategories
  const template = templates.find((item) => item.id === templateId)
  return template?.categories.map((item) => ({
    id: scope === 'dataset' ? item.dataset_id : item.model_id,
    name: item.name,
  })) || []
}

function validCategories(categories: CategoryDefinition[]) {
  const ids = categories.map((item) => item.id)
  const names = categories.map((item) => item.name.trim().toLocaleLowerCase())
  return Boolean(
    categories.length
    && categories.every((item) => Number.isInteger(item.id) && item.id >= 0 && item.name.trim())
    && new Set(ids).size === ids.length
    && new Set(names).size === names.length
  )
}

function categoryCompatibilityIssue(dataset: Dataset, model: ModelVersion) {
  if (!dataset.categories.length || !model.categories.length) return '类别尚未配置'
  const datasetNames = new Set(dataset.categories.map((item) => item.name.trim().toLocaleLowerCase()))
  const modelNames = new Set(model.categories.map((item) => item.name.trim().toLocaleLowerCase()))
  const missing = dataset.categories.filter((item) => !modelNames.has(item.name.trim().toLocaleLowerCase())).map((item) => item.name)
  const extra = model.categories.filter((item) => !datasetNames.has(item.name.trim().toLocaleLowerCase())).map((item) => item.name)
  if (missing.length) return `模型缺少：${missing.join('、')}`
  if (extra.length) return `模型包含数据集未登记类别：${extra.join('、')}`
  return ''
}

type InferenceProperty = { default?: string | number; const?: string | number }

const inferenceParameterLabels: Record<string, string> = {
  confidence: '置信度',
  nms_iou: 'NMS IoU',
  image_size: '方形推理尺寸',
  input_height: '输入高度',
  input_width: '输入宽度',
  max_detections: '最大检测数',
  batch_size: '批大小',
  warmup: '预热次数',
  precision: '推理精度',
}

function modelInferenceProperties(model: ModelVersion | undefined, adapters: Adapter[]) {
  const adapter = model && adapters.find((item) => item.id === model.adapter_id)
  return (adapter?.parameter_schema?.properties || {}) as Record<string, InferenceProperty>
}

function inferenceDefault(
  properties: Record<string, InferenceProperty>,
  name: string,
  fallback: number,
) {
  const value = properties[name]?.const ?? properties[name]?.default
  return typeof value === 'number' ? value : fallback
}

function CategoryConfiguration({
  templates,
  templateId,
  scope,
  customCategories,
  onTemplateChange,
  onCustomChange,
}: {
  templates: CategoryTemplate[]
  templateId: string
  scope: CategoryScope
  customCategories: CategoryDefinition[]
  onTemplateChange: (value: string) => void
  onCustomChange: (value: CategoryDefinition[]) => void
}) {
  const categoryFileRef = useRef<HTMLInputElement>(null)
  const categories = categoriesFromSelection(templates, templateId, scope, customCategories)
  const update = (index: number, field: 'id' | 'name', value: number | string | null) => {
    onCustomChange(customCategories.map((item, itemIndex) => itemIndex === index ? { ...item, [field]: field === 'id' ? Number(value ?? 0) : String(value) } : item))
  }
  const add = () => {
    const start = scope === 'dataset' ? 1 : 0
    const id = Math.max(start - 1, ...customCategories.map((item) => item.id)) + 1
    onCustomChange([...customCategories, { id, name: '' }])
  }
  const importFile = async (file?: File) => {
    if (!file) return
    try {
      const imported = parseCategoryFile(await file.text(), file.name)
      onCustomChange(imported)
      message.success(`已从 ${file.name} 读取 ${imported.length} 个类别`)
    } catch (error) {
      message.error(error instanceof Error ? error.message : '类别文件读取失败')
    } finally {
      if (categoryFileRef.current) categoryFileRef.current.value = ''
    }
  }
  return <Space direction="vertical" size={12} style={{ width: '100%' }}>
    <Select value={templateId} onChange={onTemplateChange} style={{ width: '100%' }} options={[...templates.map((item) => ({ value: item.id, label: item.name })), { value: 'custom', label: '自定义类别' }]} />
    {templateId === 'custom' ? <>
      <input ref={categoryFileRef} hidden type="file" accept=".json,.csv,.txt,application/json,text/csv,text/plain" onChange={(event) => void importFile(event.target.files?.[0])} />
      <Button block icon={<ImportOutlined />} onClick={() => categoryFileRef.current?.click()}>从类别文件读取</Button>
      <Typography.Text type="secondary">支持 JSON 类别数组、COCO JSON 的 categories 字段，以及每行 id,name 的 CSV/TXT 文件；导入后仍可编辑。</Typography.Text>
      <Table size="small" pagination={false} rowKey={(_, index) => String(index)} dataSource={customCategories} columns={[
        { title: scope === 'dataset' ? '数据集类别 ID' : '模型输出 ID', width: 180, render: (_, row, index) => <InputNumber min={0} precision={0} value={row.id} onChange={(value) => update(index, 'id', value)} style={{ width: '100%' }} /> },
        { title: '类别名称', render: (_, row, index) => <Input value={row.name} maxLength={64} placeholder="例如 car" onChange={(event) => update(index, 'name', event.target.value)} /> },
        { title: '操作', width: 70, render: (_, __, index) => <Button danger type="text" icon={<DeleteOutlined />} disabled={customCategories.length === 1} onClick={() => onCustomChange(customCategories.filter((_, itemIndex) => itemIndex !== index))} /> },
      ]} />
      <Button block icon={<PlusOutlined />} onClick={add}>添加类别</Button>
      {!validCategories(customCategories) && <Alert type="warning" showIcon message="类别 ID 和名称必须非空且各自唯一" />}
    </> : <Alert type="success" showIcon message={`已加载 ${categories.length} 个标准类别`} description={<Typography.Paragraph ellipsis={{ rows: 3, expandable: true, symbol: '展开全部' }} style={{ marginBottom: 0 }}>{categories.map((item) => `${item.id}:${item.name}`).join(' · ')}</Typography.Paragraph>} />}
    <Typography.Text type="secondary">评测按类别名称匹配，{scope === 'dataset' ? '这里的 ID 对应数据集标注' : '这里的 ID 必须对应模型原始输出'}。</Typography.Text>
  </Space>
}

export function OverviewPage({ dark, navigate, overview }: PageProps & { overview?: Overview }) {
  const results = useResource<ResultResponse>('/api/results', { count: 0, groups: [], runs: [], dimensions: resultDimensions() })
  if (!overview) return <Card loading />
  const metrics = [
    { label: '数据集版本', value: overview.counts.datasets, suffix: '个', icon: <DatabaseOutlined />, color: 'blue' },
    { label: '模型版本', value: overview.counts.models, suffix: '个', icon: <RobotOutlined />, color: 'purple' },
    { label: '活动任务', value: overview.counts.running, suffix: '个', icon: <ThunderboltOutlined />, color: 'cyan' },
    { label: '已完成运行', value: overview.counts.completed, suffix: '次', icon: <CheckCircleOutlined />, color: 'green' },
  ]
  return (
    <Space direction="vertical" size={18} style={{ width: '100%' }}>
      <Alert type="info" showIcon message="当前运行在隔离工作区模式" description="平台写入仅限项目目录；现有 conda 环境按只读策略登记。页面中的参考模型均明确标记为流程样例。" action={<Button size="small" onClick={() => navigate('environment')}>查看环境边界</Button>} />
      <Row gutter={[16, 16]}>
        {metrics.map((item) => (
          <Col xs={12} xl={6} key={item.label}>
            <Card className={`metric-card metric-${item.color}`}>
              <div className="metric-icon">{item.icon}</div>
              <Statistic title={item.label} value={item.value} suffix={item.suffix} />
              <div className="metric-foot">较上次刷新保持同步</div>
            </Card>
          </Col>
        ))}
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={10}>
          <Card title="实验流水线" extra={<Tag color="processing">闭环就绪</Tag>} className="full-card">
            <Timeline
              className="pipeline-timeline"
              items={overview.pipeline.map((item, index) => ({
                color: item.status === 'complete' || item.status === 'ready' ? 'green' : 'blue',
                dot: item.status === 'active' ? <PlayCircleOutlined /> : undefined,
                children: <div><Typography.Text strong>{item.name}</Typography.Text><br /><Typography.Text type="secondary">{['数据来源统一接入', 'COCO/YOLO 真值状态', '数据集与模型评测矩阵', '条件查询与效能图表'][index]}</Typography.Text></div>,
              }))}
            />
            <Divider />
            <Space wrap>
              <Button type="primary" icon={<FileImageOutlined />} onClick={() => navigate('builder')}>新建数据构建</Button>
              <Button icon={<ExperimentOutlined />} onClick={() => navigate('evaluation')}>新建模型评测</Button>
              <Button icon={<EyeOutlined />} onClick={() => navigate('explorer')}>打开效能对比</Button>
            </Space>
          </Card>
        </Col>
        <Col xs={24} xl={14}>
          <Card title="mAP — 时延 Pareto" extra={<Space><DemoTag /><Button type="link" onClick={() => navigate('explorer')}>查看全部</Button></Space>} className="full-card">
            <ParetoChart groups={results.data.groups} dark={dark} height={338} />
          </Card>
        </Col>
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={14}>
          <Card title="最近数据样例" extra={<Button type="link" onClick={() => navigate('datasets')}>数据集版本</Button>}>
            <Gallery images={overview.recent_images} height={118} />
          </Card>
        </Col>
        <Col xs={24} xl={10}>
          <Card title="最近任务与告警" extra={<Button type="link" onClick={() => navigate('tasks')}>任务中心</Button>}>
            {overview.recent_jobs.length ? <List dataSource={overview.recent_jobs} renderItem={(job) => <List.Item><List.Item.Meta title={<Space><StatusTag status={job.status} /><span>{job.stage}</span></Space>} description={job.id} /><Progress type="circle" size={36} percent={Math.round(job.progress)} /></List.Item>} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无用户任务" />}
          </Card>
        </Col>
      </Row>
    </Space>
  )
}

const sourceCards = [
  { id: 'local-import', source: 'REAL', icon: <ImportOutlined />, title: '本地数据导入', description: '导入本地数据。' },
  { id: 'adapter_basegen', source: 'GENERATIVE', icon: <RobotOutlined />, title: '基础图像生成', description: '调用生成式模型生成感知数据。' },
  { id: 'adapter_condition', source: 'REAL_TRANSFORMED', icon: <CloudOutlined />, title: '非理想条件生成', description: '运动模糊、弱光、...' },
  { id: 'airsim-future', source: 'SIMULATOR', icon: <CodeOutlined />, title: 'AirSim / UE', description: '通过独立 RPC 服务采集图像与真值。', disabled: true },
]

const CUSTOM_SCENE_DOMAIN = '__custom__'
const localSceneDomains = ['无人机航拍', '卫星遥感', '城市驾驶']

const datasetSourceLabels: Record<string, string> = {
  REAL: '导入',
  GENERATIVE: '基础图像生成',
  REAL_TRANSFORMED: '非理想条件生成',
  SIMULATOR: '仿真数据',
  REPLAY_FIXTURE: '测试回放',
}

function DatasetSourceTag({ value }: { value: string }) {
  const color = value === 'REPLAY_FIXTURE' ? 'gold' : value === 'REAL_TRANSFORMED' ? 'cyan' : value === 'GENERATIVE' ? 'purple' : value === 'SIMULATOR' ? 'geekblue' : 'blue'
  return <Tag color={color}>{datasetSourceLabels[value] || value}</Tag>
}

function DatasetPathValue({ value }: { value?: string }) {
  return value ? <Typography.Text code copyable>{value}</Typography.Text> : <Tag color="error">路径不可用</Tag>
}

function datasetLocationItems(dataset: Dataset) {
  const address = dataset.source_path || dataset.dataset_path || dataset.platform_path || dataset.artifact_path
  return [
    { key: 'path', label: '数据集实际地址', children: <DatasetPathValue value={address} />, span: 2 },
    ...(dataset.platform_path && dataset.platform_path !== address
      ? [{ key: 'platform-path', label: '平台数据目录', children: <DatasetPathValue value={dataset.platform_path} />, span: 2 }]
      : []),
  ]
}

type BaseGenSelection =
  | { mode: 'random' }
  | { mode: 'fixed'; value: string }
  | { mode: 'fixed'; values: string[] }

const RANDOM_VALUE = '__random__'

const motionPresetOptions = [
  { value: 'forward', label: '向前飞行', description: '画面内容向下' },
  { value: 'backward', label: '向后飞行', description: '画面内容向上' },
  { value: 'fly-left', label: '向左飞行', description: '画面内容向右' },
  { value: 'fly-right', label: '向右飞行', description: '画面内容向左' },
  { value: 'ascend', label: '上升', description: '内容向中心收缩' },
  { value: 'descend', label: '下降', description: '内容从中心扩张' },
  { value: 'yaw-left', label: '向左偏航', description: '画面顺时针旋转' },
  { value: 'yaw-right', label: '向右偏航', description: '画面逆时针旋转' },
  { value: 'tilt-up', label: '云台上倾', description: '向上透视运动' },
  { value: 'tilt-down', label: '云台下倾', description: '向下透视运动' },
  { value: 'tilt-left', label: '云台左倾', description: '向左透视运动' },
  { value: 'tilt-right', label: '云台右倾', description: '向右透视运动' },
  { value: 'vibration', label: '复合振动', description: '平移、缩放与旋转' },
]

function MotionPresetDiagram({ motion }: { motion: string }) {
  const markerId = `motion-arrow-${motion}`
  const arrow = (x1: number, y1: number, x2: number, y2: number, key: string) => (
    <line key={key} x1={x1} y1={y1} x2={x2} y2={y2} className="motion-diagram-arrow" markerEnd={`url(#${markerId})`} />
  )
  const paths = (() => {
    if (motion === 'forward' || motion === 'backward') {
      const [y1, y2] = motion === 'forward' ? [24, 72] : [72, 24]
      return [42, 80, 118].map((x) => arrow(x, y1, x, y2, String(x)))
    }
    if (motion === 'fly-left' || motion === 'fly-right') {
      const [x1, x2] = motion === 'fly-left' ? [42, 118] : [118, 42]
      return [27, 48, 69].map((y) => arrow(x1, y, x2, y, String(y)))
    }
    if (motion === 'ascend') {
      return [[28, 20, 68, 42], [132, 20, 92, 42], [28, 76, 68, 54], [132, 76, 92, 54]].map(([x1, y1, x2, y2], index) => arrow(x1, y1, x2, y2, String(index)))
    }
    if (motion === 'descend') {
      return [[68, 42, 28, 20], [92, 42, 132, 20], [68, 54, 28, 76], [92, 54, 132, 76]].map(([x1, y1, x2, y2], index) => arrow(x1, y1, x2, y2, String(index)))
    }
    if (motion === 'yaw-left' || motion === 'yaw-right') {
      const clockwise = motion === 'yaw-left'
      return [
        <path key="yaw-top" d={clockwise ? 'M 45 27 A 48 30 0 0 1 122 39' : 'M 122 39 A 48 30 0 0 0 45 27'} className="motion-diagram-arrow" markerEnd={`url(#${markerId})`} />,
        <path key="yaw-bottom" d={clockwise ? 'M 115 70 A 48 30 0 0 1 38 57' : 'M 38 57 A 48 30 0 0 0 115 70'} className="motion-diagram-arrow" markerEnd={`url(#${markerId})`} />,
      ]
    }
    if (motion === 'tilt-up' || motion === 'tilt-down') {
      const upward = motion === 'tilt-up'
      const values = upward
        ? [[43, 74, 55, 27], [80, 74, 80, 23], [117, 74, 105, 27]]
        : [[55, 23, 43, 70], [80, 23, 80, 74], [105, 23, 117, 70]]
      return values.map(([x1, y1, x2, y2], index) => arrow(x1, y1, x2, y2, String(index)))
    }
    if (motion === 'tilt-left' || motion === 'tilt-right') {
      const leftward = motion === 'tilt-left'
      const values = leftward
        ? [[125, 27, 42, 34], [125, 48, 35, 48], [125, 69, 42, 62]]
        : [[35, 34, 118, 27], [35, 48, 125, 48], [35, 62, 118, 69]]
      return values.map(([x1, y1, x2, y2], index) => arrow(x1, y1, x2, y2, String(index)))
    }
    return [
      <path key="v1" d="M 26 28 C 48 10, 60 48, 82 28" className="motion-diagram-mixed" markerEnd={`url(#${markerId})`} />,
      <path key="v2" d="M 107 20 C 140 36, 105 55, 136 70" className="motion-diagram-mixed" markerEnd={`url(#${markerId})`} />,
      <path key="v3" d="M 30 72 C 52 48, 70 82, 96 59" className="motion-diagram-arrow" markerEnd={`url(#${markerId})`} />,
    ]
  })()
  const showCenter = ['ascend', 'descend', 'yaw-left', 'yaw-right', 'vibration'].includes(motion)
  return (
    <svg className="motion-preset-diagram" viewBox="0 0 160 96" role="img" aria-label={`${motion} 图像表观运动示意`}>
      <defs><marker id={markerId} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" className="motion-diagram-head" /></marker></defs>
      <rect x="1" y="1" width="158" height="94" rx="8" className="motion-diagram-bg" />
      <path d="M 54 1 V 95 M 106 1 V 95 M 1 32 H 159 M 1 64 H 159" className="motion-diagram-grid" />
      {showCenter && <circle cx="80" cy="48" r="4" className="motion-diagram-center" />}
      {paths}
    </svg>
  )
}

function conditionDatasetItems(dataset: Dataset, datasets: Dataset[] = []) {
  const conditions = dataset.sensor_conditions || {}
  const degradation = String(conditions.degradation || '')
  const recordedCondition = String(conditions.recorded_condition || '')
  const legacyStrength = typeof conditions.motion_blur === 'number' ? conditions.motion_blur : Number.NaN
  const hasMotionBlur = conditions.motion_blur === true
    || (Number.isFinite(legacyStrength) && legacyStrength > 0)
    || conditions.degradation === 'ID-Blau UAV Motion Blur'
  const hasFog = Object.prototype.hasOwnProperty.call(conditions, 'fog_strength')
    || Object.prototype.hasOwnProperty.call(conditions, 'fog_density')
    || degradation.toLowerCase().includes('fog')
  const hasDayToNight = conditions.day_to_night === true
    || degradation.toLowerCase().includes('day-to-night')
  const sourceDatasetId = String(conditions.source_dataset_id || '')
  const sourceDataset = datasets.find((item) => item.id === sourceDatasetId)
  const sourceItem = sourceDatasetId
    ? [{ key: 'condition-source', label: '源数据集', children: sourceDataset?.name || sourceDatasetId }]
    : []

  if (hasDayToNight) {
    const model = String(conditions.day_to_night_model || degradation || '')
    const checkpoint = String(conditions.day_to_night_checkpoint || '')
    const method = String(conditions.day_to_night_method || '')
    const isDrivingDayToNight = method === 'unpaired' || ['城市驾驶', '自动驾驶', 'autonomous-driving'].includes(dataset.scene_domain)
    const tileSize = Number(conditions.day_to_night_tile_size)
    const overlap = Number(conditions.day_to_night_overlap)
    const imagePrep = String(conditions.day_to_night_image_prep || '')
    return [
      { key: 'condition-type', label: '非理想条件', children: <Tag color="geekblue">{isDrivingDayToNight ? '自动驾驶弱光' : '无人机弱光'}</Tag> },
      { key: 'time-domain', label: '光照条件', children: '白天 → 弱光' },
      ...(model ? [{ key: 'condition-model', label: '生成模型', children: model }] : []),
      ...(method ? [{ key: 'condition-method', label: '推理方法', children: method }] : []),
      ...(checkpoint ? [{ key: 'condition-checkpoint', label: '权重版本', children: checkpoint }] : []),
      ...(Number.isFinite(tileSize) && Number.isFinite(overlap) ? [{ key: 'condition-tiling', label: '分块参数', children: `${tileSize}px / 重叠 ${overlap}px` }] : []),
      ...(imagePrep ? [{ key: 'condition-image-prep', label: '模型输入', children: imagePrep === 'resize_640x640' ? '整图缩放至 640×640' : imagePrep === 'resize_512x512' ? '整图缩放至 512×512' : imagePrep }] : []),
      ...sourceItem,
    ]
  }
  if (hasMotionBlur) {
    const motion = String(conditions.motion || '未记录')
    const motionLabel = motionPresetOptions.find((item) => item.value === motion)?.label || motion
    const strength = Number(conditions.motion_blur_strength ?? legacyStrength)
    const steps = Number(conditions.motion_blur_sample_timesteps)
    const model = String(conditions.motion_blur_model || degradation || '')
    const conditionDirectory = String(conditions.motion_condition_directory || '')
    const matchedConditions = Number(conditions.motion_condition_matched)
    const fallbackConditions = Number(conditions.motion_condition_fallback_count)
    return [
      { key: 'condition-type', label: '非理想条件', children: <Tag color="magenta">无人机运动模糊</Tag> },
      { key: 'motion-type', label: '运动条件', children: conditionDirectory ? '条件文件（缺失时随机回退）' : motionLabel },
      { key: 'condition-strength', label: conditionDirectory ? '回退模糊强度' : '条件强度', children: Number.isFinite(strength) ? strength.toFixed(2) : '未记录' },
      ...(conditionDirectory ? [{ key: 'motion-condition-directory', label: '运动条件目录', children: <Typography.Text code copyable>{conditionDirectory}</Typography.Text>, span: 2 }] : []),
      ...(conditionDirectory && Number.isFinite(matchedConditions) && Number.isFinite(fallbackConditions)
        ? [{ key: 'motion-condition-matching', label: '文件匹配', children: `匹配 ${matchedConditions} 张 / 随机回退 ${fallbackConditions} 张` }]
        : []),
      ...(model ? [{ key: 'condition-model', label: '生成模型', children: model }] : []),
      ...(Number.isFinite(steps) ? [{ key: 'condition-sampler', label: '采样参数', children: `DDIM ${steps} 步` }] : []),
      ...sourceItem,
    ]
  }
  if (hasFog) {
    const strength = Number(conditions.fog_strength ?? conditions.fog_density)
    const model = String(conditions.fog_model || degradation || '')
    const method = String(conditions.fog_method || '')
    const checkpoint = String(conditions.fog_checkpoint || '')
    const isUavAerosol = String(conditions.condition_label || '') === '无人机气雾'
      || degradation === 'DiffusionDegrade UAV Fog'
    const isDrivingAerosol = String(conditions.condition_label || '') === '自动驾驶气雾'
      || method === 'paired'
      || degradation === 'WarpI2I Driving Fog'
    return [
      { key: 'condition-type', label: '非理想条件', children: <Tag color="cyan">{isUavAerosol ? '无人机气雾' : isDrivingAerosol ? '自动驾驶气雾' : '雾'}</Tag> },
      ...(Number.isFinite(strength) ? [{ key: 'condition-strength', label: '条件强度', children: strength.toFixed(2) }] : []),
      ...(model ? [{ key: 'condition-model', label: '生成模型', children: model }] : []),
      ...(method ? [{ key: 'condition-method', label: '推理方法', children: method }] : []),
      ...(checkpoint ? [{ key: 'condition-checkpoint', label: '权重版本', children: checkpoint }] : []),
      ...sourceItem,
    ]
  }
  if (recordedCondition) {
    return [{ key: 'condition-type', label: '非理想条件', children: <Tag>{recordedCondition}</Tag> }]
  }
  return [{ key: 'condition-type', label: '非理想条件', children: <Tag>未记录</Tag> }]
}

function datasetConditionSummary(dataset: Dataset) {
  if (dataset.source_type !== 'REAL_TRANSFORMED') return dataset.weather
  const conditions = dataset.sensor_conditions || {}
  const conditionLabel = String(conditions.condition_label || '')
  if (conditionLabel) return conditionLabel
  const degradation = String(conditions.degradation || '').toLowerCase()
  if (conditions.motion_blur === true || degradation.includes('motion blur')) return '无人机运动模糊'
  if (conditions.day_to_night === true || degradation.includes('day-to-night') || degradation.includes('low-light')) {
    return ['城市驾驶', '自动驾驶', 'autonomous-driving'].includes(dataset.scene_domain) ? '自动驾驶弱光' : '无人机弱光'
  }
  if (degradation.includes('fog') || 'fog_strength' in conditions || 'fog_density' in conditions) {
    return ['城市驾驶', '自动驾驶', 'autonomous-driving'].includes(dataset.scene_domain) ? '自动驾驶气雾' : '无人机气雾'
  }
  return dataset.weather
}

function defaultGeneratedDatasetName(datasetName: string, conditionEnglishName: string) {
  const suffix = `_${conditionEnglishName}_generated`
  return `${datasetName.trim().slice(0, 120 - suffix.length)}${suffix}`
}

interface BaseGenPreview {
  model_path: string
  device_policy: string
  images: Array<{
    seed: number
    scene: Record<string, string | string[]>
    template_id: string
    prompt: string
    width: number
    height: number
  }>
}

export function DataBuilderPage({ navigate, refresh }: PageProps) {
  const [step, setStep] = useState(0)
  const [source, setSource] = useState(sourceCards[0])
  const [domain, setDomain] = useState('无人机航拍')
  const [localDomainSelection, setLocalDomainSelection] = useState('无人机航拍')
  const [customLocalDomain, setCustomLocalDomain] = useState('')
  const [weather, setWeather] = useState('晴朗')
  const [resolution, setResolution] = useState('1920×1080')
  const [blur, setBlur] = useState(0.3)
  const [fogStrength, setFogStrength] = useState(1)
  const [conditionEffect, setConditionEffect] = useState<'fog' | 'motion_blur' | 'day_to_night' | 'driving_fog' | 'driving_day_to_night'>('fog')
  const [motionConditionMode, setMotionConditionMode] = useState<'preset' | 'files'>('preset')
  const [motionConditionDirectory, setMotionConditionDirectory] = useState('')
  const [motionPreset, setMotionPreset] = useState('forward')
  const [motionStrength, setMotionStrength] = useState(0.14)
  const [samples, setSamples] = useState(12)
  const [seeds, setSeeds] = useState([1001, 1002, 1003])
  const [generatorSeed, setGeneratorSeed] = useState(1001)
  const [generatorSteps, setGeneratorSteps] = useState(9)
  const [devicePolicy, setDevicePolicy] = useState('cuda')
  const [basegenSelections, setBasegenSelections] = useState<Record<string, BaseGenSelection>>({})
  const [basegenCustom, setBasegenCustom] = useState('')
  const [inputDatasetId, setInputDatasetId] = useState<string>()
  const [conditionDatasetName, setConditionDatasetName] = useState('')
  const [localDatasetName, setLocalDatasetName] = useState('')
  const [localNonidealCondition, setLocalNonidealCondition] = useState('无')
  const [localDirectory, setLocalDirectory] = useState('')
  const [annotationPath, setAnnotationPath] = useState('')
  const [annotationMode, setAnnotationMode] = useState<'coco' | 'yolo' | 'visdrone'>('coco')
  const [detectedAnnotationCategories, setDetectedAnnotationCategories] = useState<DatasetAnnotationCategories>()
  const [detectingAnnotationCategories, setDetectingAnnotationCategories] = useState(false)
  const [annotationCategoryError, setAnnotationCategoryError] = useState('')
  const [categoryTemplateId, setCategoryTemplateId] = useState('visdrone')
  const [customDatasetCategories, setCustomDatasetCategories] = useState<CategoryDefinition[]>([{ id: 1, name: '' }])
  const [localDataPicker, setLocalDataPicker] = useState<{
    field: 'directory' | 'annotation_path' | 'motion_condition_directory'
    title: string
    kind: 'directory' | 'annotation'
    initialPath?: string
  }>()
  const [jobId, setJobId] = useState<string>()
  const [finishedJob, setFinishedJob] = useState<Job>()
  const [submitting, setSubmitting] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [basegenPreview, setBasegenPreview] = useState<BaseGenPreview>()
  const datasets = useResource<Dataset[]>('/api/datasets', [])
  const categoryTemplates = useResource<CategoryTemplate[]>('/api/category-templates', [])
  const sourceRuntimes = useResource<Adapter[]>('/api/adapters', [])
  const basegenSchema = useResource<BaseGenSceneSchema>(
    '/api/adapters/adapter_basegen/scene-schema',
    { version: '1.0', domains: [] },
  )

  const isBaseGen = source.id === 'adapter_basegen'
  const isCondition = source.id === 'adapter_condition'
  const isMotionBlur = isCondition && conditionEffect === 'motion_blur'
  const isUavDayToNight = isCondition && conditionEffect === 'day_to_night'
  const isDrivingFog = isCondition && conditionEffect === 'driving_fog'
  const isDrivingDayToNight = isCondition && conditionEffect === 'driving_day_to_night'
  const isDayToNight = isUavDayToNight || isDrivingDayToNight
  const isDrivingCondition = isDrivingFog || isDrivingDayToNight
  const conditionSceneDomain = isDrivingCondition ? '城市驾驶' : '无人机航拍'
  const conditionEnglishName = isMotionBlur ? 'motion_blur' : isDayToNight ? 'low_light' : 'fog'
  const conditionAdapterId = isMotionBlur
    ? 'adapter_motion_blur'
    : isUavDayToNight
      ? 'adapter_day_to_night'
      : isDrivingFog
        ? 'adapter_warpi2i_fog'
        : isDrivingDayToNight
          ? 'adapter_warpi2i_day_to_night'
          : 'adapter_condition'
  const motionPresetLabel = motionPresetOptions.find((item) => item.value === motionPreset)?.label || motionPreset
  const currentBasegenDomain = basegenSchema.data.domains.find((item) => item.label_zh === domain)
  const localSceneDomain = localDomainSelection === CUSTOM_SCENE_DOMAIN
    ? customLocalDomain.trim()
    : localDomainSelection
  const localSceneDomainReady = Boolean(localSceneDomain)
  const domainOptions = isBaseGen
    ? basegenSchema.data.domains.map((item) => item.label_zh)
    : isCondition ? [conditionSceneDomain] : localSceneDomains
  const resolutionOptions = isBaseGen
    ? [
        { label: '方形', options: ['512×512', '768×768', '1024×1024'].map((value) => ({ value })) },
        { label: '横屏 16:9', options: ['768×432', '1024×576', '1280×720', '1536×864'].map((value) => ({ value })) },
        { label: '竖屏 9:16', options: ['432×768', '576×1024', '720×1280', '864×1536'].map((value) => ({ value })) },
      ]
    : [{ label: '常用分辨率', options: ['1920×1080', '1280×720', '640×640'].map((value) => ({ value })) }]
  const weatherOptions = ['晴朗', '雾', '雨', '夜间']
  const combinationCount = isBaseGen || isCondition ? 1 : seeds.length
  const conditionInputDatasets = datasets.data.filter((item) =>
    ['REAL', 'GENERATIVE'].includes(item.source_type)
    && (isDrivingCondition
      ? ['城市驾驶', '自动驾驶', 'autonomous-driving'].includes(item.scene_domain)
      : ['无人机航拍', '低空无人机', 'low-altitude-uav'].includes(item.scene_domain)),
  )
  const inputDataset = datasets.data.find((item) => item.id === inputDatasetId)
  const selectedCategories = isCondition
    ? inputDataset?.categories || []
    : source.id === 'local-import' && annotationPath
      ? detectedAnnotationCategories?.categories || []
    : categoriesFromSelection(categoryTemplates.data, categoryTemplateId, 'dataset', customDatasetCategories)
  const categoriesReady = validCategories(selectedCategories)
  const outputSampleCount = isCondition ? inputDataset?.sample_count || 0 : samples
  const conditionHasTruth = Boolean(
    inputDataset && ['CANDIDATE', 'VERIFIED'].includes(inputDataset.annotation_status),
  )

  useEffect(() => {
    if (!isBaseGen || !currentBasegenDomain) return
    setBasegenSelections((current) => Object.fromEntries(
      currentBasegenDomain.fields
        .filter((field) => field.kind !== 'text')
        .map((field) => {
          const selection = current[field.name]
          if (!selection || selection.mode === 'random') return [field.name, { mode: 'random' }]
          const allowed = new Set(field.options.map((option) => option.value))
          const values = 'values' in selection ? selection.values : [selection.value]
          return [field.name, values.every((value) => allowed.has(value)) ? selection : { mode: 'random' }]
        }),
    ))
  }, [isBaseGen, currentBasegenDomain?.value])

  const optionFor = (field: BaseGenSceneField, value: string) =>
    field.options.find((option) => option.value === value)

  const fixedValues = (selection?: BaseGenSelection): string[] => {
    if (!selection || selection.mode === 'random') return []
    return 'values' in selection ? selection.values : [selection.value]
  }

  const optionDisabled = (field: BaseGenSceneField, option: BaseGenSceneOption) => {
    if (!currentBasegenDomain) return false
    const environmentSelection = basegenSelections.environment
    const fixedEnvironment = fixedValues(environmentSelection)[0]
    if (field.name !== 'environment') {
      return Boolean(fixedEnvironment && option.environments && !option.environments.includes(fixedEnvironment))
    }
    return currentBasegenDomain.fields.some((dependentField) => {
      if (dependentField.name === 'environment' || dependentField.kind === 'text') return false
      return fixedValues(basegenSelections[dependentField.name]).some((value) => {
        const selectedOption = optionFor(dependentField, value)
        return Boolean(selectedOption?.environments && !selectedOption.environments.includes(option.value))
      })
    })
  }

  const changeBasegenSelection = (field: BaseGenSceneField, selection: BaseGenSelection) => {
    const next = { ...basegenSelections, [field.name]: selection }
    let adjusted = false
    if (field.name === 'environment' && selection.mode === 'fixed' && 'value' in selection && currentBasegenDomain) {
      for (const dependentField of currentBasegenDomain.fields) {
        if (dependentField.name === 'environment' || dependentField.kind === 'text') continue
        const current = next[dependentField.name]
        if (!current || current.mode === 'random') continue
        const compatible = fixedValues(current).filter((value) => {
          const selectedOption = optionFor(dependentField, value)
          return !selectedOption?.environments || selectedOption.environments.includes(selection.value)
        })
        if (compatible.length === fixedValues(current).length) continue
        next[dependentField.name] = dependentField.kind === 'multi' && compatible.length
          ? { mode: 'fixed', values: compatible }
          : { mode: 'random' }
        adjusted = true
      }
    }
    setBasegenSelections(next)
    if (adjusted) message.info('与新环境不兼容的场景选项已切换为随机')
  }

  const weatherSummary = (() => {
    if (!currentBasegenDomain) return '随机'
    const field = currentBasegenDomain.fields.find((item) => item.name === 'weather')
    const values = fixedValues(basegenSelections.weather)
    return values.length && field ? optionFor(field, values[0])?.label_zh || values[0] : '随机'
  })()

  const basegenSceneSummary = currentBasegenDomain?.fields
    .filter((field) => field.kind !== 'text')
    .map((field) => {
      const values = fixedValues(basegenSelections[field.name])
      const value = values.length
        ? values.map((item) => optionFor(field, item)?.label_zh || item).join('、')
        : '随机'
      return `${field.label_zh}：${value}`
    }) || []

  const selectSource = (item: typeof sourceCards[number]) => {
    if (item.disabled) return
    setSource(item)
    if (item.id === 'adapter_basegen') {
      const available = basegenSchema.data.domains.map((entry) => entry.label_zh)
      const preferred = basegenSchema.data.domains.find((entry) => entry.value === 'low-altitude-uav')
      const nextDomain = available.includes(domain) ? domain : preferred?.label_zh || available[0] || '低空无人机'
      setDomain(nextDomain)
      setResolution(basegenSchema.data.domains.find((entry) => entry.label_zh === nextDomain)?.default_resolution || '1024×1024')
    } else if (item.id === 'adapter_condition') {
      setDomain('无人机航拍')
      setWeather('雾')
      setResolution('原始分辨率')
    } else {
      if (!['无人机航拍', '卫星遥感', '城市驾驶'].includes(domain)) setDomain('无人机航拍')
      setResolution('1920×1080')
    }
  }
  const healthSource = async (id: string) => {
    try {
      const adapterId = id === 'adapter_condition' ? conditionAdapterId : id
      const result = await post<{ healthy: boolean }>(`/api/adapters/${adapterId}/health-check`)
      result.healthy ? message.success('数据来源接口测试通过') : message.warning('数据来源当前不可用')
      await sourceRuntimes.reload()
    } catch (error) {
      message.error((error as Error).message)
    }
  }
  const selectDomain = (value: string) => {
    setDomain(value)
    if (isBaseGen) {
      setResolution(basegenSchema.data.domains.find((entry) => entry.label_zh === value)?.default_resolution || '1024×576')
    }
  }
  const changeAnnotationMode = (value: string | number) => {
    const mode = String(value) as 'coco' | 'yolo' | 'visdrone'
    setAnnotationMode(mode)
    if (mode === 'visdrone') setCategoryTemplateId('visdrone')
    setAnnotationPath('')
    setDetectedAnnotationCategories(undefined)
    setAnnotationCategoryError('')
  }
  const loadAnnotationCategories = async (path: string) => {
    setDetectingAnnotationCategories(true)
    setDetectedAnnotationCategories(undefined)
    setAnnotationCategoryError('')
    try {
      const query = new URLSearchParams({ annotation_path: path, annotation_format: annotationMode.toUpperCase() })
      const detected = await api<DatasetAnnotationCategories>(`/api/local-dataset-annotation-categories?${query}`)
      setDetectedAnnotationCategories(detected)
      setCategoryTemplateId(detected.category_template)
      message.success(`已从标注读取 ${detected.categories.length} 个类别`)
    } catch (error) {
      setAnnotationCategoryError((error as Error).message)
    } finally {
      setDetectingAnnotationCategories(false)
    }
  }
  const selectLocalDataResource = (path: string) => {
    if (localDataPicker?.field === 'directory') setLocalDirectory(path)
    if (localDataPicker?.field === 'motion_condition_directory') {
      setMotionConditionDirectory(path)
    }
    if (localDataPicker?.field === 'annotation_path') {
      setAnnotationPath(path)
      void loadAnnotationCategories(path)
    }
    setLocalDataPicker(undefined)
  }
  const acquisitionPayload = () => ({
    name: isCondition ? conditionDatasetName.trim() : `${domain} · ${source.title}`,
    adapter_id: isCondition ? conditionAdapterId : source.id,
    source_type: source.source,
    sample_count: isCondition ? inputDataset?.sample_count || 1 : samples,
    seeds: isBaseGen ? [generatorSeed] : isCondition ? [isMotionBlur ? 2023 : isDayToNight || isDrivingFog ? 42 : 1001] : seeds,
    conditions: {
      scene: isBaseGen ? {
        domain: currentBasegenDomain?.value,
        domain_label: currentBasegenDomain?.label_zh,
        weather: weatherSummary,
        fields: basegenSelections,
        custom: basegenCustom,
      } : isCondition ? { domain: conditionSceneDomain, weather: isMotionBlur ? inputDataset?.weather || '未记录' : isDayToNight ? '弱光' : '雾' } : { domain, weather },
      sensor: isBaseGen ? { resolution } : isCondition ? { resolution: inputDataset?.resolution || '原始分辨率' } : { resolution, motion_blur: blur, fog_density: weather === '雾' ? 0.4 : 0 },
    },
    model_parameters: isBaseGen ? { steps: generatorSteps, guidance_scale: 0, device_policy: devicePolicy, local_files_only: false } : isMotionBlur ? { effect: 'motion_blur', domain: 'uav_aerial', motion: motionPreset, strength: motionStrength, sample_timesteps: 20, precision: 'FP32', checkpoint: 'ID_Blau.pth', ...(motionConditionMode === 'files' ? { condition_directory: motionConditionDirectory, condition_matching: 'filename', fallback_motion: 'random-preset' } : {}) } : isDrivingFog ? { effect: 'fog', domain: 'autonomous_driving', method: 'paired', image_prep: 'multiple_of_8', precision: 'FP16', checkpoint: 'foggy_1.pkl' } : isDrivingDayToNight ? { effect: 'day_to_night', domain: 'autonomous_driving', method: 'unpaired', direction: 'a2b', image_prep: 'resize_512x512', precision: 'FP16', checkpoint: 'BDD100K_day2night.pkl' } : isUavDayToNight ? { effect: 'day_to_night', domain: 'uav_aerial', direction: 'a2b', image_prep: 'resize_640x640', model_size: 640, precision: 'FP16', checkpoint: 'uav_daynight_sichuan_3125_model_3125' } : isCondition ? { effect: 'fog', domain: 'uav_aerial', image_prep: 'resize_512x512', precision: 'FP16', fog_strength: fogStrength, checkpoint: 'uav_fog_content15_model_2501' } : {},
    input_dataset_id: isCondition ? inputDatasetId : null,
    category_template: isCondition ? inputDataset?.category_template || 'custom' : categoryTemplateId,
    categories: selectedCategories.map(({ id, name }) => ({ id, name })),
  })
  const previewBasegen = async () => {
    setPreviewing(true)
    try {
      setBasegenPreview(await post<BaseGenPreview>('/api/adapters/adapter_basegen/preview', acquisitionPayload()))
    } catch (error) { message.error((error as Error).message) } finally { setPreviewing(false) }
  }
  const submit = async () => {
    setSubmitting(true)
    try {
      let job: Job
      if (source.id === 'local-import') {
        job = await post<Job>('/api/datasets/import', {
          name: localDatasetName.trim(),
          directory: localDirectory,
          annotation_path: annotationPath || null,
          annotation_format: annotationMode.toUpperCase(),
          scene_domain: localSceneDomain,
          nonideal_condition: localNonidealCondition,
          category_template: detectedAnnotationCategories?.category_template || categoryTemplateId,
          categories: selectedCategories.map(({ id, name }) => ({ id, name })),
        })
      } else {
        job = await post<Job>('/api/acquisition-jobs', acquisitionPayload())
      }
      setJobId(job.id)
      setStep(3)
      refresh()
    } catch (error) { message.error((error as Error).message) } finally { setSubmitting(false) }
  }
  const finish = useCallback((job: Job) => { setFinishedJob(job); refresh() }, [refresh])
  const freeze = async () => {
    const datasetId = finishedJob?.result?.dataset_id
    if (!datasetId || typeof datasetId !== 'string') return
    try { await post(`/api/datasets/${datasetId}/freeze`); message.success('数据集已校核并冻结'); setStep(4); refresh() } catch (error) { message.error((error as Error).message) }
  }
  const conditionModelTitle = isMotionBlur ? '运动模糊模型' : isDayToNight ? '弱光模型' : '气雾模型'
  const conditionModelItems = isMotionBlur
    ? [{ key: 'model', label: '模型', children: 'DiffusionBlur' }, { key: 'sampler', label: '采样器', children: 'DDIM / 20 步' }, { key: 'output', label: '模型输出', children: '保持原图分辨率' }, { key: 'precision', label: '推理精度', children: 'CUDA / FP32' }]
    : isDrivingFog
      ? [{ key: 'model', label: '模型', children: 'WarpI2I · 自动驾驶气雾' }, { key: 'method', label: '方法', children: 'paired' }, { key: 'checkpoint', label: '权重', children: '2_24_drive_v2_warped_128 / foggy_1.pkl' }, { key: 'prep', label: '模型输入', children: '原图调整为 8 的倍数' }, { key: 'output', label: '模型输出', children: '恢复原图分辨率和文件名' }, { key: 'precision', label: '推理精度', children: 'CUDA / FP16' }]
      : isDrivingDayToNight
        ? [{ key: 'model', label: '模型', children: 'WarpI2I · 自动驾驶弱光' }, { key: 'method', label: '方法', children: 'unpaired / a2b' }, { key: 'checkpoint', label: '权重', children: 'BDD100K_day2night.pkl' }, { key: 'prep', label: '模型输入', children: '缩放至 512×512' }, { key: 'output', label: '模型输出', children: '恢复原图分辨率和文件名' }, { key: 'precision', label: '推理精度', children: 'CUDA / FP16' }]
        : isUavDayToNight
        ? [{ key: 'model', label: '模型', children: 'DiffusionDegrade' }, { key: 'checkpoint', label: '权重', children: 'model_3125' }, { key: 'model-size', label: '模型输入', children: '整图缩放至 640×640' }, { key: 'output', label: '模型输出', children: '恢复原图分辨率和文件名' }, { key: 'precision', label: '推理精度', children: 'CUDA / FP16' }]
      : [{ key: 'model', label: '模型', children: 'DiffusionDegrade · 无人机气雾' }, { key: 'checkpoint', label: '权重', children: 'content15 / model_2501' }, { key: 'prep', label: '模型输入', children: '缩放至 512×512' }, { key: 'output', label: '模型输出', children: '恢复原图分辨率' }, { key: 'precision', label: '推理精度', children: 'CUDA / FP16' }]
  const conditionBuildName = isMotionBlur ? '非理想条件生成 · 无人机运动模糊' : isDrivingFog ? '非理想条件生成 · 自动驾驶气雾' : isDrivingDayToNight ? '非理想条件生成 · 自动驾驶弱光' : isUavDayToNight ? '非理想条件生成 · 无人机弱光' : '非理想条件生成 · 无人机气雾'
  const conditionSceneSummary = isMotionBlur ? '无人机航拍 / 保持源天气' : `${conditionSceneDomain} / ${isDayToNight ? '弱光' : '气雾'}`
  const conditionSensorSummary = isMotionBlur
    ? `${inputDataset?.resolution || '原始分辨率'} / ${motionConditionMode === 'files' ? '条件文件（缺失时随机回退）' : motionPresetLabel} / ${motionConditionMode === 'files' ? '回退强度' : '强度'} ${motionStrength.toFixed(2)} / DDIM 20 步`
    : isDrivingFog
      ? `${inputDataset?.resolution || '原始分辨率'} / 自动驾驶气雾 / WarpI2I paired / FP16`
      : isDrivingDayToNight
        ? `${inputDataset?.resolution || '原始分辨率'} / 自动驾驶弱光 / WarpI2I unpaired / 512×512 / FP16`
        : isUavDayToNight
      ? `${inputDataset?.resolution || '原始分辨率'} / 整图 640×640 / FP16`
      : `${inputDataset?.resolution || '原始分辨率'} / DiffusionDegrade 无人机气雾 / FP16 / 强度 ${fogStrength.toFixed(1)}`
  const conditionSeedSummary = isMotionBlur ? '固定为 2023' : isDayToNight || isDrivingFog ? '固定为 42' : '固定为 1001'
  const conditionExecutionMessage = isMotionBlur ? '本任务将调用 DiffusionBlur 执行无人机运动模糊生成' : isDrivingFog ? '本任务将调用 WarpI2I paired 方法执行自动驾驶气雾生成' : isDrivingDayToNight ? '本任务将调用 WarpI2I 执行自动驾驶弱光生成' : isUavDayToNight ? '本任务将调用 DiffusionDegrade 执行无人机弱光生成' : '本任务将调用 DiffusionDegrade 执行无人机气雾生成'

  return (
    <Space direction="vertical" size={18} style={{ width: '100%' }}>
      <Steps current={step} items={(source.id === 'local-import' ? ['选择来源', '导入配置', '导入确认', '执行导入', '真值冻结'] : ['选择来源', '配置条件', '组合预览', '执行与浏览', '真值冻结']).map((title) => ({ title }))} />
      {step === 0 && <Card title="选择数据来源">
        <Row gutter={[16, 16]}>{sourceCards.map((item) => {
          const runtimeId = item.id === 'adapter_condition' ? conditionAdapterId : item.id
          const runtime = sourceRuntimes.data.find((entry) => entry.id === runtimeId)
          return <Col xs={24} md={12} xl={6} key={item.id}><Card hoverable={!item.disabled} className={`source-card ${source.id === item.id ? 'source-selected' : ''} ${item.disabled ? 'source-disabled' : ''}`} onClick={() => selectSource(item)}><div className="source-icon">{item.icon}</div><Typography.Title level={4}>{item.title}</Typography.Title><Typography.Paragraph type="secondary">{item.description}</Typography.Paragraph><Space wrap>{item.id === 'local-import' ? <StatusTag status="HEALTHY" /> : item.disabled ? <Tag>未接入</Tag> : <><StatusTag status={runtime?.status || (sourceRuntimes.loading ? 'CHECKING' : 'UNAVAILABLE')} /><Button size="small" onClick={(event) => { event.stopPropagation(); healthSource(item.id) }}>接口测试</Button></>}</Space></Card></Col>
        })}</Row>
        <div className="wizard-actions"><Button type="primary" onClick={() => setStep(1)}>下一步：{source.id === 'local-import' ? '导入配置' : '配置条件'} <ArrowRightOutlined /></Button></div>
      </Card>}
      {step === 1 && <Row gutter={16}>
        <Col xs={24} xl={source.id === 'local-import' ? 24 : 12}>
          <Card title={source.id === 'local-import' ? '选择本地数据' : '场景条件'}>
            <Form layout="vertical">
              {source.id === 'local-import' && <>
                <Form.Item label="数据集名称" required>
                  <Input value={localDatasetName} onChange={(event) => setLocalDatasetName(event.target.value)} maxLength={120} showCount placeholder="例如 VisDrone2019 测试集" />
                </Form.Item>
                <Form.Item label="非理想条件">
                  <Select value={localNonidealCondition} onChange={setLocalNonidealCondition} options={['无', '无人机气雾', '无人机运动模糊', '无人机弱光', '自动驾驶气雾', '自动驾驶弱光'].map((value) => ({ value }))} />
                </Form.Item>
                <Form.Item label="图像目录" required>
                  <Space.Compact block>
                    <Input readOnly value={localDirectory} placeholder="选择服务器上的图像目录" />
                    <Button onClick={() => setLocalDataPicker({ field: 'directory', title: '选择服务器图像目录', kind: 'directory', initialPath: localDirectory || undefined })}>选择目录</Button>
                    <Button danger disabled={!localDirectory} icon={<DeleteOutlined />} onClick={() => setLocalDirectory('')}>清除</Button>
                  </Space.Compact>
                </Form.Item>
                <Form.Item label="标注（可选）">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Segmented block value={annotationMode} onChange={changeAnnotationMode} options={[{ value: 'coco', label: 'COCO 格式json' }, { value: 'yolo', label: 'YOLO 格式' }, { value: 'visdrone', label: 'VisDrone 格式' }]} />
                    <Space.Compact block>
                      <Input readOnly value={annotationPath} placeholder={annotationMode === 'coco' ? '选择服务器上的 COCO JSON 文件' : `选择服务器上的 ${annotationMode.toUpperCase()} 标注目录`} />
                      <Button onClick={() => setLocalDataPicker({ field: 'annotation_path', title: annotationMode === 'coco' ? '选择 COCO 标注文件' : `选择 ${annotationMode.toUpperCase()} 标注目录`, kind: annotationMode === 'coco' ? 'annotation' : 'directory', initialPath: annotationMode === 'coco' ? localDirectory || undefined : annotationPath || localDirectory || undefined })}>{annotationMode === 'coco' ? '选择文件' : '选择目录'}</Button>
                      <Button danger disabled={!annotationPath} icon={<DeleteOutlined />} onClick={() => { setAnnotationPath(''); setDetectedAnnotationCategories(undefined); setAnnotationCategoryError('') }}>清除</Button>
                    </Space.Compact>
                  </Space>
                </Form.Item>
              </>}
              {isCondition && <Form.Item label="非理想条件类型"><Select value={conditionEffect} onChange={(value) => { setConditionEffect(value); setInputDatasetId(undefined); setConditionDatasetName('') }} options={[{ value: 'fog', label: '无人机气雾' }, { value: 'motion_blur', label: '无人机运动模糊' }, { value: 'day_to_night', label: '无人机弱光' }, { value: 'driving_fog', label: '自动驾驶气雾' }, { value: 'driving_day_to_night', label: '自动驾驶弱光' }]} /></Form.Item>}
              <Form.Item label="场景域" required={source.id === 'local-import'}><Select disabled={isCondition} loading={isBaseGen && basegenSchema.loading} value={source.id === 'local-import' ? localDomainSelection : isCondition ? conditionSceneDomain : domain} onChange={source.id === 'local-import' ? setLocalDomainSelection : selectDomain} options={source.id === 'local-import' ? [...localSceneDomains.map((value) => ({ value })), { value: CUSTOM_SCENE_DOMAIN, label: '自定义' }] : domainOptions.map((value) => ({ value }))} /></Form.Item>
              {source.id === 'local-import' && localDomainSelection === CUSTOM_SCENE_DOMAIN && <Form.Item label="自定义场景域" required><Input value={customLocalDomain} onChange={(event) => setCustomLocalDomain(event.target.value)} maxLength={80} showCount placeholder="例如：港口监控、室内仓储" /></Form.Item>}
              {isBaseGen && !basegenSchema.loading && !currentBasegenDomain && <Alert type="error" showIcon message="未找到当前领域的 BaseGen 场景目录" />}
              {isBaseGen && currentBasegenDomain?.fields.map((field) => {
                if (field.kind === 'text') {
                  return <Form.Item key={field.name} label={field.label_zh} extra={field.description_zh}><Input.TextArea value={basegenCustom} onChange={(event) => setBasegenCustom(event.target.value)} rows={3} maxLength={500} showCount placeholder="可选；建议使用英文描述" /></Form.Item>
                }
                const selection = basegenSelections[field.name] || { mode: 'random' }
                if (field.kind === 'multi') {
                  const values = fixedValues(selection)
                  return (
                    <Form.Item key={field.name} label={field.label_zh} extra={field.description_zh}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Segmented block value={selection.mode} options={[{ value: 'random', label: field.weighted ? '随机组合（按权重）' : '随机组合' }, { value: 'fixed', label: '手动选择' }]} onChange={(value) => changeBasegenSelection(field, value === 'random' ? { mode: 'random' } : { mode: 'fixed', values })} />
                        {selection.mode === 'fixed' && <Select mode="multiple" allowClear value={values} maxTagCount="responsive" placeholder="最多选择四项" onChange={(next) => changeBasegenSelection(field, { mode: 'fixed', values: next.slice(0, 4) })} options={field.options.map((option) => ({ value: option.value, label: option.label_zh, disabled: optionDisabled(field, option) }))} />}
                      </Space>
                    </Form.Item>
                  )
                }
                const value = fixedValues(selection)[0] || RANDOM_VALUE
                return (
                  <Form.Item key={field.name} label={field.label_zh} extra={field.description_zh}>
                    <Select value={value} onChange={(next) => changeBasegenSelection(field, next === RANDOM_VALUE ? { mode: 'random' } : { mode: 'fixed', value: next })} options={[{ value: RANDOM_VALUE, label: '随机' }, ...field.options.map((option) => ({ value: option.value, label: option.label_zh, disabled: optionDisabled(field, option) }))]} />
                  </Form.Item>
                )
              })}
              {source.id !== 'local-import' && !isBaseGen && !isCondition && <Form.Item label="天气 / 环境"><Segmented block value={weather} onChange={(value) => setWeather(String(value))} options={weatherOptions} /></Form.Item>}
              {source.id !== 'local-import' && !isCondition && <Form.Item label="输出数量"><InputNumber value={samples} onChange={(value) => setSamples(value || 1)} min={1} max={1000} addonAfter="张" style={{ width: '100%' }} /></Form.Item>}
            </Form>
          </Card>
        </Col>
        {source.id !== 'local-import' && <Col xs={24} xl={12}>
          {isCondition
            ? <Card title={conditionModelTitle}>
                  <Descriptions column={1} size="small" bordered items={conditionModelItems} />
                  {isMotionBlur
                    ? <Form layout="vertical" className="top-gap">
                        <Form.Item label="运动条件来源">
                          <Segmented block value={motionConditionMode} onChange={(value) => setMotionConditionMode(value as 'preset' | 'files')} options={[{ value: 'preset', label: '预设运动模式' }, { value: 'files', label: '运动条件文件' }]} />
                        </Form.Item>
                        {motionConditionMode === 'preset'
                          ? <Form.Item label="无人机运动类型"><div className="motion-preset-grid" role="radiogroup" aria-label="无人机运动类型">{motionPresetOptions.map((option) => <button key={option.value} type="button" role="radio" aria-checked={motionPreset === option.value} className={`motion-preset-card ${motionPreset === option.value ? 'selected' : ''}`} onClick={() => setMotionPreset(option.value)}><MotionPresetDiagram motion={option.value} /><strong>{option.label}</strong><small>{option.description}</small></button>)}</div></Form.Item>
                          : <Form.Item label={<Space size={6}>运动条件 NPY 目录<Tooltip placement="right" title={<div><div>平台将按图片文件名匹配运动条件</div><div>例如 image.jpg 匹配 image.npy 或 image_condition.npy；会递归扫描目录。未匹配到 NPY 的图片会按随机种子选取一种预设运动模式。</div></div>}><QuestionCircleOutlined aria-label="运动条件 NPY 目录说明" style={{ color: '#1677ff', cursor: 'help' }} /></Tooltip></Space>} required>
                              <Space.Compact block>
                                <Input readOnly value={motionConditionDirectory} placeholder="请选择服务器上的运动条件目录" />
                                {motionConditionDirectory && <Button onClick={() => setMotionConditionDirectory('')}>清除</Button>}
                                <Button type="primary" onClick={() => setLocalDataPicker({ field: 'motion_condition_directory', title: '选择运动条件 NPY 目录', kind: 'directory', initialPath: motionConditionDirectory || undefined })}>选择目录</Button>
                              </Space.Compact>
                            </Form.Item>}
                        <Form.Item label={`${motionConditionMode === 'files' ? '回退运动模糊强度' : '运动模糊强度'} ${motionStrength.toFixed(2)}`} extra={motionConditionMode === 'preset' ? 'ID-Blau 归一化条件强度，不表示像素位移' : undefined}><Slider value={motionStrength} onChange={setMotionStrength} min={0.01} max={0.35} step={0.01} marks={{ 0.01: '轻微', 0.14: '默认', 0.35: '强烈' }} /></Form.Item>
                      </Form>
                    : isDrivingDayToNight
                      ? <Alert className="top-gap" type="info" showIcon message="当前模型不提供夜间强度控制" description="使用 WarpI2I BDD100K 权重执行固定的白天到夜晚域转换。" />
                      : isUavDayToNight
                        ? null
                      : isDrivingFog
                        ? <Alert className="top-gap" type="info" showIcon message="自动驾驶气雾为固定模型效果" description="当前 WarpI2I paired 权重不提供连续气雾强度控制。" />
                        : <Form layout="vertical" className="top-gap"><Form.Item label={`气雾强度 ${fogStrength.toFixed(1)}`}><Slider value={fogStrength} onChange={setFogStrength} min={0} max={1} step={0.1} marks={{ 0: '0', 1: '1' }} /></Form.Item></Form>}
              </Card>
            : <Card title={isBaseGen ? '生成参数' : '传感器与成像条件'}><Form layout="vertical"><Form.Item label="图像分辨率"><Select value={resolution} onChange={setResolution} options={resolutionOptions} /></Form.Item>{isBaseGen ? <><Form.Item label="起始随机种子"><InputNumber value={generatorSeed} onChange={(value) => setGeneratorSeed(value || 0)} min={0} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="推理步数"><InputNumber value={generatorSteps} onChange={(value) => setGeneratorSteps(value || 1)} min={1} max={100} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="设备策略"><Select value={devicePolicy} onChange={setDevicePolicy} options={[{ value: 'cuda', label: '全量 CUDA（推荐）' }, { value: 'cpu-offload', label: 'CPU Offload（节省显存）' }]} /></Form.Item></> : <><Form.Item label={`运动模糊强度 ${blur.toFixed(1)}`}><Slider value={blur} onChange={setBlur} min={0} max={1} step={0.1} marks={{ 0: '清洁', 0.5: '中等', 1: '严重' }} /></Form.Item><Form.Item label="固定随机种子"><Checkbox.Group options={[1001, 1002, 1003, 1004].map((value) => ({ label: value, value }))} value={seeds} onChange={(values) => setSeeds(values as number[])} /></Form.Item></>}</Form></Card>}
        </Col>}
        {isCondition && <Col span={24}><Card title="选择非理想条件输入数据集"><Form layout="vertical"><Form.Item label="基础数据集"><Select value={inputDatasetId} onChange={(value) => { setInputDatasetId(value); const selectedDataset = conditionInputDatasets.find((item) => item.id === value); setConditionDatasetName(selectedDataset ? defaultGeneratedDatasetName(selectedDataset.name, conditionEnglishName) : '') }} style={{ width: '100%' }} placeholder={`选择${conditionSceneDomain}域输入数据集`} options={conditionInputDatasets.map((item) => ({ value: item.id, label: `${item.name} · ${item.sample_count} 张`, disabled: !item.categories.length }))} /></Form.Item><Form.Item label="生成数据集名称" required><Input value={conditionDatasetName} onChange={(event) => setConditionDatasetName(event.target.value)} maxLength={120} showCount placeholder="选择基础数据集后自动生成，可自定义修改" /></Form.Item></Form><Typography.Paragraph type="secondary" style={{ marginTop: 10, marginBottom: 0 }}>支持本地导入和基础图像生成数据集；任务处理全部图像并保持文件名和原始分辨率。</Typography.Paragraph></Card></Col>}
        <Col span={24}><Card title="目标检测类别">
          {source.id === 'adapter_condition'
            ? inputDataset
              ? <Alert type="info" showIcon message={`已继承 ${inputDataset.categories.length} 个类别`} description={inputDataset.categories.map((item) => `${item.id}:${item.name}`).join(' · ')} />
              : <Alert type="warning" showIcon message="请先选择输入数据集" />
            : source.id === 'local-import' && annotationPath
              ? detectingAnnotationCategories
                ? <Spin tip="正在从标注读取类别"><div style={{ height: 56 }} /></Spin>
                : detectedAnnotationCategories
                  ? <Alert type="success" showIcon message={`已从标注读取 ${detectedAnnotationCategories.categories.length} 个类别`} description={<Space direction="vertical" size={4}><Typography.Text type="secondary">来源：{detectedAnnotationCategories.source}</Typography.Text><Typography.Paragraph ellipsis={{ rows: 3, expandable: true, symbol: '展开全部' }} style={{ marginBottom: 0 }}>{detectedAnnotationCategories.categories.map((item) => `${item.id}:${item.name}`).join(' · ')}</Typography.Paragraph></Space>} />
                  : <Alert type="error" showIcon message="无法从标注读取类别" description={annotationCategoryError || '请重新选择包含类别定义的标注文件或目录'} />
              : source.id === 'local-import'
                ? <Space direction="vertical" size={12} style={{ width: '100%' }}><Alert type="info" showIcon message="未标注数据集：请指定计划检测的类别" description="这些类别将用于人工标注、自动标注模型兼容性校验和后续评测。" /><CategoryConfiguration templates={categoryTemplates.data} templateId={categoryTemplateId} scope="dataset" customCategories={customDatasetCategories} onTemplateChange={setCategoryTemplateId} onCustomChange={setCustomDatasetCategories} /></Space>
                : <CategoryConfiguration templates={categoryTemplates.data} templateId={categoryTemplateId} scope="dataset" customCategories={customDatasetCategories} onTemplateChange={setCategoryTemplateId} onCustomChange={setCustomDatasetCategories} />}
        </Card></Col>
        <Col span={24}><div className="wizard-actions"><Button onClick={() => setStep(0)}>上一步</Button><Button type="primary" disabled={!categoriesReady || (source.id === 'local-import' ? localDatasetName.trim().length < 2 || !localDirectory || !localSceneDomainReady : isBaseGen ? !currentBasegenDomain || generatorSeed < 0 || generatorSteps < 1 : isCondition ? !inputDatasetId || conditionDatasetName.trim().length < 2 || (isMotionBlur && motionConditionMode === 'files' && !motionConditionDirectory) : !seeds.length)} onClick={() => setStep(2)}>下一步：{source.id === 'local-import' ? '导入确认' : '组合预览'}</Button></div></Col>
      </Row>}
      {step === 2 && <Card title="提交前确认" extra={source.id === 'local-import' ? <Tag color="blue">本地真实数据</Tag> : isBaseGen ? <Tag color="purple">基础图像生成</Tag> : isCondition ? <Tag color="cyan">{isMotionBlur ? 'DiffusionBlur' : isDrivingCondition ? 'WarpI2I' : 'DiffusionDegrade'}</Tag> : <DemoTag />}>
        <Row gutter={[16, 16]}><Col xs={24} md={8}><Statistic title={source.id === 'local-import' ? '导入任务' : '配置单元'} value={source.id === 'local-import' ? 1 : combinationCount} suffix="个" /></Col><Col xs={24} md={8}><Statistic title="输出样本" value={source.id === 'local-import' ? '目录内图像' : outputSampleCount} suffix={source.id === 'local-import' ? undefined : '张'} /></Col><Col xs={24} md={8}><Statistic title="真值入口" value={source.id === 'local-import' ? (annotationPath ? '已提供' : '未提供') : isBaseGen ? '未标注' : isCondition ? (conditionHasTruth ? '继承为候选真值' : '未标注') : '候选框'} /></Col></Row><Divider />
        <Descriptions column={{ xs: 1, md: 2 }} bordered size="small" items={[{ key: 'source', label: '构建方式', children: isCondition ? conditionBuildName : source.title }, { key: 'scene', label: '场景', children: isCondition ? conditionSceneSummary : source.id === 'local-import' ? localSceneDomain : isBaseGen ? domain : `${domain} / ${weather}` }, ...(source.id === 'local-import' ? [{ key: 'nonideal-condition', label: '非理想条件', children: localNonidealCondition }] : []), { key: 'sensor', label: source.id === 'local-import' ? '输入目录' : isBaseGen ? '生成参数' : '成像条件', children: source.id === 'local-import' ? localDirectory : isBaseGen ? `${resolution} / ${generatorSteps} 步 / ${devicePolicy}` : isCondition ? conditionSensorSummary : `${resolution} / 模糊 ${blur}` }, { key: 'categories', label: '检测类别', span: 2, children: <Space direction="vertical" size={4} style={{ width: '100%' }}><Typography.Text strong>{selectedCategories.length} 类</Typography.Text><Typography.Paragraph ellipsis={{ rows: 3, expandable: true, symbol: '展开全部' }} style={{ marginBottom: 0 }}>{selectedCategories.map((item) => `${item.id}:${item.name}`).join(' · ')}</Typography.Paragraph></Space> }, { key: 'seed', label: '随机种子', children: source.id === 'local-import' ? '不适用' : isBaseGen ? `${generatorSeed} 起连续 ${samples} 个` : isCondition ? conditionSeedSummary : seeds.join(', ') }, { key: 'truth', label: '真值策略', children: source.id === 'local-import' ? (annotationPath ? '导入后作为候选真值' : '未提供') : isBaseGen ? '未标注，需另行标注后评测' : isCondition ? (conditionHasTruth ? '继承标注为候选真值，需抽查后冻结' : '源数据无标注，输出保持未标注') : '候选框，完成后需校核冻结' }, { key: 'official', label: '数据来源', children: source.id === 'local-import' ? <Tag color="blue">真实采集数据</Tag> : isBaseGen ? <Tag color="purple">真实模型生成</Tag> : isCondition ? <Tag color="cyan">真实模型退化</Tag> : <DemoTag /> }]} />
        {isBaseGen && <Card size="small" title="场景字段规则" className="top-gap"><Space wrap>{basegenSceneSummary.map((item) => <Tag key={item}>{item}</Tag>)}{basegenCustom && <Tag color="blue">自定义：{basegenCustom}</Tag>}</Space></Card>}
        {isBaseGen && basegenPreview && <Card size="small" title="随机计划预览（不加载模型）" className="top-gap"><List dataSource={basegenPreview.images} renderItem={(item) => <List.Item><Space direction="vertical" style={{ width: '100%' }}><Space wrap><Tag color="blue">seed {item.seed}</Tag><Tag>{item.width}×{item.height}</Tag>{currentBasegenDomain?.fields.filter((field) => field.kind !== 'text').map((field) => { const raw = item.scene[field.name]; const values = Array.isArray(raw) ? raw : [raw]; return <Tag key={field.name}>{field.label_zh}：{values.map((value) => optionFor(field, value)?.label_zh || value).join('、')}</Tag> })}</Space><Typography.Paragraph copyable={{ text: item.prompt }} ellipsis={{ rows: 3, expandable: true, symbol: '展开 prompt' }} style={{ marginBottom: 0 }}>{item.prompt}</Typography.Paragraph></Space></List.Item>} /></Card>}
        {source.id !== 'local-import' && <Alert className="inline-alert" type={isBaseGen || isCondition ? 'info' : 'warning'} showIcon message={isBaseGen ? '本任务将调用 BaseGen 真实生成图像' : isCondition ? conditionExecutionMessage : '本任务使用固定样例验证流程'} description={isBaseGen ? '模型在独立 gen 环境中运行；纯文本生成不提供目标框等真值。' : isCondition ? '外部项目、环境和权重均只读；输出写入平台 Artifact，并保留输入文件名和数据谱系。' : '输出会保留完整数据谱系，但不能作为生成模型能力结论。'} />}
        <div className="wizard-actions"><Button onClick={() => { setBasegenPreview(undefined); setStep(1) }}>上一步</Button>{isBaseGen && <Button loading={previewing} onClick={previewBasegen}>预览 3 个随机场景</Button>}<Button type="primary" loading={submitting} icon={<PlayCircleOutlined />} onClick={submit}>提交构建任务</Button></div>
      </Card>}
      {step === 3 && <Space direction="vertical" size={16} style={{ width: '100%' }}><JobProgress jobId={jobId} onFinish={finish} />{finishedJob?.status === 'SUCCEEDED' && <Card><Result status="success" title={source.id === 'local-import' ? '本地图像已导入' : '数据构建任务已完成'} subTitle={finishedJob.result?.annotation_status === 'UNLABELED' ? '当前没有真值，需进入数据集完成标注后才能正式评测。' : '输出当前仍是候选真值，冻结前不会进入正式评测。'} extra={finishedJob.result?.annotation_status === 'UNLABELED' ? [<Button key="datasets" type="primary" onClick={() => navigate('datasets')}>打开数据集</Button>] : [<Button key="freeze" type="primary" icon={<LockOutlined />} onClick={freeze}>校核并冻结数据版本</Button>, <Button key="datasets" onClick={() => navigate('datasets')}>打开数据集</Button>]} /></Card>}</Space>}
      {step === 4 && <Card><Result status="success" title="数据版本已冻结" subTitle="该版本不可变；后续修改真值需要创建新版本。" extra={[<Button type="primary" key="eval" onClick={() => navigate('evaluation')}>进入评测中心</Button>, <Button key="again" onClick={() => { setStep(0); setJobId(undefined); setFinishedJob(undefined) }}>继续构建数据</Button>]} /></Card>}
      {localDataPicker && <LocalResourcePicker
        open
        title={localDataPicker.title}
        scope="dataset"
        kind={localDataPicker.kind}
        initialPath={localDataPicker.initialPath}
        onClose={() => setLocalDataPicker(undefined)}
        onSelect={selectLocalDataResource}
      />}
    </Space>
  )
}

function DatasetBrowser({ dataset, onClose }: { dataset?: Dataset; onClose: () => void }) {
  const pageSize = 50
  const [items, setItems] = useState<DatasetSamplePage['items']>([])
  const [total, setTotal] = useState(0)
  const [declaredCount, setDeclaredCount] = useState(0)
  const [pageNumber, setPageNumber] = useState(1)
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const loadingRef = useRef(false)
  const activeDatasetId = useRef<string>()
  const browserRef = useRef<HTMLDivElement>(null)

  const load = async (datasetId: string, nextPage: number) => {
    if (loadingRef.current) return
    loadingRef.current = true
    setLoading(true)
    try {
      const page = await api<DatasetSamplePage>(`/api/datasets/${datasetId}/samples?offset=${(nextPage - 1) * pageSize}&limit=${pageSize}`)
      if (activeDatasetId.current !== datasetId) return
      setItems(page.items)
      setTotal(page.total)
      setDeclaredCount(page.declared_count)
      setPageNumber(nextPage)
      setError('')
    } catch (loadError) {
      if (activeDatasetId.current === datasetId) setError((loadError as Error).message)
    } finally {
      if (activeDatasetId.current === datasetId) {
        loadingRef.current = false
        setLoading(false)
      }
    }
  }

  useEffect(() => {
    activeDatasetId.current = dataset?.id
    loadingRef.current = false
    setItems([])
    setTotal(0)
    setPageNumber(1)
    setShowAnnotations(true)
    setDeclaredCount(dataset?.sample_count || 0)
    setError('')
    if (dataset) load(dataset.id, 1)
  }, [dataset?.id])

  const changePage = (nextPage: number) => {
    if (!dataset || loadingRef.current) return
    setItems([])
    browserRef.current?.scrollTo({ top: 0 })
    void load(dataset.id, nextPage)
  }
  const annotationBoxes = (item: DatasetSamplePage['items'][number]) => (
    <div className="dataset-browser-boxes">
      {item.boxes.map((box, index) => (
        <div
          className="dataset-browser-box"
          key={`${box.label}-${index}`}
          style={{
            left: `${Math.max(0, Math.min(1, box.x)) * 100}%`,
            top: `${Math.max(0, Math.min(1, box.y)) * 100}%`,
            width: `${Math.max(0, Math.min(1 - Math.max(0, box.x), box.width)) * 100}%`,
            height: `${Math.max(0, Math.min(1 - Math.max(0, box.y), box.height)) * 100}%`,
            borderColor: box.color,
          }}
        >
          <span style={{ background: box.color }}>{box.label}</span>
        </div>
      ))}
    </div>
  )

  return (
    <Drawer open={Boolean(dataset)} width="90vw" title={dataset ? `${dataset.name} · 全部样本` : '全部样本'} onClose={onClose} extra={<Space><Checkbox checked={showAnnotations} onChange={(event) => setShowAnnotations(event.target.checked)}>显示标注</Checkbox><Tag>第 {pageNumber} 页 · 本页 {items.length} / 共 {total || declaredCount} 张</Tag></Space>}>
      <div ref={browserRef} className="dataset-browser">
        {error && <Alert type="error" showIcon message="样本加载失败" description={error} />}
        {!loading && !error && total !== declaredCount && <Alert type="warning" showIcon message="登记样本数与磁盘文件数不一致" description={`登记 ${declaredCount} 个，当前 Artifact 目录找到 ${total} 个可浏览图片。`} />}
        {!loading && !error && total === 0 ? <Empty description="该数据集没有可浏览的图片文件" /> : (
          <Image.PreviewGroup preview={{
            imageRender: (originalNode, info) => {
              const item = items[info.current]
              if (!item) return originalNode
              const { x, y, rotate, scale, flipX, flipY } = info.transform
              return (
                <div
                  className="dataset-browser-preview-image"
                  style={{
                    aspectRatio: item.width && item.height ? `${item.width} / ${item.height}` : '16 / 9',
                    transform: `translate3d(${x}px, ${y}px, 0) scale3d(${flipX ? -scale : scale}, ${flipY ? -scale : scale}, 1) rotate(${rotate}deg)`,
                  }}
                >
                  {cloneElement(originalNode, { style: { ...originalNode.props.style, transform: 'none', transitionDuration: '0s' } })}
                  {showAnnotations && annotationBoxes(item)}
                </div>
              )
            },
          }}>
            <div className="dataset-browser-grid">
              {items.map((item) => (
                <div className="dataset-browser-sample" key={item.url}>
                  <div className="dataset-browser-image" style={{ aspectRatio: item.width && item.height ? `${item.width} / ${item.height}` : '16 / 9' }}>
                    <Image src={item.url} alt={item.name} loading="lazy" preview={{ mask: item.name }} />
                    {showAnnotations && annotationBoxes(item)}
                  </div>
                  <div className="dataset-browser-caption">
                    <Typography.Text ellipsis={{ tooltip: item.name }}>{item.name}</Typography.Text>
                    {item.annotation_source ? <Tag color={item.boxes.length ? 'blue' : 'default'}>{item.boxes.length} 框</Tag> : <Tag>无标注</Tag>}
                  </div>
                </div>
              ))}
            </div>
          </Image.PreviewGroup>
        )}
        {loading && <div className="dataset-browser-loading"><Spin /><Typography.Text type="secondary">正在加载图片…</Typography.Text></div>}
        {!error && total > 0 && <div className="dataset-browser-pagination"><Pagination current={pageNumber} pageSize={pageSize} total={total} showSizeChanger={false} showQuickJumper disabled={loading} showTotal={(count) => `共 ${count} 张`} onChange={changePage} /></div>}
      </div>
    </Drawer>
  )
}

function EvaluationVisualizationDrawer({ group, onClose }: { group?: ResultGroup; onClose: () => void }) {
  const pageSize = 12
  const [runId, setRunId] = useState<string>()
  const [data, setData] = useState<EvaluationVisualization>()
  const [pageNumber, setPageNumber] = useState(1)
  const [threshold, setThreshold] = useState(0.25)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setRunId(group?.run_ids[0])
    setData(undefined)
    setPageNumber(1)
    setThreshold(0.25)
    setError('')
  }, [group?.comparison_id])

  useEffect(() => {
    if (!runId) return
    let active = true
    setLoading(true)
    setError('')
    api<EvaluationVisualization>(`/api/evaluation-runs/${runId}/visualization?offset=${(pageNumber - 1) * pageSize}&limit=${pageSize}`)
      .then((result) => {
        if (!active) return
        setData(result)
        setThreshold((current) => Math.max(current, result.inference_confidence))
      })
      .catch((loadError) => {
        if (active) setError((loadError as Error).message)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => { active = false }
  }, [runId, pageNumber])

  const visibleBoxes = (item: EvaluationVisualization['items'][number]) => item.boxes.filter((box) => box.score >= threshold)
  const predictionBoxes = (item: EvaluationVisualization['items'][number]) => (
    <div className="dataset-browser-boxes">
      {visibleBoxes(item).map((box, index) => (
        <div
          className="dataset-browser-box"
          key={`${box.category_id}-${index}`}
          style={{
            left: `${box.x * 100}%`,
            top: `${box.y * 100}%`,
            width: `${box.width * 100}%`,
            height: `${box.height * 100}%`,
            borderColor: box.color,
          }}
        >
          <span style={{ background: box.color }}>{box.label} {box.score.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )

  return (
    <Drawer open={Boolean(group)} width="90vw" title="推理结果可视化" onClose={onClose}>
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Row gutter={[16, 12]} align="middle">
          <Col xs={24} md={10}><Typography.Text type="secondary">评测运行</Typography.Text><Select value={runId} onChange={(value) => { setRunId(value); setPageNumber(1); setThreshold(0.25) }} style={{ width: '100%', marginTop: 6 }} options={group?.run_ids.map((value, index) => ({ value, label: `运行 ${index + 1} · ${value}` }))} /></Col>
          <Col xs={24} md={6}><Typography.Text type="secondary">可视化置信度阈值</Typography.Text><InputNumber value={threshold} onChange={(value) => setThreshold(value ?? 0)} min={0} max={1} step={0.01} precision={3} style={{ width: '100%', marginTop: 6 }} /></Col>
          <Col xs={24} md={8}><Typography.Text type="secondary">当前内容</Typography.Text><div style={{ marginTop: 8 }}>{data ? <Space wrap><Tag>{data.model_name}</Tag><Tag>{data.dataset_name}</Tag><Tag>第 {pageNumber} 页</Tag></Space> : '—'}</div></Col>
        </Row>
        {data && <Alert type="info" showIcon message={`本次推理置信度阈值：${data.inference_confidence}`} description="可视化阈值只过滤已保存的预测框，不会重新推理、修改预测文件或重新计算评测指标；低于推理阈值的框未被保存，调低可视化阈值无法恢复。" />}
        {error && <Alert type="error" showIcon message="推理结果加载失败" description={error} />}
        {!loading && !error && data?.total === 0 && <Empty description="该运行没有可视化图片" />}
        {data && data.total > 0 && (
          <Image.PreviewGroup preview={{
            imageRender: (originalNode, info) => {
              const item = data.items[info.current]
              if (!item) return originalNode
              const { x, y, rotate, scale, flipX, flipY } = info.transform
              return (
                <div
                  className="dataset-browser-preview-image"
                  style={{
                    aspectRatio: item.width && item.height ? `${item.width} / ${item.height}` : '16 / 9',
                    transform: `translate3d(${x}px, ${y}px, 0) scale3d(${flipX ? -scale : scale}, ${flipY ? -scale : scale}, 1) rotate(${rotate}deg)`,
                  }}
                >
                  {cloneElement(originalNode, { style: { ...originalNode.props.style, transform: 'none', transitionDuration: '0s' } })}
                  {predictionBoxes(item)}
                </div>
              )
            },
          }}>
            <div className="dataset-browser-grid">
              {data.items.map((item) => {
                const count = visibleBoxes(item).length
                return <div className="dataset-browser-sample" key={`${item.image_id}-${item.url}`}><div className="dataset-browser-image" style={{ aspectRatio: item.width && item.height ? `${item.width} / ${item.height}` : '16 / 9' }}><Image src={item.url} alt={item.name} loading="lazy" preview={{ mask: item.name }} />{predictionBoxes(item)}</div><div className="dataset-browser-caption"><Typography.Text ellipsis={{ tooltip: item.name }}>{item.name}</Typography.Text><Tag color={count ? 'blue' : 'default'}>{count} / {item.boxes.length} 框</Tag></div></div>
              })}
            </div>
          </Image.PreviewGroup>
        )}
        {loading && <div className="dataset-browser-loading"><Spin /><Typography.Text type="secondary">正在加载推理结果…</Typography.Text></div>}
        {!error && data && data.total > 0 && <div className="dataset-browser-pagination"><Pagination current={pageNumber} pageSize={pageSize} total={data.total} showSizeChanger={false} showQuickJumper disabled={loading} showTotal={(count) => `共 ${count} 张`} onChange={setPageNumber} /></div>}
      </Space>
    </Drawer>
  )
}

type AnnotationAction =
  | { type: 'draw'; id: string; startX: number; startY: number; before: DetectionBox[] }
  | { type: 'move'; id: string; startX: number; startY: number; original: DetectionBox; before: DetectionBox[] }
  | { type: 'resize'; id: string; handle: 'nw' | 'ne' | 'sw' | 'se'; original: DetectionBox; before: DetectionBox[] }

const annotationColors = ['#1677FF', '#13A8A8', '#722ED1', '#EB2F96', '#52C41A', '#FA8C16', '#F5222D', '#2F54EB']

function AnnotationWorkspace({ dataset, onClose, onChanged }: { dataset?: Dataset; onClose: () => void; onChanged: () => void }) {
  const [session, setSession] = useState<AnnotationSession>()
  const [sampleIndex, setSampleIndex] = useState(0)
  const [boxes, setBoxes] = useState<DetectionBox[]>([])
  const [completed, setCompleted] = useState(false)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [selectedBoxId, setSelectedBoxId] = useState<string>()
  const [selectedCategoryId, setSelectedCategoryId] = useState(1)
  const [newCategoryName, setNewCategoryName] = useState('')
  const [zoom, setZoom] = useState(1)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState('')
  const svgRef = useRef<SVGSVGElement>(null)
  const boxesRef = useRef<DetectionBox[]>([])
  const completedRef = useRef(false)
  const dimensionsRef = useRef({ width: 0, height: 0 })
  const dirtyRef = useRef(false)
  const actionRef = useRef<AnnotationAction>()

  const currentSample = session?.samples[sampleIndex]
  const editable = Boolean(dataset && !dataset.frozen)
  const selectedBox = boxes.find((box) => box.id === selectedBoxId)

  const setBoxState = (next: DetectionBox[]) => {
    boxesRef.current = next
    setBoxes(next)
  }

  const markChanged = (next: DetectionBox[]) => {
    setBoxState(next)
    if (completedRef.current) {
      completedRef.current = false
      setCompleted(false)
    }
    dirtyRef.current = true
  }

  const loadSession = useCallback(async (datasetId: string) => {
    const next = await api<AnnotationSession>(`/api/datasets/${datasetId}/annotations`)
    setSession(next)
    setSelectedCategoryId((current) => next.categories.some((category) => category.id === current) ? current : next.categories[0]?.id ?? 1)
  }, [])

  useEffect(() => {
    setSession(undefined)
    setSampleIndex(0)
    setSavedAt('')
    if (dataset) loadSession(dataset.id).catch((error) => message.error(error.message))
  }, [dataset?.id, loadSession])

  useEffect(() => {
    if (!dataset || !currentSample) return
    let active = true
    setLoading(true)
    setSelectedBoxId(undefined)
    setZoom(1)
    setBoxState([])
    setCompleted(false)
    completedRef.current = false
    dimensionsRef.current = { width: 0, height: 0 }
    setDimensions({ width: 0, height: 0 })
    dirtyRef.current = false
    api<SampleAnnotation>(`/api/datasets/${dataset.id}/samples/${encodeURIComponent(currentSample.name)}/annotations`)
      .then((annotation) => {
        if (!active) return
        setBoxState(annotation.boxes)
        completedRef.current = annotation.completed
        setCompleted(annotation.completed)
        if (annotation.width && annotation.height) {
          dimensionsRef.current = { width: annotation.width, height: annotation.height }
          setDimensions(dimensionsRef.current)
        }
        dirtyRef.current = false
      })
      .catch((error) => active && message.error(error.message))
      .finally(() => active && setLoading(false))
    return () => { active = false }
  }, [dataset?.id, currentSample?.name])

  const updateSummary = (sampleName: string, savedBoxes: DetectionBox[], isCompleted: boolean) => {
    setSession((current) => {
      if (!current) return current
      const samples = current.samples.map((sample) => sample.name === sampleName
        ? { ...sample, completed: isCompleted, box_count: savedBoxes.length }
        : sample)
      return {
        ...current,
        samples,
        progress: {
          total: samples.length,
          completed: samples.filter((sample) => sample.completed).length,
        },
      }
    })
  }

  const persist = async (forceCompleted?: boolean) => {
    if (!dataset || !currentSample || !editable) return true
    if (forceCompleted !== undefined && forceCompleted !== completedRef.current) {
      completedRef.current = forceCompleted
      setCompleted(forceCompleted)
      dirtyRef.current = true
    }
    if (!dirtyRef.current) return true
    const size = dimensionsRef.current
    if (!size.width || !size.height) {
      message.warning('图片尺寸尚未加载完成')
      return false
    }
    const savedBoxes = boxesRef.current.map((box) => ({
      ...box,
      x: Number(box.x.toFixed(2)),
      y: Number(box.y.toFixed(2)),
      width: Number(box.width.toFixed(2)),
      height: Number(box.height.toFixed(2)),
    }))
    const savedCompleted = completedRef.current
    dirtyRef.current = false
    setSaving(true)
    try {
      await api<SampleAnnotation>(`/api/datasets/${dataset.id}/samples/${encodeURIComponent(currentSample.name)}/annotations`, {
        method: 'PUT',
        body: JSON.stringify({ ...size, boxes: savedBoxes, completed: savedCompleted }),
      })
      updateSummary(currentSample.name, savedBoxes, savedCompleted)
      setSavedAt(new Date().toLocaleTimeString('zh-CN', { hour12: false }))
      if (dataset.annotation_status !== 'ANNOTATING') onChanged()
      return true
    } catch (error) {
      dirtyRef.current = true
      message.error((error as Error).message)
      return false
    } finally {
      setSaving(false)
    }
  }

  const goToSample = async (next: number) => {
    if (!session || next < 0 || next >= session.samples.length) return
    if (!await persist()) return
    setSampleIndex(next)
  }

  const imagePoint = (event: React.PointerEvent<SVGSVGElement | SVGElement>) => {
    const svg = svgRef.current
    const matrix = svg?.getScreenCTM()
    if (!svg || !matrix) return undefined
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return {
      x: Math.max(0, Math.min(dimensionsRef.current.width, point.x)),
      y: Math.max(0, Math.min(dimensionsRef.current.height, point.y)),
    }
  }

  const beginDraw = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!editable || event.button !== 0 || !dimensions.width || !session?.categories.length) return
    const point = imagePoint(event)
    if (!point) return
    const id = `box_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
    const before = boxesRef.current
    setSelectedBoxId(id)
    setBoxState([...before, { id, category_id: selectedCategoryId, x: point.x, y: point.y, width: 0.01, height: 0.01 }])
    actionRef.current = { type: 'draw', id, startX: point.x, startY: point.y, before }
    svgRef.current?.setPointerCapture(event.pointerId)
  }

  const beginMove = (event: React.PointerEvent<SVGRectElement>, box: DetectionBox) => {
    event.stopPropagation()
    setSelectedBoxId(box.id)
    setSelectedCategoryId(box.category_id)
    if (!editable || event.button !== 0) return
    const point = imagePoint(event)
    if (!point) return
    actionRef.current = { type: 'move', id: box.id, startX: point.x, startY: point.y, original: box, before: boxesRef.current }
    svgRef.current?.setPointerCapture(event.pointerId)
  }

  const beginResize = (event: React.PointerEvent<SVGRectElement>, box: DetectionBox, handle: 'nw' | 'ne' | 'sw' | 'se') => {
    event.stopPropagation()
    if (!editable || event.button !== 0) return
    const point = imagePoint(event)
    if (!point) return
    actionRef.current = { type: 'resize', id: box.id, handle, original: box, before: boxesRef.current }
    svgRef.current?.setPointerCapture(event.pointerId)
  }

  const movePointer = (event: React.PointerEvent<SVGSVGElement>) => {
    const action = actionRef.current
    const point = imagePoint(event)
    if (!action || !point) return
    if (action.type === 'draw') {
      setBoxState(boxesRef.current.map((box) => box.id === action.id ? {
        ...box,
        x: Math.min(action.startX, point.x),
        y: Math.min(action.startY, point.y),
        width: Math.abs(point.x - action.startX),
        height: Math.abs(point.y - action.startY),
      } : box))
      return
    }
    if (action.type === 'move') {
      const x = Math.max(0, Math.min(dimensions.width - action.original.width, action.original.x + point.x - action.startX))
      const y = Math.max(0, Math.min(dimensions.height - action.original.height, action.original.y + point.y - action.startY))
      setBoxState(boxesRef.current.map((box) => box.id === action.id ? { ...box, x, y } : box))
      return
    }
    const original = action.original
    const left = action.handle.includes('w') ? point.x : original.x
    const right = action.handle.includes('e') ? point.x : original.x + original.width
    const top = action.handle.includes('n') ? point.y : original.y
    const bottom = action.handle.includes('s') ? point.y : original.y + original.height
    setBoxState(boxesRef.current.map((box) => box.id === action.id ? {
      ...box,
      x: Math.min(left, right),
      y: Math.min(top, bottom),
      width: Math.abs(right - left),
      height: Math.abs(bottom - top),
    } : box))
  }

  const endPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    const action = actionRef.current
    if (!action) return
    actionRef.current = undefined
    if (svgRef.current?.hasPointerCapture(event.pointerId)) svgRef.current.releasePointerCapture(event.pointerId)
    const next = boxesRef.current.filter((box) => box.width >= 2 && box.height >= 2)
    const changed = JSON.stringify(next) !== JSON.stringify(action.before)
    if (!changed) {
      setBoxState(action.before)
      return
    }
    markChanged(next)
    void persist()
  }

  const deleteSelected = () => {
    if (!editable || !selectedBoxId) return
    const next = boxesRef.current.filter((box) => box.id !== selectedBoxId)
    setSelectedBoxId(undefined)
    markChanged(next)
    void persist()
  }

  const changeSelectedCategory = (categoryId: number) => {
    setSelectedCategoryId(categoryId)
    if (!editable || !selectedBoxId) return
    const next = boxesRef.current.map((box) => box.id === selectedBoxId ? { ...box, category_id: categoryId } : box)
    markChanged(next)
    void persist()
  }

  const saveCategories = async (categories: AnnotationCategory[]) => {
    if (!dataset) return
    try {
      const result = await api<{ categories: AnnotationCategory[] }>(`/api/datasets/${dataset.id}/annotation-schema`, {
        method: 'PUT',
        body: JSON.stringify({ categories }),
      })
      setSession((current) => current ? { ...current, categories: result.categories } : current)
      if (!result.categories.some((category) => category.id === selectedCategoryId)) setSelectedCategoryId(result.categories[0].id)
      setNewCategoryName('')
    } catch (error) { message.error((error as Error).message) }
  }

  const addCategory = () => {
    if (!session || !newCategoryName.trim()) return
    const id = Math.max(0, ...session.categories.map((category) => category.id)) + 1
    void saveCategories([...session.categories, {
      id,
      name: newCategoryName.trim(),
      color: annotationColors[(id - 1) % annotationColors.length],
    }])
  }

  const markCompleteAndNext = async () => {
    if (!await persist(true)) return
    if (session && sampleIndex < session.samples.length - 1) setSampleIndex(sampleIndex + 1)
  }

  const submitAnnotations = async () => {
    if (!dataset || !await persist()) return
    try {
      const result = await post<{ images: number; annotations: number }>(`/api/datasets/${dataset.id}/annotations/complete`)
      message.success(`已导出 COCO：${result.images} 张图片，${result.annotations} 个目标框`)
      await loadSession(dataset.id)
      onChanged()
    } catch (error) { message.error((error as Error).message) }
  }

  const keyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const tag = (event.target as HTMLElement).tagName
    if (tag === 'INPUT' || tag === 'TEXTAREA') return
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault()
      void persist()
    } else if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault()
      deleteSelected()
    } else if (event.key.toLowerCase() === 'a') {
      void goToSample(sampleIndex - 1)
    } else if (event.key.toLowerCase() === 'd') {
      void goToSample(sampleIndex + 1)
    }
  }

  const handleSize = Math.max(6, Math.min(dimensions.width || 1000, dimensions.height || 1000) / 90)
  return (
    <Drawer
      open={Boolean(dataset)}
      width="100vw"
      title={dataset ? `${editable ? '目标检测标注' : '只读标注'} · ${dataset.name}` : '目标检测标注'}
      onClose={onClose}
      destroyOnClose
      extra={session && <Space><Tag color="blue">{session.progress.completed} / {session.progress.total} 已完成</Tag>{saving ? <Tag color="processing">保存中</Tag> : savedAt && <Tag color="success">已保存 {savedAt}</Tag>}</Space>}
    >
      {!session ? <div className="annotation-loading"><Spin /><Typography.Text type="secondary">正在加载标注工作区…</Typography.Text></div> : (
        <div className="annotation-workspace" tabIndex={0} onKeyDown={keyDown}>
          <aside className="annotation-samples">
            <div className="annotation-panel-heading"><strong>图片</strong><Progress percent={session.progress.total ? Math.round(session.progress.completed / session.progress.total * 100) : 0} size="small" /></div>
            <div className="annotation-sample-list">
              {session.samples.map((sample, index) => (
                <button key={sample.name} type="button" className={`annotation-sample ${index === sampleIndex ? 'active' : ''}`} onClick={() => void goToSample(index)}>
                  <img src={sample.url} alt="" loading="lazy" />
                  <span><b>{index + 1}. {sample.name}</b><small>{sample.completed ? '已完成' : '待确认'} · {sample.box_count} 框</small></span>
                  <i className={sample.completed ? 'done' : ''} />
                </button>
              ))}
            </div>
          </aside>
          <main className="annotation-main">
            <div className="annotation-toolbar">
              <Space>
                <Button icon={<LeftOutlined />} disabled={sampleIndex === 0} onClick={() => void goToSample(sampleIndex - 1)}>上一张</Button>
                <Button icon={<RightOutlined />} disabled={sampleIndex >= session.samples.length - 1} onClick={() => void goToSample(sampleIndex + 1)}>下一张</Button>
                <Typography.Text type="secondary">{currentSample?.name}</Typography.Text>
              </Space>
              <Space>
                <Typography.Text type="secondary">滚轮缩放</Typography.Text>
                <Slider min={0.5} max={4} step={0.1} value={zoom} onChange={setZoom} style={{ width: 130 }} />
                <Tag>{Math.round(zoom * 100)}%</Tag>
                {editable && <Button icon={<SaveOutlined />} loading={saving} onClick={() => void persist()}>保存</Button>}
              </Space>
            </div>
            <div className="annotation-canvas-scroll" onWheel={(event) => { event.preventDefault(); setZoom((value) => Math.max(0.5, Math.min(4, value + (event.deltaY < 0 ? 0.1 : -0.1)))) }}>
              {loading || !dimensions.width ? <Spin /> : (
                <svg
                  ref={svgRef}
                  className="annotation-canvas"
                  viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                  style={{ width: `${zoom * 100}%`, height: 'auto', aspectRatio: `${dimensions.width} / ${dimensions.height}` }}
                  onPointerDown={beginDraw}
                  onPointerMove={movePointer}
                  onPointerUp={endPointer}
                >
                  <image href={currentSample?.url} width={dimensions.width} height={dimensions.height} pointerEvents="none" />
                  {boxes.map((box) => {
                    const category = session.categories.find((item) => item.id === box.category_id)
                    const selected = box.id === selectedBoxId
                    return <g key={box.id}>
                      <rect x={box.x} y={box.y} width={box.width} height={box.height} fill={`${category?.color || '#1677FF'}20`} stroke={category?.color || '#1677FF'} strokeWidth={selected ? 4 : 2} vectorEffect="non-scaling-stroke" onPointerDown={(event) => beginMove(event, box)} />
                      <text x={box.x + 3} y={Math.max(14, box.y + 15)} fill="#fff" stroke="#000" strokeWidth={3} paintOrder="stroke" fontSize={14} pointerEvents="none">{category?.name || box.category_id}{box.confidence !== undefined ? ` ${(box.confidence * 100).toFixed(0)}%` : ''}</text>
                      {selected && editable && (['nw', 'ne', 'sw', 'se'] as const).map((handle) => {
                        const x = handle.includes('w') ? box.x : box.x + box.width
                        const y = handle.includes('n') ? box.y : box.y + box.height
                        return <rect key={handle} x={x - handleSize / 2} y={y - handleSize / 2} width={handleSize} height={handleSize} fill="#fff" stroke={category?.color || '#1677FF'} strokeWidth={2} vectorEffect="non-scaling-stroke" onPointerDown={(event) => beginResize(event, box, handle)} />
                      })}
                    </g>
                  })}
                </svg>
              )}
              {currentSample && <img className="annotation-image-probe" src={currentSample.url} alt="" onLoad={(event) => {
                if (!dimensionsRef.current.width) {
                  dimensionsRef.current = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight }
                  setDimensions(dimensionsRef.current)
                }
              }} />}
            </div>
            <div className="annotation-footer">
              <Typography.Text type="secondary">在图片上拖拽创建框；拖动框可移动，拖动四角可缩放。快捷键：A/D 切图，Delete 删除，Ctrl+S 保存。</Typography.Text>
              {editable && <Space><Checkbox checked={completed} onChange={(event) => { completedRef.current = event.target.checked; setCompleted(event.target.checked); dirtyRef.current = true; void persist() }}>本图已确认（允许零目标）</Checkbox><Button type="primary" onClick={() => void markCompleteAndNext()}>完成并下一张</Button></Space>}
            </div>
          </main>
          <aside className="annotation-categories">
            <div className="annotation-panel-heading"><strong>目标类别</strong>{selectedBox && editable && <Button danger type="text" size="small" icon={<DeleteOutlined />} onClick={deleteSelected}>删除框</Button>}</div>
            <div className="annotation-category-list">
              {session.categories.map((category) => (
                <div key={category.id} className={`annotation-category ${selectedCategoryId === category.id ? 'active' : ''}`} onClick={() => changeSelectedCategory(category.id)}>
                  <span className="annotation-color" style={{ background: category.color }} />
                  <b>{category.name}</b>
                  <small>{boxes.filter((box) => box.category_id === category.id).length}</small>
                  {editable && session.categories.length > 1 && <Popconfirm title="删除这个类别？" description="已被任意目标框使用的类别不能删除。" onConfirm={() => void saveCategories(session.categories.filter((item) => item.id !== category.id))}><Button danger type="text" size="small" icon={<DeleteOutlined />} onClick={(event) => event.stopPropagation()} /></Popconfirm>}
                </div>
              ))}
            </div>
            {editable && <Space.Compact block className="annotation-category-add"><Input value={newCategoryName} maxLength={64} placeholder="新增类别名称" onChange={(event) => setNewCategoryName(event.target.value)} onPressEnter={addCategory} /><Button icon={<PlusOutlined />} onClick={addCategory} /></Space.Compact>}
            <Divider />
            <Alert type={editable ? 'info' : 'warning'} showIcon message={editable ? '标注规则' : '数据集已冻结'} description={editable ? '每张图片都需要确认；无目标图片也需勾选“本图已确认”。修改已完成图片会自动恢复为待确认。' : '当前只允许查看目标框，不能修改类别或标注。'} />
            {editable && <Button block type="primary" className="top-gap" disabled={!session.progress.total || session.progress.completed !== session.progress.total} onClick={() => void submitAnnotations()}>完成标注并导出 COCO</Button>}
            {editable && session.progress.completed !== session.progress.total && <Typography.Paragraph type="secondary" style={{ marginTop: 10 }}>还有 {session.progress.total - session.progress.completed} 张图片未确认，完成后才能提交。</Typography.Paragraph>}
          </aside>
        </div>
      )}
    </Drawer>
  )
}

function DatasetStatisticsPanel({ datasetId, dark }: { datasetId: string; dark: boolean }) {
  const { data, loading } = useResource<DatasetStatistics | undefined>(`/api/datasets/${datasetId}/statistics`, undefined)
  if (loading) return <Card loading />
  if (!data) return <Alert type="warning" showIcon message="暂时无法读取数据集统计" />
  const textColor = dark ? '#c6d1df' : '#4b5565'
  const splitColor = dark ? '#263449' : '#eef1f5'
  const zoom = (count: number) => count > 12 ? [{ type: 'inside', yAxisIndex: 0, start: 0, end: 1200 / count }, { type: 'slider', yAxisIndex: 0, right: 4, width: 12, start: 0, end: 1200 / count }] : []
  const categoryOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 120, right: data.category_counts.length > 12 ? 38 : 18, top: 12, bottom: 34 },
    xAxis: { type: 'value', minInterval: 1, name: '目标数', splitLine: { lineStyle: { color: splitColor } }, axisLabel: { color: textColor } },
    yAxis: { type: 'category', inverse: true, data: data.category_counts.map((item) => `${item.id ?? '?'}:${item.name}`), axisLabel: { color: textColor, width: 108, overflow: 'truncate' } },
    dataZoom: zoom(data.category_counts.length),
    series: [{ type: 'bar', data: data.category_counts.map((item) => item.count), itemStyle: { color: '#1677FF', borderRadius: [0, 4, 4, 0] } }],
  }
  const resolutionOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 100, right: data.resolutions.length > 12 ? 38 : 18, top: 12, bottom: 34 },
    xAxis: { type: 'value', minInterval: 1, name: '图像数', splitLine: { lineStyle: { color: splitColor } }, axisLabel: { color: textColor } },
    yAxis: { type: 'category', inverse: true, data: data.resolutions.map((item) => item.label), axisLabel: { color: textColor } },
    dataZoom: zoom(data.resolutions.length),
    series: [{ type: 'bar', data: data.resolutions.map((item) => item.count), itemStyle: { color: '#13A8A8', borderRadius: [0, 4, 4, 0] } }],
  }
  const scaleOption = {
    tooltip: { trigger: 'item', formatter: '{b}<br/>{c} 个（{d}%）' },
    legend: { bottom: 0, textStyle: { color: textColor } },
    series: [{ type: 'pie', radius: ['42%', '68%'], center: ['50%', '44%'], label: { formatter: '{b}\n{c}', color: textColor }, data: data.scales.filter((item) => item.count).map((item) => ({ name: item.label, value: item.count })) }],
  }
  const relativeScaleOption = {
    tooltip: { trigger: 'item', formatter: '{b}<br/>{c} 个（{d}%）' },
    legend: { bottom: 0, textStyle: { color: textColor } },
    series: [{ type: 'pie', radius: ['42%', '68%'], center: ['50%', '44%'], label: { formatter: '{b}\n{c}', color: textColor }, data: data.relative_scales.filter((item) => item.count).map((item) => ({ name: item.label, value: item.count })) }],
  }
  return <Space direction="vertical" size={14} style={{ width: '100%' }}>
    <Row gutter={12}>
      <Col span={8}><Card size="small"><Statistic title="实际图像" value={data.image_count} suffix="张" /></Card></Col>
      <Col span={8}><Card size="small"><Statistic title="有标注图像" value={data.annotated_image_count} suffix="张" /></Card></Col>
      <Col span={8}><Card size="small"><Statistic title="目标框" value={data.object_count} suffix="个" /></Card></Col>
    </Row>
    <Row gutter={[12, 12]}>
      <Col xs={24} xl={12}><Card size="small" title="各类别目标数量">{data.category_counts.length ? <ReactECharts option={categoryOption} style={{ height: 330 }} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚未配置类别" />}</Card></Col>
      <Col xs={24} xl={12}><Card size="small" title="图像分辨率分布">{data.resolutions.length ? <ReactECharts option={resolutionOption} style={{ height: 330 }} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有可统计的图像" />}</Card></Col>
      <Col xs={24} xl={12}><Card size="small" title="COCO 目标尺度" extra={<Typography.Text type="secondary">绝对像素面积</Typography.Text>}>{data.object_count ? <ReactECharts option={scaleOption} style={{ height: 310 }} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无目标框" />}</Card></Col>
      <Col xs={24} xl={12}><Card size="small" title="目标相对面积占比" extra={<Typography.Text type="secondary">框面积 ÷ 图像面积，非 COCO 指标</Typography.Text>}>{data.object_count ? <ReactECharts option={relativeScaleOption} style={{ height: 310 }} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无目标框" />}</Card></Col>
    </Row>
  </Space>
}

function AutoAnnotationModal({
  dataset,
  models,
  adapters,
  onClose,
  onCompleted,
  onOpenAnnotations,
  onOpenTasks,
}: {
  dataset?: Dataset
  models: ModelVersion[]
  adapters: Adapter[]
  onClose: () => void
  onCompleted: () => void
  onOpenAnnotations: (dataset: Dataset) => void
  onOpenTasks: () => void
}) {
  const availableModels = models.filter((model) => !model.is_demo && model.status !== 'UNAVAILABLE')
  const [modelId, setModelId] = useState('')
  const [confidence, setConfidence] = useState(0.25)
  const [nmsIou, setNmsIou] = useState(0.7)
  const [imageSize, setImageSize] = useState(1280)
  const [inputHeight, setInputHeight] = useState(960)
  const [inputWidth, setInputWidth] = useState(1280)
  const [maxDetections, setMaxDetections] = useState(300)
  const [batchSize, setBatchSize] = useState(1)
  const [warmup, setWarmup] = useState(0)
  const [precision, setPrecision] = useState<'FP16' | 'FP32'>('FP16')
  const [submitting, setSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string>()
  const [finishedJob, setFinishedJob] = useState<Job>()
  const selectedModel = availableModels.find((model) => model.id === modelId)
  const inferenceProperties = modelInferenceProperties(selectedModel, adapters)
  const supports = (name: string) => Boolean(inferenceProperties[name])
  const compatibilityIssue = dataset && selectedModel ? categoryCompatibilityIssue(dataset, selectedModel) : ''

  useEffect(() => {
    setModelId('')
    setJobId(undefined)
    setFinishedJob(undefined)
  }, [dataset?.id])

  const selectModel = (value: string) => {
    setModelId(value)
    const model = availableModels.find((item) => item.id === value)
    const properties = modelInferenceProperties(model, adapters)
    setPrecision(model?.precision === 'FP32' ? 'FP32' : 'FP16')
    setConfidence(inferenceDefault(properties, 'confidence', 0.25))
    setNmsIou(inferenceDefault(properties, 'nms_iou', 0.7))
    setImageSize(inferenceDefault(properties, 'image_size', 1280))
    setInputHeight(inferenceDefault(properties, 'input_height', 960))
    setInputWidth(inferenceDefault(properties, 'input_width', 1280))
    setMaxDetections(inferenceDefault(properties, 'max_detections', 300))
    setBatchSize(inferenceDefault(properties, 'batch_size', 1))
    setWarmup(inferenceDefault(properties, 'warmup', 0))
  }
  const submit = async () => {
    if (!dataset || !modelId || compatibilityIssue) return
    setSubmitting(true)
    try {
      const job = await post<Job>(`/api/datasets/${dataset.id}/auto-annotations`, {
        model_id: modelId,
        confidence,
        nms_iou: nmsIou,
        image_size: imageSize,
        input_height: inputHeight,
        input_width: inputWidth,
        max_detections: maxDetections,
        batch_size: batchSize,
        warmup,
        precision,
      })
      setJobId(job.id)
      message.success('自动标注任务已提交')
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }
  const finish = useCallback((job: Job) => {
    setFinishedJob(job)
    if (job.status === 'SUCCEEDED') {
      message.success('候选框已生成，请逐图人工确认')
      onCompleted()
    }
  }, [onCompleted])

  return (
    <Modal open={Boolean(dataset)} width={720} title={dataset ? `自动标注 · ${dataset.name}` : '自动标注'} onCancel={onClose} footer={null} destroyOnClose>
      {!jobId ? <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Alert type="info" showIcon message="模型预测只作为候选标注" description="只处理尚无标注记录的图片，不覆盖人工标注；生成后每张图片仍需人工确认，才能导出COCO真值。" />
        <Form layout="vertical">
          <Form.Item label="闭集检测模型" required>
            <Select value={modelId || undefined} onChange={selectModel} placeholder="选择已注册的真实检测模型" optionFilterProp="label" options={availableModels.map((model) => {
              const issue = dataset ? categoryCompatibilityIssue(dataset, model) : ''
              return { value: model.id, label: `${model.name} · ${model.version}${issue ? `（${issue}）` : ''}`, disabled: Boolean(issue) }
            })} />
          </Form.Item>
          {!availableModels.length && <Alert type="warning" showIcon message="没有可用的真实检测模型" description="请先在模型版本页面注册本地检测模型。" />}
          {compatibilityIssue && <Alert type="error" showIcon message="模型与数据集类别不兼容" description={compatibilityIssue} />}
          <Row gutter={16} className="top-gap">
            {supports('confidence') && <Col span={12}><Form.Item label={`置信度阈值 ${confidence.toFixed(2)}`}><Slider min={0.01} max={1} step={0.01} value={confidence} onChange={setConfidence} marks={{ 0.01: '宽松', 0.5: '中等', 1: '严格' }} /></Form.Item></Col>}
            {supports('nms_iou') && <Col span={12}><Form.Item label={`NMS IoU阈值 ${nmsIou.toFixed(2)}`}><Slider min={0.1} max={1} step={0.05} value={nmsIou} onChange={setNmsIou} marks={{ 0.1: '抑制强', 0.7: '默认', 1: '抑制弱' }} /></Form.Item></Col>}
            {supports('image_size') && <Col span={8}><Form.Item label="方形推理尺寸"><InputNumber min={32} max={8192} step={32} value={imageSize} onChange={(value) => setImageSize(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
            {supports('input_height') && <Col span={8}><Form.Item label="输入高度"><InputNumber min={32} max={8192} step={32} value={inputHeight} onChange={(value) => setInputHeight(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
            {supports('input_width') && <Col span={8}><Form.Item label="输入宽度"><InputNumber min={32} max={8192} step={32} value={inputWidth} onChange={(value) => setInputWidth(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
            {supports('max_detections') && <Col span={8}><Form.Item label="每图最大检测数"><InputNumber min={1} max={5000} precision={0} value={maxDetections} onChange={(value) => setMaxDetections(value || 1)} style={{ width: '100%' }} /></Form.Item></Col>}
            {supports('batch_size') && <Col span={8}><Form.Item label="批大小"><InputNumber min={1} max={64} value={batchSize} onChange={(value) => setBatchSize(value || 1)} style={{ width: '100%' }} /></Form.Item></Col>}
            {supports('warmup') && <Col span={8}><Form.Item label="预热次数"><InputNumber min={0} max={200} value={warmup} onChange={(value) => setWarmup(value || 0)} style={{ width: '100%' }} /></Form.Item></Col>}
            <Col span={8}><Form.Item label="推理精度"><Select value={precision} onChange={setPrecision} options={[{ value: 'FP16' }, { value: 'FP32' }]} /></Form.Item></Col>
          </Row>
        </Form>
        <Space><Button type="primary" icon={<RobotOutlined />} loading={submitting} disabled={!modelId || Boolean(compatibilityIssue)} onClick={submit}>启动自动标注</Button><Button onClick={onClose}>取消</Button></Space>
      </Space> : <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <JobProgress jobId={jobId} onFinish={finish} />
        {finishedJob?.status === 'SUCCEEDED' && dataset && <Result status="success" title="候选标注已写入" subTitle={`${String(finishedJob.result?.annotated_images ?? 0)} 张图片，${String(finishedJob.result?.accepted_boxes ?? 0)} 个候选框；请进入标注工作区逐图确认。`} extra={<Button type="primary" onClick={() => onOpenAnnotations(dataset)}>打开标注工作区</Button>} />}
        <Button onClick={onOpenTasks}>前往任务中心</Button>
      </Space>}
    </Modal>
  )
}

export function DatasetsPage({ dark, navigate }: PageProps) {
  const { data, loading, reload } = useResource<Dataset[]>('/api/datasets', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const adapters = useResource<Adapter[]>('/api/adapters', [])
  const [selected, setSelected] = useState<Dataset>()
  const [browserDataset, setBrowserDataset] = useState<Dataset>()
  const [annotationDataset, setAnnotationDataset] = useState<Dataset>()
  const [autoAnnotationDataset, setAutoAnnotationDataset] = useState<Dataset>()
  const [datasetPageSize, setDatasetPageSize] = useState(20)
  const freeze = async (dataset: Dataset) => {
    try { await post(`/api/datasets/${dataset.id}/freeze`); message.success('数据版本已冻结'); reload() } catch (error) { message.error((error as Error).message) }
  }
  const remove = async (dataset: Dataset) => {
    try {
      await api(`/api/datasets/${dataset.id}`, { method: 'DELETE' })
      if (selected?.id === dataset.id) setSelected(undefined)
      if (browserDataset?.id === dataset.id) setBrowserDataset(undefined)
      if (annotationDataset?.id === dataset.id) setAnnotationDataset(undefined)
      if (autoAnnotationDataset?.id === dataset.id) setAutoAnnotationDataset(undefined)
      message.success('数据集已移入回收站')
      reload()
    } catch (error) { message.error((error as Error).message) }
  }
  return <><Card title="不可变数据版本" extra={<Space><Tag>{data.length} 个版本</Tag><Typography.Text type="secondary">每页</Typography.Text><InputNumber size="small" min={1} max={200} precision={0} value={datasetPageSize} onChange={(value) => setDatasetPageSize(value ?? 20)} addonAfter="条" style={{ width: 100 }} /><Button icon={<ReloadOutlined />} onClick={reload}>刷新</Button></Space>}><Table loading={loading} rowKey="id" dataSource={data} pagination={{ pageSize: datasetPageSize, showSizeChanger: false, showTotal: (total) => `共 ${total} 条` }} columns={[
    { title: '数据集', dataIndex: 'name', render: (value, row) => <Space direction="vertical" size={1}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.id}</Typography.Text></Space> },
    { title: '来源', dataIndex: 'source_type', render: (value) => <DatasetSourceTag value={String(value)} /> },
    { title: '场景 / 条件', render: (_, row) => `${row.scene_domain} / ${datasetConditionSummary(row)}` },
    { title: '分辨率', dataIndex: 'resolution' }, { title: '样本', dataIndex: 'sample_count' },
    { title: '真值', dataIndex: 'annotation_status', render: (value) => <StatusTag status={value} /> },
    { title: '版本', render: (_, row) => row.frozen ? <Tag icon={<LockOutlined />} color="success">{row.version} 已冻结</Tag> : <Tag>草稿</Tag> },
    { title: '操作', render: (_, row) => <Space><Button type="link" onClick={() => setSelected(row)}>查看</Button><Button type="link" icon={<FileImageOutlined />} onClick={() => setBrowserDataset(row)}>浏览全部</Button><Button type="link" onClick={() => setAnnotationDataset(row)}>{row.frozen ? '查看标注' : '目标标注'}</Button>{!row.frozen && ['UNLABELED', 'ANNOTATING'].includes(row.annotation_status) && <Button type="link" icon={<RobotOutlined />} onClick={() => setAutoAnnotationDataset(row)}>自动标注</Button>}{!row.frozen && <Button type="link" onClick={() => freeze(row)}>冻结</Button>}<Popconfirm title="确认删除这个数据集？" description={`${row.name} · ${row.sample_count} 个样本${row.frozen ? '（已冻结）' : ''}将移入回收站；仍有评测结果或运行中的评测任务时不会删除。`} okText="移入回收站" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => remove(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm></Space> },
      ]} /></Card><Drawer open={Boolean(selected)} width={1040} title={selected?.name} onClose={() => setSelected(undefined)}>{selected && <Space direction="vertical" size={18} style={{ width: '100%' }}><Gallery images={selected.preview_images} height={140} /><Button block icon={<FileImageOutlined />} onClick={() => { setSelected(undefined); setBrowserDataset(selected) }}>浏览全部图片</Button><Button block onClick={() => { setSelected(undefined); setAnnotationDataset(selected) }}>{selected.frozen ? '查看目标检测标注' : '开始目标检测标注'}</Button>{!selected.frozen && ['UNLABELED', 'ANNOTATING'].includes(selected.annotation_status) && <Button block type="primary" icon={<RobotOutlined />} onClick={() => setAutoAnnotationDataset(selected)}>使用检测模型自动标注</Button>}<Descriptions bordered column={2} items={[{ key: 'source', label: '来源', children: <DatasetSourceTag value={selected.source_type} /> }, ...datasetLocationItems(selected), { key: 'scene', label: '场景', children: selected.scene_domain }, { key: 'weather', label: '天气', children: selected.weather }, { key: 'resolution', label: '分辨率', children: selected.resolution }, { key: 'truth', label: '真值', children: <StatusTag status={selected.annotation_status} /> }, { key: 'frozen', label: '不可变', children: selected.frozen ? '是' : '否' }, ...conditionDatasetItems(selected), { key: 'categories', label: '检测类别', children: selected.categories.length ? <Space wrap>{selected.categories.map((item) => <Tag key={item.id}>{item.id}:{item.name}</Tag>)}</Space> : <Tag color="warning">待配置</Tag>, span: 2 }]} /><Divider>数据统计</Divider><DatasetStatisticsPanel datasetId={selected.id} dark={dark} /></Space>}</Drawer><DatasetBrowser dataset={browserDataset} onClose={() => setBrowserDataset(undefined)} /><AnnotationWorkspace dataset={annotationDataset} onClose={() => setAnnotationDataset(undefined)} onChanged={reload} /><AutoAnnotationModal dataset={autoAnnotationDataset} models={models.data} adapters={adapters.data} onClose={() => setAutoAnnotationDataset(undefined)} onCompleted={reload} onOpenAnnotations={(dataset) => { setAutoAnnotationDataset(undefined); setSelected(undefined); setAnnotationDataset(dataset) }} onOpenTasks={() => navigate('tasks')} /></>
}

type LocalResourceKind = 'directory' | 'entrypoint' | 'weight' | 'annotation'
type LocalResourceScope = 'model' | 'environment' | 'dataset'

interface LocalResourceListing {
  root: string
  current: string
  parent?: string
  entries: Array<{
    name: string
    path: string
    is_directory: boolean
  }>
}

function LocalResourcePicker({
  open,
  title,
  scope,
  kind,
  initialPath,
  onClose,
  onSelect,
}: {
  open: boolean
  title: string
  scope: LocalResourceScope
  kind: LocalResourceKind
  initialPath?: string
  onClose: () => void
  onSelect: (path: string) => void
}) {
  const [listing, setListing] = useState<LocalResourceListing>()
  const [loading, setLoading] = useState(false)
  const load = useCallback(async (path?: string) => {
    setLoading(true)
    try {
      const query = new URLSearchParams({ kind })
      if (scope !== 'dataset') query.set('scope', scope)
      if (path) query.set('path', path)
      const endpoint = scope === 'dataset' ? '/api/local-dataset-resources' : '/api/local-model-resources'
      setListing(await api<LocalResourceListing>(`${endpoint}?${query}`))
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setLoading(false)
    }
  }, [scope, kind])
  useEffect(() => {
    if (open) load(initialPath).catch(() => undefined)
  }, [open, initialPath, load])
  return (
    <Modal
      open={open}
      width={760}
      title={title}
      onCancel={onClose}
      footer={[
        <Button key="parent" disabled={!listing?.parent} onClick={() => listing?.parent && load(listing.parent)}>上一级</Button>,
        kind === 'directory' && <Button key="select" type="primary" disabled={!listing} onClick={() => listing && onSelect(listing.current)}>选择当前目录</Button>,
        <Button key="cancel" onClick={onClose}>取消</Button>,
      ]}
    >
      <Typography.Paragraph copyable={{ text: listing?.current || '' }}>
        <Typography.Text type="secondary">当前位置：</Typography.Text>{listing?.current || '加载中'}
      </Typography.Paragraph>
      <List
        bordered
        loading={loading}
        dataSource={listing?.entries || []}
        locale={{ emptyText: '当前目录没有可选择的资源' }}
        renderItem={(entry) => (
          <List.Item
            actions={[
              entry.is_directory
                ? <Button type="link" onClick={() => load(entry.path)}>打开</Button>
                : <Button type="link" onClick={() => onSelect(entry.path)}>选择</Button>,
            ]}
          >
            <Space>
              <Tag color={entry.is_directory ? 'blue' : 'default'}>{entry.is_directory ? '目录' : '文件'}</Tag>
              <Typography.Text>{entry.name}</Typography.Text>
            </Space>
          </List.Item>
        )}
      />
    </Modal>
  )
}

interface LocalModelDraft {
  name: string
  family: string
  architecture: string
  backbone: string
  detector_head: string
  training_dataset: string
  pretrained_dataset: string
  version: string
  precision: 'FP32' | 'FP16'
  project_directory: string
  working_directory: string
  runtime_prefix: string
  command_arguments: string
  inference_defaults: {
    confidence: number
    nms_iou: number
    image_size: number
    input_height: number
    input_width: number
    max_detections: number
    batch_size: number
    warmup: number
  }
  weight_path: string
}

const emptyLocalModelDraft = (): LocalModelDraft => ({
  name: '',
  family: '',
  architecture: '',
  backbone: '',
  detector_head: '',
  training_dataset: '',
  pretrained_dataset: '',
  version: 'v1',
  precision: 'FP16',
  project_directory: '',
  working_directory: '',
  runtime_prefix: '',
  command_arguments: '',
  inference_defaults: {
    confidence: 0.25,
    nms_iou: 0.7,
    image_size: 1280,
    input_height: 960,
    input_width: 1280,
    max_detections: 300,
    batch_size: 1,
    warmup: 20,
  },
  weight_path: '',
})

type LocalCommandExecution = {
  mode?: string
  working_directory?: string
  arguments?: string[]
}

function copiedLocalModelDraft(model: ModelVersion, adapter: Adapter): LocalModelDraft | undefined {
  const properties = (adapter.parameter_schema?.properties || {}) as Record<string, InferenceProperty>
  const execution = (adapter.parameter_schema?.execution || {}) as LocalCommandExecution
  const projectDirectory = properties.project_directory?.const
  if (execution.mode !== 'command' || !Array.isArray(execution.arguments) || typeof projectDirectory !== 'string') return undefined
  const precision = model.precision === 'FP32' ? 'FP32' : 'FP16'
  return {
    name: `${model.name} 副本`,
    family: model.family,
    architecture: model.architecture,
    backbone: model.backbone,
    detector_head: model.detector_head,
    training_dataset: model.training_dataset,
    pretrained_dataset: model.pretrained_dataset,
    version: model.version,
    precision,
    project_directory: projectDirectory,
    working_directory: execution.working_directory || projectDirectory,
    runtime_prefix: adapter.runtime_prefix || '',
    command_arguments: execution.arguments.join('\n'),
    inference_defaults: {
      confidence: inferenceDefault(properties, 'confidence', 0.25),
      nms_iou: inferenceDefault(properties, 'nms_iou', 0.7),
      image_size: inferenceDefault(properties, 'image_size', 1280),
      input_height: inferenceDefault(properties, 'input_height', 960),
      input_width: inferenceDefault(properties, 'input_width', 1280),
      max_detections: inferenceDefault(properties, 'max_detections', 300),
      batch_size: inferenceDefault(properties, 'batch_size', 1),
      warmup: inferenceDefault(properties, 'warmup', 0),
    },
    weight_path: model.weight_path || '',
  }
}

export function RegistryPage({ refresh }: PageProps) {
  const adapters = useResource<Adapter[]>('/api/adapters', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const categoryTemplates = useResource<CategoryTemplate[]>('/api/category-templates', [])
  const [registerOpen, setRegisterOpen] = useState(false)
  const [registering, setRegistering] = useState(false)
  const [copySourceName, setCopySourceName] = useState<string>()
  const [editSource, setEditSource] = useState<ModelVersion>()
  const [selectedModel, setSelectedModel] = useState<ModelVersion>()
  const [categoryTemplateId, setCategoryTemplateId] = useState('visdrone')
  const [customModelCategories, setCustomModelCategories] = useState<CategoryDefinition[]>([{ id: 0, name: '' }])
  const [draft, setDraft] = useState<LocalModelDraft>(emptyLocalModelDraft)
  const [picker, setPicker] = useState<{
    field: 'project_directory' | 'working_directory' | 'runtime_prefix' | 'weight_path'
    title: string
    scope: LocalResourceScope
    kind: LocalResourceKind
    initialPath?: string
  }>()
  const setField = <K extends keyof LocalModelDraft>(field: K, value: LocalModelDraft[K]) =>
    setDraft((current) => ({ ...current, [field]: value }))
  const setInferenceDefault = (field: keyof LocalModelDraft['inference_defaults'], value: number | null) =>
    setDraft((current) => ({ ...current, inference_defaults: { ...current.inference_defaults, [field]: value ?? 0 } }))
  const commandSupports = (name: string) => draft.command_arguments.includes(`{${name}}`)
  const modelCategories = categoriesFromSelection(categoryTemplates.data, categoryTemplateId, 'model', customModelCategories)
  const selectedInferenceProperties = modelInferenceProperties(selectedModel, adapters.data)
  const openBlankRegistration = () => {
    setDraft(emptyLocalModelDraft())
    setCategoryTemplateId('visdrone')
    setCustomModelCategories([{ id: 0, name: '' }])
    setCopySourceName(undefined)
    setEditSource(undefined)
    setRegisterOpen(true)
  }
  const copyRegistration = (model: ModelVersion) => {
    const adapter = adapters.data.find((item) => item.id === model.adapter_id)
    const copiedDraft = adapter && copiedLocalModelDraft(model, adapter)
    if (!copiedDraft) {
      message.error('该模型没有可复制的本地命令配置')
      return
    }
    const templateId = categoryTemplates.data.some((item) => item.id === model.category_template)
      ? model.category_template
      : 'custom'
    setDraft(copiedDraft)
    setCategoryTemplateId(templateId)
    setCustomModelCategories(model.categories.map((item) => ({ ...item })))
    setCopySourceName(model.name)
    setEditSource(undefined)
    setSelectedModel(undefined)
    setRegisterOpen(true)
  }
  const editModel = (model: ModelVersion) => {
    const adapter = adapters.data.find((item) => item.id === model.adapter_id)
    const editableDraft = adapter && copiedLocalModelDraft(model, adapter)
    if (!editableDraft) {
      message.error('该模型没有可编辑的本地命令配置')
      return
    }
    const templateId = categoryTemplates.data.some((item) => item.id === model.category_template)
      ? model.category_template
      : 'custom'
    setDraft({ ...editableDraft, name: model.name })
    setCategoryTemplateId(templateId)
    setCustomModelCategories(model.categories.map((item) => ({ ...item })))
    setCopySourceName(undefined)
    setEditSource(model)
    setSelectedModel(undefined)
    setRegisterOpen(true)
  }
  const selectResource = (path: string) => {
    if (!picker) return
    if (picker.field === 'project_directory') {
      setDraft((current) => ({
        ...current,
        project_directory: path,
        working_directory: path,
        runtime_prefix: current.runtime_prefix || `${path}/.venv`,
        weight_path: '',
      }))
    } else if (picker.field === 'runtime_prefix') {
      setField('runtime_prefix', path)
    } else {
      setField(picker.field, path)
    }
    setPicker(undefined)
  }
  const health = async (id: string) => {
    try {
      const result = await post<{ healthy: boolean }>(`/api/adapters/${id}/health-check`)
      result.healthy ? message.success('模型运行环境接口测试通过') : message.warning('模型运行环境当前不可用')
      await adapters.reload()
    } catch (error) {
      message.error((error as Error).message)
    }
  }
  const removeModel = async (model: ModelVersion) => {
    try {
      await api(`/api/models/${model.id}`, { method: 'DELETE' })
      message.success(`已删除模型：${model.name}`)
      if (selectedModel?.id === model.id) setSelectedModel(undefined)
      await Promise.all([models.reload(), adapters.reload()])
      refresh()
    } catch (error) {
      message.error((error as Error).message)
    }
  }
  const register = async () => {
    setRegistering(true)
    try {
      if (editSource) {
        await api<ModelVersion>(`/api/local-detector-models/${editSource.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            name: draft.name,
            architecture: draft.architecture,
            backbone: draft.backbone,
            detector_head: draft.detector_head,
            training_dataset: draft.training_dataset,
            pretrained_dataset: draft.pretrained_dataset,
            precision: draft.precision,
            working_directory: draft.working_directory,
            runtime_prefix: draft.runtime_prefix,
            inference_defaults: draft.inference_defaults,
          }),
        })
        try {
          const result = await post<{ healthy: boolean }>(`/api/adapters/${editSource.adapter_id}/health-check`)
          result.healthy ? message.success('模型信息已保存，接口测试通过') : message.warning('模型信息已保存，但接口测试未通过')
        } catch (healthError) {
          message.warning(`模型信息已保存，但接口测试请求失败：${(healthError as Error).message}`)
        }
      } else {
        await post<ModelVersion>('/api/local-detector-models', {
          ...draft,
          category_template: categoryTemplateId,
          categories: modelCategories.map(({ id, name }) => ({ id, name })),
          command_arguments: draft.command_arguments
            .split('\n')
            .map((value) => value.trim())
            .filter(Boolean),
        })
        message.success('本地检测模型已注册')
      }
      setRegisterOpen(false)
      setDraft(emptyLocalModelDraft())
      setCategoryTemplateId('visdrone')
      setCustomModelCategories([{ id: 0, name: '' }])
      setCopySourceName(undefined)
      setEditSource(undefined)
      await Promise.all([models.reload(), adapters.reload()])
      refresh()
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setRegistering(false)
    }
  }
  const ready = Boolean(
    draft.name.trim().length >= 2
    && draft.family.trim()
    && draft.architecture.trim()
    && draft.backbone.trim()
    && draft.detector_head.trim()
    && validCategories(modelCategories)
    && draft.training_dataset.trim()
    && draft.pretrained_dataset.trim()
    && draft.project_directory
    && draft.working_directory
    && draft.runtime_prefix
    && draft.command_arguments.trim()
    && draft.weight_path,
  )
  const modelsCard = (
    <Card extra={<Button type="primary" icon={<PlusOutlined />} onClick={openBlankRegistration}>注册本地检测模型</Button>}>
      <Table rowKey="id" loading={models.loading} dataSource={models.data} pagination={{ defaultPageSize: 20, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100], showTotal: (total) => `共 ${total} 条` }} scroll={{ x: 950 }} onRow={(row) => ({ onClick: () => setSelectedModel(row), style: { cursor: 'pointer' } })} columns={[
        { title: '模型', dataIndex: 'name', width: 260, render: (value, row) => <Space direction="vertical" size={4}><Typography.Text strong style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>{value}</Typography.Text><Space size={4}><Tag>{row.version}</Tag>{row.is_demo && <DemoTag />}</Space></Space> },
        { title: '模型架构', dataIndex: 'architecture', width: 260, render: (value) => <Typography.Text style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>{value || '—'}</Typography.Text> },
        { title: 'Backbone', dataIndex: 'backbone', width: 160, render: (value) => <Typography.Text ellipsis={{ tooltip: value }} style={{ maxWidth: 145 }}>{value || '—'}</Typography.Text> },
        { title: '类别数', dataIndex: 'class_count', width: 90, render: (value) => value || '—' },
        { title: '操作', width: 180, render: (_, row) => <span onClick={(event) => event.stopPropagation()}><Space><Button type="link" onClick={() => setSelectedModel(row)}>查看详情</Button><Popconfirm title="确认删除这个模型？" description="仅删除平台注册记录，不删除模型项目、环境或权重；存在评测引用时会拒绝删除。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => removeModel(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm></Space></span> },
      ]} />
    </Card>
  )
  return (
    <>
      {modelsCard}
      <Drawer
        open={registerOpen}
        width={760}
        title={editSource ? `编辑“${editSource.name}”` : copySourceName ? `基于“${copySourceName}”复制注册` : '注册本地检测模型'}
        onClose={() => setRegisterOpen(false)}
      >
        <Alert
          type="info"
          showIcon
          message={editSource ? '编辑当前模型版本的信息' : copySourceName ? '已复制已有模型配置' : '配置结构化执行命令'}
          description={editSource ? '可以修正描述信息、推理参数默认值和运行路径。权重、类别、模型族、版本及命令参数属于模型版本标识，请使用“复制注册”创建新版本。' : copySourceName ? '请重点检查模型名称、版本、结构差异、权重和命令中的固定配置参数，然后注册为新的模型记录。' : '平台按参数数组直接启动进程，不经过 Shell。命令只需在指定位置生成 COCO predictions.json；同一项目的不同模型可以登记不同权重和参数。'}
        />
        <Form layout="vertical" className="top-gap">
          <Row gutter={12}>
            <Col span={12}><Form.Item label="模型名称" required><Input value={draft.name} onChange={(event) => setField('name', event.target.value)} maxLength={120} placeholder="例如 YOLOv8m VisDrone" /></Form.Item></Col>
            <Col span={12}><Form.Item label="模型族" required><Input disabled={Boolean(editSource)} value={draft.family} onChange={(event) => setField('family', event.target.value)} maxLength={80} placeholder="例如 YOLOv8" /></Form.Item></Col>
            <Col span={8}><Form.Item label="模型架构" required><Input value={draft.architecture} onChange={(event) => setField('architecture', event.target.value)} maxLength={80} placeholder="例如 DETR" /></Form.Item></Col>
            <Col span={8}><Form.Item label="Backbone" required><Input value={draft.backbone} onChange={(event) => setField('backbone', event.target.value)} maxLength={80} placeholder="例如 HGNetv2-B2" /></Form.Item></Col>
            <Col span={8}><Form.Item label="检测头" required><Input value={draft.detector_head} onChange={(event) => setField('detector_head', event.target.value)} maxLength={80} placeholder="例如 D-FINE Transformer" /></Form.Item></Col>
            <Col span={24}><Form.Item label="版本"><Input disabled={Boolean(editSource)} value={draft.version} onChange={(event) => setField('version', event.target.value)} maxLength={40} /></Form.Item></Col>
            <Col span={12}><Form.Item label="训练数据" required><Input value={draft.training_dataset} onChange={(event) => setField('training_dataset', event.target.value)} maxLength={120} placeholder="例如 VisDrone2019-DET" /></Form.Item></Col>
            <Col span={12}><Form.Item label="预训练数据" required><Input value={draft.pretrained_dataset} onChange={(event) => setField('pretrained_dataset', event.target.value)} maxLength={120} placeholder="例如 Objects365；没有则填写无" /></Form.Item></Col>
          </Row>
          <Form.Item label={`检测类别（${modelCategories.length} 类）`} required>
            {editSource
              ? <Space wrap>{modelCategories.map((item) => <Tag key={item.id}>{item.id}:{item.name}</Tag>)}</Space>
              : <CategoryConfiguration templates={categoryTemplates.data} templateId={categoryTemplateId} scope="model" customCategories={customModelCategories} onTemplateChange={setCategoryTemplateId} onCustomChange={setCustomModelCategories} />}
          </Form.Item>
          <Form.Item label={<>模型项目目录 <Typography.Text code>{'{project_directory}'}</Typography.Text></>} required>
            <Space.Compact block>
              <Input readOnly value={draft.project_directory} placeholder="从服务器模型库选择目录" />
              {!editSource && <Button onClick={() => setPicker({ field: 'project_directory', title: '选择模型项目目录', scope: 'model', kind: 'directory', initialPath: draft.project_directory || undefined })}>选择目录</Button>}
            </Space.Compact>
          </Form.Item>
          <Form.Item label="命令工作目录" required>
            <Space.Compact block>
              <Input readOnly value={draft.working_directory} placeholder="默认使用模型项目目录" />
              <Button disabled={!draft.project_directory} onClick={() => setPicker({ field: 'working_directory', title: '选择命令工作目录', scope: 'model', kind: 'directory', initialPath: draft.working_directory || draft.project_directory })}>选择目录</Button>
            </Space.Compact>
          </Form.Item>
          <Form.Item label="Python 环境目录" required extra={draft.runtime_prefix ? <>平台将使用 <Typography.Text code>{`${draft.runtime_prefix.replace(/\/+$/, '')}/bin/python`}</Typography.Text> 执行模型命令。</> : '目录内必须存在 bin/python；也可以选择模型目录中的 .venv。'}>
            <Space.Compact block>
              <Input value={draft.runtime_prefix} onChange={(event) => setField('runtime_prefix', event.target.value)} placeholder="/path/to/conda/env" />
              <Button onClick={() => setPicker({ field: 'runtime_prefix', title: '选择 Conda 环境目录', scope: 'environment', kind: 'directory' })}>选择环境</Button>
            </Space.Compact>
          </Form.Item>
          <Form.Item label={<>模型权重 <Typography.Text code>{'{weight_path}'}</Typography.Text></>} required>
            <Space.Compact block>
              <Input readOnly value={draft.weight_path} placeholder="选择模型权重文件" />
              {!editSource && <Button disabled={!draft.project_directory} onClick={() => setPicker({ field: 'weight_path', title: '选择模型权重', scope: 'model', kind: 'weight', initialPath: draft.project_directory || undefined })}>选择权重</Button>}
            </Space.Compact>
          </Form.Item>
          <Form.Item
            label={<Space size={6}>命令参数<Tooltip placement="right" title={<div><div>每行填写一个参数，平台不会经过Shell。</div><div>运行任务时，平台会把 {'{placeholder}'} 替换为当前模型、数据集和推理任务的实际值。</div><div>D-FINE入口脚本、config和固定数值直接填写，不加大括号。</div></div>}><QuestionCircleOutlined aria-label="命令参数说明" style={{ color: '#1677ff', cursor: 'help' }} /></Tooltip></Space>}
            required
            extra="每行一个参数。参数会原样传给可执行程序，不要输入整段 Shell 命令。"
          >
            <Input.TextArea
              value={draft.command_arguments}
              disabled={Boolean(editSource)}
              onChange={(event) => setField('command_arguments', event.target.value)}
              autoSize={{ minRows: 9, maxRows: 18 }}
              placeholder={'tools/evaluate.py\n--weights\n{weight_path}\n--images\n{image_directory}\n--annotations\n{annotation_path}\n--output\n{predictions_path}'}
            />
          </Form.Item>
          <Card size="small" title="推理参数默认值" extra={<Typography.Text type="secondary">命令包含对应占位符后启用；任务阶段可以覆盖</Typography.Text>}>
            <Row gutter={12}>
              <Col span={8}><Form.Item label="推理精度"><Select disabled={!commandSupports('precision')} value={draft.precision} onChange={(value) => setField('precision', value)} options={[{ value: 'FP16' }, { value: 'FP32' }]} /></Form.Item></Col>
              <Col span={8}><Form.Item label="置信度"><InputNumber disabled={!commandSupports('confidence')} min={0} max={1} step={0.01} value={draft.inference_defaults.confidence} onChange={(value) => setInferenceDefault('confidence', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="NMS IoU"><InputNumber disabled={!commandSupports('nms_iou')} min={0} max={1} step={0.05} value={draft.inference_defaults.nms_iou} onChange={(value) => setInferenceDefault('nms_iou', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="方形推理尺寸"><InputNumber disabled={!commandSupports('image_size')} min={32} max={8192} step={32} value={draft.inference_defaults.image_size} onChange={(value) => setInferenceDefault('image_size', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="输入高度"><InputNumber disabled={!commandSupports('input_height')} min={32} max={8192} step={32} value={draft.inference_defaults.input_height} onChange={(value) => setInferenceDefault('input_height', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="输入宽度"><InputNumber disabled={!commandSupports('input_width')} min={32} max={8192} step={32} value={draft.inference_defaults.input_width} onChange={(value) => setInferenceDefault('input_width', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="每图最大检测数"><InputNumber disabled={!commandSupports('max_detections')} min={1} max={5000} value={draft.inference_defaults.max_detections} onChange={(value) => setInferenceDefault('max_detections', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="批大小"><InputNumber disabled={!commandSupports('batch_size')} min={1} max={64} value={draft.inference_defaults.batch_size} onChange={(value) => setInferenceDefault('batch_size', value)} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item label="预热次数"><InputNumber disabled={!commandSupports('warmup')} min={0} max={200} value={draft.inference_defaults.warmup} onChange={(value) => setInferenceDefault('warmup', value)} style={{ width: '100%' }} /></Form.Item></Col>
            </Row>
          </Card>
          <Typography.Paragraph type="secondary">
            可用占位符：模型权重 <Typography.Text code>{'{weight_path}'}</Typography.Text> · 模型项目目录 <Typography.Text code>{'{project_directory}'}</Typography.Text> · 图片目录 <Typography.Text code>{'{image_directory}'}</Typography.Text> · 标注文件 <Typography.Text code>{'{annotation_path}'}</Typography.Text> · 预测结果 <Typography.Text code>{'{predictions_path}'}</Typography.Text> · 输出目录 <Typography.Text code>{'{output_directory}'}</Typography.Text> · 设备 <Typography.Text code>{'{device}'}</Typography.Text> · 推理精度 <Typography.Text code>{'{precision}'}</Typography.Text> · 批大小 <Typography.Text code>{'{batch_size}'}</Typography.Text> · 置信度 <Typography.Text code>{'{confidence}'}</Typography.Text> · NMS阈值 <Typography.Text code>{'{nms_iou}'}</Typography.Text> · 方形推理尺寸 <Typography.Text code>{'{image_size}'}</Typography.Text> · 输入高度 <Typography.Text code>{'{input_height}'}</Typography.Text> · 输入宽度 <Typography.Text code>{'{input_width}'}</Typography.Text> · 最大检测数 <Typography.Text code>{'{max_detections}'}</Typography.Text> · 预热次数 <Typography.Text code>{'{warmup}'}</Typography.Text> · 数据集ID <Typography.Text code>{'{dataset_id}'}</Typography.Text> · 模型ID <Typography.Text code>{'{model_id}'}</Typography.Text> · 请求文件 <Typography.Text code>{'{request_path}'}</Typography.Text> · 结果文件 <Typography.Text code>{'{result_path}'}</Typography.Text>
          </Typography.Paragraph>
          <Space>
            <Button type="primary" loading={registering} disabled={!ready} onClick={register}>{editSource ? '保存并接口测试' : '注册模型'}</Button>
            <Button onClick={() => setRegisterOpen(false)}>取消</Button>
          </Space>
        </Form>
      </Drawer>
      <Drawer
        open={Boolean(selectedModel)}
        width={720}
        title={selectedModel ? `${selectedModel.name} · 模型参数` : '模型参数'}
        onClose={() => setSelectedModel(undefined)}
      >
        {selectedModel && (
          <Space direction="vertical" size={18} style={{ width: '100%' }}>
            <Space wrap>
              {!selectedModel.is_demo && <><Button onClick={() => editModel(selectedModel)}>编辑信息</Button><Button onClick={() => copyRegistration(selectedModel)}>复制注册</Button></>}
              <Button onClick={() => health(selectedModel.adapter_id)}>接口测试</Button>
              <Popconfirm title="确认删除这个模型？" description="仅删除平台注册记录，不删除模型项目、环境或权重；存在评测引用时会拒绝删除。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => removeModel(selectedModel)}><Button danger icon={<DeleteOutlined />}>删除</Button></Popconfirm>
            </Space>
            <Descriptions bordered column={2} items={[
              { key: 'name', label: '模型名称', children: selectedModel.name, span: 2 },
              { key: 'family', label: '模型族', children: selectedModel.family },
              { key: 'architecture', label: '模型架构', children: selectedModel.architecture || '未记录' },
              { key: 'backbone', label: 'Backbone', children: selectedModel.backbone || '未记录' },
              { key: 'head', label: '检测头', children: selectedModel.detector_head || '未记录' },
              { key: 'classes', label: '类别数', children: selectedModel.class_count || '未记录' },
              { key: 'version', label: '版本', children: selectedModel.version },
              { key: 'training', label: '训练数据', children: selectedModel.training_dataset || '未记录' },
              { key: 'pretrained', label: '预训练数据', children: selectedModel.pretrained_dataset || '未记录' },
              { key: 'precision', label: '默认精度', children: <Tag>{selectedModel.precision}</Tag> },
              { key: 'status', label: '状态', children: <StatusTag status={selectedModel.status} /> },
              { key: 'runtime', label: '运行环境', children: <StatusTag status={adapters.data.find((item) => item.id === selectedModel.adapter_id)?.status || (adapters.loading ? 'CHECKING' : 'UNAVAILABLE')} />, span: 2 },
              { key: 'inference', label: '可变推理参数', children: Object.entries(selectedInferenceProperties).filter(([name]) => inferenceParameterLabels[name]).length ? <Space wrap>{Object.entries(selectedInferenceProperties).filter(([name]) => inferenceParameterLabels[name]).map(([name, specification]) => <Tag key={name}>{inferenceParameterLabels[name]}：{String(specification.default ?? specification.const ?? '任务指定')}</Tag>)}</Space> : <Tag>未声明</Tag>, span: 2 },
              { key: 'categories', label: '检测类别', children: selectedModel.categories.length ? <Space wrap>{selectedModel.categories.map((item) => <Tag key={item.id}>{item.id}:{item.name}</Tag>)}</Space> : <Tag color="warning">待配置</Tag>, span: 2 },
              { key: 'weight', label: '权重路径', children: selectedModel.weight_path ? <Typography.Text code copyable>{selectedModel.weight_path}</Typography.Text> : '未记录', span: 2 },
              { key: 'sha', label: '权重 SHA-256', children: selectedModel.weight_sha256 ? <Typography.Text code copyable>{selectedModel.weight_sha256}</Typography.Text> : '未记录', span: 2 },
              { key: 'id', label: '模型 ID', children: <Typography.Text code copyable>{selectedModel.id}</Typography.Text>, span: 2 },
            ]} />
          </Space>
        )}
      </Drawer>
      {picker && (
        <LocalResourcePicker
          open
          title={picker.title}
          scope={picker.scope}
          kind={picker.kind}
          initialPath={picker.initialPath}
          onClose={() => setPicker(undefined)}
          onSelect={selectResource}
        />
      )}
    </>
  )
}

export function EvaluationPage({ navigate, refresh }: PageProps) {
  const datasets = useResource<Dataset[]>('/api/datasets', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const adapters = useResource<Adapter[]>('/api/adapters', [])
  const [datasetIds, setDatasetIds] = useState<string[]>([])
  const [modelIds, setModelIds] = useState<string[]>([])
  const [evaluationCategories, setEvaluationCategories] = useState<string[]>([])
  const [precision, setPrecision] = useState('FP16')
  const [confidence, setConfidence] = useState(0.001)
  const [nmsIou, setNmsIou] = useState(0.7)
  const [imageSize, setImageSize] = useState(1280)
  const [inputHeight, setInputHeight] = useState(960)
  const [inputWidth, setInputWidth] = useState(1280)
  const [maxDetections, setMaxDetections] = useState(300)
  const [batchSize, setBatchSize] = useState(1)
  const warmup = 20
  const [jobId, setJobId] = useState<string>()
  const [submitting, setSubmitting] = useState(false)
  const selectedModels = models.data.filter((item) => modelIds.includes(item.id))
  const selectedDatasets = datasets.data.filter((item) => datasetIds.includes(item.id))
  const datasetOptions = datasets.data.map((item) => ({
    value: item.id,
    label: `${item.name} · ${item.scene_domain}/${item.weather} · ${item.sample_count.toLocaleString('zh-CN')}张 · #${item.id.slice(-6)}${item.categories.length ? '' : '（类别待配置）'}`,
    title: `完整ID：${item.id}\n创建时间：${new Date(item.created_at).toLocaleString('zh-CN')}\n数据集地址：${item.dataset_path || item.platform_path || item.source_path || '未记录'}`,
    disabled: !item.frozen || !item.categories.length,
  }))
  const commonEvaluationCategories = useMemo(() => {
    if (!selectedDatasets.length || !selectedModels.length) return []
    const categoryGroups = [...selectedDatasets, ...selectedModels].map((item) => item.categories)
    const [first, ...rest] = categoryGroups
    return first.filter((category) => {
      const name = category.name.trim().toLocaleLowerCase()
      return rest.every((categories) => categories.some((item) => item.name.trim().toLocaleLowerCase() === name))
    }).map((category) => category.name)
  }, [datasetIds, modelIds, datasets.data, models.data])
  const commonCategorySignature = commonEvaluationCategories.join('\u0000')
  const hasRealDetector = selectedModels.some((item) => !item.is_demo)
  const selectedInferenceProperties = selectedModels.filter((item) => !item.is_demo).map((model) => modelInferenceProperties(model, adapters.data))
  const supportsInference = (name: string) => selectedInferenceProperties.some((properties) => Boolean(properties[name]))
  const selectedDefault = (name: string, fallback: number) => {
    const properties = selectedInferenceProperties.find((item) => Boolean(item[name]))
    return properties ? inferenceDefault(properties, name, fallback) : fallback
  }
  useEffect(() => { if (hasRealDetector && precision === 'INT8') setPrecision('FP16') }, [hasRealDetector, precision])
  useEffect(() => { setEvaluationCategories(commonEvaluationCategories) }, [commonCategorySignature])
  useEffect(() => {
    const properties = selectedInferenceProperties[0]
    if (!properties) return
    setConfidence(0.001)
    setNmsIou(selectedDefault('nms_iou', 0.7))
    setImageSize(selectedDefault('image_size', 1280))
    setInputHeight(selectedDefault('input_height', 960))
    setInputWidth(selectedDefault('input_width', 1280))
    setMaxDetections(selectedDefault('max_detections', 300))
    setBatchSize(selectedDefault('batch_size', 1))
  }, [modelIds.join(','), adapters.data])
  const count = datasetIds.length * modelIds.length
  const commonNames = new Set(commonEvaluationCategories.map((name) => name.trim().toLocaleLowerCase()))
  const unavailableCategories = evaluationCategories.filter((name) => !commonNames.has(name.trim().toLocaleLowerCase()))
  const categoryIssues = count && !commonEvaluationCategories.length
    ? ['所选数据集和模型没有共同类别']
    : count && !evaluationCategories.length
      ? ['请至少选择一个评测类别']
      : unavailableCategories.length
        ? [`以下类别不再是共同类别：${unavailableCategories.join('、')}`]
        : []
  const submit = async () => {
    setSubmitting(true)
    try {
      const existing = await api<ResultResponse>('/api/results')
      const existingCounts = new Map<string, number>()
      existing.runs.forEach((run) => {
        const key = `${run.dataset_id}\u0000${run.model_id}`
        existingCounts.set(key, (existingCounts.get(key) || 0) + 1)
      })
      const duplicatePairs = selectedDatasets.flatMap((dataset) => selectedModels.flatMap((model) => {
        const runCount = existingCounts.get(`${dataset.id}\u0000${model.id}`) || 0
        return runCount ? [{ dataset: dataset.name, model: model.name, runCount }] : []
      }))
      if (duplicatePairs.length) {
        const confirmed = await new Promise<boolean>((resolve) => {
          Modal.confirm({
            title: '检测到已评测组合',
            width: 620,
            content: <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Typography.Text>以下数据集与模型已在效能模型库中存在结果。继续评测会新增记录，不会覆盖历史结果。</Typography.Text>
              <List
                size="small"
                bordered
                dataSource={duplicatePairs.slice(0, 8)}
                renderItem={(item) => <List.Item><Typography.Text>{item.dataset} / {item.model}</Typography.Text><Tag>{item.runCount} 次历史评测</Tag></List.Item>}
              />
              {duplicatePairs.length > 8 && <Typography.Text type="secondary">另有 {duplicatePairs.length - 8} 个重复组合未展开。</Typography.Text>}
            </Space>,
            okText: '继续评测',
            cancelText: '取消',
            onOk: () => resolve(true),
            onCancel: () => resolve(false),
          })
        })
        if (!confirmed) return
      }
      const plan = await post<{ id: string }>('/api/evaluation-plans', { name: `感知效能评测 ${new Date().toLocaleString('zh-CN')}`, dataset_ids: datasetIds, model_ids: modelIds, evaluation_categories: evaluationCategories, seeds: [1001], blur_levels: [0], batch_size: batchSize, precision, warmup, confidence, nms_iou: nmsIou, image_size: imageSize, input_height: inputHeight, input_width: inputWidth, max_detections: maxDetections })
      const job = await post<Job>(`/api/evaluation-plans/${plan.id}/runs`)
      setJobId(job.id)
      refresh()
      message.success('评测矩阵已提交')
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }
  return <Space direction="vertical" size={18} style={{ width: '100%' }}>
    {hasRealDetector && <Alert
      type="info"
      showIcon
      message="已选择本地真实检测模型"
      description="使用模型注册时登记的推理入口、Python 环境和权重执行真实推理，并用 pycocotools 计算 COCO 指标。"
    />}
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={12}><Card title="数据版本"><Select mode="multiple" value={datasetIds} onChange={setDatasetIds} optionFilterProp="label" style={{ width: '100%' }} options={datasetOptions} /></Card></Col>
      <Col xs={24} xl={12}><Card title="模型版本"><Select mode="multiple" value={modelIds} onChange={setModelIds} style={{ width: '100%' }} options={models.data.map((item) => ({ value: item.id, label: `${item.name}${item.is_demo ? '（流程样例）' : ''}${item.categories.length ? '' : '（类别待配置）'}`, disabled: item.status === 'UNAVAILABLE' || !item.categories.length }))} /></Card></Col>
      <Col span={24}><Card title="评测类别" extra={<Typography.Text type="secondary">已选 {evaluationCategories.length} / 共同 {commonEvaluationCategories.length} 类</Typography.Text>}><Space direction="vertical" size={12} style={{ width: '100%' }}><Select mode="multiple" allowClear maxTagCount="responsive" value={evaluationCategories} onChange={setEvaluationCategories} disabled={!commonEvaluationCategories.length} placeholder={count ? '请选择参与指标计算的类别' : '请先选择数据集和模型'} style={{ width: '100%' }} options={commonEvaluationCategories.map((value) => ({ value, label: value }))} /><Space><Button size="small" disabled={!commonEvaluationCategories.length} onClick={() => setEvaluationCategories(commonEvaluationCategories)}>全选</Button><Button size="small" disabled={!evaluationCategories.length} onClick={() => setEvaluationCategories([])}>清空</Button><Typography.Text type="secondary">模型仍执行完整推理，所选类别仅决定 mAP、精确率和召回率的计算范围。</Typography.Text></Space></Space></Card></Col>
      <Col span={24}><Card title="本次推理参数" extra={<Typography.Text type="secondary">仅覆盖模型声明支持的参数；最终值写入运行记录</Typography.Text>}><Form layout="vertical"><Row gutter={16}>
        <Col span={24}><Form.Item label="精度模式"><Segmented block value={precision} onChange={(value) => setPrecision(String(value))} options={[{ value: 'FP32', label: 'FP32' }, { value: 'FP16', label: 'FP16' }, { value: 'INT8', label: 'INT8', disabled: hasRealDetector }]} /></Form.Item></Col>
        {supportsInference('confidence') && <Col xs={24} md={12}><Form.Item label="置信度阈值"><InputNumber min={0} max={1} step={0.001} precision={3} value={confidence} onChange={(value) => setConfidence(value ?? 0.001)} style={{ width: '100%' }} /></Form.Item></Col>}
        {supportsInference('nms_iou') && <Col xs={24} md={12}><Form.Item label={`NMS IoU阈值 ${nmsIou.toFixed(2)}`}><Slider min={0.1} max={1} step={0.05} value={nmsIou} onChange={setNmsIou} /></Form.Item></Col>}
        {supportsInference('image_size') && <Col xs={12} md={6}><Form.Item label="方形推理尺寸"><InputNumber min={32} max={8192} step={32} value={imageSize} onChange={(value) => setImageSize(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
        {supportsInference('input_height') && <Col xs={12} md={6}><Form.Item label="输入高度"><InputNumber min={32} max={8192} step={32} value={inputHeight} onChange={(value) => setInputHeight(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
        {supportsInference('input_width') && <Col xs={12} md={6}><Form.Item label="输入宽度"><InputNumber min={32} max={8192} step={32} value={inputWidth} onChange={(value) => setInputWidth(value || 32)} style={{ width: '100%' }} /></Form.Item></Col>}
        {supportsInference('max_detections') && <Col xs={12} md={6}><Form.Item label="每图最大检测数"><InputNumber min={1} max={5000} value={maxDetections} onChange={(value) => setMaxDetections(value || 1)} style={{ width: '100%' }} /></Form.Item></Col>}
        {supportsInference('batch_size') && <Col xs={12} md={6}><Form.Item label="批大小"><InputNumber min={1} max={64} value={batchSize} onChange={(value) => setBatchSize(value || 1)} style={{ width: '100%' }} /></Form.Item></Col>}
      </Row></Form></Card></Col>
    </Row>
    {categoryIssues.length > 0 && <Alert type="error" showIcon message="评测类别不可用，无法启动评测" description={<Space direction="vertical" size={2}>{categoryIssues.map((item) => <Typography.Text key={item}>{item}</Typography.Text>)}</Space>} />}
    <Card className="matrix-preview"><Row align="middle" gutter={[18, 18]}><Col flex="auto"><Typography.Title level={4}>评测组合</Typography.Title><Typography.Text type="secondary">{datasetIds.length} 数据版本 × {modelIds.length} 模型 · 评测 {evaluationCategories.length} 类</Typography.Text></Col><Col><Button type="primary" size="large" icon={<PlayCircleOutlined />} disabled={!count || categoryIssues.length > 0} loading={submitting} onClick={submit}>启动批量评测</Button></Col></Row></Card>
    {jobId && <JobProgress jobId={jobId} onFinish={() => refresh()} />}
    {jobId && <Card><Button type="primary" onClick={() => navigate('explorer')}>打开效能模型库</Button></Card>}
  </Space>
}

const resultDimensions = () => ({ scenes: [], conditions: [], resolutions: [], models: [], model_options: [], dataset_options: [], condition_types: [], hardware: [] })

function evaluationCategoryLabel(categories: string[]) {
  if (!categories.length) return '历史结果（类别范围未记录）'
  if (categories.length <= 4) return categories.join('、')
  return `${categories.length} 类（${categories.slice(0, 3).join('、')}…）`
}

const precisionRecallExplanation = '取自 COCOeval 在评测 IoU=0.5 下生成的 PR 曲线：先对全部评测类别的 Precision 求平均，再选择 F1 最大的曲线点。该点没有记录对应的置信度阈值，不代表 confidence=0.001 的固定工作点；NMS IoU=0.7 仅用于预测框去重。'

function precisionRecallMetricLabel(label: string) {
  return <Space size={4}>{label}<Tooltip title={precisionRecallExplanation}><QuestionCircleOutlined aria-label={`${label}取值依据`} style={{ color: '#1677ff', cursor: 'help' }} /></Tooltip></Space>
}

function runInferenceResolution(config: Record<string, unknown>) {
  if (config.input_width && config.input_height) return `${config.input_width}×${config.input_height}`
  if (config.input_resolution) return String(config.input_resolution)
  if (config.image_size) return `${config.image_size}×${config.image_size}`
  return '默认'
}

function runParameterValue(config: Record<string, unknown>, key: string) {
  const value = config[key]
  return value === undefined || value === null || value === '' ? '默认' : String(value)
}

function hasMeasuredPerformance(run: ResultRun) {
  return run.performance_status === 'MEASURED'
}

function performanceValue(run: ResultRun, value: number | null | undefined, suffix = '') {
  return hasMeasuredPerformance(run) && value !== null && value !== undefined
    ? `${Number(value).toFixed(2)}${suffix}`
    : '—'
}

function compactMetric(value: unknown, unit: string) {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) return '—'
  return `${(number / 1e9).toFixed(2)} ${unit}`
}

function parameterMetric(value: unknown) {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) return '—'
  return `${(number / 1e6).toFixed(2)} M`
}

const resultParameterLabels: Record<string, string> = {
  input_resolution: '推理分辨率',
  confidence: '置信度阈值',
  nms_iou: 'NMS IoU',
  precision: '推理精度',
  batch_size: '批大小',
  max_detections: '最大检测数',
  warmup: '预热次数',
}

function resultInferenceParameters(run: ResultRun) {
  return {
    input_resolution: runInferenceResolution(run.config),
    confidence: runParameterValue(run.config, 'confidence'),
    nms_iou: runParameterValue(run.config, 'nms_iou'),
    precision: runParameterValue(run.config, 'precision'),
    batch_size: runParameterValue(run.config, 'batch_size'),
    max_detections: runParameterValue(run.config, 'max_detections'),
    warmup: runParameterValue(run.config, 'warmup'),
  }
}

function parameterComparisonIssue(runs: ResultRun[]) {
  if (runs.length < 2) return '请至少选择两次评测结果'
  if (runs.some((run) => !hasMeasuredPerformance(run))) return '所选结果包含未采集真实性能数据的运行'
  const first = runs[0]
  const categories = [...first.evaluation_categories].sort().join('\u0000')
  const hardware = first.environment_fingerprint || JSON.stringify(first.hardware_profile)
  const protocol = String(first.config.metric_protocol || '')
  const adapter = String(first.config.adapter_id || '')
  for (const run of runs.slice(1)) {
    if (run.model_id !== first.model_id) return '只能比较同一个模型版本的评测结果'
    if (run.dataset_id !== first.dataset_id) return '只能比较同一个数据集版本的评测结果'
    if ([...run.evaluation_categories].sort().join('\u0000') !== categories) return '所选结果的评测类别范围不一致'
    if ((run.environment_fingerprint || JSON.stringify(run.hardware_profile)) !== hardware) return '所选结果的硬件或运行环境不一致'
    if (String(run.config.metric_protocol || '') !== protocol) return '所选结果的指标计算协议不一致'
    if (String(run.config.adapter_id || '') !== adapter) return '所选结果使用的推理适配器不一致'
  }
  return ''
}

type ParameterComparisonRow = {
  key: string
  label: string
  parameters: Record<string, string>
  runs: ResultRun[]
  map: number
  map50: number
  map75: number
  latencyP50: number
  latencyP95: number
  fps: number
  peakMemory: number
  curves: { recall: number[]; precision: number[] }
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / Math.max(1, values.length)
}

function aggregateParameterRuns(runs: ResultRun[]): ParameterComparisonRow[] {
  const grouped = new Map<string, ResultRun[]>()
  runs.forEach((run) => {
    const key = JSON.stringify(resultInferenceParameters(run))
    grouped.set(key, [...(grouped.get(key) || []), run])
  })
  return [...grouped.entries()].map(([key, groupedRuns], index) => {
    const first = groupedRuns[0]
    const precisionLength = Math.max(...groupedRuns.map((run) => run.curves.precision.length), 0)
    const recallLength = Math.max(...groupedRuns.map((run) => run.curves.recall.length), 0)
    return {
      key,
      label: `配置 ${index + 1}`,
      parameters: resultInferenceParameters(first),
      runs: groupedRuns,
      map: average(groupedRuns.map((run) => run.map)),
      map50: average(groupedRuns.map((run) => run.map50)),
      map75: average(groupedRuns.map((run) => run.map75)),
      latencyP50: average(groupedRuns.map((run) => run.latency_p50)),
      latencyP95: average(groupedRuns.map((run) => run.latency_p95)),
      fps: average(groupedRuns.map((run) => run.fps)),
      peakMemory: average(groupedRuns.map((run) => run.peak_memory)),
      curves: {
        recall: Array.from({ length: recallLength }, (_, point) => average(groupedRuns.map((run) => run.curves.recall[point]).filter((value) => Number.isFinite(value)))),
        precision: Array.from({ length: precisionLength }, (_, point) => average(groupedRuns.map((run) => run.curves.precision[point]).filter((value) => Number.isFinite(value)))),
      },
    }
  })
}

export function ExplorerPage({ dark }: PageProps) {
  const [data, setData] = useState<ResultResponse>({ count: 0, groups: [], runs: [], dimensions: resultDimensions() })
  const [loading, setLoading] = useState(true)
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([])
  const [comparisonRuns, setComparisonRuns] = useState<ResultRun[]>([])
  const [baselineKey, setBaselineKey] = useState('')
  const [selectedRun, setSelectedRun] = useState<ResultRun>()
  const [visualizationGroup, setVisualizationGroup] = useState<ResultGroup>()
  const [runPage, setRunPage] = useState(1)
  const [runPageSize, setRunPageSize] = useState(20)
  const load = async () => { setLoading(true); try { setData(await api<ResultResponse>('/api/results')) } catch (error) { message.error((error as Error).message) } finally { setLoading(false) } }
  useEffect(() => { load() }, [])
  useEffect(() => {
    const available = new Set(data.runs.map((run) => run.run_id))
    setSelectedRunIds((current) => current.filter((runId) => available.has(runId)))
  }, [data.runs])
  const natureTag = (result: { is_demo: boolean; is_official: boolean }) => result.is_demo ? <DemoTag /> : result.is_official ? <Tag color="green">真实模型 · 正式结果</Tag> : null
  const runs = [...data.runs].sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
  const comparisonRows = useMemo(() => aggregateParameterRuns(comparisonRuns), [comparisonRuns])
  const changedParameterKeys = Object.keys(resultParameterLabels).filter((key) => new Set(comparisonRows.map((row) => row.parameters[key])).size > 1)
  const commonParameterKeys = Object.keys(resultParameterLabels).filter((key) => comparisonRows.length && !changedParameterKeys.includes(key))
  const baseline = comparisonRows.find((row) => row.key === baselineKey) || comparisonRows[0]
  const openParameterComparison = () => {
    const selectedRuns = runs.filter((run) => selectedRunIds.includes(run.run_id))
    const issue = parameterComparisonIssue(selectedRuns)
    if (issue) {
      message.warning(issue)
      return
    }
    const grouped = aggregateParameterRuns(selectedRuns)
    if (grouped.length < 2) {
      message.warning('所选结果的推理参数完全相同，请选择至少两种不同配置')
      return
    }
    if (grouped.length > 4) {
      message.warning('单次最多对比四种推理配置，请减少选择')
      return
    }
    setComparisonRuns(selectedRuns)
    setBaselineKey(grouped[0].key)
  }
  const visualize = (run: ResultRun) => {
    const group = data.groups.find((item) => item.run_ids.includes(run.run_id))
    if (group) setVisualizationGroup({ ...group, comparison_id: `${group.comparison_id}:${run.run_id}`, run_ids: [run.run_id] })
  }
  const removeRun = async (run: ResultRun) => {
    try {
      await api(`/api/evaluation-runs/${run.run_id}`, { method: 'DELETE' })
      if (selectedRun?.run_id === run.run_id) setSelectedRun(undefined)
      if (visualizationGroup?.run_ids.includes(run.run_id)) setVisualizationGroup(undefined)
      setSelectedRunIds((current) => current.filter((runId) => runId !== run.run_id))
      if (comparisonRuns.some((item) => item.run_id === run.run_id)) setComparisonRuns([])
      message.success('评测结果已移入回收站')
      await load()
    } catch (error) { message.error((error as Error).message) }
  }
  const singleParameterKey = changedParameterKeys.length === 1 ? changedParameterKeys[0] : undefined
  const orderedComparisonRows = [...comparisonRows].sort((left, right) => {
    if (!singleParameterKey) return 0
    return left.parameters[singleParameterKey].localeCompare(right.parameters[singleParameterKey], 'zh-CN', { numeric: true })
  })
  const comparisonTextColor = dark ? '#c6d1df' : '#4b5565'
  const comparisonSplitColor = dark ? '#263449' : '#eef1f5'
  const comparisonChartBase = {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: comparisonTextColor } },
    grid: { left: 64, right: 24, top: 48, bottom: 48 },
    xAxis: { type: 'category', name: singleParameterKey ? resultParameterLabels[singleParameterKey] : '', data: orderedComparisonRows.map((row) => singleParameterKey ? row.parameters[singleParameterKey] : row.label), axisLabel: { color: comparisonTextColor } },
  }
  const accuracyComparisonOption = {
    ...comparisonChartBase,
    yAxis: { type: 'value', name: '精度', min: 0, max: 1, axisLabel: { formatter: (value: number) => `${Math.round(value * 100)}%`, color: comparisonTextColor }, splitLine: { lineStyle: { color: comparisonSplitColor } } },
    series: [
      { name: 'mAP', type: 'line', data: orderedComparisonRows.map((row) => row.map), smooth: true },
      { name: 'AP50', type: 'line', data: orderedComparisonRows.map((row) => row.map50), smooth: true },
      { name: 'AP75', type: 'line', data: orderedComparisonRows.map((row) => row.map75), smooth: true },
    ],
  }
  const latencyComparisonOption = {
    ...comparisonChartBase,
    yAxis: { type: 'value', name: '时延 / ms', min: 0, axisLabel: { color: comparisonTextColor }, splitLine: { lineStyle: { color: comparisonSplitColor } } },
    series: [
      { name: 'P50', type: 'line', data: orderedComparisonRows.map((row) => row.latencyP50), smooth: true },
      { name: 'P95', type: 'line', data: orderedComparisonRows.map((row) => row.latencyP95), smooth: true },
    ],
  }
  const relativeDelta = (value: number, base: number) => base ? `${value >= base ? '+' : ''}${((value / base - 1) * 100).toFixed(1)}%` : '—'
  const mapDelta = (value: number, base: number) => `${value >= base ? '+' : ''}${((value - base) * 100).toFixed(2)}pp`
  return <Space direction="vertical" size={18} style={{ width: '100%' }}>
      <Card title="模型对比" extra={<Space><Button type="primary" disabled={selectedRunIds.length < 2} onClick={openParameterComparison}>参数对比{selectedRunIds.length ? `（${selectedRunIds.length}）` : ''}</Button><Button icon={<ReloadOutlined />} onClick={load} loading={loading}>刷新结果</Button></Space>}>
        <Table<ResultRun>
          size="small"
          loading={loading}
          rowKey="run_id"
          dataSource={runs}
          rowSelection={{ fixed: true, selectedRowKeys: selectedRunIds, preserveSelectedRowKeys: true, onChange: (keys) => setSelectedRunIds(keys.map(String)) }}
          pagination={{ current: runPage, pageSize: runPageSize, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100], showTotal: (total) => `共 ${total} 次评测`, onChange: (page, pageSize) => { setRunPage(page); setRunPageSize(pageSize) } }}
          scroll={{ x: 2060 }}
          columns={[
            { title: '序号', fixed: 'left', width: 70, render: (_, _row, index) => (runPage - 1) * runPageSize + index + 1 },
            { title: '模型', dataIndex: 'model_name', fixed: 'left', width: 190, render: (value) => <Typography.Text strong>{value}</Typography.Text> },
            { title: '数据集', dataIndex: 'dataset_name', width: 180 },
            { title: '场景 / 条件', width: 190, render: (_, row) => `${row.scene_domain} / ${row.condition_type || '无'}` },
            { title: '推理分辨率', width: 120, render: (_, row) => runInferenceResolution(row.config) },
            { title: 'mAP', dataIndex: 'map', width: 90, sorter: (a, b) => a.map - b.map, render: (value) => <Typography.Text strong className="map-value">{percent(value)}</Typography.Text> },
            { title: 'AP50', dataIndex: 'map50', width: 85, sorter: (a, b) => a.map50 - b.map50, render: percent },
            { title: 'AP75', dataIndex: 'map75', width: 85, sorter: (a, b) => a.map75 - b.map75, render: percent },
            { title: '端到端P50', dataIndex: 'latency_p50', width: 115, sorter: (a, b) => a.latency_p50 - b.latency_p50, render: (value, row) => performanceValue(row, value, ' ms') },
            { title: '端到端P95', dataIndex: 'latency_p95', width: 115, render: (value, row) => performanceValue(row, value, ' ms') },
            { title: '推理P50', dataIndex: 'inference_latency_p50', width: 105, render: (value, row) => performanceValue(row, value, ' ms') },
            { title: 'FPS', dataIndex: 'throughput_fps', width: 90, render: (value, row) => performanceValue(row, value) },
            { title: '进程峰值显存', dataIndex: 'nvml_process_peak', width: 125, render: (value, row) => performanceValue(row, value, ' MB') },
            { title: '参数量', width: 95, render: (_, row) => hasMeasuredPerformance(row) ? parameterMetric(row.metrics.parameters_total) : '—' },
            { title: '计算量', width: 105, render: (_, row) => hasMeasuredPerformance(row) ? compactMetric(row.metrics.flops, 'GFLOPs') : '—' },
            { title: '操作', fixed: 'right', width: 220, render: (_, row) => <Space size={0}><Button type="link" icon={<EyeOutlined />} onClick={() => setSelectedRun(row)}>查看结果</Button><Popconfirm title="确认删除这次评测结果？" description="运行记录、指标、预测结果和工作目录将移入回收站；模型与数据集不受影响。" okText="移入回收站" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => removeRun(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm></Space> },
          ]}
        />
      </Card>
      <Drawer open={Boolean(selectedRun)} onClose={() => setSelectedRun(undefined)} title="模型详情" width={760}>
        {selectedRun && <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <Space wrap>
            {natureTag(selectedRun)}
            <Typography.Text code copyable>{selectedRun.run_id}</Typography.Text>
            <Button icon={<EyeOutlined />} disabled={selectedRun.is_demo} onClick={() => visualize(selectedRun)}>查看推理结果</Button>
          </Space>
          {!hasMeasuredPerformance(selectedRun) && <Alert type="warning" showIcon message="本次运行未采集真实性能数据" description="该结果仍可用于准确率分析，但 FPS、时延、显存、参数量和计算量不会使用整进程耗时或 0 值代替。请使用支持 result.json 性能协议的适配器重新评测。" />}
          <Row gutter={12}>
            <Col span={8}><Card><Statistic title="mAP" value={selectedRun.map * 100} precision={2} suffix="%" /></Card></Col>
            <Col span={8}><Card><Statistic title="端到端时延P50" value={performanceValue(selectedRun, selectedRun.latency_p50, ' ms')} /></Card></Col>
            <Col span={8}><Card><Statistic title="FPS" value={performanceValue(selectedRun, selectedRun.throughput_fps)} /></Card></Col>
          </Row>
          <Descriptions bordered size="small" column={3} items={[
            { key: 'map50', label: 'AP50', children: percent(selectedRun.map50) },
            { key: 'map75', label: 'AP75', children: percent(selectedRun.map75) },
            { key: 'precision', label: precisionRecallMetricLabel('最佳F1点 Precision'), children: percent(selectedRun.precision) },
            { key: 'recall', label: precisionRecallMetricLabel('最佳F1点 Recall'), children: percent(selectedRun.recall) },
            { key: 'f1', label: precisionRecallMetricLabel('最佳 F1'), children: percent(selectedRun.f1) },
            { key: 'latency95', label: '端到端时延P95', children: performanceValue(selectedRun, selectedRun.latency_p95, ' ms') },
            { key: 'inference50', label: '纯推理时延P50', children: performanceValue(selectedRun, selectedRun.inference_latency_p50, ' ms') },
            { key: 'inference95', label: '纯推理时延P95', children: performanceValue(selectedRun, selectedRun.inference_latency_p95, ' ms') },
            { key: 'allocated', label: 'PyTorch 已分配峰值', children: performanceValue(selectedRun, selectedRun.torch_peak_allocated, ' MB') },
            { key: 'reserved', label: 'PyTorch 保留峰值', children: performanceValue(selectedRun, selectedRun.torch_peak_reserved, ' MB') },
            { key: 'nvml', label: '进程峰值显存', children: performanceValue(selectedRun, selectedRun.nvml_process_peak, ' MB') },
            { key: 'parameters', label: '模型参数量', children: hasMeasuredPerformance(selectedRun) ? parameterMetric(selectedRun.metrics.parameters_total) : '—' },
            { key: 'flops', label: '计算量', children: hasMeasuredPerformance(selectedRun) ? compactMetric(selectedRun.metrics.flops, 'GFLOPs') : '—' },
            { key: 'inferencePrecision', label: '推理精度', children: runParameterValue(selectedRun.config, 'precision') },
            { key: 'confidence', label: '置信度阈值', children: runParameterValue(selectedRun.config, 'confidence') },
            { key: 'nmsIou', label: 'NMS IoU', children: runParameterValue(selectedRun.config, 'nms_iou') },
          ]} />
          <Descriptions bordered column={1} items={[
            { key: 'model', label: '模型', children: selectedRun.model_name },
            { key: 'architecture', label: '架构 / Backbone', children: `${selectedRun.family} / ${selectedRun.backbone}` },
            { key: 'dataset', label: '数据集', children: selectedRun.dataset_name },
            { key: 'categories', label: '评测类别', children: selectedRun.evaluation_categories.length ? <Space wrap>{selectedRun.evaluation_categories.map((name) => <Tag key={name}>{name}</Tag>)}</Space> : evaluationCategoryLabel(selectedRun.evaluation_categories) },
            { key: 'scene', label: '场景', children: selectedRun.scene_domain },
            { key: 'condition', label: '条件', children: selectedRun.condition_type || '无' },
            { key: 'resolution', label: '数据分辨率', children: selectedRun.resolution },
            { key: 'hardware', label: '硬件环境', children: <Typography.Text code copyable>{JSON.stringify(selectedRun.hardware_profile)}</Typography.Text> },
            { key: 'fingerprint', label: '环境指纹', children: <Typography.Text code copyable>{selectedRun.environment_fingerprint || '未记录'}</Typography.Text> },
          ]} />
          <Collapse size="small" items={[
            { key: 'conditions', label: '条件参数', children: Object.keys(selectedRun.sensor_conditions).length ? <Typography.Text code copyable>{JSON.stringify(selectedRun.sensor_conditions)}</Typography.Text> : '未记录' },
            { key: 'config', label: '推理配置', children: <Typography.Text code copyable>{JSON.stringify(selectedRun.config)}</Typography.Text> },
          ]} />
          <Card title="PR 曲线"><PRChart groups={[selectedRun]} dark={dark} height={320} /></Card>
        </Space>}
      </Drawer>
      <Drawer open={Boolean(comparisonRuns.length)} onClose={() => setComparisonRuns([])} title="推理参数对比" width="94vw">
        {comparisonRows.length > 1 && baseline && <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <Alert type="info" showIcon message={`${comparisonRuns[0].model_name} · ${comparisonRuns[0].dataset_name}`} description={`评测类别：${evaluationCategoryLabel(comparisonRuns[0].evaluation_categories)}；相同推理配置的重复运行已自动计算均值。`} />
          <Space wrap>
            <Typography.Text strong>基准配置</Typography.Text>
            <Select value={baseline.key} onChange={setBaselineKey} style={{ minWidth: 260 }} options={comparisonRows.map((row) => ({ value: row.key, label: `${row.label} · ${changedParameterKeys.map((key) => `${resultParameterLabels[key]}=${row.parameters[key]}`).join('，')}` }))} />
          </Space>
          <Card size="small" title="共同推理参数"><Space wrap>{commonParameterKeys.map((key) => <Tag key={key}>{resultParameterLabels[key]}：{comparisonRows[0].parameters[key]}</Tag>)}</Space></Card>
          <Table<ParameterComparisonRow>
            rowKey="key"
            pagination={false}
            dataSource={comparisonRows}
            scroll={{ x: 1380 }}
            columns={[
              { title: '基准', width: 70, fixed: 'left', render: (_, row) => <Radio checked={row.key === baseline.key} onChange={() => setBaselineKey(row.key)} /> },
              { title: '配置', dataIndex: 'label', width: 100, fixed: 'left', render: (value, row) => <Space direction="vertical" size={0}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.runs.length} 次运行</Typography.Text></Space> },
              ...changedParameterKeys.map((key) => ({ title: resultParameterLabels[key], width: 130, render: (_: unknown, row: ParameterComparisonRow) => row.parameters[key] })),
              { title: 'mAP', width: 110, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text strong>{percent(row.map)}</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{mapDelta(row.map, baseline.map)}</Typography.Text>}</Space> },
              { title: 'AP50', width: 100, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{percent(row.map50)}</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{mapDelta(row.map50, baseline.map50)}</Typography.Text>}</Space> },
              { title: 'AP75', width: 100, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{percent(row.map75)}</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{mapDelta(row.map75, baseline.map75)}</Typography.Text>}</Space> },
              { title: '时延P50', width: 115, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{row.latencyP50.toFixed(2)} ms</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{relativeDelta(row.latencyP50, baseline.latencyP50)}</Typography.Text>}</Space> },
              { title: '时延P95', width: 115, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{row.latencyP95.toFixed(2)} ms</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{relativeDelta(row.latencyP95, baseline.latencyP95)}</Typography.Text>}</Space> },
              { title: 'FPS', width: 95, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{row.fps.toFixed(2)}</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{relativeDelta(row.fps, baseline.fps)}</Typography.Text>}</Space> },
              { title: '峰值显存', width: 110, render: (_, row) => <Space direction="vertical" size={0}><Typography.Text>{row.peakMemory.toFixed(0)} MB</Typography.Text>{row.key !== baseline.key && <Typography.Text type="secondary">{relativeDelta(row.peakMemory, baseline.peakMemory)}</Typography.Text>}</Space> },
            ]}
          />
          {singleParameterKey
            ? <Row gutter={[16, 16]}><Col xs={24} xl={12}><Card title={`${resultParameterLabels[singleParameterKey]}—精度`}><ReactECharts option={accuracyComparisonOption} style={{ height: 340 }} /></Card></Col><Col xs={24} xl={12}><Card title={`${resultParameterLabels[singleParameterKey]}—时延`}><ReactECharts option={latencyComparisonOption} style={{ height: 340 }} /></Card></Col></Row>
            : <Alert type="warning" showIcon message="同时变化了多个推理参数" description="平台仅展示结果差异，不将性能变化归因于某一个参数。若要查看单参数趋势，请选择其他参数完全一致的评测结果。" />}
          <Card title="PR 曲线对比"><PRChart groups={comparisonRows.map((row) => ({ model_name: `${row.label} ${changedParameterKeys.map((key) => row.parameters[key]).join(' / ')}`, curves: row.curves }))} dark={dark} height={360} /></Card>
        </Space>}
      </Drawer>
      <EvaluationVisualizationDrawer group={visualizationGroup} onClose={() => setVisualizationGroup(undefined)} />
    </Space>
}

export function TasksPage(_: PageProps) {
  const { data, loading, reload } = useResource<Job[]>('/api/jobs', [])
  useEffect(() => { const timer = window.setInterval(reload, 2000); return () => window.clearInterval(timer) }, [reload])
  const cancel = async (job: Job) => { try { await post(`/api/runs/${job.id}/cancel`); message.success('已请求取消'); reload() } catch (error) { message.error((error as Error).message) } }
  const remove = async (job: Job) => { try { const result = await api<{ cleanup_errors: string[] }>(`/api/jobs/${job.id}`, { method: 'DELETE' }); result.cleanup_errors.length ? message.warning(`任务记录已删除，但有 ${result.cleanup_errors.length} 个文件未能清理`) : message.success('任务已删除'); reload() } catch (error) { message.error((error as Error).message) } }
  const terminal = (job: Job) => ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status)
  return <Card title="持久任务队列" extra={<Button icon={<ReloadOutlined />} onClick={reload}>刷新</Button>}><Table loading={loading} rowKey="id" dataSource={data} expandable={{ expandedRowRender: (job) => <Space direction="vertical" style={{ width: '100%' }}><Typography.Text code copyable>{job.id}</Typography.Text>{job.error && <Alert type="error" message={job.error} />}</Space> }} columns={[{ title: '类型', dataIndex: 'type', render: (value) => <Tag>{value}</Tag> }, { title: '状态', dataIndex: 'status', render: (value) => <StatusTag status={value} /> }, { title: '阶段', dataIndex: 'stage' }, { title: '进度', render: (_, row) => <Progress percent={Math.round(row.progress)} size="small" style={{ minWidth: 150 }} status={row.status === 'FAILED' ? 'exception' : undefined} /> }, { title: '创建时间', dataIndex: 'created_at', render: (value) => new Date(value).toLocaleString('zh-CN') }, { title: '操作', render: (_, row) => terminal(row) ? <Popconfirm title="确认删除这个任务？" description="任务记录、工作区和日志将被删除；已生成的数据集和评测结果不受影响。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => remove(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm> : <Button danger type="link" onClick={() => cancel(row)}>取消</Button> }]} /></Card>
}

export function EnvironmentPage(_: PageProps) {
  const { data, loading, reload } = useResource<EnvironmentStatus | undefined>('/api/environment/status', undefined)
  if (loading || !data) return <Card loading />
  const diskPercent = Math.round(data.disk.used / data.disk.total * 100)
  return <Space direction="vertical" size={18} style={{ width: '100%' }}><Alert type="success" showIcon message="工作区隔离已启用" description="平台不修改 base、.condarc、.bashrc 或已有 conda 环境。所有平台环境、缓存、数据库和 Artifact 均位于项目目录。" action={<Button icon={<ReloadOutlined />} onClick={reload}>重新检查</Button>} /><Row gutter={[16, 16]}><Col xs={24} md={8}><Card><Statistic title="隔离模式" value="Workspace" prefix={<SafetyCertificateOutlined />} /><Typography.Text type="secondary">写入工作区外：{data.isolation.writes_outside_workspace ? '是' : '否'}</Typography.Text></Card></Col><Col xs={24} md={8}><Card><Statistic title="GPU 状态" value={data.gpu.available ? '可用' : '当前受限'} prefix={<ThunderboltOutlined />} /><Typography.Text type="secondary"></Typography.Text></Card></Col><Col xs={24} md={8}><Card><Statistic title="磁盘使用" value={diskPercent} suffix="%" /><Progress percent={diskPercent} showInfo={false} /><Typography.Text type="secondary">剩余 {formatBytes(data.disk.free)}</Typography.Text></Card></Col></Row><Card title="写入边界"><Descriptions bordered column={1} items={[{ key: 'runtime', label: '平台环境与缓存', children: <Typography.Text code copyable>{data.isolation.runtime_dir}</Typography.Text> }, { key: 'data', label: '数据库与 Artifact', children: <Typography.Text code copyable>{data.isolation.data_dir}</Typography.Text> }, { key: 'shell', label: 'Shell 配置修改', children: data.isolation.shell_configuration_modified ? <Tag color="error">有</Tag> : <Tag color="success">无</Tag> }, { key: 'conda', label: '外部环境策略', children: <Tag icon={<LockOutlined />} color="success">只读调用</Tag> }]} /></Card><Card title={`发现 ${data.conda.envs.length} 个 conda 环境`} extra={<Typography.Text type="secondary">仅读取 conda-meta/history 指纹</Typography.Text>}><Table rowKey="prefix" pagination={false} dataSource={data.conda.envs} columns={[{ title: '名称', dataIndex: 'name', render: (value) => <Typography.Text strong>{value}</Typography.Text> }, { title: '路径', dataIndex: 'prefix', render: (value) => <Typography.Text code copyable>{value}</Typography.Text> }, { title: '策略', render: () => <Tag icon={<LockOutlined />} color="success">external_read_only</Tag> }, { title: '环境指纹', dataIndex: 'fingerprint', render: (value) => value ? <Typography.Text code>{value}</Typography.Text> : '未找到' }, { title: '状态', dataIndex: 'exists', render: (value) => <StatusTag status={value ? 'HEALTHY' : 'UNAVAILABLE'} /> }]} /></Card>{!data.gpu.available && <Alert type="warning" showIcon message="当前进程无法访问 NVIDIA 驱动" description={data.gpu.error || '宿主机预检失败。Web 和 CPU 功能仍可使用，GPU Adapter 会被安全阻止。'} />}</Space>
}
