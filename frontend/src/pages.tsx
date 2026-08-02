import { cloneElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Carousel,
  Checkbox,
  Col,
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
  FullscreenExitOutlined,
  ImportOutlined,
  LeftOutlined,
  LockOutlined,
  PlayCircleOutlined,
  PlusOutlined,
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
import type { Adapter, AnnotationCategory, AnnotationSession, BaseGenSceneField, BaseGenSceneOption, BaseGenSceneSchema, CategoryDefinition, CategoryTemplate, Dataset, DatasetSamplePage, DetectionBox, EnvironmentStatus, Job, ModelVersion, Overview, ResultGroup, ResultResponse, SampleAnnotation } from './types'

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
  const results = useResource<ResultResponse>('/api/results', { count: 0, groups: [], runs: [], dimensions: { scenes: [], conditions: [], resolutions: [], models: [] } })
  if (!overview) return <Card loading />
  const metrics = [
    { label: '数据集版本', value: overview.counts.datasets, suffix: '个', icon: <DatabaseOutlined />, color: 'blue' },
    { label: '模型版本', value: overview.counts.models, suffix: '个', icon: <RobotOutlined />, color: 'purple' },
    { label: '活动任务', value: overview.counts.running, suffix: '个', icon: <ThunderboltOutlined />, color: 'cyan' },
    { label: '已完成运行', value: overview.counts.completed, suffix: '次', icon: <CheckCircleOutlined />, color: 'green' },
  ]
  return (
    <Space direction="vertical" size={18} style={{ width: '100%' }}>
      <Alert type="info" showIcon message="当前运行在隔离工作区模式" description="平台写入仅限项目目录；现有 conda 环境按只读策略登记。页面中的参考模型与回放图像均明确标记为流程样例。" action={<Button size="small" onClick={() => navigate('environment')}>查看环境边界</Button>} />
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
  { id: 'local-import', source: 'REAL', icon: <ImportOutlined />, title: '本地数据', description: '导入本地 PNG/JPEG图像及可选 COCO/YOLO/VisDrone格式标注文件。', recommended: true },
  { id: 'adapter_replay', source: 'REPLAY_FIXTURE', icon: <PlayCircleOutlined />, title: '测试回放', description: '固定样例验证生成任务、标注与评测闭环。', recommended: true },
  { id: 'adapter_condition', source: 'REAL_TRANSFORMED', icon: <CloudOutlined />, title: '条件退化', description: '读取已导入的真实图像，产生模糊、雾化和噪声条件。' },
  { id: 'adapter_basegen', source: 'GENERATIVE', icon: <RobotOutlined />, title: 'Z-Image-Turbo', description: '通过独立 gen 环境调用 BaseGen 生成未标注感知图像。' },
  { id: 'airsim-future', source: 'SIMULATOR', icon: <CodeOutlined />, title: 'AirSim / UE', description: '通过独立 RPC 服务采集图像与真值。', disabled: true },
]

type BaseGenSelection =
  | { mode: 'random' }
  | { mode: 'fixed'; value: string }
  | { mode: 'fixed'; values: string[] }

const RANDOM_VALUE = '__random__'

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
  const [weather, setWeather] = useState('晴朗')
  const [resolution, setResolution] = useState('1920×1080')
  const [blur, setBlur] = useState(0.3)
  const [samples, setSamples] = useState(12)
  const [seeds, setSeeds] = useState([1001, 1002, 1003])
  const [generatorSeed, setGeneratorSeed] = useState(1001)
  const [generatorSteps, setGeneratorSteps] = useState(9)
  const [devicePolicy, setDevicePolicy] = useState('cuda')
  const [basegenSelections, setBasegenSelections] = useState<Record<string, BaseGenSelection>>({})
  const [basegenCustom, setBasegenCustom] = useState('')
  const [inputDatasetId, setInputDatasetId] = useState<string>()
  const [localDatasetName, setLocalDatasetName] = useState('')
  const [localDirectory, setLocalDirectory] = useState('')
  const [annotationPath, setAnnotationPath] = useState('')
  const [localImageFiles, setLocalImageFiles] = useState<File[]>([])
  const [annotationMode, setAnnotationMode] = useState<'coco' | 'yolo' | 'visdrone'>('coco')
  const [categoryTemplateId, setCategoryTemplateId] = useState('visdrone')
  const [customDatasetCategories, setCustomDatasetCategories] = useState<CategoryDefinition[]>([{ id: 1, name: '' }])
  const [localAnnotationFile, setLocalAnnotationFile] = useState<File>()
  const [localYoloAnnotationFiles, setLocalYoloAnnotationFiles] = useState<File[]>([])
  const annotationFileInput = useRef<HTMLInputElement>(null)
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
  const currentBasegenDomain = basegenSchema.data.domains.find((item) => item.label_zh === domain)
  const domainOptions = isBaseGen
    ? basegenSchema.data.domains.map((item) => item.label_zh)
    : ['无人机航拍', '卫星遥感', '城市驾驶']
  const resolutionOptions = isBaseGen
    ? ['1024×1024', '1024×576']
    : ['1920×1080', '1280×720', '640×640']
  const weatherOptions = ['晴朗', '雾', '雨', '夜间']
  const combinationCount = isBaseGen ? 1 : seeds.length
  const inputDataset = datasets.data.find((item) => item.id === inputDatasetId)
  const selectedCategories = source.id === 'adapter_condition'
    ? inputDataset?.categories || []
    : categoriesFromSelection(categoryTemplates.data, categoryTemplateId, 'dataset', customDatasetCategories)
  const categoriesReady = validCategories(selectedCategories)

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
        : field.weighted ? '随机（按权重）' : '随机'
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
    } else {
      if (!['无人机航拍', '卫星遥感', '城市驾驶'].includes(domain)) setDomain('无人机航拍')
      setResolution('1920×1080')
    }
  }
  const healthSource = async (id: string) => {
    try {
      const result = await post<{ healthy: boolean }>(`/api/adapters/${id}/health-check`)
      result.healthy ? message.success('数据来源健康检查通过') : message.warning('数据来源当前不可用')
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
  const selectLocalDirectory = (event: React.ChangeEvent<HTMLInputElement>) => {
    const supported = new Set(['.jpg', '.jpeg', '.png', '.webp', '.svg'])
    const files = Array.from(event.currentTarget.files || []).filter((file) => {
      const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
      return supported.has(suffix)
    })
    setLocalImageFiles(files)
    const firstRelative = files[0] ? (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath : ''
    const directoryName = firstRelative?.split('/')[0] || ''
    setLocalDirectory(files.length ? `${directoryName || '所选目录'} · ${files.length} 张图像` : '')
    if (!files.length) message.warning('所选目录中没有 PNG、JPEG、WebP 或 SVG 图像')
  }
  const selectLocalAnnotation = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    setLocalAnnotationFile(file)
    setAnnotationPath(file?.name || '')
  }
  const selectLocalYoloAnnotations = (event: React.ChangeEvent<HTMLInputElement>) => {
    const supported = new Set(['.txt', '.yaml', '.yml', '.names'])
    const files = Array.from(event.currentTarget.files || []).filter((file) => {
      const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
      return supported.has(suffix)
    })
    setLocalYoloAnnotationFiles(files)
    const firstRelative = files[0] ? (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath : ''
    const directoryName = firstRelative?.split('/')[0] || '所选目录'
    setAnnotationPath(files.length ? `${directoryName} · ${files.length} 个标注文件` : '')
    if (!files.length) message.warning('所选目录中没有 TXT、YAML 或 NAMES 标注文件')
  }
  const changeAnnotationMode = (value: string | number) => {
    const mode = String(value) as 'coco' | 'yolo' | 'visdrone'
    setAnnotationMode(mode)
    if (mode === 'visdrone') setCategoryTemplateId('visdrone')
    setLocalAnnotationFile(undefined)
    setLocalYoloAnnotationFiles([])
    setAnnotationPath('')
    if (annotationFileInput.current) annotationFileInput.current.value = ''
  }
  const acquisitionPayload = () => ({
    name: `${domain} · ${source.title} · ${new Date().toLocaleDateString('zh-CN')}`,
    adapter_id: source.id,
    source_type: source.source,
    sample_count: samples,
    seeds: isBaseGen ? [generatorSeed] : seeds,
    conditions: {
      scene: isBaseGen ? {
        domain: currentBasegenDomain?.value,
        domain_label: currentBasegenDomain?.label_zh,
        weather: weatherSummary,
        fields: basegenSelections,
        custom: basegenCustom,
      } : { domain, weather },
      sensor: isBaseGen ? { resolution } : { resolution, motion_blur: blur, fog_density: weather === '雾' ? 0.4 : 0 },
    },
    model_parameters: isBaseGen ? { steps: generatorSteps, guidance_scale: 0, device_policy: devicePolicy, local_files_only: false } : {},
    input_dataset_id: source.id === 'adapter_condition' ? inputDatasetId : null,
    category_template: source.id === 'adapter_condition' ? inputDataset?.category_template || 'custom' : categoryTemplateId,
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
        const body = new FormData()
        body.append('name', localDatasetName.trim())
        body.append('scene_domain', domain)
        body.append('annotation_format', annotationMode.toUpperCase())
        body.append('category_template', categoryTemplateId)
        body.append('categories_json', JSON.stringify(selectedCategories.map(({ id, name }) => ({ id, name }))))
        const imageRelativePaths: string[] = []
        for (const file of localImageFiles) {
          const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
          body.append('images', file, file.name)
          imageRelativePaths.push(relativePath)
        }
        body.append('relative_paths_json', JSON.stringify(imageRelativePaths))
        if (annotationMode === 'coco' && localAnnotationFile) {
          body.append('annotation', localAnnotationFile, localAnnotationFile.name)
        }
        const annotationRelativePaths: string[] = []
        if (annotationMode !== 'coco') {
          for (const file of localYoloAnnotationFiles) {
            const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
            body.append('annotation_files', file, file.name)
            annotationRelativePaths.push(relativePath)
          }
        }
        body.append('annotation_relative_paths_json', JSON.stringify(annotationRelativePaths))
        job = await api<Job>('/api/datasets/import-upload', { method: 'POST', body })
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

  return (
    <Space direction="vertical" size={18} style={{ width: '100%' }}>
      <Steps current={step} items={['选择来源', '配置条件', '组合预览', '执行与浏览', '真值冻结'].map((title) => ({ title }))} />
      {step === 0 && <Card title="选择数据来源" extra={<Tag color="purple">BaseGen 已接入</Tag>}>
        <Row gutter={[16, 16]}>{sourceCards.map((item) => {
          const runtime = sourceRuntimes.data.find((entry) => entry.id === item.id)
          return <Col xs={24} md={12} xl={6} key={item.id}><Card hoverable={!item.disabled} className={`source-card ${source.id === item.id ? 'source-selected' : ''} ${item.disabled ? 'source-disabled' : ''}`} onClick={() => selectSource(item)}><div className="source-icon">{item.icon}</div><Space><Typography.Title level={4}>{item.title}</Typography.Title>{item.recommended && <Tag color="cyan">推荐</Tag>}</Space><Typography.Paragraph type="secondary">{item.description}</Typography.Paragraph><Space wrap>{item.id === 'local-import' ? <StatusTag status="HEALTHY" /> : item.disabled ? <Tag>未接入</Tag> : <><StatusTag status={runtime?.status || (sourceRuntimes.loading ? 'CHECKING' : 'UNAVAILABLE')} /><Button size="small" onClick={(event) => { event.stopPropagation(); healthSource(item.id) }}>健康检查</Button></>}</Space></Card></Col>
        })}</Row>
        <div className="wizard-actions"><Button type="primary" onClick={() => setStep(1)}>下一步：配置条件 <ArrowRightOutlined /></Button></div>
      </Card>}
      {step === 1 && <Row gutter={16}>
        <Col xs={24} xl={12}>
          <Card title={source.id === 'local-import' ? '选择本地数据' : '场景条件'}>
            <Form layout="vertical">
              {source.id === 'local-import' && <>
                <Form.Item label="数据集名称" required>
                  <Input value={localDatasetName} onChange={(event) => setLocalDatasetName(event.target.value)} maxLength={120} showCount placeholder="例如 VisDrone2019 测试集" />
                </Form.Item>
                <Form.Item label="图像目录" required>
                  <input
                    className="native-resource-picker"
                    type="file"
                    multiple
                    onChange={selectLocalDirectory}
                    {...({ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                  />
                  <Typography.Paragraph type={localDirectory ? 'success' : 'secondary'} style={{ margin: '8px 0 0' }}>
                    {localDirectory || '目录模式下图像文件显示为灰色是正常的；进入目标目录后，点击对话框右下角的“选择/打开”。'}
                  </Typography.Paragraph>
                </Form.Item>
                <Form.Item label="标注（可选）">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Segmented block value={annotationMode} onChange={changeAnnotationMode} options={[{ value: 'coco', label: 'COCO 格式json' }, { value: 'yolo', label: 'YOLO 格式' }, { value: 'visdrone', label: 'VisDrone 格式' }]} />
                    {annotationMode === 'coco' ? (
                      <input ref={annotationFileInput} className="native-resource-picker" type="file" onChange={selectLocalAnnotation} />
                    ) : (
                      <input
                        className="native-resource-picker"
                        type="file"
                        multiple
                        onChange={selectLocalYoloAnnotations}
                        {...({ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                      />
                    )}
                    {annotationPath && <Typography.Text type="success">已选择：{annotationPath}</Typography.Text>}
                  </Space>
                </Form.Item>
              </>}
              <Form.Item label="场景域"><Select loading={isBaseGen && basegenSchema.loading} value={domain} onChange={selectDomain} options={domainOptions.map((value) => ({ value }))} /></Form.Item>
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
                    <Select value={value} onChange={(next) => changeBasegenSelection(field, next === RANDOM_VALUE ? { mode: 'random' } : { mode: 'fixed', value: next })} options={[{ value: RANDOM_VALUE, label: field.weighted ? '随机（按 BaseGen 权重）' : '随机' }, ...field.options.map((option) => ({ value: option.value, label: option.label_zh, disabled: optionDisabled(field, option) }))]} />
                  </Form.Item>
                )
              })}
              {source.id !== 'local-import' && !isBaseGen && <Form.Item label="天气 / 环境"><Segmented block value={weather} onChange={(value) => setWeather(String(value))} options={weatherOptions} /></Form.Item>}
              {source.id !== 'local-import' && <Form.Item label="输出数量"><InputNumber value={samples} onChange={(value) => setSamples(value || 1)} min={1} max={1000} addonAfter="张" style={{ width: '100%' }} /></Form.Item>}
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={12}>{source.id === 'local-import' ? <Card title="导入校验"><Timeline items={[{ color: 'blue', children: '通过系统资源管理器选择图像目录' }, { color: 'blue', children: '上传 PNG、JPEG、WebP 和 SVG 到工作区暂存区' }, { color: 'blue', children: '标注作为候选真值导入并等待校核' }, { color: 'green', children: '导入完成后自动清理暂存文件，不修改原始目录' }]} /><Alert type="info" showIcon message="浏览器安全选择" description="浏览器不会向平台暴露本机绝对路径；只上传你在资源管理器中明确选择的文件。" /></Card> : <Card title={isBaseGen ? '生成参数' : '传感器与成像条件'}><Form layout="vertical"><Form.Item label="图像分辨率"><Select value={resolution} onChange={setResolution} options={resolutionOptions.map((value) => ({ value }))} /></Form.Item>{isBaseGen ? <><Form.Item label="起始随机种子"><InputNumber value={generatorSeed} onChange={(value) => setGeneratorSeed(value || 0)} min={0} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="推理步数"><InputNumber value={generatorSteps} onChange={(value) => setGeneratorSteps(value || 1)} min={1} max={100} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="设备策略"><Select value={devicePolicy} onChange={setDevicePolicy} options={[{ value: 'cuda', label: '全量 CUDA（推荐）' }, { value: 'cpu-offload', label: 'CPU Offload（节省显存）' }]} /></Form.Item></> : <><Form.Item label={`运动模糊强度 ${blur.toFixed(1)}`}><Slider value={blur} onChange={setBlur} min={0} max={1} step={0.1} marks={{ 0: '清洁', 0.5: '中等', 1: '严重' }} /></Form.Item><Form.Item label="固定随机种子"><Checkbox.Group options={[1001, 1002, 1003, 1004].map((value) => ({ label: value, value }))} value={seeds} onChange={(values) => setSeeds(values as number[])} /></Form.Item></>}</Form></Card>}</Col>
        {source.id === 'adapter_condition' && <Col span={24}><Card title="选择退化输入数据集"><Select value={inputDatasetId} onChange={setInputDatasetId} style={{ width: '100%' }} placeholder="选择已导入的 PNG/JPEG/WebP 数据集" options={datasets.data.filter((item) => item.source_type === 'REAL').map((item) => ({ value: item.id, label: `${item.name} · ${item.sample_count} 张`, disabled: !item.categories.length }))} /><Typography.Paragraph type="secondary" style={{ marginTop: 10, marginBottom: 0 }}>条件算子保持几何位置不变，并自动继承输入数据集的类别。</Typography.Paragraph></Card></Col>}
        <Col span={24}><Card title="目标检测类别">{source.id === 'adapter_condition' ? inputDataset ? <Alert type="info" showIcon message={`已继承 ${inputDataset.categories.length} 个类别`} description={inputDataset.categories.map((item) => `${item.id}:${item.name}`).join(' · ')} /> : <Alert type="warning" showIcon message="请先选择输入数据集" /> : <CategoryConfiguration templates={categoryTemplates.data} templateId={categoryTemplateId} scope="dataset" customCategories={customDatasetCategories} onTemplateChange={setCategoryTemplateId} onCustomChange={setCustomDatasetCategories} />}</Card></Col>
        <Col span={24}><div className="wizard-actions"><Button onClick={() => setStep(0)}>上一步</Button><Button type="primary" disabled={!categoriesReady || (source.id === 'local-import' ? localDatasetName.trim().length < 2 || !localImageFiles.length : isBaseGen ? !currentBasegenDomain || generatorSeed < 0 || generatorSteps < 1 : source.id === 'adapter_condition' ? !inputDatasetId || !seeds.length : !seeds.length)} onClick={() => setStep(2)}>下一步：组合预览</Button></div></Col>
      </Row>}
      {step === 2 && <Card title="提交前确认" extra={source.id === 'local-import' ? <Tag color="blue">本地真实数据</Tag> : isBaseGen ? <Tag color="purple">Z-Image-Turbo</Tag> : <DemoTag />}>
        <Row gutter={[16, 16]}><Col xs={24} md={8}><Statistic title={source.id === 'local-import' ? '导入任务' : '配置单元'} value={source.id === 'local-import' ? 1 : combinationCount} suffix="个" /></Col><Col xs={24} md={8}><Statistic title="输出样本" value={source.id === 'local-import' ? '目录内图像' : samples} suffix={source.id === 'local-import' ? undefined : '张'} /></Col><Col xs={24} md={8}><Statistic title="真值入口" value={source.id === 'local-import' ? (annotationPath ? '已提供' : '未提供') : isBaseGen ? '未标注' : '候选框'} /></Col></Row><Divider />
        <Descriptions column={{ xs: 1, md: 2 }} bordered size="small" items={[{ key: 'source', label: '来源', children: source.title }, { key: 'scene', label: '场景', children: source.id === 'local-import' || isBaseGen ? domain : `${domain} / ${weather}` }, { key: 'sensor', label: source.id === 'local-import' ? '输入目录' : isBaseGen ? '生成参数' : '成像条件', children: source.id === 'local-import' ? localDirectory : isBaseGen ? `${resolution} / ${generatorSteps} 步 / ${devicePolicy}` : `${resolution} / 模糊 ${blur}` }, { key: 'categories', label: '检测类别', children: `${selectedCategories.length} 类` }, { key: 'seed', label: '随机种子', children: source.id === 'local-import' ? '不适用' : isBaseGen ? `${generatorSeed} 起连续 ${samples} 个` : seeds.join(', ') }, { key: 'truth', label: '真值策略', children: source.id === 'local-import' ? (annotationPath ? '导入后作为候选真值' : '未提供') : isBaseGen ? '未标注，需另行标注后评测' : '候选框，完成后需校核冻结' }, { key: 'official', label: '结果性质', children: source.id === 'local-import' ? <Tag color="blue">真实采集数据</Tag> : isBaseGen ? <Tag color="purple">真实模型生成</Tag> : <DemoTag /> }]} />
        {isBaseGen && <Card size="small" title="场景字段规则" className="top-gap"><Space wrap>{basegenSceneSummary.map((item) => <Tag key={item}>{item}</Tag>)}{basegenCustom && <Tag color="blue">自定义：{basegenCustom}</Tag>}</Space></Card>}
        {isBaseGen && basegenPreview && <Card size="small" title="随机计划预览（不加载模型）" className="top-gap"><List dataSource={basegenPreview.images} renderItem={(item) => <List.Item><Space direction="vertical" style={{ width: '100%' }}><Space wrap><Tag color="blue">seed {item.seed}</Tag><Tag>{item.width}×{item.height}</Tag>{currentBasegenDomain?.fields.filter((field) => field.kind !== 'text').map((field) => { const raw = item.scene[field.name]; const values = Array.isArray(raw) ? raw : [raw]; return <Tag key={field.name}>{field.label_zh}：{values.map((value) => optionFor(field, value)?.label_zh || value).join('、')}</Tag> })}</Space><Typography.Paragraph copyable={{ text: item.prompt }} ellipsis={{ rows: 3, expandable: true, symbol: '展开 prompt' }} style={{ marginBottom: 0 }}>{item.prompt}</Typography.Paragraph></Space></List.Item>} /></Card>}
        {source.id !== 'local-import' && <Alert className="inline-alert" type={isBaseGen ? 'info' : 'warning'} showIcon message={isBaseGen ? '本任务将调用 BaseGen 真实生成图像' : source.id === 'adapter_condition' ? '本任务将创建真实图像的条件退化版本' : '本任务使用固定样例验证流程'} description={isBaseGen ? '模型在独立 gen 环境中运行；纯文本生成不提供目标框等真值。' : source.id === 'adapter_condition' ? '输出记录原数据集和退化参数，正式冻结前仍需抽查真值。' : '输出会保留完整数据谱系，但不能作为生成模型能力结论。'} />}
        <div className="wizard-actions"><Button onClick={() => { setBasegenPreview(undefined); setStep(1) }}>上一步</Button>{isBaseGen && <Button loading={previewing} onClick={previewBasegen}>预览 3 个随机场景</Button>}<Button type="primary" loading={submitting} icon={<PlayCircleOutlined />} onClick={submit}>提交构建任务</Button></div>
      </Card>}
      {step === 3 && <Space direction="vertical" size={16} style={{ width: '100%' }}><JobProgress jobId={jobId} onFinish={finish} />{finishedJob?.status === 'SUCCEEDED' && <Card><Result status="success" title={source.id === 'local-import' ? '本地图像已导入' : '数据构建任务已完成'} subTitle={finishedJob.result?.annotation_status === 'UNLABELED' ? '当前没有真值，需进入数据集完成标注后才能正式评测。' : '输出当前仍是候选真值，冻结前不会进入正式评测。'} extra={finishedJob.result?.annotation_status === 'UNLABELED' ? [<Button key="datasets" type="primary" onClick={() => navigate('datasets')}>打开数据集</Button>] : [<Button key="freeze" type="primary" icon={<LockOutlined />} onClick={freeze}>校核并冻结数据版本</Button>, <Button key="datasets" onClick={() => navigate('datasets')}>打开数据集</Button>]} /></Card>}</Space>}
      {step === 4 && <Card><Result status="success" title="数据版本已冻结" subTitle="该版本不可变；后续修改真值需要创建新版本。" extra={[<Button type="primary" key="eval" onClick={() => navigate('evaluation')}>进入评测中心</Button>, <Button key="again" onClick={() => { setStep(0); setJobId(undefined); setFinishedJob(undefined) }}>继续构建数据</Button>]} /></Card>}
    </Space>
  )
}

function DatasetBrowser({ dataset, onClose }: { dataset?: Dataset; onClose: () => void }) {
  const [items, setItems] = useState<DatasetSamplePage['items']>([])
  const [total, setTotal] = useState(0)
  const [declaredCount, setDeclaredCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const loadingRef = useRef(false)
  const activeDatasetId = useRef<string>()
  const browserRef = useRef<HTMLDivElement>(null)
  const loadMoreRef = useRef<HTMLDivElement>(null)

  const load = async (datasetId: string, offset: number, replace: boolean) => {
    if (loadingRef.current) return
    loadingRef.current = true
    setLoading(true)
    try {
      const page = await api<DatasetSamplePage>(`/api/datasets/${datasetId}/samples?offset=${offset}&limit=48`)
      if (activeDatasetId.current !== datasetId) return
      setItems((current) => replace ? page.items : [...current, ...page.items])
      setTotal(page.total)
      setDeclaredCount(page.declared_count)
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
    setDeclaredCount(dataset?.sample_count || 0)
    setError('')
    if (dataset) load(dataset.id, 0, true)
  }, [dataset?.id])

  const scroll = (event: React.UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget
    if (dataset && items.length < total && element.scrollHeight - element.scrollTop - element.clientHeight < 320) {
      load(dataset.id, items.length, false)
    }
  }
  useEffect(() => {
    const root = browserRef.current
    const target = loadMoreRef.current
    if (!dataset || !root || !target || loading || items.length >= total) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && !loadingRef.current) {
          load(dataset.id, items.length, false)
        }
      },
      { root, rootMargin: '480px 0px' },
    )
    observer.observe(target)
    return () => observer.disconnect()
  }, [dataset?.id, items.length, total, loading])
  const annotationSourceLabels = {
    MANUAL: '平台标注',
    COCO: 'COCO',
    YOLO: 'YOLO',
    VISDRONE: 'VisDrone',
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
    <Drawer open={Boolean(dataset)} width="90vw" title={dataset ? `${dataset.name} · 全部样本` : '全部样本'} onClose={onClose} extra={<Tag>已加载 {items.length} / 总计 {total || declaredCount}</Tag>}>
      <div ref={browserRef} className="dataset-browser" onScroll={scroll}>
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
                  {annotationBoxes(item)}
                </div>
              )
            },
          }}>
            <div className="dataset-browser-grid">
              {items.map((item) => (
                <div className="dataset-browser-sample" key={item.url}>
                  <div className="dataset-browser-image" style={{ aspectRatio: item.width && item.height ? `${item.width} / ${item.height}` : '16 / 9' }}>
                    <Image src={item.url} alt={item.name} loading="lazy" preview={{ mask: item.name }} />
                    {annotationBoxes(item)}
                  </div>
                  <div className="dataset-browser-caption">
                    <Typography.Text ellipsis={{ tooltip: item.name }}>{item.name}</Typography.Text>
                    {item.annotation_source ? <Tag color={item.boxes.length ? 'blue' : 'default'}>{annotationSourceLabels[item.annotation_source]} · {item.boxes.length} 框</Tag> : <Tag>无标注</Tag>}
                  </div>
                </div>
              ))}
            </div>
          </Image.PreviewGroup>
        )}
        {loading && <div className="dataset-browser-loading"><Spin /><Typography.Text type="secondary">正在加载图片…</Typography.Text></div>}
        {!loading && items.length < total && <div ref={loadMoreRef} className="dataset-browser-load-more"><Button onClick={() => dataset && load(dataset.id, items.length, false)}>加载更多（剩余 {total - items.length} 张）</Button></div>}
        {!loading && total > 0 && items.length >= total && <div className="dataset-browser-end">已加载全部 {total} 张图片</div>}
      </div>
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
                      <text x={box.x + 3} y={Math.max(14, box.y + 15)} fill="#fff" stroke="#000" strokeWidth={3} paintOrder="stroke" fontSize={14} pointerEvents="none">{category?.name || box.category_id}</text>
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

export function DatasetsPage(_: PageProps) {
  const { data, loading, reload } = useResource<Dataset[]>('/api/datasets', [])
  const [selected, setSelected] = useState<Dataset>()
  const [browserDataset, setBrowserDataset] = useState<Dataset>()
  const [annotationDataset, setAnnotationDataset] = useState<Dataset>()
  const freeze = async (dataset: Dataset) => {
    try { await post(`/api/datasets/${dataset.id}/freeze`); message.success('数据版本已冻结'); reload() } catch (error) { message.error((error as Error).message) }
  }
  const remove = async (dataset: Dataset) => {
    try {
      await api(`/api/datasets/${dataset.id}`, { method: 'DELETE' })
      if (selected?.id === dataset.id) setSelected(undefined)
      if (browserDataset?.id === dataset.id) setBrowserDataset(undefined)
      if (annotationDataset?.id === dataset.id) setAnnotationDataset(undefined)
      message.success('数据集已移入回收站')
      reload()
    } catch (error) { message.error((error as Error).message) }
  }
  return <><Card title="不可变数据版本" extra={<Space><Tag>{data.length} 个版本</Tag><Button icon={<ReloadOutlined />} onClick={reload}>刷新</Button></Space>}><Table loading={loading} rowKey="id" dataSource={data} pagination={{ pageSize: 8 }} columns={[
    { title: '数据集', dataIndex: 'name', render: (value, row) => <Space direction="vertical" size={1}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.id}</Typography.Text></Space> },
    { title: '来源', dataIndex: 'source_type', render: (value) => <Tag color={value === 'REPLAY_FIXTURE' ? 'gold' : value === 'REAL_TRANSFORMED' ? 'cyan' : 'blue'}>{value}</Tag> },
    { title: '场景 / 条件', render: (_, row) => `${row.scene_domain} / ${row.weather}` },
    { title: '分辨率', dataIndex: 'resolution' }, { title: '样本', dataIndex: 'sample_count' },
    { title: '真值', dataIndex: 'annotation_status', render: (value) => <StatusTag status={value} /> },
    { title: '版本', render: (_, row) => row.frozen ? <Tag icon={<LockOutlined />} color="success">{row.version} 已冻结</Tag> : <Tag>草稿</Tag> },
    { title: '操作', render: (_, row) => <Space><Button type="link" onClick={() => setSelected(row)}>查看</Button><Button type="link" icon={<FileImageOutlined />} onClick={() => setBrowserDataset(row)}>浏览全部</Button><Button type="link" onClick={() => setAnnotationDataset(row)}>{row.frozen ? '查看标注' : '目标标注'}</Button>{!row.frozen && <Button type="link" onClick={() => freeze(row)}>冻结</Button>}{!row.frozen && <Popconfirm title="确认删除这个数据集？" description={`${row.name} · ${row.sample_count} 个样本将移入回收站。`} okText="移入回收站" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => remove(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm>}</Space> },
  ]} /></Card><Drawer open={Boolean(selected)} width={760} title={selected?.name} onClose={() => setSelected(undefined)}>{selected && <Space direction="vertical" size={18} style={{ width: '100%' }}><Gallery images={selected.preview_images} height={140} /><Button block icon={<FileImageOutlined />} onClick={() => { setSelected(undefined); setBrowserDataset(selected) }}>浏览全部图片</Button><Button block onClick={() => { setSelected(undefined); setAnnotationDataset(selected) }}>{selected.frozen ? '查看目标检测标注' : '开始目标检测标注'}</Button><Descriptions bordered column={2} items={[{ key: 'source', label: '来源', children: selected.source_type }, { key: 'scene', label: '场景', children: selected.scene_domain }, { key: 'weather', label: '天气', children: selected.weather }, { key: 'resolution', label: '分辨率', children: selected.resolution }, { key: 'truth', label: '真值', children: <StatusTag status={selected.annotation_status} /> }, { key: 'frozen', label: '不可变', children: selected.frozen ? '是' : '否' }, { key: 'categories', label: '检测类别', children: selected.categories.length ? <Space wrap>{selected.categories.map((item) => <Tag key={item.id}>{item.id}:{item.name}</Tag>)}</Space> : <Tag color="warning">待配置</Tag>, span: 2 }]} /><Alert type="info" showIcon message="数据谱系" description="所有样本均记录来源、条件、seed、Adapter 版本和文件摘要。流程样例不会被标记为正式生成数据。" /></Space>}</Drawer><DatasetBrowser dataset={browserDataset} onClose={() => setBrowserDataset(undefined)} /><AnnotationWorkspace dataset={annotationDataset} onClose={() => setAnnotationDataset(undefined)} onChanged={reload} /></>
}

type LocalResourceKind = 'directory' | 'entrypoint' | 'weight'
type LocalResourceScope = 'model' | 'environment'

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
      const query = new URLSearchParams({ scope, kind })
      if (path) query.set('path', path)
      setListing(await api<LocalResourceListing>(`/api/local-model-resources?${query}`))
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
  weight_path: '',
})

export function RegistryPage({ refresh }: PageProps) {
  const adapters = useResource<Adapter[]>('/api/adapters', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const categoryTemplates = useResource<CategoryTemplate[]>('/api/category-templates', [])
  const [registerOpen, setRegisterOpen] = useState(false)
  const [registering, setRegistering] = useState(false)
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
  const modelCategories = categoriesFromSelection(categoryTemplates.data, categoryTemplateId, 'model', customModelCategories)
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
      result.healthy ? message.success('模型运行环境健康检查通过') : message.warning('模型运行环境当前不可用')
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
      setRegisterOpen(false)
      setDraft(emptyLocalModelDraft())
      setCategoryTemplateId('visdrone')
      setCustomModelCategories([{ id: 0, name: '' }])
      await Promise.all([models.reload(), adapters.reload()])
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
    <Card extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setRegisterOpen(true)}>注册本地检测模型</Button>}>
      <Table rowKey="id" loading={models.loading} dataSource={models.data} scroll={{ x: 1650 }} onRow={(row) => ({ onClick: () => setSelectedModel(row), style: { cursor: 'pointer' } })} columns={[
        { title: '模型', dataIndex: 'name', render: (value, row) => <Space><Typography.Text strong>{value}</Typography.Text>{row.is_demo && <DemoTag />}</Space> },
        { title: '模型族', dataIndex: 'family' },
        { title: '模型架构', dataIndex: 'architecture' },
        { title: 'Backbone', dataIndex: 'backbone' },
        { title: '检测头', dataIndex: 'detector_head' },
        { title: '类别数', dataIndex: 'class_count', render: (value) => value || '—' },
        { title: '训练数据', dataIndex: 'training_dataset' },
        { title: '预训练数据', dataIndex: 'pretrained_dataset' },
        { title: '版本', dataIndex: 'version' },
        { title: '精度', dataIndex: 'precision' },
        { title: '权重', dataIndex: 'weight_path', render: (value) => value ? <Typography.Text ellipsis={{ tooltip: value }} style={{ maxWidth: 180 }}>{String(value).split('/').pop()}</Typography.Text> : '—' },
        { title: '状态', dataIndex: 'status', render: (value) => <StatusTag status={value} /> },
        { title: '运行环境', render: (_, row) => <StatusTag status={adapters.data.find((item) => item.id === row.adapter_id)?.status || (adapters.loading ? 'CHECKING' : 'UNAVAILABLE')} /> },
        { title: '操作', render: (_, row) => <span onClick={(event) => event.stopPropagation()}><Space><Button type="link" onClick={() => health(row.adapter_id)}>健康检查</Button><Popconfirm title="确认删除这个模型？" description="仅删除平台注册记录，不删除模型项目、环境或权重；存在评测引用时会拒绝删除。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => removeModel(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm></Space></span> },
      ]} />
    </Card>
  )
  return (
    <>
      {modelsCard}
      <Drawer
        open={registerOpen}
        width={760}
        title="注册本地检测模型"
        onClose={() => setRegisterOpen(false)}
      >
        <Alert
          type="info"
          showIcon
          message="配置结构化执行命令"
          description="平台按参数数组直接启动进程，不经过 Shell。命令只需在指定位置生成 COCO predictions.json；同一项目的不同模型可以登记不同权重和参数。"
        />
        <Form layout="vertical" className="top-gap">
          <Row gutter={12}>
            <Col span={12}><Form.Item label="模型名称" required><Input value={draft.name} onChange={(event) => setField('name', event.target.value)} maxLength={120} placeholder="例如 YOLOv8m VisDrone" /></Form.Item></Col>
            <Col span={12}><Form.Item label="模型族" required><Input value={draft.family} onChange={(event) => setField('family', event.target.value)} maxLength={80} placeholder="例如 YOLOv8" /></Form.Item></Col>
            <Col span={8}><Form.Item label="模型架构" required><Input value={draft.architecture} onChange={(event) => setField('architecture', event.target.value)} maxLength={80} placeholder="例如 DETR" /></Form.Item></Col>
            <Col span={8}><Form.Item label="Backbone" required><Input value={draft.backbone} onChange={(event) => setField('backbone', event.target.value)} maxLength={80} placeholder="例如 HGNetv2-B2" /></Form.Item></Col>
            <Col span={8}><Form.Item label="检测头" required><Input value={draft.detector_head} onChange={(event) => setField('detector_head', event.target.value)} maxLength={80} placeholder="例如 D-FINE Transformer" /></Form.Item></Col>
            <Col span={12}><Form.Item label="版本"><Input value={draft.version} onChange={(event) => setField('version', event.target.value)} maxLength={40} /></Form.Item></Col>
            <Col span={12}><Form.Item label="精度"><Select value={draft.precision} onChange={(value) => setField('precision', value)} options={[{ value: 'FP16' }, { value: 'FP32' }]} /></Form.Item></Col>
            <Col span={12}><Form.Item label="训练数据" required><Input value={draft.training_dataset} onChange={(event) => setField('training_dataset', event.target.value)} maxLength={120} placeholder="例如 VisDrone2019-DET" /></Form.Item></Col>
            <Col span={12}><Form.Item label="预训练数据" required><Input value={draft.pretrained_dataset} onChange={(event) => setField('pretrained_dataset', event.target.value)} maxLength={120} placeholder="例如 Objects365；没有则填写无" /></Form.Item></Col>
          </Row>
          <Form.Item label={`检测类别（${modelCategories.length} 类）`} required>
            <CategoryConfiguration templates={categoryTemplates.data} templateId={categoryTemplateId} scope="model" customCategories={customModelCategories} onTemplateChange={setCategoryTemplateId} onCustomChange={setCustomModelCategories} />
          </Form.Item>
          <Form.Item label="模型项目目录" required>
            <Space.Compact block>
              <Input readOnly value={draft.project_directory} placeholder="从服务器模型库选择目录" />
              <Button onClick={() => setPicker({ field: 'project_directory', title: '选择模型项目目录', scope: 'model', kind: 'directory', initialPath: draft.project_directory || undefined })}>选择目录</Button>
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
          <Form.Item label="模型权重" required>
            <Space.Compact block>
              <Input readOnly value={draft.weight_path} placeholder="选择模型权重文件" />
              <Button disabled={!draft.project_directory} onClick={() => setPicker({ field: 'weight_path', title: '选择模型权重', scope: 'model', kind: 'weight', initialPath: draft.project_directory || undefined })}>选择权重</Button>
            </Space.Compact>
          </Form.Item>
          <Form.Item
            label="命令参数"
            required
            extra="每行一个参数。参数会原样传给可执行程序，不要输入整段 Shell 命令。"
          >
            <Input.TextArea
              value={draft.command_arguments}
              onChange={(event) => setField('command_arguments', event.target.value)}
              autoSize={{ minRows: 9, maxRows: 18 }}
              placeholder={'tools/evaluate.py\n--weights\n{weight_path}\n--images\n{image_directory}\n--annotations\n{annotation_path}\n--output\n{predictions_path}'}
            />
          </Form.Item>
          <Typography.Paragraph type="secondary">
            可用占位符：<Typography.Text code>{'{weight_path}'}</Typography.Text> <Typography.Text code>{'{image_directory}'}</Typography.Text> <Typography.Text code>{'{annotation_path}'}</Typography.Text> <Typography.Text code>{'{predictions_path}'}</Typography.Text> <Typography.Text code>{'{output_directory}'}</Typography.Text> <Typography.Text code>{'{device}'}</Typography.Text> <Typography.Text code>{'{precision}'}</Typography.Text> <Typography.Text code>{'{request_path}'}</Typography.Text> <Typography.Text code>{'{result_path}'}</Typography.Text>
          </Typography.Paragraph>
          <Space>
            <Button type="primary" loading={registering} disabled={!ready} onClick={register}>注册模型</Button>
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
              { key: 'kind', label: '记录类型', children: selectedModel.is_demo ? <DemoTag /> : <Tag color="purple">本地真实模型</Tag> },
              { key: 'runtime', label: '运行环境', children: <StatusTag status={adapters.data.find((item) => item.id === selectedModel.adapter_id)?.status || (adapters.loading ? 'CHECKING' : 'UNAVAILABLE')} />, span: 2 },
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
  const [datasetIds, setDatasetIds] = useState<string[]>([])
  const [modelIds, setModelIds] = useState<string[]>([])
  const [precision, setPrecision] = useState('FP16')
  const [jobId, setJobId] = useState<string>()
  const [submitting, setSubmitting] = useState(false)
  const selectedModels = models.data.filter((item) => modelIds.includes(item.id))
  const selectedDatasets = datasets.data.filter((item) => datasetIds.includes(item.id))
  const hasRealDetector = selectedModels.some((item) => !item.is_demo)
  useEffect(() => { if (hasRealDetector && precision === 'INT8') setPrecision('FP16') }, [hasRealDetector, precision])
  const count = datasetIds.length * modelIds.length
  const categoryIssues = selectedDatasets.flatMap((dataset) => selectedModels.flatMap((model) => {
    if (!dataset.categories.length || !model.categories.length) return [`${dataset.name} × ${model.name}：类别尚未配置`]
    const datasetNames = new Set(dataset.categories.map((item) => item.name.trim().toLocaleLowerCase()))
    const modelNames = new Set(model.categories.map((item) => item.name.trim().toLocaleLowerCase()))
    const missing = dataset.categories.filter((item) => !modelNames.has(item.name.trim().toLocaleLowerCase())).map((item) => item.name)
    const extra = model.categories.filter((item) => !datasetNames.has(item.name.trim().toLocaleLowerCase())).map((item) => item.name)
    if (!missing.length && !extra.length) return []
    return [`${dataset.name} × ${model.name}：${missing.length ? `模型缺少 ${missing.join('、')}` : ''}${missing.length && extra.length ? '；' : ''}${extra.length ? `模型多出 ${extra.join('、')}` : ''}`]
  }))
  const submit = async () => { setSubmitting(true); try { const plan = await post<{ id: string }>('/api/evaluation-plans', { name: `感知效能评测 ${new Date().toLocaleString('zh-CN')}`, dataset_ids: datasetIds, model_ids: modelIds, seeds: [1001], blur_levels: [0], batch_size: 1, precision, warmup: 0 }); const job = await post<Job>(`/api/evaluation-plans/${plan.id}/runs`); setJobId(job.id); refresh(); message.success('评测矩阵已提交') } catch (error) { message.error((error as Error).message) } finally { setSubmitting(false) } }
  return <Space direction="vertical" size={18} style={{ width: '100%' }}>
    <Alert
      type={hasRealDetector ? 'info' : 'warning'}
      showIcon
      message={hasRealDetector ? '已选择本地真实检测模型' : '当前选择的是参考流程模型'}
      description={hasRealDetector ? '使用模型注册时登记的推理入口、Python 环境和权重执行真实推理，并用 pycocotools 计算 COCO 指标。' : '页面中的 mAP 和时延为确定性流程样例。可在“模型版本”中注册本地检测模型。'}
    />
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={12}><Card title="数据版本"><Select mode="multiple" value={datasetIds} onChange={setDatasetIds} optionFilterProp="label" style={{ width: '100%' }} options={datasets.data.map((item) => ({ value: item.id, label: `${item.name}${item.categories.length ? '' : '（类别待配置）'}`, disabled: !item.frozen || !item.categories.length }))} /><Divider /><Typography.Text type="secondary">真实检测要求数据集已冻结，并具有 COCO 或 VisDrone 目标框标注。</Typography.Text></Card></Col>
      <Col xs={24} xl={12}><Card title="模型版本"><Select mode="multiple" value={modelIds} onChange={setModelIds} style={{ width: '100%' }} options={models.data.map((item) => ({ value: item.id, label: `${item.name}${item.is_demo ? '（流程样例）' : '（真实推理）'}${item.categories.length ? '' : '（类别待配置）'}`, disabled: item.status === 'UNAVAILABLE' || !item.categories.length }))} /><Divider />{hasRealDetector ? <Tag color="purple">本地真实模型</Tag> : <Space><DemoTag /><Typography.Text type="secondary">可选择已注册的本地检测模型。</Typography.Text></Space>}</Card></Col>
      <Col span={24}><Card title="标准化推理协议"><Form layout="vertical"><Form.Item label="精度模式"><Segmented block value={precision} onChange={(value) => setPrecision(String(value))} options={[{ value: 'FP32', label: 'FP32' }, { value: 'FP16', label: 'FP16' }, { value: 'INT8', label: 'INT8', disabled: hasRealDetector }]} /></Form.Item></Form></Card></Col>
    </Row>
    {categoryIssues.length > 0 && <Alert type="error" showIcon message="类别不一致，无法启动评测" description={<Space direction="vertical" size={2}>{categoryIssues.map((item) => <Typography.Text key={item}>{item}</Typography.Text>)}</Space>} />}
    <Card className="matrix-preview"><Row align="middle" gutter={[18, 18]}><Col flex="auto"><Typography.Title level={4}>组合矩阵预览</Typography.Title><Typography.Text type="secondary">{datasetIds.length} 数据版本 × {modelIds.length} 模型</Typography.Text></Col><Col><Statistic value={count} suffix="次运行" /></Col><Col><Button type="primary" size="large" icon={<PlayCircleOutlined />} disabled={!count || categoryIssues.length > 0} loading={submitting} onClick={submit}>启动批量评测</Button></Col></Row></Card>
    {jobId && <JobProgress jobId={jobId} onFinish={() => refresh()} />}
    {jobId && <Card><Button type="primary" onClick={() => navigate('explorer')}>打开效能探索器</Button></Card>}
  </Space>
}

export function ExplorerPage({ dark }: PageProps) {
  const [data, setData] = useState<ResultResponse>({ count: 0, groups: [], runs: [], dimensions: { scenes: [], conditions: [], resolutions: [], models: [] } })
  const [loading, setLoading] = useState(true)
  const [scene, setScene] = useState<string>()
  const [condition, setCondition] = useState<string>()
  const [resolution, setResolution] = useState<string>()
  const [selected, setSelected] = useState<ResultGroup>()
  const load = async () => { setLoading(true); try { const query = new URLSearchParams(); if (scene) query.set('scene', scene); if (condition) query.set('condition', condition); if (resolution) query.set('resolution', resolution); setData(await api<ResultResponse>(`/api/results?${query}`)) } catch (error) { message.error((error as Error).message) } finally { setLoading(false) } }
  useEffect(() => { load() }, [])
  const robustness = useMemo(() => ({ tooltip: { trigger: 'axis' }, legend: { textStyle: { color: dark ? '#c5d0de' : '#4b5565' } }, grid: { left: 55, right: 20, top: 48, bottom: 38 }, xAxis: { type: 'category', name: '条件数据集', data: [...new Set(data.groups.map((item) => item.dataset_name.replace('无人机航拍 · ', '')))], axisLabel: { color: dark ? '#9dafc7' : '#687386' } }, yAxis: { type: 'value', min: 0.45, max: 0.9, axisLabel: { formatter: (value: number) => `${Math.round(value * 100)}%`, color: dark ? '#9dafc7' : '#687386' }, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } } }, series: [...new Set(data.groups.map((item) => item.model_name))].slice(0, 4).map((name) => ({ name: name.split('·')[0], type: 'line', smooth: true, data: [...new Set(data.groups.map((item) => item.dataset_name))].map((datasetName) => data.groups.find((item) => item.model_name === name && item.dataset_name === datasetName)?.map_mean ?? null) })) }), [data.groups, dark])
  const natureTag = (group: ResultGroup) => group.is_demo ? <DemoTag /> : group.is_official ? <Tag color="green">真实模型 · 正式结果</Tag> : <Tag color="purple">真实模型 · 实验性结果</Tag>
  const containsRealResults = data.groups.some((item) => !item.is_demo)
  return <Space direction="vertical" size={18} style={{ width: '100%' }}>
    <Card className="filter-card"><Row gutter={[12, 12]} align="bottom"><Col xs={24} md={5}><Typography.Text type="secondary">场景域</Typography.Text><Select allowClear value={scene} onChange={setScene} placeholder="全部场景" style={{ width: '100%', marginTop: 6 }} options={['无人机航拍', '城市驾驶', '卫星遥感'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">环境条件</Typography.Text><Select allowClear value={condition} onChange={setCondition} placeholder="全部条件" style={{ width: '100%', marginTop: 6 }} options={['晴朗', '雾', '雨', '夜间'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">分辨率</Typography.Text><Select allowClear value={resolution} onChange={setResolution} placeholder="保留为分组维度" style={{ width: '100%', marginTop: 6 }} options={['1920×1080', '1280×720', '640×640'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">硬件范围</Typography.Text><Select value="current" style={{ width: '100%', marginTop: 6 }} options={[{ value: 'current', label: '相同参考设备' }]} /></Col><Col xs={24} md={4}><Button block type="primary" icon={<EyeOutlined />} onClick={load} loading={loading}>查询效能</Button></Col></Row></Card>
    <Alert type="info" showIcon message={`找到 ${data.count} 次 seed 运行，聚合为 ${data.groups.length} 个可比单元`} description={!resolution ? '分辨率未限定，结果按分辨率分别展示，不进行隐式平均。' : '当前为精确分辨率查询。'} action={containsRealResults ? <Tag color="purple">包含真实模型结果</Tag> : <DemoTag />} />
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={12}><Card title="模型排行榜"><Table size="small" loading={loading} rowKey={(row) => `${row.dataset_id}-${row.model_id}`} dataSource={data.groups} pagination={{ pageSize: 6 }} onRow={(record) => ({ onClick: () => setSelected(record), style: { cursor: 'pointer' } })} columns={[{ title: '模型', dataIndex: 'model_name', render: (value) => <Typography.Text strong>{value.split('·')[0]}</Typography.Text> }, { title: '条件单元', dataIndex: 'dataset_name' }, { title: '分辨率', dataIndex: 'resolution' }, { title: 'mAP', dataIndex: 'map_mean', sorter: (a, b) => a.map_mean - b.map_mean, render: (value) => <Typography.Text strong className="map-value">{percent(value)}</Typography.Text> }, { title: 'σ', dataIndex: 'map_std', render: (value) => percent(value) }, { title: '时延', dataIndex: 'latency_mean', render: (value) => `${value} ms` }]} /></Card></Col>
      <Col xs={24} xl={12}><Card title="mAP — 时延 Pareto"><ParetoChart groups={data.groups} dark={dark} height={320} /></Card></Col>
      <Col xs={24} xl={12}><Card title="条件鲁棒性"><ReactECharts option={robustness} style={{ height: 300 }} /></Card></Col>
      <Col xs={24} xl={12}><Card title="PR 曲线"><PRChart groups={data.groups} dark={dark} height={300} /></Card></Col>
    </Row>
    <Drawer open={Boolean(selected)} onClose={() => setSelected(undefined)} title="效能单元详情" width={620}>
      {selected && <Space direction="vertical" size={18} style={{ width: '100%' }}><Space>{natureTag(selected)}<StatusTag status="VERIFIED" /></Space><Row gutter={12}><Col span={8}><Card><Statistic title="mAP" value={selected.map_mean * 100} precision={1} suffix="%" /></Card></Col><Col span={8}><Card><Statistic title="seed σ" value={selected.map_std * 100} precision={2} suffix="%" /></Card></Col><Col span={8}><Card><Statistic title="时延" value={selected.latency_mean} precision={1} suffix="ms" /></Card></Col></Row><Descriptions bordered column={1} items={[{ key: 'model', label: '模型', children: selected.model_name }, { key: 'backbone', label: 'Backbone', children: selected.backbone }, { key: 'dataset', label: '数据版本', children: selected.dataset_name }, { key: 'scene', label: '场景', children: `${selected.scene_domain} / ${selected.weather}` }, { key: 'resolution', label: '分辨率', children: selected.resolution }, { key: 'seeds', label: '重复次数', children: `${selected.seed_count} 个 seed` }, { key: 'official', label: '结果性质', children: natureTag(selected) }]} /><PRChart groups={[selected]} dark={dark} /></Space>}
    </Drawer>
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
  return <Space direction="vertical" size={18} style={{ width: '100%' }}><Alert type="success" showIcon message="工作区隔离已启用" description="平台不修改 base、.condarc、.bashrc 或已有 conda 环境。所有平台环境、缓存、数据库和 Artifact 均位于项目目录。" action={<Button icon={<ReloadOutlined />} onClick={reload}>重新检查</Button>} /><Row gutter={[16, 16]}><Col xs={24} md={8}><Card><Statistic title="隔离模式" value="Workspace" prefix={<SafetyCertificateOutlined />} /><Typography.Text type="secondary">写入工作区外：{data.isolation.writes_outside_workspace ? '是' : '否'}</Typography.Text></Card></Col><Col xs={24} md={8}><Card><Statistic title="GPU 状态" value={data.gpu.available ? '可用' : '当前受限'} prefix={<ThunderboltOutlined />} /><Typography.Text type="secondary">失败时不会尝试修改驱动</Typography.Text></Card></Col><Col xs={24} md={8}><Card><Statistic title="磁盘使用" value={diskPercent} suffix="%" /><Progress percent={diskPercent} showInfo={false} /><Typography.Text type="secondary">剩余 {formatBytes(data.disk.free)}</Typography.Text></Card></Col></Row><Card title="写入边界"><Descriptions bordered column={1} items={[{ key: 'runtime', label: '平台环境与缓存', children: <Typography.Text code copyable>{data.isolation.runtime_dir}</Typography.Text> }, { key: 'data', label: '数据库与 Artifact', children: <Typography.Text code copyable>{data.isolation.data_dir}</Typography.Text> }, { key: 'shell', label: 'Shell 配置修改', children: data.isolation.shell_configuration_modified ? <Tag color="error">有</Tag> : <Tag color="success">无</Tag> }, { key: 'conda', label: '外部环境策略', children: <Tag icon={<LockOutlined />} color="success">只读调用</Tag> }]} /></Card><Card title={`发现 ${data.conda.envs.length} 个 conda 环境`} extra={<Typography.Text type="secondary">仅读取 conda-meta/history 指纹</Typography.Text>}><Table rowKey="prefix" pagination={false} dataSource={data.conda.envs} columns={[{ title: '名称', dataIndex: 'name', render: (value) => <Typography.Text strong>{value}</Typography.Text> }, { title: '路径', dataIndex: 'prefix', render: (value) => <Typography.Text code copyable>{value}</Typography.Text> }, { title: '策略', render: () => <Tag icon={<LockOutlined />} color="success">external_read_only</Tag> }, { title: '环境指纹', dataIndex: 'fingerprint', render: (value) => value ? <Typography.Text code>{value}</Typography.Text> : '未找到' }, { title: '状态', dataIndex: 'exists', render: (value) => <StatusTag status={value ? 'HEALTHY' : 'UNAVAILABLE'} /> }]} /></Card>{!data.gpu.available && <Alert type="warning" showIcon message="当前进程无法访问 NVIDIA 驱动" description={data.gpu.error || '宿主机预检失败。Web 和 CPU 功能仍可使用，GPU Adapter 会被安全阻止。'} />}</Space>
}

export function PresentationPage({ onExit }: { dark: boolean; onExit: () => void }) {
  const overviewResource = useResource<Overview | undefined>('/api/overview', undefined)
  const resultResource = useResource<ResultResponse>('/api/results?scene=无人机航拍', { count: 0, groups: [], runs: [], dimensions: { scenes: [], conditions: [], resolutions: [], models: [] } })
  const datasetResource = useResource<Dataset[]>('/api/datasets', [])
  const overview = overviewResource.data
  return <div className="presentation-shell"><div className="presentation-top"><div className="presentation-brand"><ExperimentOutlined /> 视觉感知效能评估平台 <Tag color="cyan">演示模式 · 只读</Tag></div><Button ghost icon={<FullscreenExitOutlined />} onClick={onExit}>退出演示</Button></div><Carousel autoplay autoplaySpeed={8000} dots className="presentation-carousel"><section className="presentation-slide"><div className="slide-kicker">GENERATIVE PERCEPTION EVALUATION</div><Typography.Title>从条件数据到感知效能结论</Typography.Title><Typography.Paragraph>统一纳管图像来源、真值、检测模型、运行环境与评测指标</Typography.Paragraph><div className="flow-band">{['场景与传感器条件', '生成 / 仿真 / 真实导入', '真值版本冻结', '多模型批量评测', '效能查询与对比'].map((item, index) => <div className="flow-node" key={item}><span>0{index + 1}</span><strong>{item}</strong>{index < 4 && <ArrowRightOutlined />}</div>)}</div><Row gutter={20} className="slide-metrics"><Col span={6}><Statistic title="数据版本" value={overview?.counts.datasets || 0} /></Col><Col span={6}><Statistic title="模型版本" value={overview?.counts.models || 0} /></Col><Col span={6}><Statistic title="完成运行" value={overview?.counts.completed || 0} /></Col><Col span={6}><Statistic title="环境策略" value="只读隔离" /></Col></Row></section><section className="presentation-slide"><div className="slide-kicker">DATA PROVENANCE</div><Typography.Title>多源图像统一纳管</Typography.Title><Typography.Paragraph>当前生成模型尚未完成，回放数据与条件算子用于验证软件闭环，并始终显著标识。</Typography.Paragraph><div className="presentation-gallery"><Gallery images={datasetResource.data.flatMap((item) => item.preview_images.slice(0, 2)).slice(0, 8)} height={178} /></div><Space size="large"><Tag color="gold">REPLAY_FIXTURE</Tag><Tag color="cyan">REAL_TRANSFORMED</Tag><Tag color="green">VERIFIED GROUND TRUTH</Tag></Space></section><section className="presentation-slide"><div className="slide-kicker">DETECTION QUALITY</div><Typography.Title>检测结果与 PR 曲线</Typography.Title><Row gutter={24} align="middle"><Col span={11}><Gallery images={datasetResource.data[0]?.preview_images.slice(0, 2) || []} height={235} /></Col><Col span={13}><PRChart groups={resultResource.data.groups} dark height={360} /></Col></Row><Alert type="warning" showIcon message="当前为流程样例指标" description="真实检测 Adapter 接入后沿用相同数据、指标和展示协议。" /></section><section className="presentation-slide"><div className="slide-kicker">PERFORMANCE EXPLORER</div><Typography.Title>精度、时延与鲁棒性联合决策</Typography.Title><Row gutter={24}><Col span={15}><ParetoChart groups={resultResource.data.groups} dark height={460} /></Col><Col span={9}><div className="rank-panel">{resultResource.data.groups.slice(0, 5).map((item, index) => <div className="rank-row" key={`${item.dataset_id}-${item.model_id}`}><span className="rank-index">0{index + 1}</span><div><strong>{item.model_name.split('·')[0]}</strong><small>{item.dataset_name}</small></div><b>{percent(item.map_mean)}</b><em>{item.latency_mean} ms</em></div>)}</div></Col></Row></section></Carousel></div>
}
