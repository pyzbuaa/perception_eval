import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  Tabs,
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
  LockOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  RobotOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { api, formatBytes, percent, post } from './api'
import { DemoTag, Gallery, JobProgress, ParetoChart, PRChart, StatusTag } from './components'
import type { Adapter, Dataset, DatasetSamplePage, EnvironmentStatus, Job, ModelVersion, Overview, ResultGroup, ResultResponse } from './types'

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
                children: <div><Typography.Text strong>{item.name}</Typography.Text><br /><Typography.Text type="secondary">{['数据来源统一接入', 'COCO/YOLO 真值状态', '模型矩阵与三 seed', '条件查询与效能图表'][index]}</Typography.Text></div>,
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
  { id: 'local-import', source: 'REAL', icon: <ImportOutlined />, title: '本地数据', status: '正式可用', description: '导入本地 PNG/JPEG/WebP/SVG 及可选 COCO/YOLO 标注。', recommended: true },
  { id: 'adapter_replay', source: 'REPLAY_FIXTURE', icon: <PlayCircleOutlined />, title: '测试回放', status: '接口已通过', description: '固定样例验证生成任务、标注与评测闭环。', recommended: true },
  { id: 'adapter_condition', source: 'REAL_TRANSFORMED', icon: <CloudOutlined />, title: '条件退化', status: '正式可用', description: '读取已导入的真实图像，产生模糊、雾化和噪声条件。' },
  { id: 'adapter_basegen', source: 'GENERATIVE', icon: <RobotOutlined />, title: 'Z-Image-Turbo', status: '实验性', description: '通过独立 gen 环境调用 BaseGen 生成未标注感知图像。' },
  { id: 'airsim-future', source: 'SIMULATOR', icon: <CodeOutlined />, title: 'AirSim / UE', status: '计划接入', description: '通过独立 RPC Adapter 采集图像与真值。', disabled: true },
]

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
  const [inputDatasetId, setInputDatasetId] = useState<string>()
  const [localDirectory, setLocalDirectory] = useState('')
  const [annotationPath, setAnnotationPath] = useState('')
  const [jobId, setJobId] = useState<string>()
  const [finishedJob, setFinishedJob] = useState<Job>()
  const [submitting, setSubmitting] = useState(false)
  const datasets = useResource<Dataset[]>('/api/datasets', [])

  const isBaseGen = source.id === 'adapter_basegen'
  const domainOptions = isBaseGen
    ? ['无人机航拍', '城市驾驶', '野外自动驾驶']
    : ['无人机航拍', '卫星遥感', '城市驾驶']
  const resolutionOptions = isBaseGen
    ? ['1024×1024', '1024×576']
    : ['1920×1080', '1280×720', '640×640']
  const weatherOptions = isBaseGen
    ? ['晴朗', '阴天', '雾', '雨', '雪', '夜间']
    : ['晴朗', '雾', '雨', '夜间']
  const combinationCount = isBaseGen ? 1 : seeds.length
  const selectSource = (item: typeof sourceCards[number]) => {
    if (item.disabled) return
    setSource(item)
    if (item.id === 'adapter_basegen') {
      const nextDomain = domain === '卫星遥感' ? '无人机航拍' : domain
      setDomain(nextDomain)
      setResolution(nextDomain === '无人机航拍' ? '1024×1024' : '1024×576')
    } else {
      if (domain === '野外自动驾驶') setDomain('无人机航拍')
      if (weather === '阴天' || weather === '雪') setWeather('晴朗')
      setResolution('1920×1080')
    }
  }
  const selectDomain = (value: string) => {
    setDomain(value)
    if (isBaseGen) setResolution(value === '无人机航拍' ? '1024×1024' : '1024×576')
  }
  const submit = async () => {
    setSubmitting(true)
    try {
      const job = source.id === 'local-import'
        ? await post<Job>('/api/datasets/import', {
            name: `${domain} · 本地导入 · ${new Date().toLocaleDateString('zh-CN')}`,
            directory: localDirectory,
            annotation_path: annotationPath || null,
            scene_domain: domain,
          })
        : await post<Job>('/api/acquisition-jobs', {
            name: `${domain} · ${source.title} · ${new Date().toLocaleDateString('zh-CN')}`,
            adapter_id: source.id,
            source_type: source.source,
            sample_count: samples,
            seeds: isBaseGen ? [generatorSeed] : seeds,
            conditions: {
              scene: { domain, weather },
              sensor: isBaseGen ? { resolution } : { resolution, motion_blur: blur, fog_density: weather === '雾' ? 0.4 : 0 },
            },
            model_parameters: isBaseGen ? { steps: generatorSteps, guidance_scale: 0, device_policy: devicePolicy, local_files_only: false } : {},
            input_dataset_id: source.id === 'adapter_condition' ? inputDatasetId : null,
          })
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
        <Row gutter={[16, 16]}>{sourceCards.map((item) => <Col xs={24} md={12} xl={6} key={item.id}><Card hoverable={!item.disabled} className={`source-card ${source.id === item.id ? 'source-selected' : ''} ${item.disabled ? 'source-disabled' : ''}`} onClick={() => selectSource(item)}><div className="source-icon">{item.icon}</div><Space><Typography.Title level={4}>{item.title}</Typography.Title>{item.recommended && <Tag color="cyan">推荐</Tag>}</Space><StatusTag status={item.status === '正式可用' ? 'READY' : item.status === '接口已通过' ? 'CONTRACT_OK' : item.status === '实验性' ? 'EXPERIMENTAL' : 'REGISTERED'} /><Typography.Paragraph type="secondary">{item.description}</Typography.Paragraph></Card></Col>)}</Row>
        <div className="wizard-actions"><Button type="primary" onClick={() => setStep(1)}>下一步：配置条件 <ArrowRightOutlined /></Button></div>
      </Card>}
      {step === 1 && <Row gutter={16}>
        <Col xs={24} xl={12}><Card title={source.id === 'local-import' ? '本地数据路径' : '场景条件'}><Form layout="vertical">{source.id === 'local-import' && <><Form.Item label="图像目录" required><Input value={localDirectory} onChange={(event) => setLocalDirectory(event.target.value)} placeholder="例如 /data/aerial/images" prefix={<DatabaseOutlined />} /></Form.Item><Form.Item label="标注文件（可选）"><Input value={annotationPath} onChange={(event) => setAnnotationPath(event.target.value)} placeholder="COCO JSON 或 YOLO 标签入口" /></Form.Item></>}<Form.Item label="场景域"><Select value={domain} onChange={selectDomain} options={domainOptions.map((value) => ({ value }))} /></Form.Item>{source.id !== 'local-import' && <><Form.Item label="天气 / 环境"><Segmented block value={weather} onChange={(value) => setWeather(String(value))} options={weatherOptions} /></Form.Item><Form.Item label="输出数量"><InputNumber value={samples} onChange={(value) => setSamples(value || 1)} min={1} max={1000} addonAfter="张" style={{ width: '100%' }} /></Form.Item></>}</Form></Card></Col>
        <Col xs={24} xl={12}>{source.id === 'local-import' ? <Card title="导入校验"><Timeline items={[{ color: 'blue', children: '扫描 PNG、JPEG、WebP 和 SVG 图像' }, { color: 'blue', children: '复制到内容受控的 Artifact 目录' }, { color: 'blue', children: '标注作为候选真值导入并等待校核' }, { color: 'green', children: '原始目录保持不变，平台不会原地修改图像' }]} /><Alert type="info" showIcon message="本地目录只读" description="平台将图像复制到工作区，不在源目录创建缓存或转换文件。" /></Card> : <Card title={isBaseGen ? '生成参数' : '传感器与成像条件'}><Form layout="vertical"><Form.Item label="图像分辨率"><Select value={resolution} onChange={setResolution} options={resolutionOptions.map((value) => ({ value }))} /></Form.Item>{isBaseGen ? <><Form.Item label="起始随机种子"><InputNumber value={generatorSeed} onChange={(value) => setGeneratorSeed(value || 0)} min={0} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="推理步数"><InputNumber value={generatorSteps} onChange={(value) => setGeneratorSteps(value || 1)} min={1} max={100} precision={0} style={{ width: '100%' }} /></Form.Item><Form.Item label="设备策略"><Select value={devicePolicy} onChange={setDevicePolicy} options={[{ value: 'cuda', label: '全量 CUDA（推荐）' }, { value: 'cpu-offload', label: 'CPU Offload（节省显存）' }]} /></Form.Item></> : <><Form.Item label={`运动模糊强度 ${blur.toFixed(1)}`}><Slider value={blur} onChange={setBlur} min={0} max={1} step={0.1} marks={{ 0: '清洁', 0.5: '中等', 1: '严重' }} /></Form.Item><Form.Item label="固定随机种子"><Checkbox.Group options={[1001, 1002, 1003, 1004].map((value) => ({ label: value, value }))} value={seeds} onChange={(values) => setSeeds(values as number[])} /></Form.Item></>}</Form></Card>}</Col>
        {source.id === 'adapter_condition' && <Col span={24}><Card title="选择退化输入数据集"><Select value={inputDatasetId} onChange={setInputDatasetId} style={{ width: '100%' }} placeholder="选择已导入的 PNG/JPEG/WebP 数据集" options={datasets.data.filter((item) => item.source_type === 'REAL').map((item) => ({ value: item.id, label: `${item.name} · ${item.sample_count} 张` }))} /><Typography.Paragraph type="secondary" style={{ marginTop: 10, marginBottom: 0 }}>条件算子保持几何位置不变，但输出真值仍需抽查后冻结。</Typography.Paragraph></Card></Col>}
        <Col span={24}><div className="wizard-actions"><Button onClick={() => setStep(0)}>上一步</Button><Button type="primary" disabled={source.id === 'local-import' ? !localDirectory : isBaseGen ? generatorSeed < 0 || generatorSteps < 1 : source.id === 'adapter_condition' ? !inputDatasetId || !seeds.length : !seeds.length} onClick={() => setStep(2)}>下一步：组合预览</Button></div></Col>
      </Row>}
      {step === 2 && <Card title="提交前确认" extra={isBaseGen ? <Tag color="purple">Z-Image-Turbo</Tag> : <DemoTag />}>
        <Row gutter={[16, 16]}><Col xs={24} md={8}><Statistic title={source.id === 'local-import' ? '导入任务' : '配置单元'} value={source.id === 'local-import' ? 1 : combinationCount} suffix="个" /></Col><Col xs={24} md={8}><Statistic title="输出样本" value={source.id === 'local-import' ? '目录内图像' : samples} suffix={source.id === 'local-import' ? undefined : '张'} /></Col><Col xs={24} md={8}><Statistic title="真值入口" value={source.id === 'local-import' ? (annotationPath ? '已提供' : '未提供') : isBaseGen ? '未标注' : '候选框'} /></Col></Row><Divider />
        <Descriptions column={{ xs: 1, md: 2 }} bordered size="small" items={[{ key: 'source', label: '来源', children: source.title }, { key: 'scene', label: '场景', children: source.id === 'local-import' ? domain : `${domain} / ${weather}` }, { key: 'sensor', label: source.id === 'local-import' ? '输入目录' : isBaseGen ? '生成参数' : '成像条件', children: source.id === 'local-import' ? localDirectory : isBaseGen ? `${resolution} / ${generatorSteps} 步 / ${devicePolicy}` : `${resolution} / 模糊 ${blur}` }, { key: 'seed', label: '随机种子', children: source.id === 'local-import' ? '不适用' : isBaseGen ? `${generatorSeed} 起连续 ${samples} 个` : seeds.join(', ') }, { key: 'truth', label: '真值策略', children: source.id === 'local-import' ? (annotationPath ? '导入后作为候选真值' : '未提供') : isBaseGen ? '未标注，需另行标注后评测' : '候选框，完成后需校核冻结' }, { key: 'official', label: '结果性质', children: source.id === 'local-import' ? <Tag color="blue">真实采集数据</Tag> : isBaseGen ? <Tag color="purple">真实模型生成</Tag> : <DemoTag /> }]} />
        {source.id !== 'local-import' && <Alert className="inline-alert" type={isBaseGen ? 'info' : 'warning'} showIcon message={isBaseGen ? '本任务将调用 BaseGen 真实生成图像' : source.id === 'adapter_condition' ? '本任务将创建真实图像的条件退化版本' : '本任务使用流程验证 Adapter'} description={isBaseGen ? '模型在独立 gen 环境中运行；纯文本生成不提供目标框等真值。' : source.id === 'adapter_condition' ? '输出记录原数据集和退化参数，正式冻结前仍需抽查真值。' : '输出会保留完整数据谱系，但不能作为生成模型能力结论。'} />}
        <div className="wizard-actions"><Button onClick={() => setStep(1)}>上一步</Button><Button type="primary" loading={submitting} icon={<PlayCircleOutlined />} onClick={submit}>提交构建任务</Button></div>
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

  return (
    <Drawer open={Boolean(dataset)} width="90vw" title={dataset ? `${dataset.name} · 全部样本` : '全部样本'} onClose={onClose} extra={<Tag>{items.length} / {total || declaredCount}</Tag>}>
      <div className="dataset-browser" onScroll={scroll}>
        {error && <Alert type="error" showIcon message="样本加载失败" description={error} />}
        {!loading && !error && total !== declaredCount && <Alert type="warning" showIcon message="登记样本数与磁盘文件数不一致" description={`登记 ${declaredCount} 个，当前 Artifact 目录找到 ${total} 个可浏览图片。`} />}
        {!loading && !error && total === 0 ? <Empty description="该数据集没有可浏览的图片文件" /> : (
          <Image.PreviewGroup>
            <div className="dataset-browser-grid">
              {items.map((item) => <Image key={item.url} src={item.url} alt={item.name} loading="lazy" preview={{ mask: item.name }} />)}
            </div>
          </Image.PreviewGroup>
        )}
        {loading && <div className="dataset-browser-loading"><Spin /><Typography.Text type="secondary">正在加载图片…</Typography.Text></div>}
        {!loading && total > 0 && items.length >= total && <div className="dataset-browser-end">已加载全部 {total} 张图片</div>}
      </div>
    </Drawer>
  )
}

export function DatasetsPage(_: PageProps) {
  const { data, loading, reload } = useResource<Dataset[]>('/api/datasets', [])
  const [selected, setSelected] = useState<Dataset>()
  const [browserDataset, setBrowserDataset] = useState<Dataset>()
  const freeze = async (dataset: Dataset) => {
    try { await post(`/api/datasets/${dataset.id}/freeze`); message.success('数据版本已冻结'); reload() } catch (error) { message.error((error as Error).message) }
  }
  const remove = async (dataset: Dataset) => {
    try {
      await api(`/api/datasets/${dataset.id}`, { method: 'DELETE' })
      if (selected?.id === dataset.id) setSelected(undefined)
      if (browserDataset?.id === dataset.id) setBrowserDataset(undefined)
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
    { title: '操作', render: (_, row) => <Space><Button type="link" onClick={() => setSelected(row)}>查看</Button><Button type="link" icon={<FileImageOutlined />} onClick={() => setBrowserDataset(row)}>浏览全部</Button>{!row.frozen && <Button type="link" onClick={() => freeze(row)}>冻结</Button>}{!row.frozen && <Popconfirm title="确认删除这个数据集？" description={`${row.name} · ${row.sample_count} 个样本将移入回收站。`} okText="移入回收站" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => remove(row)}><Button danger type="link" icon={<DeleteOutlined />}>删除</Button></Popconfirm>}</Space> },
  ]} /></Card><Drawer open={Boolean(selected)} width={760} title={selected?.name} onClose={() => setSelected(undefined)}>{selected && <Space direction="vertical" size={18} style={{ width: '100%' }}><Gallery images={selected.preview_images} height={140} /><Button block icon={<FileImageOutlined />} onClick={() => { setSelected(undefined); setBrowserDataset(selected) }}>浏览全部图片</Button><Descriptions bordered column={2} items={[{ key: 'source', label: '来源', children: selected.source_type }, { key: 'scene', label: '场景', children: selected.scene_domain }, { key: 'weather', label: '天气', children: selected.weather }, { key: 'resolution', label: '分辨率', children: selected.resolution }, { key: 'truth', label: '真值', children: <StatusTag status={selected.annotation_status} /> }, { key: 'frozen', label: '不可变', children: selected.frozen ? '是' : '否' }]} /><Alert type="info" showIcon message="数据谱系" description="所有样本均记录来源、条件、seed、Adapter 版本和文件摘要。流程样例不会被标记为正式生成数据。" /></Space>}</Drawer><DatasetBrowser dataset={browserDataset} onClose={() => setBrowserDataset(undefined)} /></>
}

export function RegistryPage(_: PageProps) {
  const adapters = useResource<Adapter[]>('/api/adapters', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const health = async (id: string) => { try { const result = await post<{ healthy: boolean }>(`/api/adapters/${id}/health-check`); result.healthy ? message.success('适配器健康检查通过') : message.warning('适配器当前不可用'); adapters.reload() } catch (error) { message.error((error as Error).message) } }
  return <Tabs items={[{ key: 'adapters', label: `适配器 ${adapters.data.length}`, children: <Card><Alert type="success" showIcon message="只读执行策略已启用" description="平台不向外部 conda 环境安装依赖；缺少依赖时应创建工作区克隆。" /><Table className="top-gap" rowKey="id" loading={adapters.loading} dataSource={adapters.data} columns={[{ title: '适配器', dataIndex: 'name', render: (value, row) => <Space direction="vertical" size={1}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.description}</Typography.Text></Space> }, { title: '类型', dataIndex: 'kind', render: (value) => <Tag>{value}</Tag> }, { title: '成熟度', dataIndex: 'maturity', render: (value) => <StatusTag status={value} /> }, { title: '运行方式', dataIndex: 'runtime_kind' }, { title: '环境策略', dataIndex: 'policy', render: () => <Tag icon={<SafetyCertificateOutlined />} color="success">只读</Tag> }, { title: '状态', dataIndex: 'status', render: (value) => <StatusTag status={value} /> }, { title: '操作', render: (_, row) => <Button type="link" onClick={() => health(row.id)}>健康检查</Button> }]} /></Card> }, { key: 'models', label: `模型版本 ${models.data.length}`, children: <Card><Table rowKey="id" loading={models.loading} dataSource={models.data} columns={[{ title: '模型', dataIndex: 'name', render: (value, row) => <Space><Typography.Text strong>{value}</Typography.Text>{row.is_demo && <DemoTag />}</Space> }, { title: '模型族', dataIndex: 'family' }, { title: 'Backbone', dataIndex: 'backbone' }, { title: '版本', dataIndex: 'version' }, { title: '精度', dataIndex: 'precision' }, { title: '状态', dataIndex: 'status', render: (value) => <StatusTag status={value} /> }]} /></Card> }]} />
}

export function EvaluationPage({ navigate, refresh }: PageProps) {
  const datasets = useResource<Dataset[]>('/api/datasets', [])
  const models = useResource<ModelVersion[]>('/api/models', [])
  const [datasetIds, setDatasetIds] = useState<string[]>([])
  const [modelIds, setModelIds] = useState<string[]>([])
  const [seeds, setSeeds] = useState([1001, 1002, 1003])
  const [blurLevels, setBlurLevels] = useState([0, 0.3, 0.5])
  const [precision, setPrecision] = useState('FP16')
  const [warmup, setWarmup] = useState(20)
  const [jobId, setJobId] = useState<string>()
  const [submitting, setSubmitting] = useState(false)
  useEffect(() => { if (!datasetIds.length && datasets.data.length) setDatasetIds(datasets.data.filter((item) => item.frozen).slice(0, 2).map((item) => item.id)) }, [datasets.data])
  useEffect(() => { if (!modelIds.length && models.data.length) setModelIds(models.data.slice(0, 2).map((item) => item.id)) }, [models.data])
  const count = datasetIds.length * modelIds.length * seeds.length * blurLevels.length
  const submit = async () => { setSubmitting(true); try { const plan = await post<{ id: string }>('/api/evaluation-plans', { name: `感知效能评测 ${new Date().toLocaleString('zh-CN')}`, dataset_ids: datasetIds, model_ids: modelIds, seeds, blur_levels: blurLevels, batch_size: 1, precision, warmup }); const job = await post<Job>(`/api/evaluation-plans/${plan.id}/runs`); setJobId(job.id); refresh(); message.success('评测矩阵已提交') } catch (error) { message.error((error as Error).message) } finally { setSubmitting(false) } }
  return <Space direction="vertical" size={18} style={{ width: '100%' }}><Alert type="warning" showIcon message="当前参考检测器只验证软件评测链路" description="页面中的 mAP 和时延均为确定性流程样例，接入真实权重后才可升级为正式结果。" /><Row gutter={[16, 16]}><Col xs={24} xl={12}><Card title="数据版本"><Select mode="multiple" value={datasetIds} onChange={setDatasetIds} optionFilterProp="label" style={{ width: '100%' }} options={datasets.data.map((item) => ({ value: item.id, label: item.name, disabled: !item.frozen }))} /><Divider /><Typography.Text type="secondary">仅允许选择已校核并冻结的数据版本。</Typography.Text></Card></Col><Col xs={24} xl={12}><Card title="模型版本"><Select mode="multiple" value={modelIds} onChange={setModelIds} style={{ width: '100%' }} options={models.data.map((item) => ({ value: item.id, label: item.name }))} /><Divider /><Space><DemoTag /><Typography.Text type="secondary">替换为真实 Adapter 后界面不变。</Typography.Text></Space></Card></Col><Col xs={24} xl={12}><Card title="条件与随机重复"><Form layout="vertical"><Form.Item label="模糊强度网格"><Checkbox.Group value={blurLevels} onChange={(values) => setBlurLevels(values as number[])} options={[0, 0.1, 0.3, 0.5].map((value) => ({ value, label: value === 0 ? '0 清洁' : value }))} /></Form.Item><Form.Item label="固定 seed"><Checkbox.Group value={seeds} onChange={(values) => setSeeds(values as number[])} options={[1001, 1002, 1003, 1004].map((value) => ({ value, label: value }))} /></Form.Item></Form></Card></Col><Col xs={24} xl={12}><Card title="标准化推理协议"><Form layout="vertical"><Form.Item label="精度模式"><Segmented block value={precision} onChange={(value) => setPrecision(String(value))} options={['FP32', 'FP16', 'INT8']} /></Form.Item><Form.Item label="预热次数"><InputNumber value={warmup} onChange={(value) => setWarmup(value || 0)} min={0} max={200} style={{ width: '100%' }} addonAfter="次" /></Form.Item></Form></Card></Col></Row><Card className="matrix-preview"><Row align="middle" gutter={[18, 18]}><Col flex="auto"><Typography.Title level={4}>组合矩阵预览</Typography.Title><Typography.Text type="secondary">{datasetIds.length} 数据版本 × {modelIds.length} 模型 × {blurLevels.length} 条件 × {seeds.length} seed</Typography.Text></Col><Col><Statistic value={count} suffix="次运行" /></Col><Col><Button type="primary" size="large" icon={<PlayCircleOutlined />} disabled={!count} loading={submitting} onClick={submit}>启动批量评测</Button></Col></Row></Card>{jobId && <JobProgress jobId={jobId} onFinish={() => refresh()} />}{jobId && <Card><Button type="primary" onClick={() => navigate('explorer')}>打开效能探索器</Button></Card>}</Space>
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
  return <Space direction="vertical" size={18} style={{ width: '100%' }}><Card className="filter-card"><Row gutter={[12, 12]} align="bottom"><Col xs={24} md={5}><Typography.Text type="secondary">场景域</Typography.Text><Select allowClear value={scene} onChange={setScene} placeholder="全部场景" style={{ width: '100%', marginTop: 6 }} options={['无人机航拍', '城市驾驶', '卫星遥感'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">环境条件</Typography.Text><Select allowClear value={condition} onChange={setCondition} placeholder="全部条件" style={{ width: '100%', marginTop: 6 }} options={['晴朗', '雾', '雨', '夜间'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">分辨率</Typography.Text><Select allowClear value={resolution} onChange={setResolution} placeholder="保留为分组维度" style={{ width: '100%', marginTop: 6 }} options={['1920×1080', '1280×720', '640×640'].map((value) => ({ value }))} /></Col><Col xs={24} md={5}><Typography.Text type="secondary">硬件范围</Typography.Text><Select value="current" style={{ width: '100%', marginTop: 6 }} options={[{ value: 'current', label: '相同参考设备' }]} /></Col><Col xs={24} md={4}><Button block type="primary" icon={<EyeOutlined />} onClick={load} loading={loading}>查询效能</Button></Col></Row></Card><Alert type="info" showIcon message={`找到 ${data.count} 次 seed 运行，聚合为 ${data.groups.length} 个可比单元`} description={!resolution ? '分辨率未限定，结果按分辨率分别展示，不进行隐式平均。' : '当前为精确分辨率查询。'} action={<DemoTag />} /><Row gutter={[16, 16]}><Col xs={24} xl={12}><Card title="模型排行榜"><Table size="small" loading={loading} rowKey={(row) => `${row.dataset_id}-${row.model_id}`} dataSource={data.groups} pagination={{ pageSize: 6 }} onRow={(record) => ({ onClick: () => setSelected(record), style: { cursor: 'pointer' } })} columns={[{ title: '模型', dataIndex: 'model_name', render: (value) => <Typography.Text strong>{value.split('·')[0]}</Typography.Text> }, { title: '条件单元', dataIndex: 'dataset_name' }, { title: '分辨率', dataIndex: 'resolution' }, { title: 'mAP', dataIndex: 'map_mean', sorter: (a, b) => a.map_mean - b.map_mean, render: (value) => <Typography.Text strong className="map-value">{percent(value)}</Typography.Text> }, { title: 'σ', dataIndex: 'map_std', render: (value) => percent(value) }, { title: '时延', dataIndex: 'latency_mean', render: (value) => `${value} ms` }]} /></Card></Col><Col xs={24} xl={12}><Card title="mAP — 时延 Pareto"><ParetoChart groups={data.groups} dark={dark} height={320} /></Card></Col><Col xs={24} xl={12}><Card title="条件鲁棒性"><ReactECharts option={robustness} style={{ height: 300 }} /></Card></Col><Col xs={24} xl={12}><Card title="PR 曲线"><PRChart groups={data.groups} dark={dark} height={300} /></Card></Col></Row><Drawer open={Boolean(selected)} onClose={() => setSelected(undefined)} title="效能单元详情" width={620}>{selected && <Space direction="vertical" size={18} style={{ width: '100%' }}><Space><DemoTag /><StatusTag status="VERIFIED" /></Space><Row gutter={12}><Col span={8}><Card><Statistic title="mAP" value={selected.map_mean * 100} precision={1} suffix="%" /></Card></Col><Col span={8}><Card><Statistic title="seed σ" value={selected.map_std * 100} precision={2} suffix="%" /></Card></Col><Col span={8}><Card><Statistic title="时延" value={selected.latency_mean} precision={1} suffix="ms" /></Card></Col></Row><Descriptions bordered column={1} items={[{ key: 'model', label: '模型', children: selected.model_name }, { key: 'backbone', label: 'Backbone', children: selected.backbone }, { key: 'dataset', label: '数据版本', children: selected.dataset_name }, { key: 'scene', label: '场景', children: `${selected.scene_domain} / ${selected.weather}` }, { key: 'resolution', label: '分辨率', children: selected.resolution }, { key: 'seeds', label: '重复次数', children: `${selected.seed_count} 个 seed` }, { key: 'official', label: '结果性质', children: <DemoTag /> }]} /><PRChart groups={[selected]} dark={dark} /></Space>}</Drawer></Space>
}

export function TasksPage(_: PageProps) {
  const { data, loading, reload } = useResource<Job[]>('/api/jobs', [])
  useEffect(() => { const timer = window.setInterval(reload, 2000); return () => window.clearInterval(timer) }, [reload])
  const cancel = async (job: Job) => { try { await post(`/api/runs/${job.id}/cancel`); message.success('已请求取消'); reload() } catch (error) { message.error((error as Error).message) } }
  return <Card title="持久任务队列" extra={<Button icon={<ReloadOutlined />} onClick={reload}>刷新</Button>}><Table loading={loading} rowKey="id" dataSource={data} expandable={{ expandedRowRender: (job) => <Space direction="vertical" style={{ width: '100%' }}><Typography.Text code copyable>{job.id}</Typography.Text>{job.error && <Alert type="error" message={job.error} />}</Space> }} columns={[{ title: '类型', dataIndex: 'type', render: (value) => <Tag>{value}</Tag> }, { title: '状态', dataIndex: 'status', render: (value) => <StatusTag status={value} /> }, { title: '阶段', dataIndex: 'stage' }, { title: '进度', render: (_, row) => <Progress percent={Math.round(row.progress)} size="small" style={{ minWidth: 150 }} status={row.status === 'FAILED' ? 'exception' : undefined} /> }, { title: '创建时间', dataIndex: 'created_at', render: (value) => new Date(value).toLocaleString('zh-CN') }, { title: '操作', render: (_, row) => !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(row.status) ? <Button danger type="link" onClick={() => cancel(row)}>取消</Button> : null }]} /></Card>
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
