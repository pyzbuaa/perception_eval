import { useEffect, useMemo, useState } from 'react'
import {
  App as AntApp,
  Avatar,
  Badge,
  Breadcrumb,
  Button,
  ConfigProvider,
  Dropdown,
  Layout,
  Menu,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  theme,
} from 'antd'
import {
  AppstoreOutlined,
  BarChartOutlined,
  BellOutlined,
  BulbOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FullscreenOutlined,
  HomeOutlined,
  LaptopOutlined,
  MoonOutlined,
  PlayCircleOutlined,
  RobotOutlined,
  SettingOutlined,
  SunOutlined,
} from '@ant-design/icons'
import { api } from './api'
import { EnvironmentBadge } from './components'
import {
  DataBuilderPage,
  DatasetsPage,
  EnvironmentPage,
  EvaluationPage,
  ExplorerPage,
  OverviewPage,
  PresentationPage,
  RegistryPage,
  TasksPage,
} from './pages'
import type { EnvironmentStatus, Overview } from './types'

const { Header, Sider, Content, Footer } = Layout

type RouteKey = 'overview' | 'builder' | 'datasets' | 'registry' | 'evaluation' | 'explorer' | 'tasks' | 'environment'

const routeTitles: Record<RouteKey, { title: string; subtitle: string }> = {
  overview: { title: '实验概览', subtitle: '数据、模型、任务与最佳效能的统一视图' },
  builder: { title: '数据集构建', subtitle: '统一纳管基础图像生成、仿真、真实导入和非理想条件生成' },
  datasets: { title: '数据集', subtitle: '查看真值状态、数据谱系和不可变版本' },
  registry: { title: '模型版本', subtitle: '注册模型信息、权重和只读执行环境' },
  evaluation: { title: '评测中心', subtitle: '配置模型矩阵与标准化时延协议' },
  explorer: { title: '效能模型库', subtitle: '按条件查询、对比精度、时延与鲁棒性' },
  tasks: { title: '任务中心', subtitle: '跟踪生成、导入和评测任务的完整生命周期' },
  environment: { title: '系统环境', subtitle: '验证工作区隔离、conda 指纹与 GPU 状态' },
}

function routeFromHash(): RouteKey {
  const value = window.location.hash.replace('#/', '') as RouteKey
  return value in routeTitles ? value : 'overview'
}

function AppShell() {
  const [route, setRoute] = useState<RouteKey>(routeFromHash())
  const [dark, setDark] = useState(() => localStorage.getItem('perception-theme') === 'dark')
  const [collapsed, setCollapsed] = useState(false)
  const [presentation, setPresentation] = useState(false)
  const [overview, setOverview] = useState<Overview>()
  const [environment, setEnvironment] = useState<EnvironmentStatus>()

  const refreshChrome = async () => {
    await Promise.allSettled([
      api<Overview>('/api/overview').then(setOverview),
      api<EnvironmentStatus>('/api/environment/status').then(setEnvironment),
    ])
  }

  useEffect(() => {
    refreshChrome().catch(() => undefined)
    const timer = window.setInterval(() => refreshChrome().catch(() => undefined), 8000)
    const onHashChange = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('hashchange', onHashChange)
    }
  }, [])

  const navigate = (next: RouteKey) => {
    window.location.hash = `#/${next}`
    setRoute(next)
  }

  const toggleTheme = () => {
    const next = !dark
    setDark(next)
    localStorage.setItem('perception-theme', next ? 'dark' : 'light')
  }

  const menuItems = [
    { key: 'overview', icon: <HomeOutlined />, label: '概览' },
    { key: 'builder', icon: <BulbOutlined />, label: '数据构建' },
    { key: 'datasets', icon: <DatabaseOutlined />, label: '数据集' },
    { key: 'registry', icon: <RobotOutlined />, label: '模型版本' },
    { key: 'evaluation', icon: <ExperimentOutlined />, label: '评测中心' },
    { key: 'explorer', icon: <BarChartOutlined />, label: '效能模型库' },
    { key: 'tasks', icon: <PlayCircleOutlined />, label: '任务中心' },
    { key: 'environment', icon: <SettingOutlined />, label: '系统环境' },
  ]

  const content = useMemo(() => {
    const common = { dark, navigate, refresh: refreshChrome }
    switch (route) {
      case 'builder': return <DataBuilderPage {...common} />
      case 'datasets': return <DatasetsPage {...common} />
      case 'registry': return <RegistryPage {...common} />
      case 'evaluation': return <EvaluationPage {...common} />
      case 'explorer': return <ExplorerPage {...common} />
      case 'tasks': return <TasksPage {...common} />
      case 'environment': return <EnvironmentPage {...common} />
      default: return <OverviewPage {...common} overview={overview} />
    }
  }, [route, dark, overview])

  if (presentation) {
    return <PresentationPage dark onExit={() => setPresentation(false)} />
  }

  return (
    <Layout className={dark ? 'app-shell theme-dark' : 'app-shell theme-light'}>
      <Header className="top-header">
        <div className="brand" onClick={() => navigate('overview')}>
          <div className="brand-mark"><ExperimentOutlined /></div>
          <div><Typography.Text className="brand-title">Perception Lab</Typography.Text><span>效能评估平台</span></div>
        </div>
        <Space className="project-switcher" size={8}>
          <Select value="生成式视觉感知评测" options={[{ value: '生成式视觉感知评测', label: '生成式视觉感知评测' }]} style={{ width: 190 }} />
          <Tag color="blue">本地工作区</Tag>
        </Space>
        <div className="header-spacer" />
        <EnvironmentBadge gpuAvailable={environment?.gpu.available} running={overview?.counts.running} />
        <Tooltip title="通知"><Badge dot={Boolean(overview?.counts.running)}><Button type="text" icon={<BellOutlined />} /></Badge></Tooltip>
        <Tooltip title={dark ? '切换浅色主题' : '切换深色主题'}><Button type="text" icon={dark ? <SunOutlined /> : <MoonOutlined />} onClick={toggleTheme} /></Tooltip>
        <Button type="primary" ghost icon={<FullscreenOutlined />} onClick={() => setPresentation(true)}>演示模式</Button>
        <Dropdown menu={{ items: [{ key: 'local', label: '单用户 · 本地模式' }] }}>
          <Avatar size="small" icon={<LaptopOutlined />} />
        </Dropdown>
      </Header>
      <Layout>
        <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed} width={218} className="side-nav">
          <Menu mode="inline" selectedKeys={[route]} items={menuItems} onClick={({ key }) => navigate(key as RouteKey)} />
          {!collapsed && <div className="side-safety"><Badge status="success" />现有 conda 环境只读<br /><small>平台写入仅限工作区</small></div>}
        </Sider>
        <Layout className="workspace-layout">
          <Content className="workspace-content">
            <div className="page-heading">
              <div>
                <Breadcrumb items={[{ title: '视觉感知效能评估' }, { title: routeTitles[route].title }]} />
                <Typography.Title level={2}>{routeTitles[route].title}</Typography.Title>
                <Typography.Text type="secondary">{routeTitles[route].subtitle}</Typography.Text>
              </div>
              <Space><Tag icon={<AppstoreOutlined />} color="cyan">MVP 0.1</Tag></Space>
            </div>
            <div className="page-body">{content || <Spin />}</div>
          </Content>
          <Footer className="app-footer">
            <span>Perception Lab v0.1.0</span>
            <span>Agent {overview ? '在线' : '连接中'}</span>
            <span>{environment?.isolation.data_dir || '工作区数据目录'}</span>
            <span>外部环境策略：只读</span>
          </Footer>
        </Layout>
      </Layout>
    </Layout>
  )
}

export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('perception-theme') === 'dark')
  useEffect(() => {
    const timer = window.setInterval(() => setDark(localStorage.getItem('perception-theme') === 'dark'), 500)
    return () => window.clearInterval(timer)
  }, [])
  return (
    <ConfigProvider theme={{
      algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm,
      token: { colorPrimary: dark ? '#22b8cf' : '#1677ff', borderRadius: 8, fontFamily: "Inter, 'Noto Sans SC', system-ui, sans-serif" },
      components: { Layout: { headerBg: dark ? '#0c1525' : '#ffffff', siderBg: dark ? '#0c1525' : '#ffffff' }, Card: { headerFontSize: 15 } },
    }}>
      <AntApp><AppShell /></AntApp>
    </ConfigProvider>
  )
}
