import { useEffect, useMemo, useState } from 'react'
import { Alert, Badge, Button, Card, Empty, Image, Progress, Space, Tag, Typography, message } from 'antd'
import ReactECharts from 'echarts-for-react'
import { api, post } from './api'
import type { Job, ResultGroup } from './types'

const statusColors: Record<string, string> = {
  SUCCEEDED: 'success',
  HEALTHY: 'success',
  READY: 'success',
  BENCHMARK_READY: 'success',
  RUNNING: 'processing',
  CHECKING: 'processing',
  QUEUED: 'default',
  CONTRACT_OK: 'cyan',
  EXPERIMENTAL: 'warning',
  REGISTERED: 'default',
  FAILED: 'error',
  UNAVAILABLE: 'error',
  CANCELLED: 'default',
  VERIFIED: 'success',
  ANNOTATING: 'processing',
  CANDIDATE: 'warning',
  UNLABELED: 'default',
}

const statusLabels: Record<string, string> = {
  SUCCEEDED: '已完成', RUNNING: '运行中', CHECKING: '检查中', QUEUED: '排队中', FAILED: '失败', CANCELLED: '已取消',
  HEALTHY: '状态正常', UNAVAILABLE: '不可用', READY: '正式可用', BENCHMARK_READY: '可正式评测',
  CONTRACT_OK: '接口已通过', EXPERIMENTAL: '实验性', REGISTERED: '已注册', VERIFIED: '真值已验证',
  ANNOTATING: '标注中', CANDIDATE: '候选真值', UNLABELED: '未标注',
}

export function StatusTag({ status }: { status: string }) {
  return <Tag color={statusColors[status] || 'default'}>{statusLabels[status] || status}</Tag>
}

export function DemoTag() {
  return <Tag color="gold">流程样例 · 非正式结果</Tag>
}

export function JobProgress({ jobId, onFinish }: { jobId?: string; onFinish?: (job: Job) => void }) {
  const [job, setJob] = useState<Job>()
  useEffect(() => {
    if (!jobId) return
    let active = true
    const load = async () => {
      try {
        const next = await api<Job>(`/api/runs/${jobId}`)
        if (!active) return
        setJob(next)
        if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(next.status)) {
          onFinish?.(next)
          return
        }
        window.setTimeout(load, 700)
      } catch {
        if (active) window.setTimeout(load, 1200)
      }
    }
    load()
    return () => { active = false }
  }, [jobId, onFinish])
  if (!jobId) return null
  if (!job) return <Card loading />
  return (
    <Card className="job-progress-card">
      <Space direction="vertical" size={14} style={{ width: '100%' }}>
        <Space><StatusTag status={job.status} /><Typography.Text strong>{job.stage}</Typography.Text></Space>
        <Progress percent={Math.round(job.progress)} status={job.status === 'FAILED' ? 'exception' : undefined} />
        <Typography.Text type="secondary" copyable={{ text: job.id }}>任务 {job.id}</Typography.Text>
        {job.error && <Alert type="error" showIcon message="任务执行失败" description={job.error} />}
        {!['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status) && (
          <Button danger onClick={() => post(`/api/runs/${job.id}/cancel`).catch((error) => message.error(error.message))}>取消任务</Button>
        )}
      </Space>
    </Card>
  )
}

export function Gallery({ images, height = 112 }: { images: string[]; height?: number }) {
  if (!images.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无图像" />
  return (
    <Image.PreviewGroup>
      <div className="image-grid">
        {images.map((src) => <Image key={src} src={src} height={height} preview={{ mask: '查看检测细节' }} />)}
      </div>
    </Image.PreviewGroup>
  )
}

export function ParetoChart({ groups, dark = false, height = 310, onSelect }: { groups: ResultGroup[]; dark?: boolean; height?: number; onSelect?: (group: ResultGroup) => void }) {
  const frontierIds = useMemo(() => {
    let bestMap = Number.NEGATIVE_INFINITY
    return new Set(
      [...groups]
        .sort((left, right) => left.latency_mean - right.latency_mean || right.map_mean - left.map_mean)
        .filter((group) => {
          if (group.map_mean <= bestMap) return false
          bestMap = group.map_mean
          return true
        })
        .map((group) => group.comparison_id),
    )
  }, [groups])
  const frontier = useMemo(() => groups.filter((group) => frontierIds.has(group.comparison_id)).sort((left, right) => left.latency_mean - right.latency_mean), [groups, frontierIds])
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', formatter: (params: { data?: { value?: [number, number, number]; model?: string; dataset?: string; config?: string } }) => params.data?.value ? `${params.data.model}<br/>${params.data.dataset}<br/>配置 ${params.data.config}<br/>时延 ${params.data.value[0]} ms<br/>mAP ${(params.data.value[1] * 100).toFixed(2)}%<br/>显存 ${params.data.value[2]} MB` : 'Pareto前沿' },
    legend: { top: 0, type: 'scroll', textStyle: { color: dark ? '#c6d1df' : '#4b5565' } },
    grid: { left: 58, right: 24, top: 48, bottom: 45 },
    xAxis: { name: '时延 / ms', nameLocation: 'middle', nameGap: 30, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } }, axisLabel: { color: dark ? '#9dafc7' : '#687386' } },
    yAxis: { name: 'mAP', scale: true, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } }, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%`, color: dark ? '#9dafc7' : '#687386' } },
    series: [
      {
        name: 'Pareto前沿',
        type: 'line',
        showSymbol: false,
        silent: true,
        lineStyle: { color: '#fa8c16', width: 2, type: 'dashed' },
        data: frontier.map((group) => [group.latency_mean, group.map_mean]),
      },
      ...[...new Set(groups.map((group) => group.model_name))].map((modelName) => ({
        name: modelName.split('·')[0],
        type: 'scatter',
        symbolSize: (value: number[]) => Math.max(12, Math.min(25, 12 + (value[2] || 0) / 1000)),
        data: groups.filter((group) => group.model_name === modelName).map((group) => ({
          value: [group.latency_mean, group.map_mean, group.peak_memory_mean] as [number, number, number],
          comparisonId: group.comparison_id,
          model: group.model_name,
          dataset: group.dataset_name,
          config: group.configuration_id,
          symbol: group.inference_config.precision === 'FP32' ? 'diamond' : 'circle',
          itemStyle: frontierIds.has(group.comparison_id) ? { borderColor: '#fa8c16', borderWidth: 4 } : { borderWidth: 1 },
        })),
      })),
    ],
  }), [groups, dark, frontier, frontierIds])
  return <ReactECharts option={option} onEvents={onSelect ? { click: (params: { data?: { comparisonId?: string } }) => { const group = groups.find((item) => item.comparison_id === params.data?.comparisonId); if (group) onSelect(group) } } : undefined} style={{ height }} />
}

export function PRChart({ groups, dark = false, height = 290 }: { groups: ResultGroup[]; dark?: boolean; height?: number }) {
  const option = useMemo(() => ({
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: dark ? '#c6d1df' : '#4b5565' } },
    grid: { left: 50, right: 20, top: 45, bottom: 35 },
    xAxis: { type: 'category', name: 'Recall', data: groups[0]?.curves.recall || [], axisLabel: { color: dark ? '#9dafc7' : '#687386' } },
    yAxis: { type: 'value', name: 'Precision', min: 0, max: 1, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } }, axisLabel: { color: dark ? '#9dafc7' : '#687386' } },
    series: groups.slice(0, 4).map((item) => ({ name: item.model_name.split('·')[0], type: 'line', smooth: true, showSymbol: false, data: item.curves.precision })),
  }), [groups, dark])
  return <ReactECharts option={option} style={{ height }} />
}

export function EnvironmentBadge({ gpuAvailable, running }: { gpuAvailable?: boolean; running?: number }) {
  return (
    <Space size={16}>
      <Badge status={gpuAvailable ? 'success' : 'warning'} text={gpuAvailable ? 'GPU 可用' : 'GPU 受限'} />
      <Badge status={running ? 'processing' : 'default'} text={`${running || 0} 个活动任务`} />
    </Space>
  )
}
