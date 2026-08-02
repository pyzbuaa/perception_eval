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
  HEALTHY: '健康', UNAVAILABLE: '不可用', READY: '正式可用', BENCHMARK_READY: '可正式评测',
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

export function ParetoChart({ groups, dark = false, height = 310 }: { groups: ResultGroup[]; dark?: boolean; height?: number }) {
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', formatter: (params: { data: [number, number, string] }) => `${params.data[2]}<br/>时延 ${params.data[0]} ms<br/>mAP ${(params.data[1] * 100).toFixed(1)}%` },
    grid: { left: 52, right: 24, top: 24, bottom: 45 },
    xAxis: { name: '时延 / ms', nameLocation: 'middle', nameGap: 30, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } }, axisLabel: { color: dark ? '#9dafc7' : '#687386' } },
    yAxis: { name: 'mAP', min: 0.45, max: 0.9, splitLine: { lineStyle: { color: dark ? '#263449' : '#eef1f5' } }, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%`, color: dark ? '#9dafc7' : '#687386' } },
    series: [{
      type: 'scatter', symbolSize: 15,
      data: groups.map((item) => [item.latency_mean, item.map_mean, item.model_name]),
      itemStyle: { color: '#22b8cf', borderColor: dark ? '#d6fbff' : '#087f9c', borderWidth: 1 },
    }],
  }), [groups, dark])
  return <ReactECharts option={option} style={{ height }} />
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
