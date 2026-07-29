export interface Job {
  id: string
  type: string
  status: string
  progress: number
  stage: string
  error?: string
  created_at: string
  finished_at?: string
  result?: Record<string, unknown>
}

export interface Overview {
  counts: { datasets: number; models: number; running: number; completed: number }
  best_result?: { map: number; latency_p50: number; model_name: string; dataset_name: string }
  recent_jobs: Job[]
  recent_images: string[]
  pipeline: Array<{ name: string; status: string }>
}

export interface Dataset {
  id: string
  name: string
  version: string
  source_type: string
  scene_domain: string
  weather: string
  sensor_conditions: Record<string, unknown>
  resolution: string
  sample_count: number
  annotation_status: string
  frozen: boolean
  preview_images: string[]
  created_at: string
}

export interface ModelVersion {
  id: string
  name: string
  family: string
  backbone: string
  version: string
  precision: string
  adapter_id: string
  is_demo: boolean
  status: string
}

export interface Adapter {
  id: string
  name: string
  kind: string
  version: string
  maturity: string
  runtime_kind: string
  runtime_prefix?: string
  policy: string
  requires_gpu: boolean
  status: string
  description: string
  parameter_schema: Record<string, unknown>
}

export interface ResultGroup {
  dataset_id: string
  dataset_name: string
  model_id: string
  model_name: string
  family: string
  backbone: string
  scene_domain: string
  weather: string
  resolution: string
  sensor_conditions: Record<string, unknown>
  source_type: string
  is_demo: boolean
  map_mean: number
  map_std: number
  latency_mean: number
  delta_map_mean: number
  seed_count: number
  curves: { recall: number[]; precision: number[] }
}

export interface ResultResponse {
  count: number
  groups: ResultGroup[]
  runs: Array<Record<string, unknown>>
  dimensions: {
    scenes: string[]
    conditions: string[]
    resolutions: string[]
    models: string[]
  }
}

export interface EnvironmentStatus {
  isolation: {
    mode: string
    runtime_dir: string
    data_dir: string
    writes_outside_workspace: boolean
    shell_configuration_modified: boolean
  }
  conda: {
    available: boolean
    executable?: string
    error?: string
    envs: Array<{ name: string; prefix: string; policy: string; exists: boolean; fingerprint?: string }>
  }
  gpu: {
    available: boolean
    error?: string
    devices: Array<{ name: string; driver: string; memory_mb: number }>
  }
  disk: { total: number; used: number; free: number }
}

