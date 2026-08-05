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
  category_template: string
  categories: CategoryDefinition[]
  preview_images: string[]
  created_at: string
}

export interface CategoryDefinition {
  id: number
  name: string
  color?: string
}

export interface CategoryTemplate {
  id: string
  name: string
  categories: Array<{
    name: string
    dataset_id: number
    model_id: number
  }>
}

export interface DatasetAnnotationCategories {
  categories: CategoryDefinition[]
  category_template: string
  source: string
}

export interface DatasetSamplePage {
  dataset_id: string
  dataset_name: string
  declared_count: number
  total: number
  offset: number
  limit: number
  has_more: boolean
  items: Array<{
    name: string
    url: string
    width: number
    height: number
    annotation_source: 'MANUAL' | 'COCO' | 'YOLO' | 'VISDRONE' | null
    boxes: Array<{
      label: string
      color: string
      x: number
      y: number
      width: number
      height: number
    }>
  }>
}

export interface DatasetStatistics {
  dataset_id: string
  image_count: number
  annotated_image_count: number
  object_count: number
  category_counts: Array<{ id: number | null; name: string; count: number }>
  resolutions: Array<{ width: number; height: number; label: string; count: number }>
  scales: Array<{ key: 'small' | 'medium' | 'large' | 'unknown'; label: string; count: number }>
}

export interface AnnotationCategory extends CategoryDefinition {
  color: string
}

export interface DetectionBox {
  id: string
  category_id: number
  x: number
  y: number
  width: number
  height: number
  confidence?: number
  source?: 'AUTO_MODEL' | 'MANUAL'
}

export interface AnnotationSession {
  dataset: Dataset
  categories: AnnotationCategory[]
  progress: { completed: number; total: number }
  samples: Array<{
    name: string
    url: string
    completed: boolean
    box_count: number
  }>
}

export interface SampleAnnotation {
  dataset_id: string
  sample_name: string
  width: number
  height: number
  boxes: DetectionBox[]
  completed: boolean
  updated_at?: string
}

export interface ModelVersion {
  id: string
  name: string
  family: string
  architecture: string
  backbone: string
  detector_head: string
  class_count: number
  category_template: string
  categories: CategoryDefinition[]
  training_dataset: string
  pretrained_dataset: string
  version: string
  precision: string
  adapter_id: string
  weight_path?: string
  weight_sha256?: string
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

export interface BaseGenSceneOption {
  value: string
  label_zh: string
  environments?: string[]
}

export interface BaseGenSceneField {
  name: string
  label_zh: string
  description_zh: string
  kind: 'single' | 'multi' | 'text'
  weighted: boolean
  options: BaseGenSceneOption[]
}

export interface BaseGenSceneDomain {
  value: string
  label_zh: string
  default_resolution: string
  fields: BaseGenSceneField[]
}

export interface BaseGenSceneSchema {
  version: string
  domains: BaseGenSceneDomain[]
}

export interface ResultGroup {
  comparison_id: string
  configuration_id: string
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
  is_official: boolean
  inference_config: Record<string, string | number | boolean>
  hardware_profile: Record<string, unknown>
  environment_fingerprint?: string
  condition_type: string
  condition_strength: number | null
  source_dataset_id?: string
  map_mean: number
  map_std: number
  map50_mean: number
  map75_mean: number
  precision_mean: number
  recall_mean: number
  f1_mean: number
  latency_mean: number
  latency_p95_mean: number
  fps_mean: number
  peak_memory_mean: number
  delta_map_mean: number
  seed_count: number
  seeds: number[]
  run_ids: string[]
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
    model_options: Array<[string, string]>
    dataset_options: Array<[string, string]>
    condition_types: string[]
    hardware: string[]
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
