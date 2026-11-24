export type Severity = 'clear' | 'caution' | 'danger';

export type SteeringAction =
  | 'CLEAR'
  | 'VEER_LEFT'
  | 'VEER_RIGHT'
  | 'STOP'
  | 'CURB_DROP';

export type SampleLabel =
  | 'clear'
  | 'obstacle_left'
  | 'obstacle_center'
  | 'obstacle_right'
  | 'curb_drop';

export interface ZoneMetrics {
  severity: Severity;
  obstacle_score: number;
  min_distance_m: number;
  coverage_ratio: number;
}

export interface CurbMetrics {
  detected: boolean;
  confidence: number;
  drop_row_ratio: number;
  gradient_peak: number;
}

export interface LatencyBreakdown {
  decode_ms: number;
  inference_ms: number;
  hazard_ms: number;
  server_total_ms: number;
}

export interface GuidancePayload {
  frame_id: number;
  client_ts: number;
  server_ts: number;
  overall_severity: Severity;
  steering: SteeringAction;
  spoken_cue: string;
  left: ZoneMetrics;
  center: ZoneMetrics;
  right: ZoneMetrics;
  curb: CurbMetrics;
  latency: LatencyBreakdown;
  device: string;
  model_name: string;
}

export interface CalibrationConfig {
  caution_threshold: number;
  danger_threshold: number;
  curb_gradient_threshold: number;
  floor_margin: number;
  ema_alpha: number;
  confirm_frames: number;
}

export interface SampleRecord {
  sample_id: string;
  label: SampleLabel;
  note: string;
  source: 'real' | 'synthetic';
  created_at: number;
  image_b64?: string | null;
}

export interface EngineBenchmarkMetrics {
  engine_name: string;
  precision: number;
  recall: number;
  f1: number;
  false_alarm_rate_clear: number;
  curb_recall: number;
  direction_accuracy: number;
  p50_ms: number;
  p95_ms: number;
}

export interface PerSampleComparison {
  sample_id: string;
  label: SampleLabel;
  note: string;
  source: 'real' | 'synthetic';
  naive_detected: boolean;
  naive_correct: boolean;
  smart_steering: SteeringAction;
  smart_severity: Severity;
  smart_curb: boolean;
  smart_correct: boolean;
  smart_ms: number;
}

export interface BenchmarkReport {
  total_samples: number;
  real_samples: number;
  synthetic_samples: number;
  device: string;
  model_name: string;
  naive: EngineBenchmarkMetrics;
  smart: EngineBenchmarkMetrics;
  samples: PerSampleComparison[];
}

export interface NetworkInfo {
  lan_ip: string;
  mobile_url: string;
  dashboard_url: string;
}

export interface DemoSceneSummary {
  scene_id: string;
  label: SampleLabel;
  note: string;
}

export interface DashboardMessage {
  type: 'status' | 'telemetry';
  phone_connected?: boolean;
  guidance?: GuidancePayload | null;
  raw_b64?: string | null;
  depth_b64?: string | null;
  simulated_note?: string;
}
