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

export type ExecutionMode = 'auto' | 'tethered' | 'on_phone';

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
  stairs_likely?: boolean;
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
  kernel_backend?: string;
  overhead_hazard?: boolean;
  wall_override?: boolean;
  lens_blocked?: boolean;
  pitch_deg?: number | null;
  memory_hint?: string | null;
  memory_veto?: boolean;
  best_escape_deg?: number | null;
  glass_barrier?: boolean;
  ultrasonic_dist_m?: number | null;
}

export type UltrasonicSurface =
  | 'open_space'
  | 'flat_wall_or_glass'
  | 'furniture_or_corner'
  | 'narrow_hallway';

export type UltrasonicZone = 'clear' | 'danger' | 'approach' | 'far';

export interface UltrasonicReading {
  active: boolean;
  calibrating: boolean;
  signalOk: boolean;
  detected: boolean;
  distanceM: number;
  snr: number;
  confidence: number;
  directLevel: number;
  sampleRate: number;
  profile64: number[];
  // Smart acoustic scene interpretation
  surface: UltrasonicSurface;
  surfaceLabel: string;
  zone: UltrasonicZone;
  approachMps: number; // negative = approaching fast!
  roomReverb: 'open' | 'room' | 'enclosed';
  peakWidthCm: number;
}

export type ElevationBand = 'down' | 'mid' | 'up' | 'sonar';

export interface MemoryCellPoint {
  gx: number;
  gy: number;
  prob: number;
  isCurb: boolean;
  band: ElevationBand;
}

export interface SpatialMemorySnapshot {
  headingDeg: number;
  pitchDeg: number;
  cameraBand: ElevationBand;
  stepCount: number;
  walkedMeters: number;
  scannedPercent: number;
  upClear: boolean;
  midClear: boolean;
  downClear: boolean;
  upSeen: boolean;
  midSeen: boolean;
  downSeen: boolean;
  filteredFlickers: number;
  motionGated: boolean;
  blindLeftDanger: boolean;
  blindRightDanger: boolean;
  bestEscapeDeg: number | null;
  memoryVetoActive: boolean;
  memoryHint: string | null;
  cells: MemoryCellPoint[];
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
