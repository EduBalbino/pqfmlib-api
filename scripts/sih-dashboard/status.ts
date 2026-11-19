export type Stage = { name: string; start: number; stop: number };
export type Trial = {
  name: string;
  fold: number | null;
  seconds: number;
  scores: Record<string, number>;
  timings: Stage[];
  cpu: number | null;
  readBytes: number | null;
};
export type Experiment = {
  id: 'wide' | 'stratified' | 'optimized' | 'dense' | 'fast';
  status: 'building' | 'queued' | 'running' | 'complete' | 'stopped';
  total: number;
  runs: Trial[];
  active: string | null;
  observedSince: number | null;
  phase: string | null;
  recent: string[];
  cpu: number | null;
  rss: number | null;
};
export type Snapshot = {
  updatedAt: number;
  disk?: { at: number; read: number; write: number }[];
  experiments: Experiment[];
  gpus: { index: number; utilization: number; memory: number; limit: number }[];
  cpuLimit: number | null;
  memory: number | null;
  memoryLimit: number | null;
};
export type Feed =
  | { kind: 'online'; snapshot: Snapshot }
  | { kind: 'offline'; message: string; snapshot: Snapshot | null };

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function nullable(value: unknown, check: (x: unknown) => boolean): boolean {
  return value === null || check(value);
}

function string(value: unknown): value is string {
  return typeof value === 'string';
}

function stage(value: unknown): value is Stage {
  return record(value) && string(value.name) && finite(value.start) &&
    finite(value.stop) && value.start <= value.stop;
}

function trial(value: unknown): value is Trial {
  return record(value) && string(value.name) && nullable(value.fold, finite) &&
    finite(value.seconds) && value.seconds > 0 && record(value.scores) &&
    Object.values(value.scores).every(finite) && Array.isArray(value.timings) &&
    value.timings.every(stage) && nullable(value.cpu, finite) && nullable(value.readBytes, finite);
}

function experiment(value: unknown): value is Experiment {
  return record(value) && ['wide', 'stratified', 'optimized', 'dense', 'fast'].some((id) => value.id === id) &&
    ['building','queued','running','complete','stopped'].some((status) => value.status === status) &&
    finite(value.total) && value.total > 0 && Array.isArray(value.runs) && value.runs.every(trial) &&
    nullable(value.active, string) && nullable(value.observedSince, finite) &&
    nullable(value.phase, string) && Array.isArray(value.recent) && value.recent.every(string) &&
    nullable(value.cpu, finite) && nullable(value.rss, finite);
}

export function isSnapshot(value: unknown): value is Snapshot {
  return record(value) && finite(value.updatedAt) &&
    (value.disk === undefined || (Array.isArray(value.disk) && value.disk.every((sample: unknown) =>
      record(sample) && finite(sample.at) && finite(sample.read) && finite(sample.write)))) &&
    Array.isArray(value.experiments) &&
    value.experiments.every(experiment) && Array.isArray(value.gpus) &&
    value.gpus.every((gpu: unknown) => record(gpu) && finite(gpu.index) &&
      finite(gpu.utilization) && finite(gpu.memory) && finite(gpu.limit)) &&
    nullable(value.cpuLimit, finite) && nullable(value.memory, finite) && nullable(value.memoryLimit, finite);
}

export function isFeed(value: unknown): value is Feed {
  if (!record(value)) return false;
  if (value.kind === 'online') return isSnapshot(value.snapshot);
  return value.kind === 'offline' && string(value.message) && nullable(value.snapshot, isSnapshot);
}

export type HardwareSample = {
  at: number;
  memory: number | null;
  cpu: number | null;
  throttled: number | null;
  rss: number | null;
  file: number | null;
  swap: number | null;
  cpuPressure: number | null;
  memoryPressure: number | null;
  ioPressure: number | null;
  read: number | null;
  write: number | null;
  gpu: [index: number, use: number, vram: number][];
};
export type HardwareHistory = {
  updatedAt: number;
  samples: HardwareSample[];
  gpuLimits: Record<string, number>;
  memoryLimit: number | null;
  cpuLimit: number | null;
};

export function isHardwareHistory(value: unknown): value is HardwareHistory {
  const numeric = ['memory','cpu','throttled','rss','file','swap','cpuPressure','memoryPressure','ioPressure','read','write'];
  return record(value) && finite(value.updatedAt) && record(value.gpuLimits) &&
    Object.values(value.gpuLimits).every(finite) && nullable(value.memoryLimit, finite) &&
    nullable(value.cpuLimit, finite) && Array.isArray(value.samples) &&
    value.samples.every((sample: unknown) => record(sample) && finite(sample.at) &&
      numeric.every((key) => nullable(sample[key], finite)) && Array.isArray(sample.gpu) &&
      sample.gpu.every((row: unknown) => Array.isArray(row) && row.length === 3 && row.every(finite)));
}
