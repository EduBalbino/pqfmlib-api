"""Publish a small, atomic status snapshot beside the hosted SIH journals."""
from collections import defaultdict, deque
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(sys.argv[1])
INTERVAL = 1
CLOCK_TICKS = os.sysconf('SC_CLK_TCK')
STAGES = {
    'inner_classic_preprocessing', 'inner_pca', 'inner_train', 'validation',
    'classic_search', 'hybrid_search', 'inner_hybrid_scale', 'outer_preprocess',
    'outer_refit_pca', 'outer_refit_projection', 'outer_refit', 'outer_test',
    'outer_scale', 'outer_ranking_fit', 'outer_exact_shap', 'outer_prediction_fit',
    'router_preprocessing', 'calibration_and_router_targets', 'router_fit',
    'classic_expert_refit', 'expert_refit_pca', 'expert_refit', 'calibration', 'analysis', 'test',
}


def follow(path, cursor):
    """Advance only past newline-terminated records, including while a writer is active."""
    try:
        with path.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            inode = (stat.st_dev, stat.st_ino)
            start = cursor[1] if cursor[0] == inode and cursor[1] <= stat.st_size else 0
            stream.seek(start)
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    break
                cursor[:] = [inode, stream.tell()]
                yield line.decode(errors='replace').strip()
    except FileNotFoundError:
        pass


def process_usage(pid):
    stat = Path(f'/proc/{pid}/stat')
    try:
        fields = stat.read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        cpu = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
        rss = int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except OSError:
            children = []
        for child in children:
            if usage := process_usage(int(child)):
                cpu += usage[0]
                rss += usage[1]
        return cpu, rss
    except (OSError, ValueError, IndexError):
        return None


def cgroup_root(pid):
    group = Path(f'/proc/{pid}/cgroup').read_text().split('::', 1)[1].strip()
    return Path('/sys/fs/cgroup') / group.lstrip('/')


def integer_stats(path):
    return {key: int(value) for key, value in map(str.split, path.read_text().splitlines())}


def pressure(root, kind):
    try:
        first = (root/f'{kind}.pressure').read_text().splitlines()[0].split()
        return float(next(value[6:] for value in first if value.startswith('avg10=')))
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def system_sample(root, previous, elapsed):
    try:
        cpu = integer_stats(root/'cpu.stat')
    except (OSError, ValueError):
        cpu = {}
    try:
        memory = integer_stats(root/'memory.stat')
        swap = int((root/'memory.swap.current').read_text())
    except (OSError, ValueError):
        memory, swap = {}, None

    def rate(key):
        if previous is None or elapsed <= 0 or key not in cpu or key not in previous:
            return None
        return max(0, (cpu[key]-previous[key]) / (elapsed * 1_000_000))

    return dict(cpu=rate('usage_usec'), throttled=rate('throttled_usec'),
                rss=memory.get('anon'), file=memory.get('file'), swap=swap,
                cpuPressure=pressure(root, 'cpu'), memoryPressure=pressure(root, 'memory'),
                ioPressure=pressure(root, 'io')), cpu


def disk_counters(pid):
    try:
        path = Path(f'/proc/{pid}/cgroup').read_text().split('::', 1)[1].strip()
        records = (Path('/sys/fs/cgroup') / path.lstrip('/') / 'io.stat').read_text().splitlines()
    except (OSError, IndexError):
        return None
    totals = dict(rbytes=0, wbytes=0)
    for record in records:
        for field in record.split()[1:]:
            key, _, value = field.partition('=')
            if key in totals:
                totals[key] += int(value)
    return totals


def read_cgroup(pid):
    try:
        path = Path(f'/proc/{pid}/cgroup').read_text().split('::', 1)[1].strip()
        root = Path('/sys/fs/cgroup') / path.lstrip('/')
        usage = int((root/'memory.current').read_text())
        limit = (root/'memory.max').read_text().strip()
        quota, period = (root/'cpu.max').read_text().split()
        return dict(memory=usage, memoryLimit=None if limit=='max' else int(limit),
                    cpuLimit=None if quota=='max' else int(quota)/int(period))
    except (OSError, ValueError, IndexError):
        return dict(memory=None, memoryLimit=None, cpuLimit=None)


def gpu_sample():
    try:
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits',
        ], text=True, timeout=3)
        return [dict(zip(('index','utilization','memory','limit'), map(int, line.split(',')), strict=True))
                for line in result.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


class Experiment:
    def __init__(self, key, log, journal, pid, total):
        self.key, self.log, self.journal, self.pid, self.total = key, log, journal, pid, total
        self.logs, self.results = [None, 0], [None, 0]
        self.recent = deque(maxlen=8)
        self.runs = {}
        self.active = {}
        self.finished = False
        self.queued = False
        self.phase = None
        self.previous_cpu = None
        self.next_pid = None

    def update(self, now):
        for line in follow(self.log, self.logs):
            if line.startswith('HILL_FINISHED') or (self.key == 'stratified' and 'STRATIFIED_FINISHED' in line):
                self.finished = True
            if line.startswith(('STRATIFIED_QUEUED', 'OPTIMIZED_QUEUED')):
                self.queued = True
            match = re.match(r'(HILL_START|STRATIFIED_START) (\S+)', line)
            if match:
                self.active[match[2]] = now
                self.phase = 'Trial started'
            if line.startswith(('PCA Heisenberg', 'Router:', 'WIDE_SHAPE', 'WIDE_BUILT', 'STRATIFIED_BUILT')):
                self.phase = line[:180]
            if line.startswith(('HILL_RESULT', 'STRATIFIED_RESULT', 'WIDE_BUILT',
                                'STRATIFIED_BUILT', 'STRATIFIED_QUEUED', 'PCA Heisenberg',
                                'Router:', 'Traceback', 'RuntimeError', 'ValueError', 'HILL_FINISHED')):
                self.recent.append(line[:400])
        for line in follow(self.journal, self.results):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'seconds' not in row:
                self.recent.append(f"{row.get('name', 'trial')}: {row.get('error', 'failed')}"[:400])
                continue
            spans = defaultdict(list)
            for event in row.get('events', []):
                if event['name'] in STAGES:
                    spans[event['name']].append((event['start'], event['stop']))
            timings = [{
                'name': name, 'start': min(a for a, _ in intervals),
                'stop': max(b for _, b in intervals),
            } for name, intervals in spans.items()]
            stats = row.get('resources', {})
            io = {key: stats.get(key, {}).get('io', {}).get('read_bytes') for key in ('before','after')}
            self.runs[row['name']] = dict(name=row['name'], fold=row.get('fold'), seconds=row['seconds'],
                                         scores=row['scores'], timings=timings,
                                         cpu=stats.get('mean_process_cores'),
                                         readBytes=io['after']-io['before'] if None not in io.values() else None)
        self.active = {key: started for key, started in self.active.items() if key not in self.runs}

    def snapshot(self, now):
        usage = process_usage(self.pid) or (process_usage(self.next_pid) if self.next_pid else None)
        cpu = None
        if usage and self.previous_cpu:
            previous_time, previous_cpu = self.previous_cpu
            cpu = max(0.0, (usage[0] - previous_cpu) / (now - previous_time))
        self.previous_cpu = (now, usage[0]) if usage else None
        completed = len(self.runs)
        if self.key == 'stratified' and usage and not self.queued:
            status = 'building'
        elif completed >= self.total or self.finished:
            status = 'complete'
        elif self.queued and usage and completed == 0 and not self.active:
            status = 'queued'
        elif usage:
            status = 'running'
        else:
            status = 'stopped'
        return dict(id=self.key, status=status, total=self.total, runs=list(self.runs.values()),
                    active=', '.join(self.active) or None, observedSince=min(self.active.values(), default=None), phase=self.phase,
                    recent=list(self.recent), cpu=cpu, rss=usage[1] if usage else None)


def main():
    experiments = [
        Experiment('wide', ROOT/'sih-wide-run.log', ROOT/'sih-wide-results.jsonl', int(sys.argv[2]), 60),
        Experiment('stratified', ROOT/'sih-stratified-run.log', ROOT/'sih-stratified-results.jsonl', int(sys.argv[3]), 4),
    ]
    if len(sys.argv) > 4:
        experiments.append(Experiment('optimized', ROOT/'sih-optimized-run.log',
                                      ROOT/'sih-optimized-results.jsonl', int(sys.argv[4]), 8))
        experiments[-1].next_pid = int(sys.argv[5]) if len(sys.argv) > 5 else None
    target = ROOT/'sih-dashboard-status.json'
    temporary = target.with_suffix('.tmp')
    try:
        disk = deque(json.loads(target.read_text()).get('disk', []), maxlen=120)
    except (OSError, ValueError):
        disk = deque(maxlen=120)
    history_path = ROOT/'sih-dashboard-hardware.json.gz'
    history_tmp = history_path.with_suffix('.tmp')
    try:
        history = deque(json.loads(gzip.decompress(history_path.read_bytes()))['samples'], maxlen=1800)
    except (OSError, ValueError, KeyError):
        history = deque(maxlen=1800)
    previous_disk = None
    last_system, last_sample = None, None
    system_root = cgroup_root(os.getpid())
    tick = 0
    while True:
        started = time.monotonic()
        now = time.time()
        memory = read_cgroup(os.getpid())
        gpu = gpu_sample()
        system, last_system = system_sample(system_root, last_system,
                                            0 if last_sample is None else started-last_sample)
        last_sample = started
        if tick % 5 == 0:
            for experiment in experiments:
                experiment.update(now)
            snapshot = dict(updatedAt=now, experiments=[e.snapshot(now) for e in experiments],
                            gpus=gpu, **memory)
        counters = disk_counters(os.getpid())
        if counters is not None:
            if previous_disk is not None:
                last, previous = previous_disk
                delta = now - last
                if delta > 0 and all(counters[k] >= previous[k] for k in counters):
                    disk.append(dict(at=now, read=(counters['rbytes']-previous['rbytes'])/delta,
                                     write=(counters['wbytes']-previous['wbytes'])/delta))
            previous_disk = now, counters
        rates = disk[-1] if disk and disk[-1]['at'] == now else {}
        history.append(dict(at=now, memory=memory['memory'], gpu=[[g['index'],g['utilization'],g['memory']] for g in gpu],
                            read=rates.get('read'), write=rates.get('write'), **system))
        if tick % 2 == 0:
            payload = json.dumps(dict(updatedAt=now, samples=list(history),
                                     gpuLimits={str(g['index']):g['limit'] for g in gpu},
                                     memoryLimit=memory['memoryLimit'], cpuLimit=memory['cpuLimit']),
                                 separators=(',', ':')).encode()
            history_tmp.write_bytes(gzip.compress(payload, compresslevel=1))
            os.replace(history_tmp, history_path)
        if tick % 5 == 0:
            snapshot['disk'] = list(disk)
            temporary.write_text(json.dumps(snapshot, separators=(',', ':')))
            os.replace(temporary, target)
        tick += 1
        time.sleep(max(0, INTERVAL-(time.monotonic()-started)))


if __name__ == '__main__':
    main()
