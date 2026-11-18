import { isFeed, isHardwareHistory, type HardwareHistory, type HardwareSample, type Experiment, type Feed, type Snapshot, type Trial } from './status';

const labels = { wide: '26 / 33 qubit encoding sweep', stratified: 'Stratified MoE', optimized: 'Shared queue benchmark' };
const number = new Intl.NumberFormat('en', { maximumFractionDigits: 1 });

function safe(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case '&': return '&amp;';
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '"': return '&quot;';
      default: return '&#39;';
    }
  });
}

function duration(seconds: number): string {
  if (seconds < 60) return seconds.toFixed(1) + ' s';
  const whole = Math.round(seconds);
  return Math.floor(whole / 60) + 'm ' + String(whole % 60).padStart(2, '0') + 's';
}

function gib(bytes: number | null): string {
  return bytes === null ? '—' : number.format(bytes / 2 ** 30) + ' GiB';
}

function score(ap: number | undefined): string {
  return ap === undefined ? '—' : ap.toFixed(5);
}

function estimate(experiment: Experiment): number | null {
  const last = experiment.runs.at(-1);
  if (!last) return null;
  const pending = Math.max(0, experiment.total - experiment.runs.length);
  if (experiment.id !== 'wide' || !last.fold) return pending * last.seconds;
  const byFold = new Map<number, number[]>();
  for (const trial of experiment.runs) {
    if (trial.fold !== null) {
      const bucket = byFold.get(trial.fold) ?? [];
      bucket.push(trial.seconds);
      byFold.set(trial.fold, bucket);
    }
  }
  const points = Array.from(byFold, ([fold, values]) => [fold, values.reduce((a, b) => a + b, 0) / values.length]);
  const first = points[0];
  const latest = points.at(-1);
  if (!first || !latest || latest[0] === first[0]) return pending * last.seconds;
  const slope = (latest[1] - first[1]) / (latest[0] - first[0]);
  let remaining = 0;
  for (let index = experiment.runs.length; index < experiment.total; index++) {
    const fold = Math.floor(index / 6) + 1;
    remaining += Math.max(0, latest[1] + slope * (fold - latest[0]));
  }
  return remaining;
}

class RunOverview extends HTMLElement {
  set snapshot(snapshot: Snapshot) {
    this.innerHTML = snapshot.experiments.map((experiment) => {
      const completed = experiment.runs.length;
      const last = experiment.runs.at(-1);
      const total = experiment.runs.reduce((seconds, trial) => seconds + trial.seconds, 0);
      const eta = experiment.status === 'running' ? estimate(experiment) : null;
      return `<article class="card run">
        <div class="section-row"><h2>${labels[experiment.id]}</h2><span class="pill ${experiment.status}">${experiment.status}</span></div>
        <p class="caption">${experiment.id === 'wide' ? 'Monthly benchmark' : experiment.id === 'optimized' ? 'GPU serial, GPU overlap, CPU classical' : 'Shuffled admission split · first fold'}</p>
        <div class="run-progress"><strong>${completed}<small> / ${experiment.total}</small></strong><progress value="${completed}" max="${experiment.total}"></progress></div>
        <dl class="summary">
          <div><dt>Last trial</dt><dd>${last ? duration(last.seconds) : '—'}</dd></div>
          <div><dt>Measured compute</dt><dd>${duration(total)}</dd></div>
          <div><dt>Estimated remaining</dt><dd>${eta === null ? '—' : '≈ ' + duration(eta)}</dd></div>
        </dl>
        <div class="activity"><strong>${experiment.active ? safe(experiment.active) : 'Latest milestone'}</strong>
          ${experiment.observedSince !== null ? `<span class="elapsed" data-start="${experiment.observedSince}"></span>` : ''}
          <p>${safe(experiment.phase ?? (last ? 'Latest recorded trial: ' + last.name : 'Waiting for training.'))}</p></div>
      </article>`;
    }).join('');
  }
}

class HardwarePanel extends HTMLElement {
  set snapshot(snapshot: Snapshot) {
    const active = snapshot.experiments.filter((experiment) => experiment.status === 'running');
    this.innerHTML = `<section class="card">
      <div class="section-row"><h2>Hosted hardware</h2><span class="caption">Latest status sample</span></div>
      <dl class="summary hardware-metrics">
        <div><dt>Container RAM</dt><dd>${gib(snapshot.memory)} / ${gib(snapshot.memoryLimit)}</dd></div>
        <div><dt>CPU budget</dt><dd>${snapshot.cpuLimit ?? '—'} cores</dd></div>
        <div><dt>Active process CPU</dt><dd>${active.map((experiment) => experiment.cpu === null ? '—' : number.format(experiment.cpu) + ' cores').join(' · ') || '—'}</dd></div>
      </dl>
      <div class="gpus">${snapshot.gpus.map((gpu) => `<div class="gpu">
        <span>GPU ${gpu.index}</span><strong>${gpu.utilization}%</strong>
        <meter min="0" max="100" value="${gpu.utilization}">${gpu.utilization}%</meter>
        <small>${number.format(gpu.memory / 1024)} / ${number.format(gpu.limit / 1024)} GiB VRAM</small></div>`).join('')}</div>
      </section>`;
  }
}

const palette = ['#7fe2be', '#8dbbff', '#ffb87b', '#e9a7e4', '#dce486', '#f19c9a'];
type Series = { label: string; color: string; value(sample: HardwareSample): number | null };

class HardwareHistoryPanel extends HTMLElement {
  #history: HardwareHistory | null = null;
  #window = 600;
  #updated = 0;
  #poll: ReturnType<typeof setTimeout> | undefined;

  connectedCallback() {
    this.innerHTML = `<section class="card hardware-history">
      <div class="section-row"><h2>Hosted hardware history</h2>
        <label class="caption">Window <select aria-label="History window"><option value="120">2 min</option><option value="600" selected>10 min</option><option value="1800">30 min</option></select></label>
      </div><p class="caption hardware-feed">Waiting for host samples.</p><div class="history-plots"></div>
      </section>`;
    this.querySelector('select')?.addEventListener('change', (event) => {
      if (!(event.target instanceof HTMLSelectElement)) return;
      this.#window = Number(event.target.value);
      this.#draw(true);
    });
    void this.#refresh();
  }

  disconnectedCallback() {
    clearTimeout(this.#poll);
  }

  async #refresh() {
    const message = this.querySelector('.hardware-feed');
    try {
      const response = await fetch('/api/hardware', { cache: 'no-store', signal: AbortSignal.timeout(3000) });
      const payload: unknown = await response.json();
      if (!response.ok || typeof payload !== 'object' || payload === null || !('history' in payload)) {
        throw new Error('Invalid history response.');
      }
      if (isHardwareHistory(payload.history)) {
        this.#history = payload.history;
        this.#draw(false);
      }
      if (message) {
        const lag = this.#history ? Math.max(0, Math.round(Date.now() / 1000 - this.#history.updatedAt)) : null;
        const error = 'error' in payload && typeof payload.error === 'string' ? payload.error : null;
        message.textContent = error || (lag === null ? 'Waiting for host samples.'
          : `1 s host sampling · ${lag} s transport lag · Linux cgroup + NVIDIA`);
      }
    } catch {
      if (message) message.textContent = 'Local history endpoint unavailable. Showing the last samples.';
    } finally {
      if (this.isConnected) this.#poll = setTimeout(() => void this.#refresh(), 1000);
    }
  }

  #draw(force: boolean) {
    const history = this.#history;
    const target = this.querySelector('.history-plots');
    if (!target || !history || (!force && this.#updated === history.updatedAt)) return;
    this.#updated = history.updatedAt;
    const begin = history.updatedAt - this.#window;
    const samples = history.samples.filter((sample) => sample.at >= begin);
    const gpuIndices = Object.keys(history.gpuLimits).map(Number);
    const series = (values: [string, Series['value']][]): Series[] => values.map(([label, value], index) =>
      ({ label, value, color: palette[index % palette.length] }));
    const gpuSeries = (metric: 1 | 2): Series[] => gpuIndices.map((index, i) => ({
      label: 'GPU ' + index, color: palette[i % palette.length],
      value: (sample) => {
        const value = sample.gpu.find((gpu) => gpu[0] === index)?.[metric];
        return value === undefined ? null : metric === 2 ? value / 1024 : value;
      },
    }));
    const convert = (value: number | null, scale: number) => value === null ? null : value / scale;
    const chart = (title: string, unit: string, lines: Series[], fixedMax?: number) => {
      const points = lines.map((line) => samples.map((sample) => ({
        at: sample.at, value: line.value(sample),
      })));
      const max = fixedMax ?? Math.max(1, ...points.flatMap((line) => line.map((point) => point.value ?? 0)));
      const paths = lines.map((line, index) => {
        let previous: number | null = null;
        const commands = points[index].map(({ at, value }) => {
          if (value === null) {
            previous = null;
            return '';
          }
          const x = (at - begin) / this.#window * 456 + 42;
          const y = 94 - Math.min(1, Math.max(0, value / max)) * 88;
          const command = previous !== null && at - previous < 3 ? 'L' : 'M';
          previous = at;
          return command + x.toFixed(2) + ' ' + y.toFixed(2);
        }).join(' ');
        return `<path d="${commands}" stroke="${line.color}" class="history-line"/>`;
      }).join('');
      const recent = samples.at(-1);
      const clock = (time: number) => new Date(time * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
      return `<article><h3>${title}</h3><svg viewBox="0 0 500 120" role="img" aria-label="${title} over time">
        <path class="disk-grid" fill="none" d="M42 6H498 M42 50H498 M42 94H498"/>
        <text x="0" y="10">${number.format(max)}</text><text x="0" y="53">${number.format(max/2)}</text>
        <text x="0" y="97">0</text><text x="42" y="115">${clock(begin)}</text>
        <text x="498" y="115" text-anchor="end">${clock(history.updatedAt)}</text>${paths}
        </svg><div class="history-legend">${lines.map((line) => {
          const latest = recent ? line.value(recent) : null;
          return `<span style="--series:${line.color}">${line.label} ${latest === null ? '—' : number.format(latest) + ' ' + unit}</span>`;
        }).join('')}</div></article>`;
    };
    target.innerHTML = [
      chart('CPU cores used', 'cores', series([['Cgroup', (s) => s.cpu]]), history.cpuLimit ?? undefined),
      chart('RAM and swap', 'GiB', series([['Total', (s) => convert(s.memory, 2 ** 30)],
        ['Anonymous', (s) => convert(s.rss, 2 ** 30)], ['File cache', (s) => convert(s.file, 2 ** 30)],
        ['Swap', (s) => convert(s.swap, 2 ** 30)]]), history.memoryLimit ? history.memoryLimit / 2 ** 30 : undefined),
      chart('GPU activity', '%', gpuSeries(1), 100),
      chart('VRAM per device', 'GiB', gpuSeries(2), Math.max(1, ...Object.values(history.gpuLimits)) / 1024),
      chart('Physical disk I/O', 'KiB/s', series([['Read', (s) => convert(s.read, 1024)], ['Write', (s) => convert(s.write, 1024)]])),
      samples.some((sample) => sample.cpuPressure !== null || sample.memoryPressure !== null || sample.ioPressure !== null)
        ? chart('Linux pressure · 10 s average', '%', series([['CPU', (s) => s.cpuPressure],
            ['Memory', (s) => s.memoryPressure], ['I/O', (s) => s.ioPressure]]))
        : chart('CPU throttling', 's/s', series([['Cgroup throttled time per second', (s) => s.throttled]])),
    ].join('');
  }
}

class TrialProfile extends HTMLElement {
  set trial(trial: Trial | undefined) {
    if (!trial) {
      this.innerHTML = '<section class="card"><h2>Stage profile</h2><p class="caption">Waiting for a completed trial.</p></section>';
      return;
    }
    const timings = trial.timings.toSorted((a, b) => a.start - b.start || b.stop - a.stop);
    const large = timings.toSorted((a, b) => (b.stop - b.start) - (a.stop - a.start))[0];
    this.innerHTML = `<section class="card profile">
      <div class="section-row"><h2>Stage profile</h2><span class="pill">${duration(trial.seconds)}</span></div>
      <p class="caption">${safe(trial.name)} · positions reflect wall-clock offsets. Concurrent stages overlap.</p>
      <dl class="summary">
        <div><dt>Longest recorded stage</dt><dd>${large ? safe(large.name.replaceAll('_', ' ')) : '—'}</dd></div>
        <div><dt>Mean CPU cores</dt><dd>${trial.cpu === null ? '—' : number.format(trial.cpu)}</dd></div>
        <div><dt>Physical reads</dt><dd>${trial.readBytes === null ? '—' : number.format(trial.readBytes / 1048576) + ' MiB'}</dd></div>
      </dl>
      <div class="stage-list">${timings.map((stage) => {
        const offset = Math.max(0, Math.min(100, 100 * stage.start / trial.seconds));
        const length = Math.max(0, Math.min(100 - offset, 100 * (stage.stop - stage.start) / trial.seconds));
        return `<div class="stage">
          <span>${safe(stage.name.replaceAll('_', ' '))}</span>
          <div class="track"><i style="left:${offset}%;width:${length}%"></i></div>
          <span>${duration(stage.stop - stage.start)}</span></div>`;
      }).join('')}</div>
      <div class="timeline"><span>0 s</span><span>${duration(trial.seconds / 2)}</span><span>${duration(trial.seconds)}</span></div>
      </section>`;
  }
}

class TrialHistory extends HTMLElement {
  show(trials: Trial[], selected: string | undefined) {
    this.innerHTML = `<section class="card">
      <div class="section-row"><h2>Completed trials</h2><span class="caption">Held-out AP</span></div>
      <div class="scroll"><table><thead><tr><th scope="col">Configuration</th><th scope="col">Wall</th>
      <th scope="col">Classic</th><th scope="col">Hybrid</th><th scope="col">MoE</th></tr></thead>
      <tbody>${trials.toReversed().map((trial) => `<tr class="${trial.name === selected ? 'selected' : ''}">
        <td><button type="button" data-trial="${safe(trial.name)}" aria-pressed="${trial.name === selected}">${safe(trial.name)}</button></td>
        <td>${duration(trial.seconds)}</td><td>${score(trial.scores.classic)}</td><td>${score(trial.scores.hybrid)}</td>
        <td>${score(trial.scores.moe ?? trial.scores.hard_moe)}</td></tr>`).join('')}</tbody></table></div>
      </section>`;
  }
}

class SihDashboard extends HTMLElement {
  #poll: ReturnType<typeof setTimeout> | undefined;
  #clock: ReturnType<typeof setInterval> | undefined;
  #selected: string | undefined;
  #feed: Feed | undefined;

  connectedCallback() {
    this.innerHTML = `<main>
      <header><div><p class="eyebrow">Experiment monitor</p><h1>SIH-SUS</h1></div><div class="connection" role="status">Connecting…</div></header>
      <run-overview></run-overview><hardware-panel></hardware-panel><hardware-history-panel></hardware-history-panel>
      <div class="detail"><trial-profile></trial-profile><trial-history></trial-history></div>
      <section class="card log"><h2>Recent hosted milestones</h2><pre></pre></section>
      </main>`;
    this.addEventListener('click', this.#onClick);
    this.#clock = setInterval(() => this.#updateClock(), 1000);
    void this.#refresh();
  }

  disconnectedCallback() {
    clearTimeout(this.#poll);
    clearInterval(this.#clock);
    this.removeEventListener('click', this.#onClick);
  }

  #onClick = (event: MouseEvent) => {
    if (!(event.target instanceof HTMLButtonElement) || !event.target.dataset.trial) return;
    this.#selected = event.target.dataset.trial;
    this.#show();
  };

  async #refresh() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store', signal: AbortSignal.timeout(4000) });
      const feed: unknown = await response.json();
      if (!response.ok || !isFeed(feed)) throw new Error('Local status response is invalid.');
      this.#feed = feed;
    } catch (error) {
      this.#feed = { kind: 'offline', message: error instanceof Error ? error.message : 'Local API unavailable.',
                     snapshot: this.#feed?.snapshot ?? null };
    }
    this.#show();
    if (this.isConnected) this.#poll = setTimeout(() => void this.#refresh(), 5000);
  }

  #updateClock() {
    for (const element of this.querySelectorAll<HTMLElement>('[data-start]')) {
      const start = Number(element.dataset.start);
      const end = this.#feed?.kind === 'offline' ? this.#feed.snapshot?.updatedAt ?? start : Date.now() / 1000;
      element.textContent = duration(Math.max(0, end - start)) + ' observed';
      element.title = 'Elapsed since the watcher detected this trial. An already-running trial can be older.';
    }
  }

  #show() {
    if (!this.#feed) return;
    const snapshot = this.#feed.snapshot;
    const status = this.querySelector('.connection');
    if (status) {
      status.classList.toggle('online', this.#feed.kind === 'online');
      status.textContent = this.#feed.kind === 'online'
        ? 'Live · ' + new Date(this.#feed.snapshot.updatedAt * 1000).toLocaleTimeString()
        : this.#feed.message + (snapshot ? ' Showing last sample.' : '');
    }
    if (!snapshot) return;
    const overview = this.querySelector('run-overview');
    if (overview instanceof RunOverview) overview.snapshot = snapshot;
    const hardware = this.querySelector('hardware-panel');
    if (hardware instanceof HardwarePanel) hardware.snapshot = snapshot;
    const trials = snapshot.experiments.flatMap((experiment) => experiment.runs);
    const selected = trials.find((trial) => trial.name === this.#selected) ?? trials.at(-1);
    const profile = this.querySelector('trial-profile');
    if (profile instanceof TrialProfile) profile.trial = selected;
    const history = this.querySelector('trial-history');
    if (history instanceof TrialHistory) history.show(trials, selected?.name);
    const log = this.querySelector('pre');
    if (log) log.textContent = snapshot.experiments.flatMap((experiment) =>
      experiment.recent.slice(-5).map((entry) => '[' + experiment.id + '] ' + entry)).join('\n');
    this.#updateClock();
  }
}

customElements.define('run-overview', RunOverview);
customElements.define('hardware-panel', HardwarePanel);
customElements.define('hardware-history-panel', HardwareHistoryPanel);
customElements.define('trial-profile', TrialProfile);
customElements.define('trial-history', TrialHistory);
customElements.define('sih-dashboard', SihDashboard);
