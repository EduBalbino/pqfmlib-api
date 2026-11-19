import { isHardwareHistory, isSnapshot, type Feed, type HardwareHistory, type Snapshot } from './status';

const port = Number(Bun.env.SIH_DASHBOARD_PORT ?? '4317');
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error('SIH_DASHBOARD_PORT must be an integer between 1024 and 65535.');
}
const token = Bun.env.PQFM_REMOTE_TOKEN;
if (!token) throw new Error('Set PQFM_REMOTE_TOKEN before starting the dashboard.');

const hostedFiles = new URL(Bun.env.SIH_DASHBOARD_HOSTED_FILES ??
  'https://127.0.0.1:19411/files/workspace/pqfm-pilot/');
const localTunnel = hostedFiles.hostname === '127.0.0.1';
const assets = new URL('./', import.meta.url);
const bundle = await Bun.build({
  entrypoints: [new URL('./components.ts', assets).pathname],
  target: 'browser',
});
const script = bundle.outputs[0];
if (!bundle.success || !script) throw new Error(bundle.logs.map(String).join('\n'));

let previous: Snapshot | null = null;
let feed: Feed = { kind: 'offline', message: 'Waiting for the first hosted sample.', snapshot: previous };

async function refresh() {
  const started = performance.now();
  try {
    const response = await fetch(
      new URL('sih-dashboard-status.json', hostedFiles),
      {
        headers: { Authorization: 'token ' + token, 'Cache-Control': 'no-cache' },
        signal: AbortSignal.timeout(16000),
        tls: { rejectUnauthorized: !localTunnel }, // The existing local tunnel terminates with a self-signed certificate.
      },
    );
    if (!response.ok) throw new Error('Hosted snapshot returned HTTP ' + response.status);
    const snapshot: unknown = await response.json();
    if (!isSnapshot(snapshot)) throw new Error('Hosted status format is invalid.');
    previous = snapshot;
    if (Date.now() / 1000 - snapshot.updatedAt > 40) throw new Error('Hosted publisher has stopped updating.');
    feed = { kind: 'online', snapshot };
  } catch (error) {
    feed = {
      kind: 'offline',
      message: error instanceof Error ? error.message : 'Hosted telemetry is unavailable.',
      snapshot: previous,
    };
  } finally {
    setTimeout(refresh, Math.max(1000, 5000 - (performance.now() - started)));
  }
}

let hardware: HardwareHistory | null = null;
let hardwareError: string | null = 'Waiting for hardware history.';

async function refreshHardware() {
  try {
    const response = await fetch(new URL('sih-dashboard-hardware.json.gz', hostedFiles), {
      headers: { Authorization: 'token ' + token, 'Cache-Control': 'no-cache' },
      signal: AbortSignal.timeout(12000),
      tls: { rejectUnauthorized: !localTunnel },
    });
    if (!response.ok) throw new Error('Hosted hardware history returned HTTP ' + response.status);
    const raw = new Uint8Array(await response.arrayBuffer());
    const data = raw[0] === 31 && raw[1] === 139 ? Bun.gunzipSync(raw) : raw;
    const value: unknown = JSON.parse(new TextDecoder().decode(data));
    if (!isHardwareHistory(value)) throw new Error('Hardware history format is invalid.');
    hardware = value;
    hardwareError = null;
  } catch (error) {
    hardwareError = error instanceof Error ? error.message : 'Hardware history unavailable.';
  } finally {
    setTimeout(refreshHardware, 2000);
  }
}

Bun.serve({
  hostname: '127.0.0.1',
  port,
  fetch(request) {
    switch (new URL(request.url).pathname) {
      case '/': return new Response(Bun.file(new URL('./index.html', assets)));
      case '/style.css': return new Response(Bun.file(new URL('./style.css', assets)));
      case '/components.js': return new Response(script, { headers: { 'Content-Type': 'application/javascript; charset=utf-8' } });
      case '/api/hardware': return Response.json({ history: hardware, error: hardwareError },
        { headers: { 'Cache-Control': 'no-store' } });
      case '/api/status': return Response.json(feed, { headers: { 'Cache-Control': 'no-store' } });
      default: return new Response('Not found', { status: 404 });
    }
  },
});
void refresh();
void refreshHardware();
