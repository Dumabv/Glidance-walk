import { useCallback, useEffect, useId, useState } from 'react';
import {
  BarChart3,
  Camera,
  Check,
  CheckCircle2,
  ClipboardCopy,
  Download,
  Play,
  Trash2,
  Upload,
  XCircle,
} from 'lucide-react';
import type {
  BenchmarkReport,
  SampleLabel,
  SampleRecord,
} from '../types/telemetry';

const LABEL_OPTIONS: { value: SampleLabel; title: string }[] = [
  { value: 'clear', title: 'Clear Tilted Floor' },
  { value: 'obstacle_left', title: 'Obstacle Left' },
  { value: 'obstacle_center', title: 'Obstacle Center / Wall' },
  { value: 'obstacle_right', title: 'Obstacle Right' },
  { value: 'curb_drop', title: 'Curb / Stairs Down' },
];

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export function BenchmarkLab() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [samples, setSamples] = useState<SampleRecord[]>([]);
  const [running, setRunning] = useState(false);
  const [includeSynthetic, setIncludeSynthetic] = useState(true);
  const [label, setLabel] = useState<SampleLabel>('clear');
  const [note, setNote] = useState('');
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [copiedSummary, setCopiedSummary] = useState(false);
  const idPrefix = useId();

  const downloadBenchmarkJson = () => {
    if (!report) return;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `glidance-benchmark-${stamp}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const copySlideSummary = async () => {
    if (!report) return;
    const summary = [
      `Glidance v2 Benchmark (${report.total_samples} scenes: ${report.real_samples} real, ${report.synthetic_samples} calibration | Device: ${report.device.toUpperCase()})`,
      `- Overall F1: Glidance ${pct(report.smart.f1)} vs Naive ${pct(report.naive.f1)}`,
      `- Curb / Drop-Off Recall: Glidance ${pct(report.smart.curb_recall)} vs Naive ${pct(report.naive.curb_recall)}`,
      `- Directional Accuracy: Glidance ${pct(report.smart.direction_accuracy)} vs Naive ${pct(report.naive.direction_accuracy)}`,
      `- Clear-Floor False Alarms: Glidance ${pct(report.smart.false_alarm_rate_clear)} vs Naive ${pct(report.naive.false_alarm_rate_clear)}`,
      `- Latency: P50 ${report.smart.p50_ms.toFixed(1)} ms | P95 ${report.smart.p95_ms.toFixed(1)} ms`,
    ].join('\n');
    try {
      await navigator.clipboard.writeText(summary);
      setCopiedSummary(true);
      window.setTimeout(() => setCopiedSummary(false), 2000);
    } catch {
      setStatusMsg('Clipboard access blocked — use Download JSON instead.');
    }
  };

  const loadSamples = useCallback(async () => {
    const res = await fetch('/api/samples');
    if (res.ok) {
      setSamples(await res.json());
    }
  }, []);

  const runEval = useCallback(
    async (withSynthetic = includeSynthetic) => {
      setRunning(true);
      setStatusMsg(null);
      try {
        const res = await fetch(
          `/api/benchmark/run?include_synthetic=${withSynthetic}`,
          { method: 'POST' }
        );
        if (!res.ok) throw new Error('Benchmark failed');
        setReport(await res.json());
      } catch (err) {
        setStatusMsg(
          err instanceof Error ? err.message : 'Failed to run benchmark'
        );
      } finally {
        setRunning(false);
      }
    },
    [includeSynthetic]
  );

  useEffect(() => {
    void loadSamples();
    void runEval(true);
  }, [loadSamples, runEval]);

  const captureLiveFrame = async () => {
    setStatusMsg(null);
    const res = await fetch('/api/samples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, note }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setStatusMsg(body.detail || 'Stream a phone frame first or upload a photo.');
      return;
    }
    setNote('');
    setStatusMsg('Saved live hallway frame!');
    await loadSamples();
    await runEval();
  };

  const uploadPhoto = async (file: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      const image_b64 = typeof reader.result === 'string' ? reader.result : '';
      const res = await fetch('/api/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          note: note || file.name,
          image_b64,
        }),
      });
      if (res.ok) {
        setNote('');
        setStatusMsg(`Uploaded ${file.name} and re-scored!`);
        await loadSamples();
        await runEval();
      }
    };
    reader.readAsDataURL(file);
  };

  const removeSample = async (sampleId: string) => {
    await fetch(`/api/samples/${sampleId}`, { method: 'DELETE' });
    await loadSamples();
    await runEval();
  };

  const metricRows = report
    ? [
        {
          name: 'Overall F1 Score (Higher is better)',
          naive: report.naive.f1,
          smart: report.smart.f1,
          lowerBetter: false,
        },
        {
          name: 'Curb / Drop-Off Recall (Higher is better)',
          naive: report.naive.curb_recall,
          smart: report.smart.curb_recall,
          lowerBetter: false,
        },
        {
          name: 'Directional Accuracy (Higher is better)',
          naive: report.naive.direction_accuracy,
          smart: report.smart.direction_accuracy,
          lowerBetter: false,
        },
        {
          name: 'Clear-Floor False-Alarm Rate (Lower is better)',
          naive: report.naive.false_alarm_rate_clear,
          smart: report.smart.false_alarm_rate_clear,
          lowerBetter: true,
        },
      ]
    : [];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 pb-12 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Midterm Evaluation &amp; Dataset Lab
          </h1>
          <p className="text-sm text-slate-400">
            Quantitative accuracy &amp; latency benchmark: Naive Proximity
            Baseline vs. Glidance Ground-Aware + Curb Engine
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-200">
            <input
              type="checkbox"
              checked={includeSynthetic}
              onChange={(e) => setIncludeSynthetic(e.target.checked)}
              className="accent-sky-400"
            />
            Include 15 Calibration Scenes
          </label>
          <button
            type="button"
            onClick={() => void runEval()}
            disabled={running}
            className="inline-flex items-center gap-2 rounded-xl bg-sky-500 px-4 py-2.5 text-sm font-bold text-slate-950 hover:bg-sky-400 disabled:opacity-60"
          >
            <Play className="h-4 w-4" aria-hidden="true" />
            {running ? 'Running Benchmark...' : 'Run Benchmark'}
          </button>
          <button
            type="button"
            onClick={downloadBenchmarkJson}
            disabled={!report}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2.5 text-xs font-bold text-slate-100 hover:border-sky-400 hover:bg-slate-800 disabled:opacity-50"
          >
            <Download className="h-4 w-4 text-sky-400" aria-hidden="true" />
            Download Benchmark JSON
          </button>
          <button
            type="button"
            onClick={() => void copySlideSummary()}
            disabled={!report}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2.5 text-xs font-bold text-slate-100 hover:border-emerald-400 hover:bg-slate-800 disabled:opacity-50"
          >
            {copiedSummary ? (
              <Check className="h-4 w-4 text-emerald-400" aria-hidden="true" />
            ) : (
              <ClipboardCopy className="h-4 w-4 text-emerald-400" aria-hidden="true" />
            )}
            {copiedSummary ? 'Copied Summary!' : 'Copy Slide Summary'}
          </button>
        </div>
      </div>

      {/* Headline Comparison Bars */}
      {report && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <h2 className="flex items-center gap-2 text-base font-bold text-white">
              <BarChart3 className="h-5 w-5 text-sky-400" aria-hidden="true" />
              Engine Comparison ({report.total_samples} scenes:{' '}
              {report.real_samples} real, {report.synthetic_samples} calibration)
            </h2>
            <div className="flex flex-wrap gap-3 text-xs font-semibold">
              <span className="rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">
                P50: {report.smart.p50_ms.toFixed(1)} ms
              </span>
              <span className="rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">
                P95: {report.smart.p95_ms.toFixed(1)} ms
              </span>
              <span className="rounded-md bg-sky-500/20 px-2.5 py-1 text-sky-300">
                Device: {report.device.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {metricRows.map((row) => (
              <div
                key={row.name}
                className="rounded-xl border border-slate-800 bg-slate-950/70 p-4"
              >
                <p className="mb-3 text-xs font-semibold text-slate-300">
                  {row.name}
                </p>

                <div className="mb-2">
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-slate-400">Naive Center Threshold</span>
                    <span className="font-mono text-slate-300">
                      {pct(row.naive)}
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full ${
                        row.lowerBetter ? 'bg-red-500/80' : 'bg-slate-500'
                      }`}
                      style={{ width: `${Math.max(4, row.naive * 100)}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="font-semibold text-sky-300">
                      Glidance Ground-Aware + Curb
                    </span>
                    <span className="font-mono font-bold text-emerald-300">
                      {pct(row.smart)}
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full ${
                        row.lowerBetter ? 'bg-emerald-400' : 'bg-sky-400'
                      }`}
                      style={{ width: `${Math.max(4, row.smart * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Hallway Sample Recorder / Uploader */}
      <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
        <h2 className="mb-3 text-base font-bold text-white">
          Add Real Hallway &amp; Stair Photos (Team Dataset)
        </h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
          <div className="flex flex-col gap-1 md:col-span-3">
            <label
              htmlFor={`${idPrefix}-label`}
              className="text-xs font-medium text-slate-300"
            >
              Ground-Truth Label
            </label>
            <select
              id={`${idPrefix}-label`}
              value={label}
              onChange={(e) => setLabel(e.target.value as SampleLabel)}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
            >
              {LABEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.title}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1 md:col-span-4">
            <label
              htmlFor={`${idPrefix}-note`}
              className="text-xs font-medium text-slate-300"
            >
              Scene Note
            </label>
            <input
              id={`${idPrefix}-note`}
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. FCIM hallway chair 1.2m ahead"
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
            />
          </div>

          <div className="flex flex-wrap items-end gap-2 md:col-span-5">
            <button
              type="button"
              onClick={() => void captureLiveFrame()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3.5 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-400"
            >
              <Camera className="h-4 w-4" aria-hidden="true" />
              Save Live Phone Frame
            </button>

            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2 text-xs font-semibold text-slate-100 hover:border-sky-400">
              <Upload className="h-4 w-4" aria-hidden="true" />
              Upload Hallway Photo
              <input
                type="file"
                accept="image/*"
                className="visually-hidden"
                onChange={(e) => void uploadPhoto(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>
        </div>

        {statusMsg && (
          <p role="status" className="mt-2.5 text-xs font-medium text-sky-300">
            {statusMsg}
          </p>
        )}

        {samples.length > 0 && (
          <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {samples.map((s) => (
              <li
                key={s.sample_id}
                className="flex flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950"
              >
                {s.image_b64 && (
                  <img
                    src={`data:image/jpeg;base64,${s.image_b64}`}
                    alt={`Captured hallway sample labeled ${s.label}`}
                    className="aspect-[4/3] w-full bg-slate-950 object-contain"
                  />
                )}
                <div className="flex items-center justify-between gap-1 p-2">
                  <div className="min-w-0">
                    <p className="truncate text-[11px] font-bold text-sky-300">
                      {s.label}
                    </p>
                    <p className="truncate text-[10px] text-slate-400">
                      {s.note}
                    </p>
                  </div>
                  <button
                    type="button"
                    aria-label={`Delete sample ${s.note}`}
                    onClick={() => void removeSample(s.sample_id)}
                    className="rounded p-1 text-slate-400 hover:bg-red-500/20 hover:text-red-300"
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Semantic Accessible Per-Sample Table */}
      {report && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <div
            role="region"
            aria-label="Per-sample benchmark breakdown"
            tabIndex={0}
            className="overflow-x-auto"
          >
            <table className="w-full min-w-[640px] border-collapse text-left text-xs">
              <caption className="mb-3 text-left text-base font-bold text-white">
                Per-Scene Evaluation Breakdown
              </caption>
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th scope="col" className="px-3 py-2">
                    Scene
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Ground Truth
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Naive Baseline
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Glidance Steering
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Glidance Result
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Latency
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {report.samples.map((row) => (
                  <tr key={row.sample_id} className="hover:bg-slate-800/40">
                    <td className="px-3 py-2 font-medium text-slate-200">
                      {row.note}{' '}
                      <span className="ml-1 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
                        {row.source}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-sky-300">
                      {row.label}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex items-center gap-1 font-semibold ${
                          row.naive_correct ? 'text-emerald-400' : 'text-red-400'
                        }`}
                      >
                        {row.naive_correct ? (
                          <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                        ) : (
                          <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        )}
                        {row.naive_detected ? 'Alert' : 'Clear'}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-slate-200">
                      {row.smart_steering}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex items-center gap-1 font-semibold ${
                          row.smart_correct ? 'text-emerald-400' : 'text-red-400'
                        }`}
                      >
                        {row.smart_correct ? (
                          <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                        ) : (
                          <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        )}
                        {row.smart_correct ? 'Pass' : 'Miss'}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-slate-300">
                      {row.smart_ms.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
