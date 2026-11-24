import { useEffect, useState } from 'react';
import { BarChart3, Compass, Radar, Smartphone } from 'lucide-react';
import { BenchmarkLab } from './pages/BenchmarkLab';
import { MissionControl } from './pages/MissionControl';
import { MobileWalker } from './pages/MobileWalker';

type ActiveTab = 'dashboard' | 'mobile' | 'benchmark';

function getInitialTab(): ActiveTab {
  if (typeof window === 'undefined') return 'dashboard';
  const mode = new URLSearchParams(window.location.search).get('mode');
  if (mode === 'mobile') return 'mobile';
  if (mode === 'benchmark') return 'benchmark';
  return 'dashboard';
}

const TAB_TITLES: Record<ActiveTab, string> = {
  dashboard: 'Mission Control | Glidance Depth Radar',
  mobile: 'Mobile Walker HUD | Glidance Depth Radar',
  benchmark: 'Midterm Benchmark Lab | Glidance Depth Radar',
};

export default function App() {
  const [tab, setTab] = useState<ActiveTab>(getInitialTab);

  useEffect(() => {
    document.title = TAB_TITLES[tab];
  }, [tab]);

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100">
      <a
        href="#main-content"
        className="visually-hidden fixed left-3 top-3 z-50 rounded-lg bg-sky-400 px-3 py-2 text-sm font-bold text-slate-950"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-500/20 text-sky-400">
              <Compass className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <span className="text-base font-extrabold tracking-tight text-white">
                Glidance
              </span>
              <span className="ml-2 hidden rounded bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-sky-300 sm:inline">
                Depth-Anything-V2 · Hallway Prototype
              </span>
            </div>
          </div>

          <nav aria-label="Primary" className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              aria-current={tab === 'dashboard' ? 'page' : undefined}
              onClick={() => setTab('dashboard')}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                tab === 'dashboard'
                  ? 'bg-sky-500 text-slate-950'
                  : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
              }`}
            >
              <Radar className="h-4 w-4" aria-hidden="true" />
              Mission Control
            </button>

            <button
              type="button"
              aria-current={tab === 'mobile' ? 'page' : undefined}
              onClick={() => setTab('mobile')}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                tab === 'mobile'
                  ? 'bg-sky-500 text-slate-950'
                  : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
              }`}
            >
              <Smartphone className="h-4 w-4" aria-hidden="true" />
              Mobile Walker
            </button>

            <button
              type="button"
              aria-current={tab === 'benchmark' ? 'page' : undefined}
              onClick={() => setTab('benchmark')}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                tab === 'benchmark'
                  ? 'bg-sky-500 text-slate-950'
                  : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
              }`}
            >
              <BarChart3 className="h-4 w-4" aria-hidden="true" />
              Benchmark Lab
            </button>
          </nav>
        </div>
      </header>

      <main id="main-content" tabIndex={-1}>
        {tab === 'dashboard' && <MissionControl />}
        {tab === 'mobile' && <MobileWalker />}
        {tab === 'benchmark' && <BenchmarkLab />}
      </main>
    </div>
  );
}
