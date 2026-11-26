export interface LatencySample {
  total: number;
  inference: number;
}

interface LatencySparklineProps {
  samples: LatencySample[];
  width?: number;
  height?: number;
}

export function LatencySparkline({
  samples,
  width = 112,
  height = 24,
}: LatencySparklineProps) {
  const padY = 2;
  const usableH = height - padY * 2;
  const maxMs = Math.max(
    40,
    ...samples.map((s) => Math.max(s.total, s.inference))
  );

  const toPoints = (values: number[]): string => {
    if (values.length === 0) return '';
    if (values.length === 1) {
      const y = height - padY - (values[0] / maxMs) * usableH;
      return `0,${y.toFixed(1)} ${width},${y.toFixed(1)}`;
    }
    const stepX = width / (values.length - 1);
    return values
      .map((v, i) => {
        const x = i * stepX;
        const y = height - padY - (v / maxMs) * usableH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  };

  const totalPts = toPoints(samples.map((s) => s.total));
  const infPts = toPoints(samples.map((s) => s.inference));
  const areaPts = totalPts
    ? `0,${height} ${totalPts} ${width},${height}`
    : '';

  return (
    <svg
      role="img"
      aria-label={
        samples.length > 0
          ? `Live latency sparkline over last ${samples.length} frames, peak ${maxMs.toFixed(0)} milliseconds`
          : 'Live latency sparkline awaiting frames'
      }
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="overflow-visible"
    >
      {/* Subtle baseline */}
      <line
        x1={0}
        y1={height - 1}
        x2={width}
        y2={height - 1}
        stroke="#334155"
        strokeWidth="1"
      />
      {samples.length > 0 ? (
        <>
          <polygon points={areaPts} fill="rgba(251, 191, 36, 0.14)" />
          <polyline
            fill="none"
            stroke="#fbbf24"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={totalPts}
          />
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="1.25"
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray="2 1.5"
            points={infPts}
          />
        </>
      ) : (
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="#475569"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
      )}
    </svg>
  );
}
