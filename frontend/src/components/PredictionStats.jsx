import React, { useMemo } from "react";
import { TrendingUp, TrendingDown, Minus, BarChart3, Zap } from "lucide-react";

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const describePastWindow = (minutes) => {
  if (minutes >= 60 && minutes % 60 === 0) return `${minutes / 60}H`;
  return `${minutes}M`;
};

const describeFutureWindow = (minutes) => `${minutes}M`;

export default function PredictionStats({ summaryStats, timeline }) {
  const stats = useMemo(() => {
    const pastPoints = timeline?.filter((p) => !p.is_future) || [];
    const futurePoints = timeline?.filter((p) => p.is_future) || [];

    const avgPastRisk = pastPoints.length
      ? (pastPoints.reduce((acc, p) => acc + (toNumber(p.risk_score) ?? 0), 0) / pastPoints.length) * 100
      : 0;
    const avgFutureRisk = futurePoints.length
      ? (futurePoints.reduce((acc, p) => acc + (toNumber(p.risk_score) ?? 0), 0) / futurePoints.length) * 100
      : 0;

    const pastScrapCount = summaryStats?.past_scrap_detected || 0;
    const futureScrapCount = summaryStats?.future_scrap_predicted || 0;
    const riskTrend = avgFutureRisk - avgPastRisk;
    const scrapGap = futureScrapCount - pastScrapCount;

    return {
      pastScrapCount,
      futureScrapCount,
      avgPastRisk,
      avgFutureRisk,
      riskTrend,
      scrapGap,
      trendText:
        riskTrend > 0 ? `+${riskTrend.toFixed(1)}%` : riskTrend < 0 ? `${riskTrend.toFixed(1)}%` : "0.0%",
      trendLabel: riskTrend > 0 ? "Upward" : riskTrend < 0 ? "Downward" : "Stable",
      trendColor: riskTrend > 0 ? "text-red-500" : riskTrend < 0 ? "text-emerald-600" : "text-slate-500",
      trendBg:
        riskTrend > 0
          ? "bg-red-50 border-red-100"
          : riskTrend < 0
            ? "bg-emerald-50 border-emerald-100"
            : "bg-slate-50 border-slate-200",
      TrendIcon: riskTrend > 0 ? TrendingUp : riskTrend < 0 ? TrendingDown : Minus,
      insight:
        scrapGap > 0
          ? "Forecasted scrap is rising above the historical baseline."
          : scrapGap < 0
            ? "Forecasted scrap is trending below the historical baseline."
            : "Forecasted scrap is holding steady against the historical baseline.",
    };
  }, [summaryStats, timeline]);

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2 animate-slide-up" style={{ animationDelay: "0.3s" }}>
      <section className="glass-card relative overflow-hidden group p-6 shadow-sm hover:shadow-md transition-shadow">
        <div className="absolute top-0 right-0 w-32 h-32 bg-brand-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="mb-5 flex items-center justify-between relative z-10">
          <h3 className="flex items-center gap-2 text-sm font-bold text-slate-700">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
              <BarChart3 size={14} className="text-brand-500" />
            </div>
            Past Scrap Analysis
          </h3>
          <span className="rounded-lg border border-brand-100 bg-brand-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-brand-600">
            Actual
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-2xl border border-slate-200/80 bg-slate-50/80 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Total Scrap</p>
            <p className="mt-2 text-4xl font-extrabold text-slate-800">
              {stats.pastScrapCount}
              <span className="ml-1.5 text-sm font-medium text-slate-400">units</span>
            </p>
            <p className="mt-2 text-[11px] text-slate-400">{`Last ${describePastWindow(summaryStats?.past_window_minutes || 60)}`}</p>
          </div>
          <div className="rounded-2xl border border-brand-100 bg-brand-50/60 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-brand-500">Avg Risk</p>
            <p className="mt-2 text-4xl font-extrabold text-brand-600">{stats.avgPastRisk.toFixed(1)}%</p>
            <p className="mt-2 text-[11px] text-brand-400">Historical baseline</p>
          </div>
        </div>
      </section>

      <section className="glass-card relative overflow-hidden group p-6 shadow-sm hover:shadow-md transition-shadow">
        <div className="absolute top-0 left-0 w-32 h-32 bg-amber-400/10 rounded-full blur-3xl -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="mb-5 flex items-center justify-between relative z-10">
          <h3 className="flex items-center gap-2 text-sm font-bold text-slate-700">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-50">
              <Zap size={14} className="text-amber-500" />
            </div>
            Future Scrap Forecast
          </h3>
          <span className="rounded-lg border border-amber-100 bg-amber-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-amber-600">
            Predicted
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-2xl border border-amber-100 bg-amber-50/60 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-amber-600">Predicted Scrap</p>
            <p className="mt-2 text-4xl font-extrabold text-amber-600">
              {stats.futureScrapCount}
              <span className="ml-1.5 text-sm font-medium text-amber-400">units</span>
            </p>
            <p className="mt-2 text-[11px] text-amber-400">{`Next ${describeFutureWindow(summaryStats?.future_window_minutes || 30)}`}</p>
          </div>
          <div className={`rounded-2xl border p-4 ${stats.trendBg}`}>
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Trend</p>
            <div className="mt-2 flex items-center gap-2">
              <stats.TrendIcon size={20} className={stats.trendColor} />
              <p className={`text-3xl font-extrabold ${stats.trendColor}`}>{stats.trendText}</p>
            </div>
            <p className="mt-2 text-[11px] text-slate-400">{stats.trendLabel} · Future vs Past</p>
          </div>
        </div>

        <div className="mt-4 rounded-2xl border border-slate-200 bg-white/70 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            Operational Readout
          </p>
          <p className="mt-2 text-sm font-medium text-slate-700">
            {stats.insight}
          </p>
        </div>
      </section>
    </div>
  );
}
