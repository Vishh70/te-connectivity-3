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
    <div className="flex flex-col gap-6 animate-slide-up" style={{ animationDelay: "0.3s" }}>
      <section className="glass-card relative overflow-hidden group p-6 shadow-sm border-slate-200">
        <div className="absolute top-0 right-0 w-32 h-32 bg-brand-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="mb-6 flex items-center justify-between relative z-10">
          <h3 className="flex items-center gap-3 text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-brand-50 border border-brand-100 shadow-sm">
              <BarChart3 size={16} className="text-brand-600" />
            </div>
            Past Analysis
          </h3>
          <span className="rounded-full bg-brand-600 px-3 py-1 text-[9px] font-black uppercase tracking-widest text-white shadow-lg shadow-brand-900/20">
            Actual
          </span>
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-100 bg-slate-50/50 p-5 group/item transition-colors hover:bg-white hover:border-brand-200">
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Total Scrap detected</p>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-4xl font-black text-slate-800 tracking-tight">
                {stats.pastScrapCount}
              </span>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">shots</span>
            </div>
            <p className="mt-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest">{`Window: ${describePastWindow(summaryStats?.past_window_minutes || 60)}`}</p>
          </div>

          <div className="rounded-2xl border border-brand-50 bg-brand-50/20 p-5 border-l-4 border-l-brand-500">
            <p className="text-[10px] font-black uppercase tracking-widest text-brand-600 opacity-80">Avg Operational Risk</p>
            <div className="mt-2 text-4xl font-black text-brand-600 tracking-tight leading-none">
              {stats.avgPastRisk.toFixed(1)}%
            </div>
            <p className="mt-3 text-[9px] font-bold text-brand-400 uppercase tracking-widest">Historical Baseline</p>
          </div>
        </div>
      </section>

      <section className="glass-card relative overflow-hidden group p-6 shadow-sm border-slate-200">
        <div className="absolute top-0 left-0 w-32 h-32 bg-amber-400/10 rounded-full blur-3xl -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="mb-6 flex items-center justify-between relative z-10">
          <h3 className="flex items-center gap-3 text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-amber-50 border border-amber-100 shadow-sm">
              <Zap size={16} className="text-amber-500" />
            </div>
            Future Forecast
          </h3>
          <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[9px] font-black uppercase tracking-widest text-amber-700">
            AI Model
          </span>
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-amber-50 bg-amber-50/20 p-5 border-l-4 border-l-amber-500">
            <p className="text-[10px] font-black uppercase tracking-widest text-amber-600 opacity-80">Predicted Scrap</p>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-4xl font-black text-amber-600 tracking-tight">
                {stats.futureScrapCount}
              </span>
              <span className="text-[10px] font-bold text-amber-500/60 uppercase tracking-widest">Units</span>
            </div>
            <p className="mt-3 text-[9px] font-bold text-amber-400 uppercase tracking-widest">{`Horizon: ${describeFutureWindow(summaryStats?.future_window_minutes || 30)}`}</p>
          </div>

          <div className={`rounded-2xl border p-5 border-l-4 ${stats.trendBg} ${stats.trendLabel === 'Upward' ? 'border-l-red-500' : stats.trendLabel === 'Downward' ? 'border-l-emerald-500' : 'border-l-slate-400'}`}>
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">Risk Variance</p>
            <div className="mt-2 flex items-center gap-3">
              <stats.TrendIcon size={24} className={stats.trendColor} />
              <p className={`text-3xl font-black tracking-tighter ${stats.trendColor}`}>{stats.trendText}</p>
            </div>
            <p className="mt-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest">{stats.trendLabel} · Projection vs History</p>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-slate-100 bg-slate-50/30 p-5 bg-gradient-to-r from-slate-50 to-white">
          <p className="text-[9px] font-black uppercase tracking-[0.25em] text-slate-400 mb-3">
            Operational Intelligence
          </p>
          <p className="text-[11px] font-bold text-slate-600 leading-relaxed uppercase tracking-wide">
            {stats.insight}
          </p>
        </div>
      </section>
    </div>
  );
}
