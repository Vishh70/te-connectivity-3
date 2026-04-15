import React, { useMemo, useState } from "react";
import { Activity, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const formatCauseName = (value) =>
  String(value || "Unknown Cause")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export default function RootCause({ rootCauses, onSelectSensor }) {
  const topCauses = useMemo(
    () => (Array.isArray(rootCauses) ? rootCauses.slice(0, 3) : []),
    [rootCauses],
  );
  const [expandedIndex, setExpandedIndex] = useState(null);

  const maxImpact = useMemo(() => {
    const impacts = topCauses
      .map((entry) => Math.abs(toNumber(entry?.impact) ?? 0))
      .filter((value) => value > 0);
    return impacts.length ? Math.max(...impacts, 1) : 1;
  }, [topCauses]);

  return (
    <section className="glass-card p-5 animate-slide-up" style={{ animationDelay: "0.2s" }}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-700">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-50">
              <AlertTriangle size={16} className="text-amber-500" />
            </div>
            Root Cause Analysis
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Ranked signals show what is most likely driving the current machine state.
          </p>
        </div>
        <span className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400 shadow-sm">
          {topCauses.length} detected
        </span>
      </div>

      {!topCauses.length ? (
        <div className="rounded-3xl border border-emerald-100 bg-emerald-50/70 px-6 py-8 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100">
            <CheckCircle2 size={28} className="text-emerald-600" />
          </div>
          <p className="mt-4 text-lg font-bold text-emerald-700">All Systems Normal</p>
          <p className="mt-2 text-sm text-emerald-600/90">
            No root causes detected right now. The machine is operating inside the expected range.
          </p>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
            <span className="rounded-full border border-emerald-200 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
              Stable signal
            </span>
            <span className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              No escalation required
            </span>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {topCauses.map((entry, index) => {
            const impact = toNumber(entry?.impact);
            const isExpanded = expandedIndex === index;
            const hasParams = Array.isArray(entry?.top_parameters) && entry.top_parameters.length > 0;
            const normalizedImpact = Math.min(1, Math.abs(impact ?? 0) / maxImpact);
            const positive = (impact ?? 0) >= 0;

            return (
              <div
                key={`${entry?.cause || "cause"}-${index}`}
                role="button"
                tabIndex={hasParams ? 0 : -1}
                aria-disabled={!hasParams}
                aria-expanded={isExpanded}
                className={`group relative overflow-hidden w-full rounded-2xl border px-5 py-4 text-left transition-all duration-300 ${
                  hasParams ? "cursor-pointer" : "cursor-default"
                } ${
                  isExpanded
                    ? "border-amber-300/80 bg-gradient-to-br from-amber-50 to-white shadow-[0_8px_30px_rgba(245,158,11,0.12)] ring-1 ring-amber-500/10"
                    : "border-amber-100/50 bg-white/40 hover:bg-amber-50/60 hover:shadow-lg hover:-translate-y-0.5 hover:border-amber-200"
                } ${!hasParams ? "opacity-80" : ""}`}
                onClick={() => hasParams && setExpandedIndex(isExpanded ? null : index)}
                onKeyDown={(event) => {
                  if (!hasParams) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setExpandedIndex(isExpanded ? null : index);
                  }
                }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-200 text-xs font-bold text-amber-800">
                        {index + 1}
                      </span>
                      <p className="truncate text-sm font-semibold text-slate-700">
                        {formatCauseName(entry?.cause)}
                      </p>
                      {hasParams &&
                        (isExpanded ? (
                          <ChevronUp size={14} className="text-amber-500" />
                        ) : (
                          <ChevronDown size={14} className="text-amber-400" />
                        ))}
                    </div>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/80">
                      <div
                        className={`h-full rounded-full ${positive ? "bg-amber-500" : "bg-emerald-500"}`}
                        style={{ width: `${Math.max(8, normalizedImpact * 100)}%` }}
                      />
                    </div>
                  </div>
                  <span className={`relative rounded-xl border px-3 py-1.5 text-xs font-bold shadow-sm backdrop-blur-sm z-10 transition-colors ${isExpanded ? "border-amber-300 bg-amber-100/80 text-amber-800" : "border-amber-200/60 bg-white/80 text-amber-700"} flex items-center gap-1.5`}>
                    <span className="text-[10px] font-black uppercase tracking-wider opacity-60">Impact</span>
                    <span>{impact !== null ? impact.toFixed(3) : "--"}</span>
                  </span>
                </div>

                {hasParams && isExpanded && (
                  <div className="mt-4 rounded-xl border border-amber-200/60 bg-white/70 p-4">
                    <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.22em] text-amber-500/70">
                      Specific Drivers
                    </p>
                    <div className="space-y-2">
                      {entry.top_parameters.map((p, pIdx) => {
                        const parameterImpact = toNumber(p?.impact) ?? 0;
                        const parameterName = p?.parameter;
                        return (
                          <button
                            key={`${parameterName || "param"}-${pIdx}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (onSelectSensor && parameterName) onSelectSensor(parameterName);
                            }}
                            className="flex w-full items-center justify-between gap-3 rounded-lg bg-slate-50 px-3 py-2 text-xs transition-colors hover:bg-slate-100 hover:shadow-sm group/param"
                          >
                            <div className="flex items-center gap-2 overflow-hidden">
                              <Activity size={12} className="shrink-0 text-slate-300 group-hover/param:text-brand-500 transition-colors" />
                              <span className="font-medium text-slate-600 truncate">
                                {formatCauseName(parameterName)}
                              </span>
                            </div>
                            <span
                              className={`font-mono font-semibold ${
                                parameterImpact > 0 ? "text-red-500" : "text-emerald-600"
                              }`}
                            >
                              {parameterImpact > 0 ? "+" : ""}
                              {parameterImpact.toFixed(4)}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
