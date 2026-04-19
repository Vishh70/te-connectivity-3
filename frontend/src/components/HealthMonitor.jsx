import React, { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart,
} from "recharts";
import { Activity } from "lucide-react";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import { mapStatus, UI_STATUS, getStatusBadgeClass, getStatusColorClass } from "../utils/statusUtils";

dayjs.extend(utc);
dayjs.extend(timezone);

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

export default function HealthMonitor({ timeline, riskScore, auditAreas = [] }) {
  const chartData = useMemo(
    () =>
      (timeline || [])
        .map((point) => {
          const risk = toNumber(point.risk_score) ?? 0;
          const isFuture = point.type === "future" || point.is_future;
          const isBridge = point.type === "bridge";
          const diagnostics = point.diagnostics || null;

          let time = null;
          if (typeof point.timestamp === "number" && Number.isFinite(point.timestamp)) {
            time = point.timestamp;
          } else if (typeof point.timestamp === "string") {
            const rawTs = point.timestamp.replace(" ", "T");
            time = Date.parse(rawTs);
          }
          if (!Number.isFinite(time)) return null;

          if (isBridge) {
            return {
              time,
              pastRisk: risk,
              futureRisk: toNumber(point.bridge_future_risk) ?? risk,
              isFuture: false,
              isBridge: true,
              diagnostics
            };
          }

          return {
            time,
            pastRisk: isFuture ? null : risk,
            futureRisk: isFuture ? risk : null,
            isFuture,
            isBridge: false,
            diagnostics
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.time - b.time),
    [timeline],
  );

  const predictionStartTime = useMemo(() => {
    const bridge = chartData.find((p) => p.isBridge);
    if (bridge) return bridge.time;
    const firstFuture = chartData.find((p) => p.isFuture);
    return firstFuture ? firstFuture.time : null;
  }, [chartData]);

  const riskBand = useMemo(() => {
    const status = mapStatus(null, riskScore);
    return { 
      label: status, 
      tone: getStatusBadgeClass(status) 
    };
  }, [riskScore]);

  if (!chartData.length) {
    return (
      <div className="glass-card flex h-64 items-center justify-center text-sm text-slate-400">
        No timeline data available.
      </div>
    );
  }

  const riskPercent = (riskScore * 100).toFixed(1);

  // High-Fidelity Diagnostic Tooltip Content
  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    const timeStr = dayjs(label).utc().format("MMM DD, HH:mm:ss");
    const riskVal = (toNumber(data.pastRisk ?? data.futureRisk) ?? 0);
    const riskText = (riskVal * 100).toFixed(2) + "%";
    
    // Find if we are currently in an audit area
    const activeAudit = auditAreas.find(a => label >= a.start && label <= a.end);

    return (
      <div className="glass-card-dark min-w-[220px] p-4 shadow-2xl backdrop-blur-xl border border-white/10 animate-in fade-in zoom-in duration-200">
        <div className="mb-3 border-b border-white/10 pb-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{timeStr} <span className="ml-2 py-0.5 px-1.5 rounded bg-white/5 text-[9px]">UTC</span></p>
          <div className="mt-1 flex items-center justify-between">
            <span className="text-xs font-semibold text-white/90">Scrap Risk</span>
            <span className={`text-sm font-black ${getStatusColorClass(mapStatus(null, riskVal))}`}>{riskText}</span>
          </div>
        </div>

        {data.diagnostics && (
          <div className="mb-3 space-y-1">
            <p className="text-[9px] font-black uppercase tracking-widest text-brand-400">Primary System Driver</p>
            <div className="flex items-center justify-between gap-3">
              <span className="text-[11px] font-bold text-slate-200 truncate max-w-[140px]">{data.diagnostics.top_sensor}</span>
              <span className="text-[11px] font-black text-brand-300">{(data.diagnostics.deviation * 100).toFixed(1)}% Δ</span>
            </div>
            <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-brand-500 transition-all duration-500" 
                style={{ width: `${Math.min(100, data.diagnostics.deviation * 100)}%` }} 
              />
            </div>
          </div>
        )}

        {activeAudit && (
          <div className="mt-3 rounded-xl bg-pink-500/10 p-2.5 border border-pink-500/20">
            <p className="text-[9px] font-black uppercase tracking-widest text-pink-400 flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-pink-500 animate-pulse" />
              {activeAudit.id} Active
            </p>
            {activeAudit.comment && (
              <p className="mt-1.5 text-[10px] italic leading-relaxed text-slate-300 line-clamp-2">
                "{activeAudit.comment}"
              </p>
            )}
            <p className="mt-1 text-[9px] font-bold text-pink-300 opacity-80 uppercase tracking-tight">Verified Scrap: {activeAudit.status}</p>
          </div>
        )}
        
        <div className="mt-2 text-[9px] font-medium text-slate-500 text-center uppercase tracking-tighter">
          {data.isFuture ? "🔮 Oracle Projection" : "📊 Historical Signal"}
        </div>
      </div>
    );
  };

  return (
    <section className="glass-card p-5 animate-slide-up" style={{ animationDelay: "0.15s" }}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-700">
            <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center shadow-inner">
              <Activity size={16} className="text-brand-500" />
            </div>
            System Health Monitor
          </h2>
          <p className="mt-1 text-xs text-slate-400 font-medium">
            Risk trajectory optimized via <span className="text-brand-600 font-bold">V9 Neural Oracle</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded-full px-3 py-1 text-[11px] font-black uppercase tracking-[0.2em] shadow-sm ${riskBand.tone}`}
          >
            {riskBand.label}
          </span>
          <div className="rounded-2xl border border-slate-200 bg-white/50 backdrop-blur-sm px-4 py-2 text-right shadow-sm">
            <span className="block text-[10px] font-bold uppercase tracking-widest text-slate-400">
              Current Risk
            </span>
            <span
              className={`text-2xl font-black ${getStatusColorClass(mapStatus(null, riskScore))}`}
            >
              {riskPercent}%
            </span>
          </div>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2 text-[10px] font-black uppercase tracking-[0.15em] text-slate-500">
        <span className="rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1 text-emerald-700">Safe 0-35%</span>
        <span className="rounded-full border border-amber-100 bg-amber-50 px-3 py-1 text-amber-700">Watch 35-60%</span>
        <span className="rounded-full border border-orange-100 bg-orange-50 px-3 py-1 text-orange-700">High 60-80%</span>
        <span className="rounded-full border border-red-100 bg-red-50 px-3 py-1 text-red-700">Critical 80-100%</span>
      </div>

      <div className="h-[22rem] rounded-2xl border border-slate-100 bg-slate-50/80 px-3 py-2 shadow-inner relative overflow-hidden">
        {/* Subtle Risk Band Backgrounds */}
        <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.03]">
          <div className="h-[35%] w-full bg-emerald-500" />
          <div className="h-[25%] w-full bg-amber-500" />
          <div className="h-[20%] w-full bg-orange-500" />
          <div className="h-[20%] w-full bg-red-500" />
        </div>

        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 20, right: 35, bottom: 0, left: 8 }}>
            <defs>
              <linearGradient id="pastGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.6} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="futureGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f97316" stopOpacity={0.6} />
                <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <ReferenceArea y1={0} y2={0.35} fill="#10b981" fillOpacity={0.02} strokeOpacity={0} />
            <ReferenceArea y1={0.35} y2={0.6} fill="#f59e0b" fillOpacity={0.02} strokeOpacity={0} />
            <ReferenceArea y1={0.6} y2={0.8} fill="#f97316" fillOpacity={0.02} strokeOpacity={0} />
            <ReferenceArea y1={0.8} y2={1} fill="#ef4444" fillOpacity={0.02} strokeOpacity={0} />
            
            {auditAreas.map((area, idx) => (
              <ReferenceArea 
                key={`audit-${idx}`} 
                x1={area.start} 
                x2={area.end} 
                fill="#ec4899" 
                fillOpacity={0.15} 
                stroke="#ec4899"
                strokeWidth={1}
                strokeDasharray="3 3"
                label={{ 
                  value: area.id, 
                  position: 'insideTopLeft', 
                  fill: '#be185d', 
                  fontSize: 10, 
                  fontWeight: '900', 
                  opacity: 0.9,
                  offset: 15
                }} 
              />
            ))}
            <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
            <XAxis
              dataKey="time"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => dayjs(t).utc().format("HH:mm")}
              tick={{ fill: "#64748b", fontSize: 11, fontWeight: 600 }}
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={{ stroke: "#cbd5e1" }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "#64748b", fontSize: 11, fontWeight: 600 }}
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={{ stroke: "#cbd5e1" }}
              label={{
                value: "Risk Probability",
                angle: -90,
                position: "insideLeft",
                offset: -10,
                fill: "#64748b",
                fontSize: 12,
                fontWeight: 700,
                style: { textTransform: 'uppercase', letterSpacing: '0.05em' }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              height={36}
              iconType="circle"
              wrapperStyle={{ fontSize: "11px", fontWeight: 700, color: "#475569", textTransform: 'uppercase', letterSpacing: '0.1em' }}
            />
            <Area
              type="monotone"
              dataKey="pastRisk"
              name="Past (Actual)"
              stroke="#3b82f6"
              strokeWidth={3}
              fill="url(#pastGradient)"
              dot={false}
              connectNulls={false}
              activeDot={{ r: 6, strokeWidth: 0, fill: '#3b82f6' }}
            />
            <Area
              type="monotone"
              dataKey="futureRisk"
              name="Future (Predicted)"
              stroke="#f97316"
              strokeWidth={3}
              strokeDasharray="8 6"
              fill="url(#futureGradient)"
              dot={{ r: 4, fill: "#f97316", stroke: "#fff", strokeWidth: 2 }}
              activeDot={{ r: 7, strokeWidth: 0, fill: '#f97316' }}
              connectNulls={false}
            />
            {predictionStartTime && (
              <ReferenceLine
                x={predictionStartTime}
                stroke="#64748b"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ 
                  value: "Live Prediction Horizon", 
                  fill: "#334155", 
                  fontSize: 10, 
                  fontWeight: 900,
                  position: "top",
                  offset: 20,
                  className: "uppercase tracking-tighter"
                }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
