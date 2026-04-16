import React, { useState, useEffect, useMemo } from "react";
import apiClient from "../utils/apiClient";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, Cell, Legend
} from "recharts";
import {
  BarChart3, Cpu, AlertTriangle, ShieldCheck, TrendingUp,
  Target, Zap, RefreshCw, Activity, CheckCircle2, XCircle
} from "lucide-react";

const STATUS_COLOR = {
  NORMAL:   { bg: "bg-emerald-500", text: "text-emerald-600", ring: "#10b981" },
  LOW:      { bg: "bg-emerald-400", text: "text-emerald-500", ring: "#34d399" },
  MEDIUM:   { bg: "bg-amber-400",   text: "text-amber-600",   ring: "#f59e0b" },
  WATCH:    { bg: "bg-amber-500",   text: "text-amber-600",   ring: "#f59e0b" },
  HIGH:     { bg: "bg-orange-500",  text: "text-orange-600",  ring: "#f97316" },
  CRITICAL: { bg: "bg-red-500",    text: "text-red-600",     ring: "#ef4444" },
  OFFLINE:  { bg: "bg-slate-400",  text: "text-slate-500",   ring: "#94a3b8" },
};

function getStatusConfig(status) {
  return STATUS_COLOR[status?.toUpperCase()] || STATUS_COLOR.OFFLINE;
}

function RiskGauge({ value, label, color }) {
  const data = [{ name: label, value, fill: color }];
  return (
    <div className="flex flex-col items-center gap-1">
      <ResponsiveContainer width={100} height={100}>
        <RadialBarChart innerRadius={32} outerRadius={48} data={data} startAngle={180} endAngle={0}>
          <RadialBar dataKey="value" maxBarSize={12} cornerRadius={6} background={{ fill: "#f1f5f9" }} />
        </RadialBarChart>
      </ResponsiveContainer>
      <span className="text-xl font-black text-slate-800 -mt-8">{value}%</span>
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-6">{label}</span>
    </div>
  );
}

const CustomBarTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-100 rounded-xl shadow-xl px-4 py-3 text-xs">
      <p className="font-black text-slate-700 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="font-semibold" style={{ color: p.fill }}>
          {p.name}: {typeof p.value === "number" ? (p.name === "Risk Score" ? (p.value * 100).toFixed(1) + "%" : p.value) : p.value}
        </p>
      ))}
    </div>
  );
};

export default function AnalyticsHub({ onViewMachine }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      const res = await apiClient.get("/api/analytics/fleet");
      setData(res.data);
      setError(null);
    } catch (err) {
      setError("Failed to load fleet analytics. Backend may be processing.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const { machines = [], fleet_summary = {} } = data || {};

  const sortedMachines = useMemo(() =>
    [...machines].sort((a, b) => b.risk_score - a.risk_score), [machines]);

  const riskChartData = useMemo(() =>
    sortedMachines.map(m => ({
      id: m.display_id || m.id,
      "Risk Score": m.risk_score,
      fill: getStatusConfig(m.status).ring,
    })), [sortedMachines]);

  const alertChartData = useMemo(() =>
    sortedMachines.map(m => ({
      id: m.display_id || m.id,
      "Active Alerts": m.active_alerts,
      "Scrap Events": m.past_scrap_detected,
    })), [sortedMachines]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <div className="w-10 h-10 rounded-full border-4 border-slate-200 border-t-brand-500 animate-spin" />
        <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Loading Fleet Analytics…</p>
        <p className="text-xs text-slate-300">This may take up to 30s on first load</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <AlertTriangle size={32} className="text-amber-400" />
        <p className="text-sm font-bold text-slate-500">{error}</p>
        <button onClick={() => fetchData()} className="px-4 py-2 bg-brand-600 text-white rounded-xl text-xs font-bold">
          Retry
        </button>
      </div>
    );
  }

  const summaryCards = [
    {
      label: "Fleet Machines",
      value: fleet_summary.total_machines ?? 0,
      icon: Cpu,
      note: "Active in registry",
      color: "bg-brand-50 text-brand-500",
      valueColor: "text-slate-800",
    },
    {
      label: "Critical / High",
      value: fleet_summary.critical_count ?? 0,
      icon: AlertTriangle,
      note: "Machines needing attention",
      color: "bg-red-50 text-red-500",
      valueColor: fleet_summary.critical_count > 0 ? "text-red-600" : "text-emerald-600",
    },
    {
      label: "Avg Fleet Risk",
      value: `${((fleet_summary.average_risk ?? 0) * 100).toFixed(1)}%`,
      icon: Activity,
      note: "Across all active machines",
      color: "bg-orange-50 text-orange-500",
      valueColor: (fleet_summary.average_risk ?? 0) > 0.5 ? "text-orange-600" : "text-slate-800",
    },
    {
      label: "Total Alerts",
      value: fleet_summary.total_active_alerts ?? 0,
      icon: Zap,
      note: "Active sensor warnings",
      color: "bg-amber-50 text-amber-500",
      valueColor: "text-slate-800",
    },
    {
      label: "Model Accuracy",
      value: `${fleet_summary.model_accuracy ?? 0}%`,
      icon: Target,
      note: `${fleet_summary.audit_matched ?? 0} / ${fleet_summary.audit_total_cases ?? 0} cases matched`,
      color: "bg-emerald-50 text-emerald-600",
      valueColor: (fleet_summary.model_accuracy ?? 0) >= 70 ? "text-emerald-600" : "text-amber-500",
    },
    {
      label: "Total Scrap Events",
      value: fleet_summary.total_scrap_events ?? 0,
      icon: TrendingUp,
      note: "Detected in 4h windows",
      color: "bg-purple-50 text-purple-500",
      valueColor: "text-slate-800",
    },
  ];

  return (
    <div className="flex flex-col gap-6 animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-slate-800 flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-brand-50 flex items-center justify-center shadow-sm">
              <BarChart3 size={20} className="text-brand-600" />
            </div>
            Fleet Analytics Hub
          </h1>
          <p className="text-xs text-slate-400 font-semibold mt-1 ml-13 pl-13">
            Live aggregated performance across all {fleet_summary.total_machines ?? 0} monitored machines
          </p>
        </div>
        <button
          onClick={() => fetchData(true)}
          disabled={refreshing}
          className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-brand-600 text-white text-xs font-black uppercase tracking-wider hover:bg-brand-700 transition-all active:scale-95 shadow-lg shadow-brand-200 disabled:opacity-50"
        >
          <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "Refreshing…" : "Refresh Data"}
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-4">
        {summaryCards.map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.label} className="glass-card p-5 flex flex-col gap-3 hover:shadow-xl transition-all hover:-translate-y-0.5">
              <div className={`w-10 h-10 rounded-2xl flex items-center justify-center ${card.color}`}>
                <Icon size={18} />
              </div>
              <div>
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">{card.label}</p>
                <p className={`text-2xl font-black mt-0.5 ${card.valueColor}`}>{card.value}</p>
              </div>
              <p className="text-[10px] text-slate-400 font-semibold border-t border-slate-100 pt-2">{card.note}</p>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Risk Score by Machine */}
        <div className="glass-card p-6">
          <h2 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-600 flex items-center gap-2 mb-4">
            <Activity size={14} className="text-brand-500" />
            Risk Score by Machine
          </h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskChartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="id" tick={{ fontSize: 10, fontWeight: 700, fill: "#64748b" }} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: "#94a3b8" }} />
                <Tooltip content={<CustomBarTooltip />} />
                <Bar dataKey="Risk Score" radius={[6, 6, 0, 0]}>
                  {riskChartData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Alerts & Scrap Events */}
        <div className="glass-card p-6">
          <h2 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-600 flex items-center gap-2 mb-4">
            <Zap size={14} className="text-amber-500" />
            Alerts & Scrap Events by Machine
          </h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={alertChartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="id" tick={{ fontSize: 10, fontWeight: 700, fill: "#64748b" }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 9, fill: "#94a3b8" }} />
                <Tooltip content={<CustomBarTooltip />} />
                <Bar dataKey="Active Alerts" fill="#f97316" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Scrap Events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-4 mt-2">
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400">
              <span className="w-3 h-3 rounded-sm bg-orange-500 inline-block" />Active Alerts
            </span>
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400">
              <span className="w-3 h-3 rounded-sm bg-violet-500 inline-block" />Scrap Events
            </span>
          </div>
        </div>
      </div>

      {/* Fleet Machine Table */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/30">
          <h2 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-800 flex items-center gap-3">
            <ShieldCheck size={16} className="text-brand-500" />
            Machine Health Matrix
          </h2>
          <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">
            {machines.length} machines · Last 4h window
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[700px]">
            <thead>
              <tr className="bg-slate-50/50">
                {["Machine", "Status", "Risk Score", "Active Alerts", "Scrap Detected", "Future Predicted", "Top Root Cause"].map(h => (
                  <th key={h} className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sortedMachines.map((m) => {
                const cfg = getStatusConfig(m.status);
                const riskPct = (m.risk_score * 100).toFixed(1);
                return (
                  <tr
                    key={m.id}
                    className="group hover:bg-brand-50/20 transition-colors cursor-pointer"
                    onClick={() => onViewMachine && onViewMachine(m.id)}
                    title="Click to view on Dashboard"
                  >
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 text-white text-[11px] font-black shadow-sm group-hover:scale-105 transition-transform">
                        {m.display_id || m.id}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider ${cfg.bg} text-white`}>
                        {m.status}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-20 h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${Math.min(m.risk_score * 100, 100)}%`, background: cfg.ring }}
                          />
                        </div>
                        <span className={`text-[12px] font-black ${cfg.text}`}>{riskPct}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-[13px] font-black ${m.active_alerts > 0 ? "text-orange-600" : "text-slate-400"}`}>
                        {m.active_alerts}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-[13px] font-black ${m.past_scrap_detected > 0 ? "text-violet-600" : "text-slate-400"}`}>
                        {m.past_scrap_detected}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-[13px] font-black ${m.future_scrap_predicted > 0 ? "text-red-500" : "text-slate-400"}`}>
                        {m.future_scrap_predicted}
                      </span>
                    </td>
                    <td className="px-6 py-4 max-w-[180px]">
                      {m.root_causes?.length > 0 ? (
                        <span className="text-[10px] font-semibold text-slate-500 line-clamp-1">{m.root_causes[0]}</span>
                      ) : (
                        <span className="text-[10px] text-slate-300 italic">–</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Accuracy Footer Card */}
      <div className="glass-card p-6 flex items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center shadow-sm">
            <Target size={24} className="text-emerald-600" />
          </div>
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">AI Model Performance</p>
            <p className="text-3xl font-black text-slate-800">{fleet_summary.model_accuracy ?? 0}%</p>
            <p className="text-xs text-slate-400 font-semibold mt-0.5">
              {fleet_summary.audit_matched ?? 0} predicted correctly out of {fleet_summary.audit_total_cases ?? 0} audited scrap cases
            </p>
          </div>
        </div>
        <div className="flex items-center gap-6 flex-wrap">
          <div className="flex flex-col items-center gap-1">
            <CheckCircle2 size={24} className="text-emerald-500" />
            <span className="text-xl font-black text-slate-800">{fleet_summary.audit_matched ?? 0}</span>
            <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Correct</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <XCircle size={24} className="text-red-400" />
            <span className="text-xl font-black text-slate-800">{(fleet_summary.audit_total_cases ?? 0) - (fleet_summary.audit_matched ?? 0)}</span>
            <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Missed</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <BarChart3 size={24} className="text-brand-500" />
            <span className="text-xl font-black text-slate-800">{fleet_summary.audit_total_cases ?? 0}</span>
            <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Total Cases</span>
          </div>
        </div>
      </div>
    </div>
  );
}
