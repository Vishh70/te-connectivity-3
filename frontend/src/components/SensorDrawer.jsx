import React, { useMemo, useState, useEffect, useCallback } from "react";
import { X, AlertCircle, CheckCircle2, Info, Activity, ShieldAlert, Zap, ClipboardList, CheckCheck, Loader2 } from "lucide-react";
import { LineChart, Line, ResponsiveContainer, YAxis, CartesianGrid, ReferenceArea, ReferenceLine, Tooltip } from "recharts";

import apiClient from "../utils/apiClient";
import { mapStatus, UI_STATUS, getStatusColorClass, getStatusBadgeClass, SENSOR_METADATA, formatSensorName } from "../utils/statusUtils";


const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const getRecommendation = (sensor, status, value, min, max) => {
  if (status === UI_STATUS.NORMAL) return "No immediate action required. Sensor is within safe operating parameters.";
  
  const name = formatSensorName(sensor).toLowerCase();
  
  if (name.includes("tmp") || name.includes("temperature") || name.includes("oil")) {
    if (value > max) return "High thermal state detected. Verify heater band integrity, cooling flow rate, or hydraulic oil heat exchange.";
    return "Low temperature detected. Check for component failure or startup stabilization delay.";
  }
  
  if (name.includes("pressure")) {
    if (value > max) return "Overpressure state. Inspect hydraulic valves and verify material viscosity settings.";
    return "Pressure drop detected. Check for seal leakage, pump cavitation, or low oil levels.";
  }
  
  if (name.includes("cushion")) {
    return "Cushion instability detected. Verify injection speed profile and check for non-return valve wear.";
  }
  
  if (name.includes("cycle time")) {
    return "Cycle time variance detected. Inspect mold opening/closing mechanism for mechanical friction.";
  }

  if (name.includes("dosage")) {
    return "Dosage time deviation. Check screw refill efficiency, back pressure settings, or material blockage.";
  }

  if (name.includes("ejector")) {
    return "Ejector torque deviation. Inspect ejector pin movement for mechanical resistance or breakage.";
  }

  if (name.includes("start pos")) {
    return "Extruder start position shift. Verify screw recovery consistency and check for non-return valve leakage.";
  }

  if (name.includes("peak")) {
    return "Peak pressure variance. Evaluate material viscosity shifts or heater band performance.";
  }

  if (name.includes("torque")) {
    return "Torque variance detected. Inspect for material contamination, screw wear, or motor load imbalance.";
  }

  if (name.includes("injection pressure")) {
    return "Injection pressure variance. Verify material viscosity consistency and inspect nozzle for obstructions.";
  }

  if (name.includes("injection time")) {
    return "Injection time shift detected. Review injection speed profile and check switchover position accuracy.";
  }

  if (name.includes("switch")) {
    return "Switchover accuracy warning. Verify position sensor calibration and hydraulic valve responsiveness.";
  }

  if (name.includes("scrap")) {
    return "Scrap threshold exceeded. Review recent parameter shifts and verify part quality metrics.";
  }

  if (name.includes("temp") || name.includes("tmp") || name.includes("cylinder")) {
    return "Thermal variance detected. Check heater zone stability, thermocouple calibration, or cooling system efficiency.";
  }

  if (status === UI_STATUS.CRITICAL) return "Immediate mechanical inspection recommended. Signal variance suggests imminent downtime.";
  return "Schedule proactive maintenance check. Sensor showing persistent deviation from baseline.";
};

export default function SensorDrawer({ isOpen, onClose, machineId, sensor, sensorData }) {
  const [trendData, setTrendData] = useState([]);
  const [trendLoading, setTrendLoading] = useState(false);
  // Acknowledge state: 'idle' | 'loading' | 'done'
  const [ackState, setAckState] = useState("idle");
  // Log modal state
  const [logModal, setLogModal] = useState(false);
  const [logNote, setLogNote] = useState("");
  const [logPriority, setLogPriority] = useState("medium");
  const [logState, setLogState] = useState("idle"); // 'idle' | 'loading' | 'done'

  // Reset action states when sensor changes
  useEffect(() => {
    setAckState("idle");
    setLogState("idle");
    setLogModal(false);
    setLogNote("");
    setLogPriority("medium");
  }, [sensor, machineId]);

  const handleAcknowledge = useCallback(async () => {
    if (ackState !== "idle") return;
    setAckState("loading");
    try {
      await apiClient.post("/api/maintenance/acknowledge", {
        machine_id: machineId,
        sensor: sensor,
        status: sensorData?.status || "UNKNOWN",
        value: sensorData?.value,
        operator: "dashboard-user",
      });
      setAckState("done");
    } catch (err) {
      console.error("Acknowledge failed:", err);
      setAckState("idle");
    }
  }, [ackState, machineId, sensor, sensorData]);

  const handleLogSubmit = useCallback(async () => {
    if (logState !== "idle") return;
    setLogState("loading");
    try {
      await apiClient.post("/api/maintenance/log", {
        machine_id: machineId,
        sensor: sensor,
        action: "Manual check scheduled",
        note: logNote,
        priority: logPriority,
        operator: "dashboard-user",
      });
      setLogState("done");
      setTimeout(() => {
        setLogModal(false);
        setLogState("idle");
        setLogNote("");
      }, 1500);
    } catch (err) {
      console.error("Log failed:", err);
      setLogState("idle");
    }
  }, [logState, machineId, sensor, logNote, logPriority]);

  useEffect(() => {
    if (isOpen && sensor && machineId) {
      const fetchTrend = async () => {
        try {
          setTrendLoading(true);
          const response = await apiClient.get(
            `/api/trend/${encodeURIComponent(machineId)}/${encodeURIComponent(sensor)}`,
          );
          setTrendData(response.data.data || []);
        } catch (err) {
          console.error("Error fetching trend:", err);
        } finally {
          setTrendLoading(false);
        }
      };
      fetchTrend();
    } else {
      setTrendData([]);
    }
  }, [isOpen, sensor, machineId]);

  const name = useMemo(() => formatSensorName(sensor), [sensor]);

  
  const status = mapStatus(sensorData?.status);
  const value = toNumber(sensorData?.value);
  const min = toNumber(sensorData?.safe_min);
  const max = toNumber(sensorData?.safe_max);
  
  const recommendation = useMemo(() => getRecommendation(sensor, status, value, min, max), [sensor, status, value, min, max]);

  const sensorDesc = useMemo(() => {
    const normKey = String(sensor || "").toLowerCase().replace(/ /g, "_");
    const matchedKey = Object.keys(SENSOR_METADATA).find(k => k === normKey || normKey.includes(k) || k.includes(normKey));
    return matchedKey && SENSOR_METADATA[matchedKey].description ? SENSOR_METADATA[matchedKey].description : "System monitoring point";
  }, [sensor]);

  const sparklineData = useMemo(() => {
    // Priority: 1. Fetched high-res trend, 2. Passed sparkline
    if (trendData.length > 0) {
      return trendData.map((d, i) => ({ 
        i, 
        value: toNumber(d.value), // Backend returns 'value' regardless of sensor name
        type: d.type,            // 'history' or 'prediction'
        timestamp: d.timestamp 
      })).filter(p => p.value !== null);
    }
    const raw = Array.isArray(sensorData?.sparkline) ? sensorData.sparkline : [];
    return raw.map((v, i) => ({ i, value: toNumber(v) })).filter(p => p.value !== null);
  }, [sensorData, trendData]);

  if (!isOpen) return null;

  const StatusIcon = status === UI_STATUS.CRITICAL ? ShieldAlert : status === UI_STATUS.WARNING || status === UI_STATUS.WATCH ? AlertCircle : CheckCircle2;
  const statusClass = getStatusColorClass(status);
  const statusBadge = getStatusBadgeClass(status);
  const isCritical = status === UI_STATUS.CRITICAL;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 z-[100] bg-slate-900/10 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />
      
      {/* Drawer Content */}
      <div className="fixed top-0 right-0 z-[101] h-screen w-full max-w-[420px] bg-white/95 backdrop-blur-2xl shadow-[-20px_0_50px_rgba(15,23,42,0.1)] border-l border-slate-200 animate-slide-in-right overflow-y-auto">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl bg-white shadow-sm border border-slate-100 ${statusClass} ${isCritical ? 'animate-pulse' : ''}`}>
                <Activity size={20} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-800">{name}</h3>
                <p className="text-xs text-slate-400 font-medium tracking-wide">SENSOR DIAGNOSTICS</p>
                <p className="text-[11px] text-slate-500 mt-1 leading-snug max-w-[280px]">
                  {sensorDesc}
                </p>
              </div>
            </div>
            <button 
              onClick={onClose}
              className="p-2 rounded-xl hover:bg-slate-100 text-slate-400 transition-colors"
            >
              <X size={20} />
            </button>
          </div>

          <div className="flex-1 p-6 space-y-8">
            {/* Status Card */}
            <div className="glass-card relative overflow-hidden group p-6 border-brand-100/50 bg-gradient-to-b from-brand-50/40 to-transparent shadow-sm">
              <div className="absolute top-0 right-0 w-32 h-32 bg-brand-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="flex items-center justify-between relative z-10">
                <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${statusBadge}`}>
                  {status}
                </span>
                <StatusIcon size={24} className={statusClass} />
              </div>
              <div className="mt-4">
                <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Current Value</p>
                <p className="text-4xl font-black text-slate-800 tracking-tight mt-1">
                  {value !== null ? value.toFixed(2) : "--"}
                </p>
              </div>
              <div className="mt-5 grid grid-cols-2 gap-4 pt-4 border-t border-slate-200/50">
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Safe Min</p>
                  <p className="text-sm font-semibold text-slate-700">{min !== null ? min.toFixed(2) : "--"}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase">Safe Max</p>
                  <p className="text-sm font-semibold text-slate-700">{max !== null ? max.toFixed(2) : "--"}</p>
                </div>
              </div>
            </div>

            {/* Recommendation */}
            <div className="space-y-3">
              <h4 className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
                <Zap size={14} className="text-brand-500" />
                Root Cause & Recommended Action
              </h4>
              <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 text-sm text-slate-600 leading-relaxed italic">
                "{recommendation}"
              </div>
            </div>

            {/* Trend History */}
            <div className="space-y-4">
              <h4 className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
                <Activity size={14} className="text-brand-500" />
                Trend History (Last {sensorData?.window_label || "1H"})
              </h4>
              <div className="h-48 rounded-2xl border border-slate-100 bg-slate-50/50 p-2 overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sparklineData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <YAxis hide domain={['auto', 'auto']} />
                    {typeof min === 'number' && typeof max === 'number' && max > min && (
                      <ReferenceArea y1={min} y2={max} fill="#10b981" fillOpacity={0.05} />
                    )}
                    {(typeof min === 'number') && (
                      <ReferenceLine y={min} stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 4" strokeOpacity={0.8} label={{ position: 'insideTopLeft', value: 'Safe Min', fill: '#f59e0b', fontSize: 10, fontWeight: 700 }} />
                    )}
                    {(typeof max === 'number') && (
                      <ReferenceLine y={max} stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" strokeOpacity={0.8} label={{ position: 'insideBottomLeft', value: 'Safe Max', fill: '#ef4444', fontSize: 10, fontWeight: 700 }} />
                    )}
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: "rgba(255,255,255,0.75)",
                        WebkitBackdropFilter: "blur(16px)",
                        backdropFilter: "blur(16px)",
                        border: "1px solid rgba(255,255,255,0.8)",
                        borderRadius: "16px",
                        color: "#1e293b",
                        boxShadow: "0 10px 40px -10px rgba(0,0,0,0.15)",
                      }}
                      labelFormatter={(t) => {
                        // Senior Pro Fix: Ensure "1 Means 1" timeline in diagnostic tooltips.
                        if (!t) return "";
                        return dayjs(t).utc().format("HH:mm:ss");
                      }}
                      formatter={(v) => [`${Number(v).toFixed(3)}`, 'Value']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke={
                        status === UI_STATUS.CRITICAL ? "#ef4444" : 
                        status === UI_STATUS.WARNING || status === UI_STATUS.WATCH ? "#f59e0b" : "#10b981"
                      } 
                      strokeWidth={3} 
                      dot={false}
                      activeDot={{ r: 5, fill: "#fff", stroke: "#3b82f6", strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* AI Insights */}
            <div className="p-4 rounded-2xl border border-emerald-100 bg-emerald-50/30 flex gap-4">
              <div className="h-8 w-8 shrink-0 rounded-lg bg-emerald-100 flex items-center justify-center text-emerald-600">
                <Info size={16} />
              </div>
              <div>
                <p className="text-xs font-bold text-emerald-800 uppercase tracking-wider">AI Signal Confidence</p>
                <p className="text-xs text-emerald-700 mt-1 leading-relaxed">
                  The predictive engine detects a higher variance than normal. This signal is currently weighted at {Math.round((sensorData?.impact || 0) * 100)}% for overall machine risk.
                </p>
              </div>
            </div>
          </div>

          {/* Footer Controls */}
          <div className="p-6 border-t border-slate-100 flex gap-3">
            {/* Acknowledge Warning Button */}
            <button
              onClick={handleAcknowledge}
              disabled={ackState !== "idle"}
              className={`flex-1 py-3 px-4 rounded-xl text-sm font-bold shadow-md transition-all duration-300 flex items-center justify-center gap-2 ${
                ackState === "done"
                  ? "bg-emerald-600 text-white"
                  : ackState === "loading"
                  ? "bg-slate-600 text-white opacity-80 cursor-not-allowed"
                  : "bg-slate-800 text-white hover:bg-slate-700 active:scale-95"
              }`}
            >
              {ackState === "loading" ? (
                <><Loader2 size={15} className="animate-spin" /> Sending...</>
              ) : ackState === "done" ? (
                <><CheckCheck size={15} /> Acknowledged</>
              ) : (
                "Acknowledge Warning"
              )}
            </button>

            {/* Log Maintenance Button */}
            <button
              onClick={() => setLogModal(true)}
              className="py-3 px-4 rounded-xl border border-slate-200 text-slate-600 text-sm font-bold hover:bg-slate-50 active:scale-95 transition-all flex items-center gap-1.5"
            >
              <ClipboardList size={15} />
              Log
            </button>
          </div>

          {/* Log Maintenance Modal */}
          {logModal && (
            <div className="absolute inset-0 z-[200] flex items-end justify-stretch bg-slate-900/20 backdrop-blur-sm">
              <div className="w-full bg-white border-t border-slate-200 shadow-2xl rounded-t-3xl p-6 animate-slide-up">
                <div className="flex items-center justify-between mb-5">
                  <h4 className="text-sm font-black text-slate-800 flex items-center gap-2">
                    <ClipboardList size={16} className="text-brand-600" />
                    Log Maintenance Event
                  </h4>
                  <button
                    onClick={() => setLogModal(false)}
                    className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400"
                  >
                    <X size={16} />
                  </button>
                </div>

                {/* Machine + Sensor info */}
                <div className="mb-4 flex gap-2">
                  <span className="px-2.5 py-1 rounded-lg bg-brand-50 text-brand-700 text-[11px] font-bold">{machineId}</span>
                  <span className="px-2.5 py-1 rounded-lg bg-slate-100 text-slate-600 text-[11px] font-bold">{name}</span>
                </div>

                {/* Priority selector */}
                <div className="mb-4">
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2">Priority</p>
                  <div className="flex gap-2">
                    {["low", "medium", "high"].map((p) => (
                      <button
                        key={p}
                        onClick={() => setLogPriority(p)}
                        className={`flex-1 py-2 rounded-xl text-xs font-bold border transition-all ${
                          logPriority === p
                            ? p === "high"
                              ? "bg-red-600 text-white border-red-600"
                              : p === "medium"
                              ? "bg-amber-500 text-white border-amber-500"
                              : "bg-emerald-600 text-white border-emerald-600"
                            : "bg-white text-slate-500 border-slate-200 hover:bg-slate-50"
                        }`}
                      >
                        {p.charAt(0).toUpperCase() + p.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Note */}
                <div className="mb-5">
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2">Operator Note (optional)</p>
                  <textarea
                    value={logNote}
                    onChange={(e) => setLogNote(e.target.value)}
                    placeholder="Describe the issue or action taken..."
                    rows={3}
                    className="w-full text-sm rounded-xl border border-slate-200 px-4 py-3 text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-2 focus:ring-brand-300 resize-none"
                  />
                </div>

                <button
                  onClick={handleLogSubmit}
                  disabled={logState !== "idle"}
                  className={`w-full py-3 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                    logState === "done"
                      ? "bg-emerald-600 text-white"
                      : logState === "loading"
                      ? "bg-brand-400 text-white opacity-80 cursor-not-allowed"
                      : "bg-brand-600 text-white hover:bg-brand-500 shadow-lg shadow-brand-200 active:scale-95"
                  }`}
                >
                  {logState === "loading" ? (
                    <><Loader2 size={15} className="animate-spin" /> Saving...</>
                  ) : logState === "done" ? (
                    <><CheckCheck size={15} /> Logged Successfully!</>
                  ) : (
                    <><ClipboardList size={15} /> Submit Maintenance Log</>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
