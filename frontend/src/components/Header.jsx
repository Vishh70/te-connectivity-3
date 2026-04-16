import React, { useState, useEffect } from "react";
import apiClient from "../utils/apiClient";
import { Activity, Bell, Clock3, Cpu, Menu, Database, ShieldCheck, Zap } from "lucide-react";
import { mapStatus, UI_STATUS, getStatusBadgeClass } from "../utils/statusUtils";
import { PAST_WINDOW_OPTIONS, FUTURE_WINDOW_OPTIONS } from "../utils/windowOptions";

export { PAST_WINDOW_OPTIONS, FUTURE_WINDOW_OPTIONS } from "../utils/windowOptions";

export default function Header({
  machineId,
  hasPendingWindowChanges,
  draftPastWindowMinutes,
  draftFutureWindowMinutes,
  pastWindowMinutes,
  futureWindowMinutes,
  onMachineChange,
  onPastWindowChange,
  onFutureWindowChange,
  onApplyWindows,
  onToggleSidebar,
  healthStatus,
  isSensorFrozen,
  machineInfo,
  lastUpdatedLabel,
}) {
  const [machines, setMachines] = useState([]);
  const [bufferStatus, setBufferStatus] = useState({ state: "Standby", count: 0 });

  useEffect(() => {
    const fetchMeta = async () => {
      try {
        const res = await apiClient.get("/api/machines");
        setMachines(res.data);
      } catch (err) {
        console.error("Meta fetch error:", err);
      }
    };
    fetchMeta();
  }, []);

  useEffect(() => {
    const checkBuffer = async () => {
      try {
        const res = await apiClient.get("/api/predict/buffer-status", { 
          params: { machine_id: machineId } 
        });
        const count = res.data.buffer_size ?? res.data.total_buffered ?? 0;
        setBufferStatus({
          state: count > 0 ? "Active" : "Ready",
          count
        });
      } catch (err) {
        setBufferStatus({ state: "Offline", count: 0 });
      }
    };
    checkBuffer();
    const interval = setInterval(checkBuffer, 10000);
    return () => clearInterval(interval);
  }, [machineId]);

  const uiStatus = mapStatus(healthStatus);
  const statusConfig = {
    cls: getStatusBadgeClass(uiStatus),
    glow: uiStatus === UI_STATUS.CRITICAL ? "shadow-glow-red" : uiStatus === UI_STATUS.HIGH ? "shadow-glow-orange" : uiStatus === UI_STATUS.NORMAL ? "shadow-glow-green" : ""
  };

  const pastLabel =
    PAST_WINDOW_OPTIONS.find((opt) => opt.value === pastWindowMinutes)?.label ||
    `${pastWindowMinutes}m Past`;
  const futureLabel =
    FUTURE_WINDOW_OPTIONS.find((opt) => opt.value === futureWindowMinutes)?.label ||
    `${futureWindowMinutes}m Future`;
  const draftLabel = `${draftPastWindowMinutes}m Past / ${draftFutureWindowMinutes}m Future`;
  const machineDisplayId = machineInfo?.display_id || machineInfo?.id || machineId;
  const machineNumberLabel =
    machineInfo?.machine_number ||
    String(machineDisplayId).match(/\d+/)?.[0] ||
    String(machineId).match(/\d+/)?.[0] ||
    machineDisplayId;
  const machineNumberChip = /^\d+$/.test(String(machineNumberLabel)) ? `Machine #${machineNumberLabel}` : `Machine ${machineNumberLabel}`;

  const machineMeta = [
    { label: machineNumberChip },
    machineInfo?.name ? { label: machineInfo.name } : null,
    { label: pastLabel },
    { label: futureLabel },
    hasPendingWindowChanges ? { label: `Pending ${draftLabel}` } : null,
    machineInfo?.tool_id ? { label: `Tool ${machineInfo.tool_id}` } : null,
    machineInfo?.part_number ? { label: `Part ${machineInfo.part_number}` } : null,
  ].filter(Boolean);

  return (
    <header
      className="sticky top-0 z-50 flex flex-wrap items-center justify-between gap-6 py-4 px-6 -mx-6 bg-white/95 backdrop-blur-xl border-b border-slate-100/80 shadow-sm animate-fade-in"
      style={{ animationDelay: "0.1s" }}
    >
      <div className="flex-1 min-w-[300px]">
        <div className="flex items-center gap-4">
          <div className="relative group">
            <div className="absolute inset-0 bg-brand-500 blur-xl opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-600 to-indigo-700 text-white shadow-lg border border-white/20">
              <Cpu size={24} />
            </div>
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-slate-900 flex items-center gap-3">
              <button 
                onClick={onToggleSidebar}
                className="lg:hidden flex h-9 w-9 items-center justify-center rounded-xl bg-white border border-slate-200 text-slate-500 shadow-sm active:scale-95 transition-all"
              >
                <Menu size={18} />
              </button>
              Predictive Control Room
            </h1>
            <p className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.1em] mt-0.5 opacity-80">
              Industrial Hub • {machineDisplayId || machineId || 'Awaiting Selection'}
            </p>
          </div>
        </div>
        
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {machineMeta.map((item) => (
            <span
              key={item.label}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200/60 bg-white/50 px-2.5 py-1 text-[9px] font-black text-slate-500 uppercase tracking-wider transition-all hover:bg-white"
            >
              <div className="w-1 h-1 rounded-full bg-brand-400" />
              {item.label}
            </span>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        {/* Themed Selectors */}
        <div className="flex items-center gap-2 p-1 rounded-2xl bg-slate-100/50 backdrop-blur-md border border-slate-200/40">
          <select
            id="machine-select"
            value={machineId}
            onChange={(e) => onMachineChange(e.target.value)}
            className="bg-transparent px-3 py-2 text-[11px] font-black text-slate-700 uppercase tracking-wider outline-none cursor-pointer hover:text-brand-600 transition-colors"
          >
            {machines.map((opt) => (
              <option key={opt.id} value={opt.id}>{opt.display_id || opt.id}</option>
            ))}
          </select>
          <div className="w-px h-4 bg-slate-300/50" />
          <select
            id="past-range-select"
            value={pastWindowMinutes}
            onChange={(e) => onPastWindowChange(Number(e.target.value))}
            className="bg-transparent px-3 py-2 text-[11px] font-black text-slate-700 uppercase tracking-wider outline-none cursor-pointer hover:text-brand-600 transition-colors"
          >
            {PAST_WINDOW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <div className="w-px h-4 bg-slate-300/50" />
          <select
            id="future-range-select"
            value={futureWindowMinutes}
            onChange={(e) => onFutureWindowChange(Number(e.target.value))}
            className="bg-transparent px-4 py-2 text-[11px] font-black text-slate-700 uppercase tracking-wider outline-none cursor-pointer hover:text-brand-600 transition-colors"
          >
            {FUTURE_WINDOW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <button
          onClick={onApplyWindows}
          disabled={!hasPendingWindowChanges}
          className={`group flex items-center gap-2 rounded-2xl px-6 py-3 text-[11px] font-black tracking-widest uppercase transition-all active:scale-95 ${
            hasPendingWindowChanges
              ? "bg-brand-600 text-white hover:bg-brand-700 shadow-xl shadow-brand-200"
              : "bg-slate-200/50 text-slate-400 cursor-not-allowed"
          }`}
        >
          {hasPendingWindowChanges && <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />}
          Apply Range
        </button>

        <div className="flex items-center gap-3 ml-2 pl-4 border-l border-slate-200/60">
          <button
            className="relative rounded-xl border border-white bg-white/60 p-3 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md hover:bg-white"
            title="Notifications"
          >
            <Bell size={18} className="text-slate-600" />
            {(healthStatus === "CRITICAL" || healthStatus === "HIGH") && (
              <span className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-red-500 rounded-full animate-pulse border-2 border-white shadow-[0_0_8px_rgba(239,68,68,0.5)]" />
            )}
          </button>

          <div className="flex flex-col items-end">
             <div className="flex items-center gap-2 mb-1">
                <div className={`flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[9px] font-black uppercase tracking-widest transition-all ${
                  bufferStatus.state === "Active" ? "bg-brand-50 border-brand-200 text-brand-600 shadow-sm" : "bg-slate-50 border-slate-100 text-slate-400"
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${bufferStatus.state === "Active" ? "bg-brand-500 animate-pulse" : "bg-slate-300"}`} />
                  V9 SYNCED {bufferStatus.count > 0 ? `[${bufferStatus.count}ms]` : ""}
                </div>
                <span
                 className={`rounded-lg px-3 py-1 text-[9px] font-black tracking-[0.2em] uppercase border ${statusConfig.cls} ${statusConfig.glow} transition-all`}
               >
                 {uiStatus}
               </span>
             </div>
            {lastUpdatedLabel && (
              <p className="text-[9px] font-black text-slate-400 mt-0.5 uppercase tracking-widest flex items-center gap-1 opacity-70">
                <Clock3 size={10} />
                {lastUpdatedLabel}
              </p>
            )}
          </div>
        </div>

        {isSensorFrozen && (
          <div className="flex items-center gap-2 rounded-2xl border border-orange-200 bg-gradient-to-r from-orange-500 to-amber-500 px-4 py-2.5 text-[10px] font-black text-white shadow-lg shadow-orange-200 animate-pulse">
            <Activity size={14} className="stroke-[3]" />
            SENSOR FROZEN
          </div>
        )}
      </div>
    </header>
  );
}
