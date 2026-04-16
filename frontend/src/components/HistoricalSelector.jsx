import React, { useState } from "react";
import { Clock, Calendar, Zap, ArrowRight, History } from "lucide-react";

/**
 * HistoricalSelector
 * Allows manual date/time entry to anchor the dashboard for replay.
 */
export default function HistoricalSelector({ onApply, onClear, currentAnchor }) {
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");

  const handleApply = () => {
    if (!date || !time) return;
    // Combine date and time into ISO string
    const isoString = new Date(`${date}T${time}Z`).toISOString();
    onApply(isoString);
  };

  if (currentAnchor) {
    // Senior Pro Fix: Use Intl.DateTimeFormat with timeZone: 'UTC' to display
    // the machine's direct raw time without regional offsets (e.g. 5.5h IST shift).
    const displayDate = new Intl.DateTimeFormat("en-GB", {
      year: "numeric",
      month: "numeric",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      hour12: true,
      timeZone: "UTC",
    }).format(new Date(currentAnchor));

    return (
      <div className="flex items-center gap-3 px-4 py-2 rounded-2xl bg-brand-600 text-white shadow-lg shadow-brand-200 animate-in fade-in zoom-in duration-300">
        <History size={16} className="animate-pulse" />
        <div className="flex flex-col">
          <span className="text-[9px] font-black uppercase tracking-widest opacity-80">Replay Mode</span>
          <span className="text-[10px] font-bold">{displayDate}</span>
        </div>
        <button 
          onClick={onClear}
          className="ml-2 p-1.5 rounded-lg bg-white/20 hover:bg-white/40 transition-colors"
          title="Return to Live"
        >
          <Zap size={14} className="fill-current" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 p-1 rounded-2xl bg-slate-100/50 backdrop-blur-md border border-slate-200/40">
      <div className="flex items-center gap-2 px-3 py-1.5">
        <Calendar size={14} className="text-slate-400" />
        <input 
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="bg-transparent border-none outline-none text-[11px] font-bold text-slate-700 w-28"
        />
      </div>
      <div className="w-px h-4 bg-slate-300/40" />
      <div className="flex items-center gap-2 px-3 py-1.5">
        <Clock size={14} className="text-slate-400" />
        <input 
          type="time"
          value={time}
          step="60"
          onChange={(e) => setTime(e.target.value)}
          className="bg-transparent border-none outline-none text-[11px] font-bold text-slate-700 w-20"
        />
      </div>
      
      <button
        onClick={handleApply}
        disabled={!date || !time}
        className={`flex items-center justify-center p-2 rounded-xl transition-all ${
          date && time 
            ? "bg-brand-500 text-white hover:bg-brand-600 shadow-md" 
            : "bg-slate-200 text-slate-400"
        }`}
        title="Jump to Time"
      >
        <ArrowRight size={14} strokeWidth={3} />
      </button>
    </div>
  );
}
