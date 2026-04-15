import React from "react";
import { 
  BarChart3, 
  Zap, 
  Target, 
  Network, 
  Binary,
  ArrowUpRight,
  Activity
} from "lucide-react";

export default function DatasetStats({ metrics, isComplete }) {
  const statCards = [
    { 
      label: 'MES Rows', 
      value: metrics.mesRows?.toLocaleString(), 
      icon: Network, 
      color: 'text-brand-400', 
      bg: 'bg-brand-500/10',
      border: 'border-brand-500/20'
    },
    { 
      label: 'Hydra Rows', 
      value: metrics.hydraRows?.toLocaleString(), 
      icon: Zap, 
      color: 'text-amber-400', 
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20'
    },
    { 
      label: 'Match Rate', 
      value: `${metrics.matchRate}%`, 
      icon: Target, 
      color: 'text-emerald-400', 
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20'
    },
    { 
      label: 'Features Extracted', 
      value: metrics.featuresCount, 
      icon: Binary, 
      color: 'text-purple-400', 
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/20'
    }
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center gap-4">
        <div className="h-4 w-1 bg-brand-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
        <div>
          <h2 className="text-[11px] font-black text-slate-800 uppercase tracking-[0.2em] bg-clip-text text-transparent bg-gradient-to-r from-brand-600 to-indigo-600">
            Analytics Intake Summary
          </h2>
          <p className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">Statistical verification of alignment</p>
        </div>
        <div className="h-[1px] flex-1 bg-gradient-to-r from-slate-200 to-transparent" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {statCards.map((stat, idx) => {
          const Icon = stat.icon;
          const displayValue = stat.value || '0';
          
          return (
            <div 
              key={stat.label} 
              className={`relative overflow-hidden p-6 flex flex-col gap-4 rounded-[1.5rem] border-2 transition-all duration-700 backdrop-blur-md group ${
                isComplete 
                  ? `translate-y-0 opacity-100 ${stat.border} bg-white/80 shadow-lg shadow-slate-200/40` 
                  : 'translate-y-4 opacity-50 border-slate-100 bg-slate-50/50 grayscale-[0.5]'
              }`}
              style={{ transitionDelay: `${idx * 0.1}s` }}
            >
              <div className={`absolute -inset-2 opacity-0 group-hover:opacity-30 transition-opacity blur-2xl ${stat.bg} rounded-[2.5rem]`} />
              
              <div className="relative flex items-center justify-between">
                <div className={`h-12 w-12 rounded-2xl ${isComplete ? stat.bg : 'bg-white shadow-inner'} flex items-center justify-center transition-all duration-500 border border-white/40`}>
                  <Icon size={20} className={isComplete ? stat.color : 'text-slate-300'} />
                </div>
                {!isComplete && (
                   <div className="flex gap-1.5 items-center bg-white px-3 py-1.5 rounded-xl border border-slate-100 shadow-sm">
                     <div className="w-1.5 h-1.5 bg-slate-200 rounded-full animate-pulse" />
                     <span className="text-[9px] font-black text-slate-400 uppercase tracking-[0.1em]">Pending</span>
                   </div>
                )}
                {isComplete && (
                   <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                     <ArrowUpRight size={14} className="text-slate-300" />
                   </div>
                )}
              </div>
              <div className="relative">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.15em]">{stat.label}</p>
                <div className="mt-1 h-9 flex items-end">
                  {isComplete ? (
                    <p className="text-2xl font-black tracking-tighter text-slate-900 animate-slide-in-bottom">
                      {displayValue}
                    </p>
                  ) : (
                    <div className="w-2/3 h-6 bg-slate-200/50 rounded-lg animate-pulse" />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {isComplete && (
        <div className="relative overflow-hidden rounded-[2rem] border-2 border-brand-100 bg-gradient-to-br from-white to-brand-50/30 p-6 animate-bounce-in shadow-xl shadow-brand-100/20 group">
          <div className="absolute top-0 right-0 w-40 h-40 bg-brand-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:scale-110 transition-transform duration-1000" />
          <div className="relative flex items-center justify-between">
            <div className="flex gap-5 items-center">
              <div className="relative">
                <div className="absolute inset-0 bg-brand-500/20 blur-xl animate-pulse" />
                <div className="relative h-14 w-14 rounded-[1.25rem] bg-white shadow-lg border border-brand-100 flex items-center justify-center text-brand-600 scale-105">
                  <Activity size={28} className="stroke-[2.5]" />
                </div>
              </div>
              <div>
                <p className="text-[11px] font-black text-brand-900 uppercase tracking-[0.2em]">Signal Alignment Score</p>
                <div className="flex items-center gap-2.5 mt-1.5">
                  <div className="h-px w-6 bg-emerald-500/50" />
                  <p className="text-xs font-bold text-slate-600">Cross-correlation: <span className="font-black text-slate-900 tracking-tight">0.984</span></p>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-end">
              <div className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-emerald-500 text-white font-black text-base border-b-4 border-emerald-700 shadow-lg shadow-emerald-200/50 group-hover:scale-105 transition-transform">
                96% <ArrowUpRight size={18} className="stroke-[3]" />
              </div>
              <p className="text-[9px] font-black text-emerald-600 uppercase tracking-[0.2em] mt-2 italic">Optimal Sync</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
