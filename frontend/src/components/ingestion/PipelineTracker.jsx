import React from "react";
import { CheckCircle2, Circle, Clock, Loader2, AlertCircle } from "lucide-react";

export default function PipelineTracker({ steps, currentStep, status }) {
  return (
    <div className="glass-card p-8 space-y-8 animate-slide-up bg-white/60 border-white/40 shadow-xl shadow-slate-200/50">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-[11px] font-black text-slate-800 uppercase tracking-[0.2em]">Flow Synchronizer</h3>
          <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-1">Multi-Stage Data Alignment</p>
        </div>
        <div className="flex items-center gap-2">
          {status === 'processing' && (
            <div className="flex items-center gap-3 text-[10px] font-black text-brand-600 bg-brand-50 px-3 py-1.5 rounded-xl border border-brand-100 shadow-sm">
              <Loader2 size={12} className="animate-spin" />
              HOT-RELOAD ACTIVE
            </div>
          )}
          {status === 'complete' && (
            <div className="flex items-center gap-3 text-[10px] font-black text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-xl border border-emerald-100 shadow-sm animate-bounce-in">
              <CheckCircle2 size={12} />
              SYNC OPTIMAL
            </div>
          )}
        </div>
      </div>

      <div className="relative px-2">
        {/* Modern Progress Line */}
        <div className="absolute left-[19px] top-4 bottom-4 w-[2px] bg-slate-100 rounded-full" />
        <div 
          className="absolute left-[19px] top-4 w-[2px] bg-gradient-to-b from-brand-500 via-indigo-500 to-emerald-500 transition-all duration-1000 ease-in-out shadow-[0_0_12px_rgba(59,130,246,0.6)] rounded-full" 
          style={{ height: `${(currentStep / (steps.length - 1)) * 92}%` }}
        >
           <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-white border-2 border-brand-500 animate-ping" />
        </div>

        {/* Steps with Staggered Entrance */}
        <div className="space-y-8 relative">
          {steps.map((step, idx) => {
            const isCompleted = idx < currentStep;
            const isActive = idx === currentStep;
            const isPending = idx > currentStep;

            return (
              <div 
                key={step.id} 
                className={`flex items-start gap-6 transition-all duration-700 ease-out ${
                  isActive ? 'translate-x-3 scale-[1.02]' : 'opacity-100'
                }`}
                style={{ transitionDelay: `${idx * 0.05}s` }}
              >
                <div className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border-2 transition-all duration-700 shadow-lg ${
                  isCompleted ? 'bg-brand-600 border-brand-600 text-white shadow-brand-200 -rotate-12' :
                  isActive ? 'bg-white border-brand-500 text-brand-500 scale-125 shadow-brand-100 rotate-0' :
                  'bg-white border-slate-100 text-slate-300'
                }`}>
                  {isCompleted ? <CheckCircle2 size={18} className="animate-fade-in" /> : 
                   isActive ? <Loader2 size={18} className="animate-spin text-brand-500" /> :
                   <Circle size={8} fill="currentColor" className="opacity-20" />}
                  
                  {isActive && (
                    <div className="absolute inset-0 rounded-2xl bg-brand-500/10 animate-pulse" />
                  )}
                </div>
                
                <div className="flex-1 pt-1">
                  <div className="flex items-center gap-3">
                    <p className={`text-[11px] font-black uppercase tracking-[0.1em] transition-all duration-500 ${
                      isActive ? 'text-brand-700' : isCompleted ? 'text-slate-800' : 'text-slate-400'
                    }`}>
                      {step.label}
                    </p>
                    {isActive && (
                      <div className="h-[1px] flex-1 bg-gradient-to-r from-brand-200 to-transparent animate-pulse" />
                    )}
                  </div>
                  <p className={`mt-1 text-[10px] font-bold leading-relaxed transition-all duration-500 ${
                     isActive ? 'text-slate-600' : 'text-slate-400'
                  }`}>
                    {step.description}
                  </p>
                </div>

                {isActive && (
                   <div className="hidden sm:flex flex-col items-end pt-1.5 animate-fade-in">
                     <span className="text-[8px] font-black text-brand-600/40 uppercase tracking-[0.3em] italic bg-brand-50 px-2 py-0.5 rounded-inner">ACTIVE_GATE</span>
                   </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
