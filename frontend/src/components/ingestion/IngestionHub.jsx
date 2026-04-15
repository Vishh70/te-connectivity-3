import React, { useState, useCallback, useMemo, useEffect } from "react";
import apiClient from "../../utils/apiClient";
import { 
  Database, 
  Layers, 
  History, 
  ArrowRight, 
  AlertCircle,
  CheckCircle2,
  Clock,
  Settings2,
  Cpu,
  BarChart3,
  Search,
  ChevronRight,
  Menu
} from "lucide-react";
import UploadPortal from "./UploadPortal";
import PipelineTracker from "./PipelineTracker";
import DatasetStats from "./DatasetStats";
import IngestionHistory from "./IngestionHistory";



const STEPS = [
  { id: 'upload', label: 'Upload', description: 'Intake raw CSV stream' },
  { id: 'validate', label: 'Validate', description: 'Schema enforcement' },
  { id: 'pair', label: 'Pair', description: 'Machine ID alignment' },
  { id: 'align', label: 'Align', description: 'Temporal synchronization' },
  { id: 'aggregate', label: 'Aggregate', description: 'Stat feature extraction' },
  { id: 'merge', label: 'Merge', description: 'Unified dataset creation' },
  { id: 'features', label: 'Features', description: 'ML engineering' },
  { id: 'normalize', label: 'Normalize', description: 'Range scaling' },
  { id: 'score', label: 'Score', description: 'Inference scoring' },
  { id: 'shap', label: 'SHAP', description: 'Explainability generation' }
];

export default function IngestionHub({ onFinish, onToggleSidebar }) {
  const [activeTab, setActiveTab] = useState('portal'); // 'portal', 'pipeline', 'history'
  const [pipelineState, setPipelineState] = useState({
    active: false,
    currentStep: 0,
    files: {
      parameter: null,
      hydra: null
    },
    status: 'idle', // 'idle', 'processing', 'complete', 'failed'
    metrics: {
      mesRows: 0,
      hydraRows: 0,
      matchRate: 0,
      alignmentQuality: 0,
      featuresCount: 0
    }
  });

  const startPipeline = useCallback(async (parameterFiles, hydraFile) => {
    setPipelineState(prev => ({
      ...prev,
      active: true,
      currentStep: 0,
      files: { parameter: parameterFiles, hydra: hydraFile },
      status: 'processing'
    }));

    try {
      // 1. Upload Files
      const formData = new FormData();
      (Array.isArray(parameterFiles) ? parameterFiles : [parameterFiles]).forEach((file) => {
        if (file) {
          formData.append("mes_files", file);
        }
      });
      formData.append("hydra_file", hydraFile);

      await apiClient.post("/api/ingest/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      // 2. Trigger Processing
      await apiClient.post("/api/ingest/process");
      
      // 3. Start Polling for status
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await apiClient.get("/api/ingest/status");
          const { status, step, metrics } = statusRes.data;
          
          setPipelineState(prev => ({ 
            ...prev, 
            currentStep: step, 
            status: status,
            metrics: status === 'complete' ? metrics : prev.metrics
          }));

          if (status === 'complete' || status === 'failed') {
            clearInterval(pollInterval);
          }
        } catch (err) {
          console.error("Polling error:", err);
          clearInterval(pollInterval);
          setPipelineState(prev => ({ ...prev, status: 'failed' }));
        }
      }, 1000);

    } catch (error) {
      console.error("Pipeline failed:", error);
      setPipelineState(prev => ({ 
        ...prev, 
        status: 'failed',
        error: error?.response?.data?.detail || error.message || "Failed to process pipeline"
      }));
    }
  }, []);

  return (
    <div id="ingestion" className="flex flex-col gap-6 animate-fade-in pb-12 scroll-mt-28">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 glass-card p-6 border-brand-100 bg-brand-50/10 mt-6 lg:mt-0">
        <div className="flex items-center gap-4">
          <button 
            onClick={onToggleSidebar}
            className="lg:hidden flex h-14 w-14 items-center justify-center rounded-2xl bg-white border border-slate-200 text-slate-500 shadow-sm active:scale-95 transition-all"
          >
            <Menu size={24} />
          </button>
          <div className="hidden lg:flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-xl shadow-brand-200">
            <Database size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight text-slate-800">Industrial Ingestion HUB</h1>
            <p className="text-sm font-medium text-slate-500">Production-grade dual-source data pipeline • TE AI Cup</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
           <button 
            onClick={() => setActiveTab('portal')}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${activeTab === 'portal' ? 'bg-brand-600 text-white shadow-lg' : 'bg-white text-slate-500 hover:bg-slate-50 shadow-sm'}`}
          >
            Data Portal
          </button>
          <button 
            onClick={() => setActiveTab('history')}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${activeTab === 'history' ? 'bg-brand-600 text-white shadow-lg' : 'bg-white text-slate-500 hover:bg-slate-50 shadow-sm'}`}
          >
            Ingestion History
          </button>
        </div>
      </div>

      <div>
        {activeTab === 'portal' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 space-y-6">
              <UploadPortal onStart={startPipeline} isProcessing={pipelineState.status === 'processing'} />
              {pipelineState.active && (
                <PipelineTracker 
                  steps={STEPS} 
                  currentStep={pipelineState.currentStep} 
                  status={pipelineState.status} 
                />
              )}
            </div>
            <div className="lg:col-span-4 space-y-6">
              <DatasetStats metrics={pipelineState.metrics} isComplete={pipelineState.status === 'complete'} />
              
              {pipelineState.status === 'failed' && pipelineState.error && (
                <div className="relative overflow-hidden group p-6 rounded-2xl bg-red-50 border border-red-200 shadow-sm">
                  <div className="flex gap-3 items-start">
                    <AlertCircle className="text-red-500 shrink-0 mt-0.5" size={20} />
                    <div>
                      <h3 className="text-sm font-black text-red-800 tracking-tight">Pipeline Execution Failed</h3>
                      <p className="mt-1 text-xs text-red-600/80 font-medium leading-relaxed">{pipelineState.error}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Pipeline Strategy Card */}
              <div className="relative overflow-hidden group p-6 rounded-2xl bg-white border border-slate-100 shadow-sm transition-all hover:shadow-md">
                <div className="absolute top-0 right-0 w-32 h-32 bg-slate-100 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 opacity-50 group-hover:bg-slate-200 transition-all" />
                <div className="relative">
                  <h3 className="text-xs font-black text-slate-800 uppercase tracking-widest flex items-center gap-2 mb-5">
                    <div className="p-1.5 rounded-md bg-brand-50 text-brand-600">
                      <Settings2 size={16} />
                    </div>
                    Pipeline Strategy
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex gap-4 items-start group/item">
                      <div className="mt-1 h-6 w-6 rounded-full bg-emerald-50 border-2 border-emerald-100 flex items-center justify-center shrink-0 transition-colors group-hover/item:border-emerald-300">
                        <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                      </div>
                      <div>
                        <p className="text-xs font-bold text-slate-800 tracking-tight">Temporal Alignment</p>
                        <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">Hydra sensor data is aggregated within +/- 50ms of each injection cycle pulse.</p>
                      </div>
                    </div>
                    
                    <div className="w-px h-6 bg-slate-100 ml-[11px] -my-2" />
                    
                    <div className="flex gap-4 items-start group/item">
                      <div className="mt-1 h-6 w-6 rounded-full bg-amber-50 border-2 border-amber-100 flex items-center justify-center shrink-0 transition-colors group-hover/item:border-amber-300">
                        <div className="h-2 w-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" />
                      </div>
                      <div>
                        <p className="text-xs font-bold text-slate-800 tracking-tight">Schema Enforcement</p>
                        <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">Mandatory index columns: [machine_id, timestamp]. Recommended: [Injection_pressure, Cycle_time].</p>
                      </div>
                    </div>
                    
                    <div className="w-px h-6 bg-slate-100 ml-[11px] -my-2" />
                    
                    <div className="flex gap-4 items-start group/item">
                      <div className="mt-1 h-6 w-6 rounded-full bg-brand-50 border-2 border-brand-100 flex items-center justify-center shrink-0 transition-colors group-hover/item:border-brand-300">
                        <div className="h-2 w-2 rounded-full bg-brand-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                      </div>
                      <div>
                        <p className="text-xs font-bold text-slate-800 tracking-tight">Inference Scoring</p>
                        <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">Current model scoring is triggered automatically after merge verification and control-room sync.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* AI Pipeline Ready Card */}
              <div className="relative overflow-hidden rounded-2xl border border-emerald-200/60 bg-gradient-to-br from-emerald-500/[0.03] to-emerald-500/[0.1] p-6 shadow-inner pointer-events-auto">
                <div className="absolute -top-10 -right-10 w-40 h-40 bg-emerald-400/20 rounded-full blur-3xl" />
                <div className="relative">
                  <h3 className="text-xs font-black text-emerald-800 uppercase tracking-widest flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-emerald-100 text-emerald-700 shadow-sm">
                      <Cpu size={16} />
                    </div>
                    AI Pipeline Ready
                  </h3>
                  <p className="mt-4 text-[11px] text-emerald-800/80 leading-relaxed font-semibold">
                    The ingestion engine is fully optimized! Features are aligned for <span className="text-emerald-700 font-bold bg-emerald-200/50 px-1 py-0.5 rounded">SHAP explainability</span> and LightGBM scrap prediction.
                  </p>
                  {pipelineState.status === 'complete' && (
                    <button 
                      onClick={onFinish}
                      className="mt-5 w-full py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 outline outline-2 outline-offset-2 outline-emerald-500/20 text-white text-xs font-black tracking-wide shadow-xl shadow-emerald-600/20 hover:shadow-emerald-600/40 hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2 group/btn"
                    >
                      GO TO SCORING DASHBOARD 
                      <ArrowRight size={16} className="group-hover/btn:translate-x-1 transition-transform" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="w-full">
            <IngestionHistory />
          </div>
        )}
      </div>
    </div>
  );
}
