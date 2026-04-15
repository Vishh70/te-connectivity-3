import React, { useState, useCallback } from "react";
import { 
  CloudUpload, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  Zap, 
  Cpu,
  Monitor,
  Droplets,
  Search,
  X,
  Activity,
  ShieldCheck
} from "lucide-react";

const PARAMETER_EXTENSIONS = ['csv', 'txt'];
const HYDRA_EXTENSIONS = ['xls', 'xlsx'];

const getFileExtension = (file) => String(file?.name || "").split(".").pop().toLowerCase();

export default function UploadPortal({ onStart, isProcessing }) {
  const [parameterFiles, setParameterFiles] = useState([]);
  const [hydraFile, setHydraFile] = useState(null);
  const [parameterVerified, setParameterVerified] = useState(false);
  const [hydraVerified, setHydraVerified] = useState(false);
  const [detecting, setDetecting] = useState(false);

  const clearFile = (type, index = null) => {
    if (type === 'parameter') {
      if (index === null) {
        setParameterFiles([]);
        setParameterVerified(false);
      } else {
        const nextFiles = parameterFiles.filter((_, fileIndex) => fileIndex !== index);
        setParameterFiles(nextFiles);
        verifyParameterBatch(nextFiles);
      }
    } else {
      setHydraFile(null);
      setHydraVerified(false);
    }
  };

  const inspectFile = async (file, type) => {
    await new Promise(r => setTimeout(r, 500));

    const ext = getFileExtension(file);
    let isVerified = false;

    if (type === 'parameter') {
      const text = await file.slice(0, 10240).text();
      const textLower = text.toLowerCase();
      const mandatory = [
        'cushion', 'cycle_time', 'cylinder_temp', 'dosage_time',
        'ejector_torque', 'extruder_start_pos', 'extruder_torque',
        'injection_pressure', 'injection_time', 'machine_status',
        'peak_pressure_pos', 'peak_pressure_time', 'scrap_counter',
        'switch_position', 'switch_pressure', 'variable_name'
      ];
      isVerified = mandatory.some(m => textLower.includes(m));
      if (!PARAMETER_EXTENSIONS.includes(ext)) {
        isVerified = false;
      }
    } else {
      isVerified = HYDRA_EXTENSIONS.includes(ext);
    }

    return isVerified;
  };

  const verifyParameterBatch = async (files) => {
    if (!files.length) {
      setParameterVerified(false);
      return;
    }

    setDetecting(true);
    try {
      const results = await Promise.all(files.map((file) => inspectFile(file, 'parameter')));
      setParameterVerified(results.length > 0 && results.every(Boolean));
    } catch (err) {
      console.error("Verification failed", err);
      setParameterVerified(false);
    } finally {
      setDetecting(false);
    }
  };

  const handleDrop = useCallback((e, type) => {
    e.preventDefault();
    if (isProcessing) return;

    const droppedFiles = Array.from(e.dataTransfer?.files || []);
    if (droppedFiles.length === 0) return;

    if (type === 'parameter') {
      const nextFiles = Array.from(new Map([...parameterFiles, ...droppedFiles].map((file) => [file.name, file])).values());
      setParameterFiles(nextFiles);
      verifyParameterBatch(nextFiles);
    } else {
      const file = droppedFiles[0];
      setHydraFile(file);
      setDetecting(true);
      inspectFile(file, type)
        .then(setHydraVerified)
        .catch((err) => {
          console.error("Verification failed", err);
          setHydraVerified(false);
        })
        .finally(() => setDetecting(false));
    }
  }, [isProcessing, parameterFiles]);

  const handleFileChange = (e, type) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    if (type === 'parameter') {
      const nextFiles = Array.from(new Map([...parameterFiles, ...files].map((file) => [file.name, file])).values());
      setParameterFiles(nextFiles);
      verifyParameterBatch(nextFiles);
      e.target.value = "";
    } else {
      const file = files[0];
      setHydraFile(file);
      setDetecting(true);
      inspectFile(file, type)
        .then(setHydraVerified)
        .catch((err) => {
          console.error("Verification failed", err);
          setHydraVerified(false);
        })
        .finally(() => setDetecting(false));
    }
  };

  const handleStart = () => {
    if (parameterFiles.length && hydraFile) {
      onStart(parameterFiles, hydraFile);
    }
  };

  const isReady = parameterFiles.length > 0 && hydraFile && parameterVerified && hydraVerified && !isProcessing;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Title */}
      <div className="flex items-center gap-4">
        <div className="h-[2px] w-8 bg-brand-500 rounded-full" />
        <h2 className="text-[11px] font-black text-slate-800 uppercase tracking-[0.3em]">Neural Ingestion Portal</h2>
        <div className="h-[1px] flex-1 bg-gradient-to-r from-slate-200 to-transparent" />
      </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Parameter Upload Pod */}
        <div 
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => handleDrop(e, 'parameter')}
          className={`relative group h-[280px] transition-all duration-500 ${parameterFiles.length ? 'scale-[1.02]' : ''}`}
        >
          {/* Intake Pod Background */}
          <div className={`absolute inset-0 rounded-[2rem] transition-all duration-500 blur-2xl opacity-20 ${
            parameterVerified ? 'bg-emerald-500' : parameterFiles.length ? 'bg-brand-500' : 'bg-slate-300 group-hover:bg-brand-400'
          }`} />
          
          <div className={`relative glass-card p-8 h-full flex flex-col items-center justify-center text-center border-0 overflow-hidden ${
            parameterFiles.length ? 'bg-white/80' : 'bg-white/40 hover:bg-white/60'
          }`}>
            <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-brand-500/20 to-transparent" />
            
             {parameterFiles.length ? (
                <div className="animate-fade-in space-y-6 w-full">
                  <div className="relative flex flex-col items-center">
                    <div className={`h-20 w-20 rounded-3xl ${parameterVerified ? 'bg-emerald-600 shadow-emerald-200' : 'bg-brand-600 shadow-brand-200'} text-white flex items-center justify-center shadow-2xl mb-4 transition-all hover:rotate-6`}>
                      {parameterVerified ? <ShieldCheck size={40} className="animate-pulse" /> : <Monitor size={40} />}
                    </div>
                    <p className="text-sm font-black text-slate-800 tracking-tight px-4 line-clamp-1">
                      {parameterFiles.length} machine file(s) selected
                    </p>
                    <div className="mt-3 flex flex-wrap justify-center gap-2 px-4 max-h-24 overflow-auto">
                      {parameterFiles.map((file, index) => (
                        <span
                          key={`${file.name}-${index}`}
                          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-[10px] font-black text-slate-600 shadow-sm"
                        >
                          <span className="max-w-[140px] truncate">{file.name}</span>
                          <button
                            type="button"
                            onClick={() => clearFile('parameter', index)}
                            className="text-red-400 hover:text-red-600"
                            aria-label={`Remove ${file.name}`}
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                       <span className={`text-[10px] font-black ${parameterVerified ? 'text-emerald-600' : 'text-brand-600'} uppercase tracking-widest bg-white/80 px-2 py-0.5 rounded-md border border-slate-100`}>
                        {parameterVerified ? 'Expert Verified' : 'Scanning Schema...'}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-center gap-3">
                    <div className="px-3 py-1.5 rounded-xl bg-slate-50 text-[10px] font-black text-slate-400 border border-slate-100 uppercase tracking-wider">
                      {parameterFiles.length} file(s)
                    </div>
                    <button onClick={() => clearFile('parameter')} className="px-4 py-1.5 rounded-xl bg-red-50 text-[10px] font-black text-red-500 border border-red-100 hover:bg-red-500 hover:text-white transition-all uppercase tracking-wider">Eject</button>
                  </div>
                </div>
              ) : (
                <>
                 <div className="relative h-20 w-20 rounded-[2.5rem] bg-slate-100/50 flex items-center justify-center text-slate-400 group-hover:scale-110 group-hover:bg-brand-50 group-hover:text-brand-500 transition-all duration-700 mb-6 border border-slate-200/50 shadow-inner">
                   <div className="absolute inset-0 rounded-[2.5rem] border border-brand-500/0 group-hover:border-brand-500/20 animate-spin-slow" />
                   <CloudUpload size={36} className="group-hover:animate-bounce" />
                 </div>
                 <h3 className="text-sm font-black text-slate-800 tracking-tight uppercase">Machine Logic Stream</h3>
                 <p className="mt-3 text-[11px] text-slate-500 leading-relaxed font-bold max-w-[200px] opacity-70 italic">Injection molding MES sensor values & cycle metadata</p>
                 <input 
                   type="file" 
                   accept=".csv,.txt" 
                   multiple
                   className="absolute inset-0 opacity-0 cursor-pointer" 
                   onChange={(e) => handleFileChange(e, 'parameter')}
                 />
                </>
              )}
          </div>
        </div>

        {/* Hydra Sensor Pod */}
        <div 
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => handleDrop(e, 'hydra')}
          className={`relative group h-[280px] transition-all duration-500 ${hydraFile ? 'scale-[1.02]' : ''}`}
        >
          <div className={`absolute inset-0 rounded-[2rem] transition-all duration-500 blur-2xl opacity-20 ${
            hydraVerified ? 'bg-amber-500' : hydraFile ? 'bg-brand-500' : 'bg-slate-300 group-hover:bg-brand-400'
          }`} />
          
          <div className={`relative glass-card p-8 h-full flex flex-col items-center justify-center text-center border-0 overflow-hidden ${
            hydraFile ? 'bg-white/80' : 'bg-white/40 hover:bg-white/60'
          }`}>
            <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-amber-500/20 to-transparent" />
            
             {hydraFile ? (
                <div className="animate-fade-in space-y-6 w-full">
                  <div className="relative flex flex-col items-center">
                    <div className={`h-20 w-20 rounded-3xl ${hydraVerified ? 'bg-amber-600 shadow-amber-200' : 'bg-brand-600 shadow-brand-200'} text-white flex items-center justify-center shadow-2xl mb-4 transition-all hover:rotate-6`}>
                      {hydraVerified ? <Activity size={40} className="animate-pulse" /> : <Droplets size={40} />}
                    </div>
                    <p className="text-sm font-black text-slate-800 tracking-tight px-4 line-clamp-1">{hydraFile.name}</p>
                    <div className="flex items-center gap-2 mt-2">
                       <span className={`text-[10px] font-black ${hydraVerified ? 'text-amber-600' : 'text-brand-600'} uppercase tracking-widest bg-white/80 px-2 py-0.5 rounded-md border border-slate-100`}>
                        {hydraVerified ? ' Expert Verified' : 'Parsing Dynamics...'}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-center gap-3">
                    <div className="px-3 py-1.5 rounded-xl bg-slate-50 text-[10px] font-black text-slate-400 border border-slate-100 uppercase tracking-wider">{(hydraFile.size / (1024 * 1024)).toFixed(1)} MB</div>
                    <button onClick={() => clearFile('hydra')} className="px-4 py-1.5 rounded-xl bg-red-50 text-[10px] font-black text-red-500 border border-red-100 hover:bg-red-500 hover:text-white transition-all uppercase tracking-wider">Eject</button>
                  </div>
                </div>
              ) : (
                <>
                 <div className="relative h-20 w-20 rounded-[2.5rem] bg-slate-100/50 flex items-center justify-center text-slate-400 group-hover:scale-110 group-hover:bg-brand-50 group-hover:text-amber-500 transition-all duration-700 mb-6 border border-slate-200/50 shadow-inner">
                   <div className="absolute inset-0 rounded-[2.5rem] border border-amber-500/0 group-hover:border-amber-500/20 animate-spin-slow" />
                   <CloudUpload size={36} className="group-hover:animate-bounce" />
                 </div>
                 <h3 className="text-sm font-black text-slate-800 tracking-tight uppercase">High-Speed Telemetry</h3>
                 <p className="mt-3 text-[11px] text-slate-500 leading-relaxed font-bold max-w-[200px] opacity-70 italic">Actuator dynamics & hydraulic pressure streams</p>
                 <input 
                   type="file" 
                   accept=".xls,.xlsx" 
                   className="absolute inset-0 opacity-0 cursor-pointer" 
                   onChange={(e) => handleFileChange(e, 'hydra')}
                 />
                </>
              )}
          </div>
        </div>
      </div>

      {/* Control Action */}
      <div className="flex flex-col items-center gap-4 pt-4">
        <div className={`w-full max-w-[760px] rounded-[2rem] border p-4 sm:p-5 shadow-lg transition-all ${isReady ? "border-brand-200 bg-white/90" : "border-slate-200 bg-white/70"}`}>
          <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-between sm:items-center">
            <div className="text-center sm:text-left">
              <p className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Pipeline Control</p>
              <p className="mt-1 text-sm font-semibold text-slate-600">
                {isReady
                  ? "Files are validated. Launch the ingestion pipeline now."
                  : "Select all machine files and the Hydra file to enable execution."}
              </p>
            </div>

            <button
              onClick={handleStart}
              disabled={!isReady}
              className={`relative w-full sm:w-auto px-8 py-4 rounded-[1.6rem] font-black tracking-[0.2em] text-xs uppercase transition-all active:scale-95 overflow-hidden ${
                isReady
                  ? "bg-gradient-to-r from-brand-600 via-indigo-600 to-emerald-600 text-white shadow-2xl shadow-brand-500/20"
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full ${isReady ? "group-hover:translate-x-full" : ""} transition-transform duration-1000`} />
              <div className="flex items-center justify-center gap-3">
                <Zap size={16} className={isReady ? "text-brand-100 fill-brand-100" : "text-slate-300 fill-slate-300"} />
                EXECUTE INGESTION PIPELINE
              </div>
            </button>
          </div>
        </div>

        {!isReady && !isProcessing && (
          <div className="flex items-center gap-3 px-8 py-3 rounded-[1.5rem] bg-slate-100 border border-slate-200/60 text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] shadow-inner">
            <div className="flex gap-1.5">
               <div className="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse" />
               <div className="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse [animation-delay:0.2s]" />
            </div>
            Awaiting Dual-Source Validation
          </div>
        )}

        {isProcessing && (
          <div className="flex flex-col items-center gap-4 animate-fade-in">
            <div className="flex gap-3">
              <div className="w-2 h-2 rounded-full bg-brand-500 animate-bounce [animation-delay:-0.3s] shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
              <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce [animation-delay:-0.15s] shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
            </div>
            <p className="text-[11px] font-black text-brand-700 uppercase tracking-[0.3em] bg-brand-50 px-4 py-1.5 rounded-full border border-brand-100 shadow-sm">Analyzing High-Dim Streams...</p>
          </div>
        )}
      </div>
    </div>
  );
}
