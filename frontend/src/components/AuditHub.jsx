import React, { useState, useEffect } from "react";
import { CheckCircle2, XCircle, ShieldCheck, Target, BarChart, AlertTriangle, Search } from "lucide-react";
import apiClient from "../utils/apiClient";

export default function AuditHub() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAudit = async () => {
      try {
        setLoading(true);
        const res = await apiClient.get("/api/audit/validation");
        setData(res.data);
      } catch (err) {
        console.error("Audit fetch failed:", err);
        setError("Failed to load audit results. Ensure the backend is running.");
      } finally {
        setLoading(false);
      }
    };
    fetchAudit();
  }, []);

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-brand-500 border-t-transparent" />
          <p className="text-sm font-bold text-slate-500 animate-pulse uppercase tracking-widest">Running Ground-Truth Audit...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card flex h-64 flex-col items-center justify-center p-8 text-center">
        <AlertTriangle size={48} className="text-amber-500 mb-4" />
        <h3 className="text-lg font-bold text-slate-800">Operational Sync Error</h3>
        <p className="mt-2 text-sm text-slate-400 max-w-md">{error}</p>
      </div>
    );
  }

  const results = data?.results || [];

  return (
    <div id="audit-hub" className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Audit Stats Header */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Target size={64} className="text-brand-600" />
          </div>
          <p className="text-[11px] font-black uppercase tracking-widest text-slate-400">Total Audit Accuracy</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-black text-slate-800 tracking-tight">{data?.accuracy}%</span>
            <span className="text-xs font-bold text-emerald-600 uppercase">Verified</span>
          </div>
          <p className="mt-2 text-xs text-slate-500 font-medium">Cross-referenced against official MES logs</p>
        </div>

        <div className="glass-card p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <ShieldCheck size={64} className="text-emerald-600" />
          </div>
          <p className="text-[11px] font-black uppercase tracking-widest text-slate-400">Scrap Events Matched</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-black text-slate-800 tracking-tight">{data?.matches}</span>
            <p className="text-xs font-bold text-slate-400 uppercase">of {data?.total_cases} Reported Instances</p>
          </div>
          <div className="mt-3 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
             <div 
               className="h-full bg-emerald-500 transition-all duration-1000" 
               style={{ width: `${data?.accuracy}%` }}
             />
          </div>
        </div>

        <div className="glass-card p-6 flex flex-col justify-center bg-gradient-to-br from-brand-600 to-brand-800 text-white border-0">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center backdrop-blur-md">
              <Search size={20} className="text-white" />
            </div>
            <div>
              <p className="text-[10px] font-black uppercase tracking-widest opacity-70">Senior Auditor</p>
              <p className="text-sm font-bold">Operational Integrity</p>
            </div>
          </div>
          <p className="text-xs leading-relaxed opacity-90">
            Audit system aligned successfully with machine telemetry. Use the results below to verify model sensitivity across different machine assets.
          </p>
        </div>
      </div>

      {/* Audit Detail Table */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between">
          <h2 className="text-sm font-black uppercase tracking-widest text-slate-800 flex items-center gap-2">
            <BarChart size={18} className="text-brand-500" />
            Performance Validation Log
          </h2>
          <div className="flex gap-2">
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 uppercase">
              <div className="w-2 h-2 rounded-full bg-emerald-500" /> Success
            </span>
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 uppercase">
              <div className="w-2 h-2 rounded-full bg-red-500" /> Missed
            </span>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50/50">
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400">Machine</th>
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400">Case ID</th>
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400 text-center">Date & Interval</th>
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400">Model Predicted?</th>
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400">Audit Status</th>
                <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-400 text-right">Max Risk</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {results.map((row) => (
                <tr key={`${row.machine}-${row.id}`} className="hover:bg-slate-50/30 transition-colors group">
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center gap-2 px-2 py-1 rounded-md bg-slate-100 text-[11px] font-black text-slate-700">
                      {row.machine}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-xs font-bold text-slate-600">{row.id}</td>
                  <td className="px-6 py-4 text-center">
                    <div className="inline-flex flex-col gap-0.5">
                      <span className="text-[11px] font-black text-slate-800">{row.date}</span>
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                        {row.start} — {row.end}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className={`text-xs font-black italic ${row.predicted === 'YES' ? 'text-brand-600' : 'text-slate-400'}`}>
                      {row.predicted === 'YES' ? 'YES (Scrap)' : 'NO (Normal)'}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {row.status === "MATCH" ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 text-[10px] font-black uppercase tracking-widest">
                        <CheckCircle2 size={12} /> SUCCESS
                      </span>
                    ) : row.status === "MISSED" ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-50 text-red-700 text-[10px] font-black uppercase tracking-widest">
                        <XCircle size={12} /> MISSED
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-slate-100 text-slate-400 text-[10px] font-black uppercase tracking-widest">
                        {row.status}
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className={`text-sm font-black ${
                      row.max_risk >= 0.8 ? 'text-red-600' : 
                      row.max_risk >= 0.6 ? 'text-orange-600' : 'text-slate-400'
                    }`}>
                      {(row.max_risk * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
