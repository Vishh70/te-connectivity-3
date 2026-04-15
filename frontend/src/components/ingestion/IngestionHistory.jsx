import React, { useState, useEffect } from "react";
import { 
  FileText, 
  Search, 
  Filter, 
  CheckCircle2, 
  Clock, 
  AlertCircle,
  Database,
  Monitor,
  Droplets,
  ArrowRight,
  History
} from "lucide-react";

import apiClient from "../../utils/apiClient";

export default function IngestionHistory() {
  const [filter, setFilter] = useState('All');
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await apiClient.get("/api/ingest/history");
        setHistory(response.data);
      } catch (err) {
        console.error("Failed to fetch history:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const filteredHistory = history.filter((item) => {
    if (filter === 'All') return true;
    return String(item.type || '').toLowerCase() === filter.toLowerCase();
  });

  return (
    <div className="glass-card p-6 space-y-6 animate-fade-in bg-white/60">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-bold text-slate-700 flex items-center gap-2">
            <History size={16} className="text-brand-500" />
            Ingestion History Log
          </h2>
          <p className="text-[10px] text-slate-400 font-medium mt-1 uppercase tracking-wider">Tracking industrial data streams (Last 30 days)</p>
        </div>

        <div className="flex items-center gap-2 overflow-x-auto pb-2 sm:pb-0">
          {['All', 'Machine', 'Hydra', 'Merged'].map(t => (
            <button 
              key={t}
              onClick={() => setFilter(t)}
              className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all ${filter === t ? 'bg-brand-600 text-white' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="pb-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Source File</th>
              <th className="pb-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest px-4">Type</th>
              <th className="pb-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest px-4">Machine</th>
              <th className="pb-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest px-4 text-right">Size</th>
              <th className="pb-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest px-4 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {loading ? (
               <tr><td colSpan="5" className="py-8 text-center text-xs font-bold text-slate-400">Loading history...</td></tr>
            ) : filteredHistory.length === 0 ? (
               <tr><td colSpan="5" className="py-8 text-center text-xs font-bold text-slate-400">No ingestion records found.</td></tr>
            ) : filteredHistory.map((item) => (
              <tr key={item.id} className="group hover:bg-slate-50/50 transition-all">
                <td className="py-4">
                  <div className="flex items-center gap-3">
                    <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${
                      item.type === 'Machine' ? 'bg-blue-50 text-blue-500' : 
                      item.type === 'Hydra' ? 'bg-amber-50 text-amber-500' : 'bg-purple-50 text-purple-500'
                    }`}>
                       {item.type === 'Machine' ? <Monitor size={14} /> : 
                        item.type === 'Hydra' ? <Droplets size={14} /> : <Database size={14} />}
                    </div>
                    <div>
                      <p className="text-xs font-bold text-slate-700">{item.name}</p>
                      <p className="text-[10px] text-slate-400 font-medium">{item.timestamp}</p>
                    </div>
                  </div>
                </td>
                <td className="py-4 px-4 text-[10px] font-bold text-slate-500 italic opacity-80">{item.type}</td>
                <td className="py-4 px-4">
                  <span className="px-2 py-1 rounded bg-slate-100 text-[10px] font-bold text-slate-600 uppercase border border-slate-200">{item.machine || 'AUTO'}</span>
                </td>
                <td className="py-4 px-4 text-right text-[10px] font-bold text-slate-500">{item.size}</td>
                <td className="py-4 px-4 text-right">
                   <div className="flex items-center justify-end gap-1.5">
                     <span className={`text-[10px] font-black uppercase tracking-wider ${item.status === 'Ready' || item.status === 'Success' ? 'text-emerald-500' : 'text-red-500'}`}>
                       {item.status}
                     </span>
                     {(item.status === 'Ready' || item.status === 'Success') ? <CheckCircle2 size={12} className="text-emerald-500" /> : <AlertCircle size={12} className="text-red-500" />}
                   </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-center pt-4">
        <button className="text-[10px] font-bold text-brand-600 hover:text-brand-700 flex items-center gap-2 group transition-all">
          LOAD ARCHIVE HISTORY <Search size={12} className="group-hover:scale-110" />
        </button>
      </div>
    </div>
  );
}
