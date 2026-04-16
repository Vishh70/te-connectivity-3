import React from "react";
import apiClient, { getWsUrl } from "../utils/apiClient";
import {
  LayoutDashboard,
  Activity,
  AlertTriangle,
  BarChart3,
  Settings,
  HelpCircle,
  Database,
  CloudUpload,
  Cpu,
  ChevronRight
} from "lucide-react";
import { mapStatus, UI_STATUS } from "../utils/statusUtils";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "telemetry", label: "Telemetry", icon: Activity },
  { id: "ingestion", label: "Data Ingest", icon: Database },
  { id: "audit", label: "Performance Audit", icon: AlertTriangle },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
];

const NAV_TARGETS = {
  dashboard: "dashboard",
  telemetry: "telemetry-grid",
  analytics: "prediction-stats",
  ingestion: "ingestion",
  audit: "audit-hub",
};

export default function Sidebar({ activeMachine, onSelectMachine, activeView, onViewChange }) {
  const [machines, setMachines] = React.useState([]);
  const [statuses, setStatuses] = React.useState({});
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    const fetchMachines = async () => {
      try {
        const res = await apiClient.get("/api/machines");
        setMachines(res.data);
      } catch (err) {
        console.error("Failed to fetch machines:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchMachines();
  }, []);

  React.useEffect(() => {
    let ws;
    let fallbackInterval;

    const pollStatuses = async () => {
      try {
        const res = await apiClient.get("/api/machines/status");
        const statusMap = {};
        res.data.forEach(m => {
          statusMap[m.id] = m.status;
        });
        setStatuses(statusMap);
      } catch (err) {
        console.error("Status poll failed:", err);
      }
    };

    const connectWs = () => {
      ws = new WebSocket(getWsUrl("/ws/machines/status"));

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const statusMap = {};
          data.forEach(m => {
            statusMap[m.id] = m.status;
          });
          setStatuses(statusMap);
        } catch (err) {
          console.error("WS Parse Error:", err);
        }
      };

      ws.onerror = (err) => {
        console.error("WS Status Error, falling back to polling", err);
        if (!fallbackInterval) {
          pollStatuses();
          fallbackInterval = setInterval(pollStatuses, 15000);
        }
      };
    };

    connectWs();

    return () => {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
      if (fallbackInterval) clearInterval(fallbackInterval);
    };
  }, []);

  const getStatusColor = (status) => {
    const uiStatus = mapStatus(status);
    switch (uiStatus) {
      case UI_STATUS.NORMAL: return "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]";
      case UI_STATUS.WARNING:
      case UI_STATUS.WATCH: return "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]";
      case UI_STATUS.CRITICAL:
      case UI_STATUS.HIGH: return "bg-red-500 animate-pulse shadow-[0_0_12px_rgba(239,68,68,0.6)]";
      default: return "bg-slate-600";
    }
  };

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
    else window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleNavClick = (itemId) => {
    onViewChange && onViewChange(itemId);
    const targetId = NAV_TARGETS[itemId];
    if (!targetId) return;
    setTimeout(() => scrollTo(targetId), 80);
  };

  return (
    <aside className="hidden lg:flex flex-col w-[280px] shrink-0 h-screen sticky top-0 z-40 bg-slate-900 shadow-2xl animate-slide-in-left">
      {/* Dark Premium Branding Area */}
      <div className="relative group px-6 py-8 border-b border-slate-800/80 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
        <div className="relative flex items-center gap-4">
          <div className="relative w-11 h-11 rounded-2xl bg-brand-600 flex items-center justify-center shadow-xl shadow-brand-900/50 border border-brand-400/20 group-hover:scale-105 transition-transform duration-500">
            <div className="absolute inset-0 rounded-2xl bg-white/10 animate-pulse" />
            <span className="relative text-white font-black text-sm tracking-tight">TE</span>
          </div>
          <div>
            <p className="text-[13px] font-black text-white leading-tight tracking-tight uppercase">Control Center</p>
            <p className="text-[9px] text-brand-400 font-bold uppercase tracking-[0.2em] mt-0.5">V3 Production AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Ecosystem - Dark Mode */}
      <nav className="flex-1 px-4 py-6 space-y-8 overflow-y-auto custom-scrollbar">
        <div>
          <p className="px-4 mb-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Navigation</p>
          <div className="space-y-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`group relative flex items-center gap-3 w-full px-4 py-2.5 text-[11px] font-black uppercase tracking-wider rounded-xl transition-all duration-300 ${
                    isActive 
                    ? "bg-brand-600 text-white shadow-lg shadow-brand-900 border-brand-500" 
                    : "text-slate-400 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <Icon size={16} className={`transition-transform duration-500 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`} />
                  {item.label}
                  {isActive && <div className="ml-auto w-1 h-4 bg-white/40 rounded-full" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Machine Selection Grid - High Contrast */}
        <div>
          <p className="px-4 mb-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Fleet Assets</p>
          <div className="space-y-1">
            {loading ? (
              <div className="px-4 py-4 text-[10px] text-slate-500 font-bold animate-pulse">Scanning Network...</div>
            ) : machines.length ? machines.map((opt) => {
              const isActive = activeMachine === opt.id;
              const status = statuses[opt.id] || "UNKNOWN";
              const displayId = opt.display_id || opt.id;
              const machineName = opt.name || `Asset ${displayId}`;
              return (
                <button
                  key={opt.id}
                  onClick={() => onSelectMachine && onSelectMachine(opt.id)}
                  className={`group relative flex items-center justify-between w-full px-4 py-2 rounded-lg transition-all duration-300 border border-transparent ${
                    isActive 
                    ? "bg-slate-800 border-slate-700 text-brand-400 shadow-sm" 
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-2 h-2 rounded-full ring-2 ring-slate-900/50 transition-all duration-500 ${getStatusColor(status)} ${isActive ? 'scale-110 ring-brand-900/40' : ''}`} />
                    <div className="flex min-w-0 flex-col items-start pt-0.5">
                      <span className="text-[11px] font-black leading-none">{displayId}</span>
                      <span className="max-w-[170px] truncate text-[9px] font-bold text-slate-600 mt-1 uppercase tracking-tighter">
                        {machineName.replace('Injection Molder ', '')}
                      </span>
                    </div>
                  </div>
                  {isActive && <ChevronRight size={14} className="text-brand-500" />}
                </button>
              );
            }) : (
              <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-900/40 px-4 py-4">
                <div className="text-xs font-bold text-slate-500">Awaiting Discovery</div>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Persistence Controls */}
      <div className="px-4 py-6 border-t border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3 w-full px-4 py-3 text-[10px] font-black uppercase text-slate-500 tracking-widest bg-slate-900/50 rounded-xl border border-slate-800">
             <Settings size={14} className="text-brand-500" />
             System Console
          </div>
      </div>
    </aside>
  );
}
