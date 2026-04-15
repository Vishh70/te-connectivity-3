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
  Cpu
} from "lucide-react";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "telemetry", label: "Telemetry", icon: Activity },
  { id: "ingestion", label: "Data Ingest", icon: Database },
  { id: "audit", label: "Performance Audit", icon: AlertTriangle },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
];

const BOTTOM_ITEMS = [
  { id: "settings", label: "Settings", icon: Settings },
  { id: "help", label: "Help", icon: HelpCircle },
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
    switch (status) {
      case "NORMAL": return "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]";
      case "WARNING":
      case "LOW":
      case "MEDIUM": return "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]";
      case "CRITICAL":
      case "HIGH": return "bg-red-500 animate-pulse shadow-[0_0_12px_rgba(239,68,68,0.6)]";
      default: return "bg-slate-300";
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
    <aside className="hidden lg:flex flex-col w-[260px] shrink-0 h-screen sticky top-0 z-30 bg-white/40 backdrop-blur-2xl border-r border-white/40 animate-slide-in-left shadow-[4px_0_24px_-12px_rgba(15,23,42,0.1)]">
      {/* Premium Branding Area */}
      <div className="relative group px-6 py-8 border-b border-slate-100/50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
        <div className="relative flex items-center gap-4">
          <div className="relative w-11 h-11 rounded-2xl bg-brand-600 flex items-center justify-center shadow-xl shadow-brand-200 border border-brand-400/20 group-hover:scale-105 transition-transform duration-500">
            <div className="absolute inset-0 rounded-2xl bg-white/20 animate-pulse" />
            <span className="relative text-white font-black text-sm tracking-tight">TE</span>
          </div>
          <div>
            <p className="text-[13px] font-black text-slate-800 leading-tight tracking-tight uppercase">Control Center</p>
            <p className="text-[9px] text-brand-600 font-bold uppercase tracking-[0.2em] mt-0.5">Predictive AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Ecosystem */}
      <nav className="flex-1 px-4 py-6 space-y-8 overflow-y-auto custom-scrollbar">
        <div>
          <p className="px-4 mb-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-400/80">Main Control</p>
          <div className="space-y-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    handleNavClick(item.id);
                  }}
                  className={`group relative flex items-center gap-3 w-full px-4 py-3 text-[13px] font-bold rounded-2xl transition-all duration-300 ${
                    isActive 
                    ? "bg-white text-brand-600 shadow-sm border border-slate-100" 
                    : "text-slate-500 hover:bg-white/60 hover:text-slate-900"
                  }`}
                >
                  {isActive && (
                    <div className="absolute left-0 w-1 h-5 bg-brand-500 rounded-r-full shadow-[2px_0_8px_rgba(59,130,246,0.6)]" />
                  )}
                  <Icon size={18} className={`transition-transform duration-500 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Machine Selection Grid */}
        <div>
          <p className="px-4 mb-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-400/80">Active Assets</p>
          <div className="space-y-1">
            {loading ? (
              <div className="px-4 py-4 text-[10px] text-slate-400 font-bold animate-pulse">Initializing Assets...</div>
            ) : machines.length ? machines.map((opt) => {
              const isActive = activeMachine === opt.id;
              const status = statuses[opt.id] || "UNKNOWN";
              const displayId = opt.display_id || opt.id;
              const machineName = opt.name || `Injection Molder ${displayId}`;
              return (
                <button
                  key={opt.id}
                  onClick={() => onSelectMachine && onSelectMachine(opt.id)}
                  className={`group relative flex items-center justify-between w-full px-4 py-2.5 rounded-xl transition-all duration-300 ${
                    isActive 
                    ? "bg-brand-50/50 text-brand-700" 
                    : "text-slate-500 hover:bg-slate-50/50 hover:text-slate-900"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-2 h-2 rounded-full transition-all duration-500 ${getStatusColor(status)} ${isActive ? 'scale-125' : ''}`} />
                    <div className="flex min-w-0 flex-col items-start">
                      <span className="text-xs font-bold leading-tight">{displayId}</span>
                      <span className="max-w-[160px] truncate text-[10px] font-medium text-slate-400 leading-tight">
                        {machineName}
                      </span>
                    </div>
                  </div>
                  {isActive && (
                    <div className="flex gap-0.5">
                      <div className="w-1 h-3 bg-brand-400/30 rounded-full animate-pulse [animation-delay:-0.2s]" />
                      <div className="w-1 h-3 bg-brand-400/30 rounded-full animate-pulse [animation-delay:-0.1s]" />
                      <div className="w-1 h-3 bg-brand-400/30 rounded-full animate-pulse" />
                    </div>
                  )}
                </button>
              );
            }) : (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-white/40 px-4 py-4">
                <div className="text-xs font-bold text-slate-700">{activeMachine || "No machines found"}</div>
                <div className="mt-1 text-[10px] font-medium text-slate-400">Backend may still be scanning files</div>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Persistence Controls */}
      <div className="px-4 py-6 border-t border-slate-100/50 bg-slate-50/30">
        <div className="space-y-1">
          {BOTTOM_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.id} className="flex items-center gap-3 w-full px-4 py-2.5 text-xs font-bold text-slate-400 transition-colors group opacity-75">
                <Icon size={16} className="group-hover:rotate-12 transition-transform" />
                {item.label}
                <span className="ml-auto text-[9px] font-black uppercase tracking-[0.2em] text-slate-300">Soon</span>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
