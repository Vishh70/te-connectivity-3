import React, { useMemo, useState, useRef, useEffect } from "react";
import { LineChart, Line } from "recharts";
import { Activity, Flame, Search, ArrowRight, Filter } from "lucide-react";
import { mapStatus, UI_STATUS, getStatusColorClass, getStatusBadgeClass, SENSOR_METADATA, formatSensorName } from "../utils/statusUtils";

const FILTERS = [
  { id: "all", label: "All" },
  { id: "warning", label: "Warnings" },
  { id: "critical", label: "Critical" },
  { id: "normal", label: "Normal" },
  { id: "root", label: "Root Causes" },
];

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};


export default function TelemetryGrid({ telemetryRows, selectedSensor, onSelectSensor }) {
  const [activeFilter, setActiveFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const gridRef = useRef(null);

  const tableData = useMemo(() => {
    const rows = Array.isArray(telemetryRows) ? telemetryRows : [];
    const statusWeight = { [UI_STATUS.CRITICAL]: 3, [UI_STATUS.WARNING]: 2, [UI_STATUS.HIGH]: 2, [UI_STATUS.WATCH]: 1.5, [UI_STATUS.NORMAL]: 1 };

    const query = searchQuery.toLowerCase().trim();

    return rows
      .map((row) => {
        const status = mapStatus(row?.status);
        return { ...(row || {}), uiStatus: status };
      })
      .filter((row) => {
        // Search filter
        if (query && !(row?.sensor || "").toLowerCase().includes(query)) return false;

        // Status filter
        if (activeFilter === "all") return true;
        if (activeFilter === "root") return row?.is_root_cause;
        return (row?.uiStatus || "").toLowerCase() === activeFilter.toLowerCase();
      })
      .sort((a, b) => (statusWeight[b.uiStatus] || 0) - (statusWeight[a.uiStatus] || 0));
  }, [telemetryRows, activeFilter, searchQuery]);

  const prevSensorRef = useRef(selectedSensor);

  // Auto-scroll logic ONLY when user changes selection, not on every live data tick
  useEffect(() => {
    if (selectedSensor && gridRef.current && selectedSensor !== prevSensorRef.current) {
      const row = gridRef.current.querySelector(`[data-sensor-id="${selectedSensor}"]`);
      if (row) {
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        prevSensorRef.current = selectedSensor;
      }
    } else if (!selectedSensor) {
      prevSensorRef.current = null;
    }
  }, [selectedSensor, tableData.length]); // Use tableData.length just to ensure DOM is ready on first load, but won't trigger continuously

  const counts = useMemo(() => {
    return tableData.reduce(
      (acc, row) => {
        const status = mapStatus(row?.status);
        if (status === UI_STATUS.CRITICAL) acc.critical += 1;
        else if (status === UI_STATUS.WARNING) acc.warning += 1;
        else acc.normal += 1;
        if (row?.is_root_cause) acc.root += 1;
        return acc;
      },
      { warning: 0, critical: 0, normal: 0, root: 0 },
    );
  }, [tableData]);

  const visibleRows = useMemo(() => {
    let filtered = tableData;
    
    if (activeFilter === "root") filtered = tableData.filter((row) => row?.is_root_cause);
    else if (activeFilter === "critical") {
      filtered = tableData.filter((row) => mapStatus(row?.status) === UI_STATUS.CRITICAL);
    }
    else if (activeFilter === "warning") {
      filtered = tableData.filter((row) => mapStatus(row?.status) === UI_STATUS.WARNING);
    }
    else if (activeFilter === "normal") {
      filtered = tableData.filter((row) => mapStatus(row?.status) === UI_STATUS.NORMAL);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(row => 
        formatSensorName(row?.sensor || "").toLowerCase().includes(q)
      );
    }

    return filtered;
  }, [tableData, activeFilter, searchQuery]);

  const selectedLabel = selectedSensor ? formatSensorName(selectedSensor) : "No sensor selected";

  const filterOptions = useMemo(
    () =>
      FILTERS.map((filter) => {
        let count = tableData.length;
        if (filter.id === "warning") count = counts.warning;
        if (filter.id === "critical") count = counts.critical;
        if (filter.id === "normal") count = counts.normal;
        if (filter.id === "root") count = counts.root;
        return { ...filter, count };
      }),
    [counts, tableData.length],
  );

  return (
    <section id="telemetry-grid" ref={gridRef} className="glass-card flex h-full flex-col p-5 animate-slide-up scroll-mt-24" style={{ animationDelay: "0.25s" }}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-700">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-50">
              <Activity size={16} className="text-emerald-500" />
            </div>
            Real-Time Telemetry Grid
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Use the filters to isolate warnings, critical sensors, and root-cause candidates.
          </p>
        </div>

        <div className="relative min-w-[280px]">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
          <input
            type="text"
            placeholder="Search sensors (e.g. Temperature)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-white/70 py-3 pl-11 pr-4 text-sm font-medium text-slate-700 shadow-sm outline-none transition-all focus:border-brand-400 focus:bg-white focus:ring-4 focus:ring-brand-500/5"
          />
        </div>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 xl:grid-cols-4">
        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Sensors</p>
          <p className="mt-1 text-2xl font-extrabold text-slate-800">{tableData.length}</p>
        </div>
        <div className="rounded-2xl border border-amber-100 bg-amber-50/70 px-4 py-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-amber-500">Warnings</p>
          <p className="mt-1 text-2xl font-extrabold text-amber-600">{counts.warning}</p>
        </div>
        <div className="rounded-2xl border border-red-100 bg-red-50/70 px-4 py-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-red-500">Critical</p>
          <p className="mt-1 text-2xl font-extrabold text-red-600">{counts.critical}</p>
        </div>
        <div className="rounded-2xl border border-emerald-100 bg-emerald-50/70 px-4 py-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-emerald-500">Root causes</p>
          <p className="mt-1 text-2xl font-extrabold text-emerald-600">{counts.root}</p>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        {filterOptions.map((filter) => {
          const isActive = activeFilter === filter.id;
          return (
            <button
              key={filter.id}
              type="button"
              onClick={() => setActiveFilter(filter.id)}
              className={`rounded-full border px-4 py-2 text-xs font-semibold transition-all ${
                isActive
                  ? "border-brand-300 bg-brand-50 text-brand-700 shadow-sm"
                  : "border-slate-200 bg-white text-slate-500 hover:border-brand-200 hover:text-brand-600"
              }`}
            >
              {filter.label}
              <span className="ml-2 rounded-full bg-white/80 px-2 py-0.5 text-[10px] font-bold text-slate-400">
                {filter.count}
              </span>
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-hidden rounded-2xl border border-slate-200/80 bg-white/60">
        <div className="flex items-center justify-between border-b border-slate-200/80 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Click a row to inspect its live history and root causes
          </p>
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-300">
            Sorted by severity
          </p>
        </div>

        <div className="h-full overflow-auto">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="sticky top-0 z-10 border-b border-slate-200 bg-slate-50/95 text-[11px] uppercase tracking-wider text-slate-500 backdrop-blur-sm">
              <tr>
                <th className="px-5 py-3.5 font-bold">Parameter</th>
                <th className="px-5 py-3.5 font-bold">Status</th>
                <th className="px-5 py-3.5 font-bold text-right">Value</th>
                <th className="px-5 py-3.5 font-bold">Trend</th>
                <th className="px-5 py-3.5 font-bold">Safe Range</th>
                <th className="px-5 py-3.5 font-bold">History</th>
                <th className="px-5 py-3.5 font-bold"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {visibleRows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-5 py-10 text-center text-sm text-slate-400">
                    No rows match the current filter.
                  </td>
                </tr>
              ) : (
                visibleRows.map((row, idx) => {
                  const status = mapStatus(row?.status);
                  const isRoot = Boolean(row?.is_root_cause);
                  const isClickable = status !== UI_STATUS.NORMAL || isRoot;
                  const isSelected = selectedSensor === row?.sensor;
                  const delta = toNumber(row?.trend_delta) ?? 0;
                  const trendDirection = row?.trend_direction;
                  const trendIcon = trendDirection === "up" ? "▲" : trendDirection === "down" ? "▼" : "—";
                  const trendColor =
                    trendDirection === "up"
                      ? "text-emerald-600"
                      : trendDirection === "down"
                        ? "text-red-500"
                        : "text-slate-400";

                  const statusBadgeClass = getStatusBadgeClass(status);

                  const sparklineData = (Array.isArray(row?.sparkline) ? row.sparkline : [])
                    .map((value, pointIndex) => ({ pointIndex, value: toNumber(value) }))
                    .filter((point) => point.value !== null);

                  const rowClassName = [
                    "group transition-all duration-300 cursor-pointer relative",
                    isRoot
                      ? "bg-amber-50/50 hover:bg-amber-50/80"
                      : status === "EXCEEDED"
                        ? "bg-red-50/50 hover:bg-red-50/80"
                        : status === "WARNING"
                          ? "bg-amber-50/20 hover:bg-amber-50/50"
                          : "hover:bg-white/80 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.6),0_4px_20px_-4px_rgba(0,0,0,0.05)] z-0 hover:z-10",
                    isSelected ? "bg-brand-50/80 border-l-4 border-l-brand-500 shadow-sm z-20" : "border-l-4 border-l-transparent",
                  ]
                    .filter(Boolean)
                    .join(" ");

                  const jitter = (toNumber(row?.variance) ?? 0) > 0.15;
                  
                  return (
                    <tr
                      key={`${row?.sensor || "sensor"}-${idx}`}
                      className={rowClassName}
                      data-sensor-id={row?.sensor}
                      onClick={() => onSelectSensor(row.sensor)}
                    >
                      <td className="px-5 py-3.5 font-semibold text-slate-700">
                        <div className="flex items-center gap-2">
                          {isRoot && (
                            <span className="flex h-5 w-5 items-center justify-center rounded bg-amber-100 shadow-sm border border-amber-200">
                              <Flame size={12} className="text-amber-500" />
                            </span>
                          )}
                          <span>{formatSensorName(row?.sensor)}</span>
                          {jitter && (
                            <div 
                              className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.6)]" 
                              title="Signal Stability Warning: Jitter detected"
                            />
                          )}
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <span
                          className={`inline-block rounded-lg px-2.5 py-1 text-[10px] font-bold tracking-widest ${statusBadgeClass}`}
                        >
                          {status}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 font-mono font-bold text-slate-800 text-right">
                        <div className="flex flex-col items-end">
                          <span>{toNumber(row?.value) !== null ? Number(row.value).toFixed(2) : "--"}</span>
                          <span className="text-[10px] text-slate-400 font-medium uppercase tracking-tighter">
                            {SENSOR_METADATA[row.sensor]?.unit || ""}
                          </span>
                        </div>
                      </td>
                      <td className={`px-5 py-3.5 font-mono font-bold text-[11px] ${trendColor}`}>
                        <div className="flex items-center gap-1">
                          {trendIcon}
                          <span>{delta > 0 ? "+" : ""}{delta.toFixed(2)}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5 text-[11px] font-bold text-slate-400">
                        <div className="flex items-center gap-1.5">
                          <span className="bg-slate-100 px-1.5 py-0.5 rounded">{toNumber(row?.safe_min) !== null ? Number(row.safe_min).toFixed(2) : "0"}</span>
                          <span className="text-slate-200">—</span>
                          <span className="bg-slate-100 px-1.5 py-0.5 rounded">{toNumber(row?.safe_max) !== null ? Number(row.safe_max).toFixed(2) : "inf"}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        {sparklineData.length ? (
                          <LineChart width={80} height={24} data={sparklineData}>
                            <Line
                              type="monotone"
                              dataKey="value"
                              stroke={isRoot ? "#f59e0b" : status === UI_STATUS.CRITICAL ? "#ef4444" : "#3b82f6"}
                              strokeWidth={2.5}
                              dot={false}
                              isAnimationActive={false}
                            />
                          </LineChart>
                        ) : (
                          <span className="text-[10px] uppercase font-bold text-slate-300 tracking-wider">No Data</span>
                        )}
                      </td>
                      <td 
                        className="px-5 py-3.5 text-right"
                        onClick={(e) => { e.stopPropagation(); onSelectSensor(row.sensor); }}
                      >
                        <ArrowRight size={14} className={`inline transition-transform duration-300 ${isSelected ? "translate-x-1 text-brand-600" : "text-slate-300 group-hover:translate-x-1 group-hover:text-brand-400"}`} />
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
