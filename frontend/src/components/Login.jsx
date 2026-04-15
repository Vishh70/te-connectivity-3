import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck, LogIn, Lock, User, Info,
  ArrowRight, Cpu, Zap, Activity, Eye, EyeOff
} from 'lucide-react';
import apiClient from '../utils/apiClient';

const TICKER_WORDS = [
  "CUSHION", "CYCLE TIME", "CYLINDER TEMP", "DOSAGE TIME",
  "INJECTION PRESSURE", "PEAK PRESSURE", "SWITCH OVER POS",
  "EXTRUDER TORQUE", "SHOT COUNTER", "SCRAP COUNTER",
  "SENSOR FUSION", "PREDICTIVE AI", "REAL-TIME SYNC"
];

const BOOT_LOGS = [
  { text: "INITIALIZING NEURAL ENGINE...", done: false },
  { text: "LOADING SENSOR MODELS...", done: false },
  { text: "ESTABLISHING SECURE TUNNEL...", done: false },
  { text: "READY — AWAITING AUTHENTICATION.", done: true },
];

const Login = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError]   = useState('');
  const [loading, setLoading] = useState(false);
  const [booting, setBooting] = useState(true);
  const [visibleLogs, setVisibleLogs] = useState([]);
  const [progress, setProgress] = useState(0);

  // Boot sequence
  useEffect(() => {
    let logIdx = 0;
    const interval = setInterval(() => {
      if (logIdx < BOOT_LOGS.length) {
        setVisibleLogs(prev => [...prev, BOOT_LOGS[logIdx]]);
        setProgress(Math.round(((logIdx + 1) / BOOT_LOGS.length) * 100));
        logIdx++;
      } else {
        clearInterval(interval);
        setTimeout(() => setBooting(false), 700);
      }
    }, 480);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Both username and access key are required.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await apiClient.post('/api/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });

      if (response.data.access_token) {
        localStorage.setItem('jwt_token', response.data.access_token);
        onLoginSuccess();
      } else {
        setError('Server refused the handshake. Please try again.');
      }
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail ? `${detail}` : 'Invalid credentials. Please check and retry.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">

      {/* ── Ambient glow orbs ── */}
      <div className="pointer-events-none absolute -top-40 -left-40 h-96 w-96 rounded-full bg-brand-500/20 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-[-8rem] right-[-6rem] h-80 w-80 rounded-full bg-emerald-500/15 blur-[100px]" />
      <div className="pointer-events-none absolute top-1/2 left-1/2 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-500/10 blur-[80px]" />

      {/* ── Scanline texture ── */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,1) 2px, rgba(255,255,255,1) 4px)',
          backgroundSize: '100% 4px',
        }}
      />

      {/* ── Ticker background ── */}
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-around py-6 opacity-[0.05] select-none">
        {[0, 1, 2].map(row => (
          <div key={row} className="flex whitespace-nowrap overflow-hidden" style={{ animation: `ticker ${55 + row * 10}s linear infinite ${row % 2 === 1 ? 'reverse' : ''}` }}>
            {[...TICKER_WORDS, ...TICKER_WORDS, ...TICKER_WORDS].map((word, i) => (
              <span key={i} className="text-5xl font-black text-white px-10 border-r border-white/10 tracking-tighter">
                {word}
              </span>
            ))}
          </div>
        ))}
      </div>

      {/* ── Boot Overlay ── */}
      <AnimatePresence>
        {booting && (
          <motion.div
            key="boot"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 1.04 }}
            transition={{ duration: 0.6 }}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-slate-950"
          >
            <div className="w-full max-w-sm px-8 space-y-4">
              {/* Logo mark */}
              <div className="flex items-center gap-3 mb-8">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500/20 border border-brand-400/30">
                  <Cpu size={20} className="text-brand-400 animate-pulse" />
                </div>
                <div>
                  <p className="text-[10px] font-black tracking-[0.5em] text-brand-400 uppercase">TE AI Cup</p>
                  <p className="text-[9px] text-slate-500 tracking-widest uppercase">Predictive Maintenance</p>
                </div>
              </div>

              {/* Boot logs */}
              <div className="space-y-2 font-mono">
                {visibleLogs.filter(Boolean).map((log, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-center gap-3 text-[11px]"
                  >
                    <span className={`shrink-0 font-black ${log?.done ? 'text-emerald-400' : 'text-brand-400'}`}>
                      {log?.done ? '✓' : '›'}
                    </span>
                    <span className={log?.done ? 'text-emerald-300' : 'text-slate-400'}>{log?.text}</span>
                  </motion.div>
                ))}
              </div>

              {/* Progress bar */}
              <div className="mt-8 h-[2px] w-full overflow-hidden rounded-full bg-slate-800">
                <motion.div
                  className="h-full bg-brand-500 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                />
              </div>
              <p className="text-right text-[10px] font-mono text-slate-600">{progress}%</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Main Login Card ── */}
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.96 }}
        animate={{ opacity: booting ? 0 : 1, y: booting ? 20 : 0, scale: booting ? 0.96 : 1 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md mx-4"
      >
        {/* Card */}
        <div
          className="relative overflow-hidden rounded-3xl border border-white/10"
          style={{
            background: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(32px) saturate(150%)',
            WebkitBackdropFilter: 'blur(32px) saturate(150%)',
            boxShadow: '0 32px 80px -16px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.07)',
          }}
        >
          {/* Top accent line */}
          <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-brand-400 to-transparent opacity-60" />

          <div className="p-10 md:p-12">
            {/* Header */}
            <div className="flex flex-col items-center mb-10">
              <motion.div whileHover={{ scale: 1.06 }} className="relative mb-6 cursor-default">
                <div className="absolute inset-0 rounded-2xl bg-brand-500/30 blur-xl" />
                <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-lg shadow-brand-500/30">
                  <ShieldCheck className="h-8 w-8 text-white" />
                </div>
              </motion.div>

              <h1 className="text-2xl font-black text-white tracking-tight">
                TE <span className="text-brand-400">AI Cup</span>
              </h1>
              <p className="mt-1.5 text-sm text-slate-400 font-medium">Predictive Maintenance Portal</p>

              {/* Live indicator */}
              <div className="mt-4 flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">System Online</span>
              </div>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-5" noValidate>

              {/* Error banner */}
              <AnimatePresence mode="wait">
                {error && (
                  <motion.div
                    key="err"
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.25 }}
                    className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3"
                  >
                    <Info className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
                    <p className="text-[12px] font-semibold text-red-300">{error}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Username */}
              <div className="space-y-2">
                <label className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">
                  Username
                </label>
                <div className="relative group">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4 text-slate-500 transition-colors duration-200 group-focus-within:text-brand-400">
                    <User size={17} />
                  </div>
                  <input
                    id="login-username"
                    type="text"
                    autoComplete="username"
                    placeholder="Enter user ID"
                    value={username}
                    onChange={(e) => { setUsername(e.target.value); setError(''); }}
                    disabled={loading}
                    className="block w-full rounded-xl border border-white/8 bg-white/5 py-4 pl-11 pr-4 text-sm font-medium text-white placeholder-slate-600 transition-all duration-200 focus:border-brand-500/50 focus:bg-white/8 focus:outline-none focus:ring-2 focus:ring-brand-500/20 disabled:opacity-50"
                  />
                </div>
              </div>

              {/* Password */}
              <div className="space-y-2">
                <label className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">
                  Access Key
                </label>
                <div className="relative group">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4 text-slate-500 transition-colors duration-200 group-focus-within:text-brand-400">
                    <Lock size={17} />
                  </div>
                  <input
                    id="login-password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => { setPassword(e.target.value); setError(''); }}
                    disabled={loading}
                    className="block w-full rounded-xl border border-white/8 bg-white/5 py-4 pl-11 pr-12 text-sm font-medium text-white placeholder-slate-600 transition-all duration-200 focus:border-brand-500/50 focus:bg-white/8 focus:outline-none focus:ring-2 focus:ring-brand-500/20 disabled:opacity-50"
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setShowPassword(v => !v)}
                    className="absolute inset-y-0 right-0 flex items-center pr-4 text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Submit */}
              <div className="pt-2">
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: loading ? 1 : 1.01 }}
                  whileTap={{ scale: loading ? 1 : 0.98 }}
                  className="relative w-full overflow-hidden rounded-xl py-4 text-sm font-black uppercase tracking-[0.25em] text-white transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed"
                  style={{
                    background: loading
                      ? 'rgb(51, 65, 85)'
                      : 'linear-gradient(135deg, #3b5cf6, #2563eb)',
                    boxShadow: loading ? 'none' : '0 8px 32px -8px rgba(59, 92, 246, 0.6)',
                  }}
                >
                  {/* Sheen effect */}
                  {!loading && (
                    <div
                      className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-500"
                      style={{
                        background: 'linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%)',
                      }}
                    />
                  )}
                  <span className="relative flex items-center justify-center gap-3">
                    {loading ? (
                      <>
                        <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Authenticating...
                      </>
                    ) : (
                      <>
                        Enter Portal
                        <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                      </>
                    )}
                  </span>
                </motion.button>
              </div>
            </form>

            {/* Footer */}
            <div className="mt-10 pt-6 border-t border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-4 text-[10px] font-bold uppercase tracking-widest text-slate-600">
                <span>Secure</span>
                <span className="text-slate-700">·</span>
                <span>v0.10.4</span>
              </div>
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-600">
                <Zap size={11} className="text-amber-500" />
                <span>AI Powered</span>
              </div>
            </div>
          </div>
        </div>

        {/* Below-card note */}
        <p className="mt-6 text-center text-[11px] text-slate-600 font-medium">
          Authorized personnel only · TE Connectivity © 2026
        </p>
      </motion.div>
    </div>
  );
};

export default Login;
