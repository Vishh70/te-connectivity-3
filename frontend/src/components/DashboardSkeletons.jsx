import React from "react";

export const Shimmer = () => (
  <div className="absolute inset-0 -translate-x-full animate-[shimmer_3s_infinite] bg-gradient-to-r from-transparent via-white/40 to-transparent skew-x-12" />
);

export const SkeletonBox = ({ className = "" }) => (
  <div className={`relative overflow-hidden rounded-2xl bg-slate-200/40 backdrop-blur-md border border-white/20 shadow-inner ${className}`}>
    <Shimmer />
  </div>
);

export const CardSkeleton = () => (
  <div className="glass-card p-6 space-y-5 bg-white/40 border-white/60">
    <div className="flex items-center gap-4">
      <SkeletonBox className="h-12 w-12 rounded-2xl" />
      <div className="space-y-2 flex-1">
        <SkeletonBox className="h-4 w-1/3" />
        <SkeletonBox className="h-3 w-1/2 opacity-60" />
      </div>
    </div>
    <SkeletonBox className="h-36 w-full opacity-80" />
  </div>
);

export const HealthMonitorSkeleton = () => (
  <div className="glass-card p-8 space-y-8 bg-white/40 border-white/60">
    <div className="flex justify-between items-start">
      <div className="space-y-3">
        <SkeletonBox className="h-6 w-56" />
        <SkeletonBox className="h-4 w-80 opacity-60" />
      </div>
      <SkeletonBox className="h-14 w-28 rounded-2xl" />
    </div>
    <div className="flex gap-3">
      {[1, 2, 3, 4].map((i) => (
        <SkeletonBox key={i} className="h-8 w-28 rounded-xl opacity-70" />
      ))}
    </div>
    <SkeletonBox className="h-[24rem] w-full rounded-[2rem] opacity-90 shadow-2xl" />
  </div>
);

export const TelemetryGridSkeleton = () => (
  <div className="glass-card p-8 space-y-6 bg-white/40 border-white/60">
    <div className="flex justify-between items-center mb-8">
      <div className="space-y-2">
        <SkeletonBox className="h-5 w-64" />
        <SkeletonBox className="h-3 w-40 opacity-50" />
      </div>
      <div className="flex gap-3">
        {[1, 2, 3].map((i) => (
          <SkeletonBox key={i} className="h-10 w-32 rounded-xl" />
        ))}
      </div>
    </div>
    <div className="space-y-3">
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <SkeletonBox key={i} className={`h-16 w-full ${i % 2 === 0 ? 'opacity-80' : 'opacity-40'}`} />
      ))}
    </div>
  </div>
);
