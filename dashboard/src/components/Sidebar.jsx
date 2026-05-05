import React from 'react';
import { LayoutDashboard, BarChart3, BrainCircuit, Search, MapPin } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { id: 'overview', label: 'Tổng quan', icon: LayoutDashboard },
  { id: 'malls', label: 'So sánh Mall', icon: MapPin },
  { id: 'benchmarks', label: 'Model AI', icon: BrainCircuit },
  { id: 'reviews', label: 'Khám phá', icon: Search },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 flex flex-col shadow-xl">
      <div className="p-6">
        <h1 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <BarChart3 className="text-red-600" />
          VinCom Analytics
        </h1>
        <p className="text-xs text-slate-500 mt-1 uppercase tracking-wider font-semibold">Sentiment Dashboard</p>
      </div>

      <nav className="flex-1 px-4 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
              activeTab === item.id
                ? "bg-red-50 text-red-600 dark:bg-red-500/20 dark:text-red-400"
                : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
            )}
          >
            <item.icon size={20} />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-200 dark:border-slate-700">
        <div className="bg-slate-50 dark:bg-slate-900/50 p-4 rounded-xl">
          <p className="text-xs text-slate-500 dark:text-slate-400">Dataset Version</p>
          <p className="text-sm font-semibold text-slate-900 dark:text-white">v1.2.0 (510 reviews)</p>
        </div>
      </div>
    </aside>
  );
}
