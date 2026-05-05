import React, { useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import data from '../data/data.json';

const COLORS = ['#ef4444', '#f97316', '#f59e0b', '#10b981', '#3b82f6', '#6366f1', '#8b5cf6', '#d946ef'];

export default function MallComparison() {
  const mallData = useMemo(() => {
    const malls = data.reviews.reduce((acc, curr) => {
      if (!acc[curr.mall_name]) {
        acc[curr.mall_name] = { 
          name: curr.mall_name, 
          positive: 0, 
          neutral: 0, 
          negative: 0, 
          total: 0,
          avgRating: 0,
          sumRating: 0
        };
      }
      acc[curr.mall_name].total += 1;
      acc[curr.mall_name][curr.sentiment] += 1;
      acc[curr.mall_name].sumRating += curr.rating;
      acc[curr.mall_name].avgRating = (acc[curr.mall_name].sumRating / acc[curr.mall_name].total).toFixed(1);
      return acc;
    }, {});

    return Object.values(malls).sort((a, b) => b.total - a.total);
  }, []);

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <header>
        <h2 className="text-3xl font-bold text-slate-900 dark:text-white">So sánh Trung tâm thương mại</h2>
        <p className="text-slate-500 mt-1">Phân tích chi tiết cảm xúc trên từng địa điểm cụ thể</p>
      </header>

      {/* Chart */}
      <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
        <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-8">Cơ cấu cảm xúc (Số lượng review)</h3>
        <div className="h-[450px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mallData} margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                fontSize={10} 
                tick={{ fill: '#64748b' }} 
                tickFormatter={(val) => val.split(',')[0]}
              />
              <YAxis fontSize={12} tick={{ fill: '#64748b' }} />
              <Tooltip 
                cursor={{ fill: 'rgba(241, 245, 249, 0.5)' }}
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
              />
              <Legend verticalAlign="top" height={36}/>
              <Bar dataKey="positive" name="Tích cực" stackId="a" fill="#22c55e" radius={[0, 0, 0, 0]} />
              <Bar dataKey="neutral" name="Trung lập" stackId="a" fill="#94a3b8" />
              <Bar dataKey="negative" name="Tiêu cực" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid of Malls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {mallData.map((mall, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
            <h4 className="font-bold text-slate-900 dark:text-white mb-4 line-clamp-1">{mall.name}</h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Rating</p>
                <p className="text-xl font-bold text-amber-500">{mall.avgRating}</p>
              </div>
              <div className="text-center">
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Hài lòng</p>
                <p className="text-xl font-bold text-green-600">{Math.round((mall.positive / mall.total) * 100)}%</p>
              </div>
              <div className="text-center">
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Tổng số</p>
                <p className="text-xl font-bold text-slate-700 dark:text-slate-300">{mall.total}</p>
              </div>
            </div>
            <div className="mt-4 h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden flex">
              <div style={{ width: `${(mall.positive / mall.total) * 100}%` }} className="h-full bg-green-500" />
              <div style={{ width: `${(mall.neutral / mall.total) * 100}%` }} className="h-full bg-slate-400" />
              <div style={{ width: `${(mall.negative / mall.total) * 100}%` }} className="h-full bg-red-500" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
