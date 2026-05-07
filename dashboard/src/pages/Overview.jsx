import React, { useMemo } from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
  AreaChart, Area, LineChart, Line
} from 'recharts';
import { Users, Star, MessageSquare, TrendingUp } from 'lucide-react';
import data from '../data/data.json';

const SENTIMENT_COLORS = {
  positive: '#22c55e',
  neutral: '#94a3b8',
  negative: '#ef4444',
};


export default function Overview() {
  const sentimentData = useMemo(() => {
    const counts = data.reviews.reduce((acc, curr) => {
      acc[curr.sentiment] = (acc[curr.sentiment] || 0) + 1;
      return acc;
    }, {});
    
    return Object.keys(counts).map(key => ({
      name: key === 'positive' ? 'Tích cực' : key === 'negative' ? 'Tiêu cực' : 'Trung lập',
      value: counts[key],
      key: key
    }));
  }, []);

  const topMalls = useMemo(() => {
    const malls = data.reviews.reduce((acc, curr) => {
      if (!acc[curr.mall_name]) acc[curr.mall_name] = { name: curr.mall_name, positive: 0, total: 0 };
      acc[curr.mall_name].total += 1;
      if (curr.sentiment === 'positive') acc[curr.mall_name].positive += 1;
      return acc;
    }, {});

    return Object.values(malls)
      .map(m => ({ ...m, rate: Math.round((m.positive / m.total) * 100) }))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 5);
  }, []);

  const statCards = useMemo(() => {
    const totalReviews = data.reviews.length;
    const avgRating = totalReviews ? (data.reviews.reduce((acc, curr) => acc + curr.rating, 0) / totalReviews).toFixed(1) : 0;
    const uniqueReviewers = new Set(data.reviews.map(r => r.reviewer_name)).size;
    const positiveCount = data.reviews.filter(r => r.sentiment === 'positive').length;
    const positiveRate = totalReviews ? Math.round((positiveCount / totalReviews) * 100) : 0;

    return [
      { label: 'Tổng Review', value: totalReviews, icon: MessageSquare, color: 'text-blue-600', bg: 'bg-blue-50' },
      { label: 'Rating Trung Bình', value: avgRating, icon: Star, color: 'text-amber-500', bg: 'bg-amber-50' },
      { label: 'Người Đánh Giá', value: uniqueReviewers, icon: Users, color: 'text-purple-600', bg: 'bg-purple-50' },
      { label: 'Tỷ Lệ Tích Cực', value: `${positiveRate}%`, icon: TrendingUp, color: 'text-green-600', bg: 'bg-green-50' },
    ];
  }, []);

  const timeSeriesData = useMemo(() => {
    const monthlyData = data.reviews.reduce((acc, curr) => {
      if (!curr.review_date) return acc;
      // Extract YYYY-MM
      const monthStr = curr.review_date.substring(0, 7); 
      
      if (!acc[monthStr]) {
        acc[monthStr] = { date: monthStr, positive: 0, negative: 0, neutral: 0 };
      }
      acc[monthStr][curr.sentiment] += 1;
      return acc;
    }, {});

    // Convert to array and sort chronologically
    return Object.values(monthlyData).sort((a, b) => a.date.localeCompare(b.date));
  }, []);

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <header>
        <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Tổng quan hệ thống</h2>
        <p className="text-slate-500 mt-1">Dữ liệu tổng hợp từ các trung tâm thương mại Vincom</p>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className={`${stat.bg} p-3 rounded-xl`}>
                <stat.icon className={stat.color} size={24} />
              </div>
            </div>
            <p className="text-sm font-medium text-slate-500">{stat.label}</p>
            <h3 className="text-2xl font-bold text-slate-900 dark:text-white mt-1">{stat.value}</h3>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Distribution */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-6">Phân bổ cảm xúc</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={SENTIMENT_COLORS[entry.key]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                />
                <Legend verticalAlign="bottom" height={36}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Malls Chart */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-6">Top 5 Mall hài lòng nhất (%)</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topMalls} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  width={150} 
                  fontSize={12} 
                  tick={{ fill: '#64748b' }}
                  tickFormatter={(val) => val.split(',')[0]} 
                />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="rate" fill="#ef4444" radius={[0, 4, 4, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Time Series Chart */}
      <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
        <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-6">Xu hướng cảm xúc theo thời gian</h3>
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeriesData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: '#64748b' }} tickFormatter={(val) => {
                const [year, month] = val.split('-');
                return `Th${month}/${year.slice(2)}`;
              }} />
              <YAxis tick={{ fill: '#64748b' }} />
              <Tooltip 
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                labelFormatter={(val) => `Tháng ${val.split('-')[1]}/${val.split('-')[0]}`}
              />
              <Legend verticalAlign="top" height={36}/>
              <Line type="monotone" dataKey="positive" name="Tích cực" stroke="#22c55e" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="negative" name="Tiêu cực" stroke="#ef4444" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="neutral" name="Trung lập" stroke="#94a3b8" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
