import React, { useMemo, useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ComposedChart, Area, Line
} from 'recharts';
import { 
  Cpu, Zap, Target, Timer, Info, Trophy, 
  ChevronRight, AlertCircle, CheckCircle2, HelpCircle, Eye
} from 'lucide-react';
import data from '../data/data.json';
import { motion, AnimatePresence } from 'framer-motion';

const METRIC_GLOSSARY = [
  { 
    title: 'Độ chính xác (Accuracy)', 
    desc: 'Tỷ lệ dự đoán đúng trên tổng số review. Chỉ số càng cao nghĩa là mô hình càng ít đoán sai.',
    icon: Target,
    color: 'text-green-500'
  },
  { 
    title: 'Macro F1-Score', 
    desc: 'Đánh giá khả năng nhận diện đều giữa các nhóm (Tích cực, Tiêu cực, Trung lập). Quan trọng khi dữ liệu bị lệch.',
    icon: Info,
    color: 'text-blue-500'
  },
  { 
    title: 'Tốc độ (Inference Time)', 
    desc: 'Thời gian mô hình xử lý một câu review (mili giây). Càng thấp thì hệ thống phản hồi càng nhanh.',
    icon: Zap,
    color: 'text-amber-500'
  },
];

const COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'];

// Custom Tooltip Styles for Dark Mode
const tooltipStyle = {
  backgroundColor: '#0f172a',
  borderRadius: '16px',
  border: '1px solid #1e293b',
  color: '#f8fafc',
  boxShadow: '0 20px 25px -5px rgba(0,0,0,0.5)'
};

export default function ModelBenchmarks() {
  const benchmarkData = data.benchmarks || [];
  const [hoveredModel, setHoveredModel] = useState(null);

  const winners = useMemo(() => {
    if (!benchmarkData.length) return {};
    
    const sortedByAcc = [...benchmarkData].sort((a, b) => (b.Accuracy || 0) - (a.Accuracy || 0));
    const sortedBySpeed = [...benchmarkData].sort((a, b) => (a['Inference Time (ms/seq)'] || 0) - (b['Inference Time (ms/seq)'] || 0));
    const sortedByF1 = [...benchmarkData].sort((a, b) => (b['Macro F1'] || 0) - (a['Macro F1'] || 0));

    return {
      bestAccuracy: sortedByAcc[0],
      fastest: sortedBySpeed[0],
      bestBalance: sortedByF1[0],
    };
  }, [benchmarkData]);

  const radarData = useMemo(() => {
    if (!benchmarkData.length) return [];
    
    const metrics = [
      { name: 'Accuracy', key: 'Accuracy', norm: 1 },
      { name: 'F1-Score', key: 'Macro F1', norm: 1 },
      { name: 'Speed', key: 'Inference Time (ms/seq)', norm: 20, invert: true },
      { name: 'Params', key: 'Params', norm: 500e6, invert: true },
    ];

    return metrics.map(m => {
      const row = { subject: m.name };
      benchmarkData.forEach(model => {
        let val = model[m.key] || 0;
        if (m.invert) {
          val = Math.max(0, 1 - (val / m.norm));
        } else {
          val = val / m.norm;
        }
        row[model.Model] = val;
      });
      return row;
    });
  }, [benchmarkData]);

  // Small multiples data
  const individualRadarData = useMemo(() => {
    return benchmarkData.map(model => {
      return [
        { subject: 'Accuracy', value: model.Accuracy },
        { subject: 'F1-Score', value: model['Macro F1'] },
        { subject: 'Speed', value: Math.max(0, 1 - (model['Inference Time (ms/seq)'] / 20)) },
        { subject: 'Params', value: Math.max(0, 1 - (model.Params / 500e6)) },
      ];
    });
  }, [benchmarkData]);

  const chartData = useMemo(() => {
    return benchmarkData.map(d => ({
      ...d,
      ParamsMillions: (d.Params || 0) / 1e6
    }));
  }, [benchmarkData]);

  if (!benchmarkData.length) {
    return <div className="p-10 text-center">No data available</div>;
  }

  return (
    <div className="space-y-12 pb-24 animate-in fade-in duration-500">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-4xl font-black text-slate-900 dark:text-white tracking-tight">Trí Tuệ Nhân Tạo</h2>
          <p className="text-slate-500 mt-2 text-lg">Phân tích chuyên sâu và so sánh hiệu năng các mô hình ngôn ngữ</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-full text-sm font-bold text-slate-600 dark:text-slate-400">
          <Timer size={16} />
          Cập nhật: {benchmarkData[benchmarkData.length - 1]?.Date?.split(' ')[0] || 'N/A'}
        </div>
      </header>

      {/* 1. Winners / Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { label: 'Chính xác nhất', model: winners.bestAccuracy?.Model || 'N/A', val: winners.bestAccuracy ? `${(winners.bestAccuracy.Accuracy * 100).toFixed(1)}%` : '0%', icon: Trophy, color: 'text-yellow-500', bg: 'bg-yellow-50 dark:bg-yellow-900/20' },
          { label: 'Tốc độ nhanh nhất', model: winners.fastest?.Model || 'N/A', val: winners.fastest ? `${winners.fastest['Inference Time (ms/seq)']}ms` : '0ms', icon: Zap, color: 'text-blue-500', bg: 'bg-blue-50 dark:bg-blue-900/20' },
          { label: 'Cân bằng nhất (F1)', model: winners.bestBalance?.Model || 'N/A', val: winners.bestBalance?.['Macro F1'] || '0.0', icon: Target, color: 'text-green-500', bg: 'bg-green-50 dark:bg-green-900/20' },
        ].map((w, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            key={i} 
            className="relative overflow-hidden bg-white dark:bg-slate-900 p-6 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none group"
          >
            <div className={`absolute top-0 right-0 p-8 ${w.color} opacity-10 group-hover:scale-110 transition-transform`}>
              <w.icon size={80} />
            </div>
            <div className={`w-12 h-12 ${w.bg} rounded-2xl flex items-center justify-center mb-4`}>
              <w.icon className={w.color} size={24} />
            </div>
            <p className="text-sm font-bold text-slate-500 uppercase tracking-widest">{w.label}</p>
            <h3 className="text-2xl font-black text-slate-900 dark:text-white mt-1">{w.model}</h3>
            <div className="mt-4 flex items-center gap-2">
              <span className="px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded-lg text-sm font-mono font-bold text-slate-700 dark:text-slate-300">
                {w.val}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* 2. Glossary & Basic Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-slate-950 p-6 rounded-3xl text-white shadow-2xl h-full border border-slate-800">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
              <Info className="text-blue-400" />
              Giải thích thông số
            </h3>
            <div className="space-y-6">
              {METRIC_GLOSSARY.map((g, i) => (
                <div key={i} className="space-y-2 bg-slate-900 p-4 rounded-2xl border border-slate-800">
                  <div className={`flex items-center gap-2 font-bold ${g.color}`}>
                    <g.icon size={18} />
                    {g.title}
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    {g.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-8">
          <div className="bg-slate-950 p-8 rounded-3xl border border-slate-800 shadow-2xl">
            <h3 className="text-xl font-bold text-white mb-2">Tương quan Độ chính xác & F1</h3>
            <p className="text-sm text-slate-400 mb-8 italic">
              So sánh khả năng dự đoán đúng (Accuracy) và khả năng giữ cân bằng giữa các nhãn (F1-Score). Nếu hai cột cao bằng nhau, mô hình đó rất ổn định.
            </p>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={benchmarkData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                  <XAxis dataKey="Model" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} dy={10} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0.7, 1]} />
                  <Tooltip 
                    cursor={{ fill: '#1e293b' }}
                    contentStyle={tooltipStyle}
                  />
                  <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', color: '#cbd5e1' }} />
                  <Bar dataKey="Accuracy" name="Accuracy" fill="#ef4444" radius={[6, 6, 0, 0]} barSize={40} />
                  <Bar dataKey="Macro F1" name="F1-Score" fill="#fca5a5" radius={[6, 6, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-slate-950 p-8 rounded-3xl border border-slate-800 shadow-2xl">
            <h3 className="text-xl font-bold text-white mb-2">Hiệu suất và Tài nguyên</h3>
            <p className="text-sm text-slate-400 mb-8 italic">
              Thể hiện sự đánh đổi giữa tốc độ xử lý (Speed) và độ cồng kềnh của mô hình (Params). Mô hình lý tưởng là mô hình có cả đường xanh và đường xám đều thấp.
            </p>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                  <XAxis dataKey="Model" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis yAxisId="left" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis yAxisId="right" orientation="right" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', color: '#cbd5e1' }} />
                  <Area yAxisId="left" type="monotone" dataKey="Inference Time (ms/seq)" name="Tốc độ (ms)" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                  <Line yAxisId="right" type="monotone" dataKey="ParamsMillions" name="Tham số (M)" stroke="#94a3b8" strokeDasharray="5 5" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Interactive Radar Comparison Section */}
      <section className="bg-slate-950 p-10 rounded-[40px] border border-slate-800 shadow-2xl">
        <div className="max-w-4xl mx-auto text-center mb-12">
          <h3 className="text-3xl font-black text-white mb-4">So sánh đa chiều (Interactive Radar)</h3>
          <p className="text-slate-400 text-lg leading-relaxed">
            Biểu đồ này giúp bạn nhìn thấy "vóc dáng" tổng thể của mô hình. Một mô hình lý tưởng sẽ có hình dạng to và cân đối, phủ rộng tất cả các góc (Accuracy, Speed, F1, Params). Di chuột qua tên bên dưới để xem chi tiết.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Models List for selection */}
          <div className="lg:col-span-4 space-y-3">
            <p className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4">Chọn mô hình để xem chi tiết</p>
            {benchmarkData.map((m, i) => (
              <button
                key={m.Model}
                onMouseEnter={() => setHoveredModel(m.Model)}
                onMouseLeave={() => setHoveredModel(null)}
                className={`w-full flex items-center justify-between p-4 rounded-2xl transition-all border ${
                  hoveredModel === m.Model 
                    ? 'bg-slate-800 text-white border-slate-700 shadow-2xl -translate-x-2' 
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % 5] }} />
                  <span className="font-bold">{m.Model}</span>
                </div>
                <Eye size={18} className={hoveredModel === m.Model ? 'opacity-100 text-blue-400' : 'opacity-0'} />
              </button>
            ))}
          </div>

          {/* BIG Radar Chart */}
          <div className="lg:col-span-8 h-[500px] relative">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="#334155" strokeWidth={1} />
                <PolarAngleAxis dataKey="subject" fontSize={14} tick={{ fill: '#cbd5e1', fontWeight: 800 }} />
                <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                {benchmarkData.map((m, i) => (
                  <Radar
                    key={m.Model}
                    name={m.Model}
                    dataKey={m.Model}
                    stroke={COLORS[i % 5]}
                    fill={COLORS[i % 5]}
                    fillOpacity={hoveredModel === null ? 0.2 : (hoveredModel === m.Model ? 0.6 : 0.05)}
                    strokeWidth={hoveredModel === m.Model ? 4 : 2}
                    animationDuration={500}
                  />
                ))}
                <Tooltip 
                  contentStyle={{ ...tooltipStyle, padding: '20px', borderRadius: '24px' }}
                />
              </RadarChart>
            </ResponsiveContainer>
            
            <AnimatePresence>
              {hoveredModel && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="absolute bottom-4 right-4 bg-slate-900 text-white p-6 rounded-3xl shadow-2xl border border-slate-700 max-w-xs"
                >
                  <h4 className="font-black text-xl mb-2">{hoveredModel}</h4>
                  <p className="text-sm text-slate-400">
                    Đang hiển thị biểu đồ phân tích chuyên sâu cho {hoveredModel}. Mô hình này có thế mạnh vượt trội về 
                    {benchmarkData.find(m => m.Model === hoveredModel)?.Accuracy > 0.85 ? ' độ chính xác' : ' tốc độ xử lý'}.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </section>

      {/* 4. Small Multiples Section */}
      <section className="space-y-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="h-8 w-2 bg-blue-600 rounded-full" />
            <h3 className="text-2xl font-black text-slate-900 dark:text-white">Góc nhìn riêng biệt (Small Multiples)</h3>
          </div>
          <p className="text-slate-500 italic pl-11">
            Tách biệt từng mô hình để dễ dàng quan sát thế mạnh riêng mà không bị rối mắt bởi các đường nét chồng chéo.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {benchmarkData.map((m, i) => (
            <div key={m.Model} className="bg-slate-950 p-6 rounded-[32px] border border-slate-800 shadow-xl hover:shadow-2xl transition-all group">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-4 rounded-full" style={{ backgroundColor: COLORS[i % 5] }} />
                <h4 className="font-bold text-white">{m.Model}</h4>
              </div>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={individualRadarData[i]}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="subject" fontSize={8} tick={{fill: '#94a3b8'}} />
                    <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                    <Radar
                      dataKey="value"
                      stroke={COLORS[i % 5]}
                      fill={COLORS[i % 5]}
                      fillOpacity={0.4}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 gap-2 text-center">
                <div>
                  <p className="text-[10px] text-slate-500 uppercase font-black">Accuracy</p>
                  <p className="font-bold text-white">{m.Accuracy}</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 uppercase font-black">Speed</p>
                  <p className="font-bold text-white">{m['Inference Time (ms/seq)']}ms</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. Comparison Table */}
      <div className="bg-white dark:bg-slate-900 rounded-[32px] border border-slate-200 dark:border-slate-800 shadow-xl overflow-hidden">
        <div className="p-8 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/30">
          <div className="flex items-center gap-2 mb-2">
            <HelpCircle className="text-blue-600" size={20} />
            <h3 className="text-xl font-bold">Bảng so sánh chi tiết (Source of Truth)</h3>
          </div>
          <p className="text-sm text-slate-500 max-w-3xl">
            Bảng này cung cấp con số chính xác tuyệt đối từ các đợt thử nghiệm. Nó giúp bạn đưa ra quyết định cuối cùng dựa trên các tiêu chí cụ thể: Dùng <b>PhoBERT</b> nếu cần độ chính xác cao nhất cho báo cáo, hoặc dùng <b>mBERT</b> nếu cần tốc độ phản hồi cực nhanh.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800/50">
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest">Mô hình</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest text-center">Accuracy</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest text-center">Tốc độ</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest text-center">Đánh giá</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {benchmarkData.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/30 transition-all">
                  <td className="px-8 py-6">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-8 bg-blue-600 rounded-full" />
                      <div>
                        <div className="font-black text-lg text-slate-900 dark:text-white">{row.Model}</div>
                        <div className="text-xs text-slate-400">Huấn luyện: {row.Date?.split(' ')[0] || 'N/A'}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-8 py-6 text-center font-mono font-bold text-green-600 text-lg">
                    {row.Accuracy}
                  </td>
                  <td className="px-8 py-6 text-center font-mono font-bold text-slate-700 dark:text-slate-300">
                    {row['Inference Time (ms/seq)']} ms
                  </td>
                  <td className="px-8 py-6">
                    <div className="flex justify-center">
                      {row.Accuracy > 0.85 ? (
                        <div className="flex items-center gap-1 text-green-600 text-xs font-bold">
                          <CheckCircle2 size={16} /> Khuyên dùng
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-slate-400 text-xs font-bold">
                          <AlertCircle size={16} /> Cần cân nhắc
                        </div>
                      )}
                    </div>
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
