import React, { useState, useMemo } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, MessageSquare } from 'lucide-react';
import data from '../data/data.json';

const SENTIMENT_LABELS = {
  positive: { text: 'Tích cực', color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' },
  neutral: { text: 'Trung lập', color: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400' },
  negative: { text: 'Tiêu cực', color: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' },
};

export default function ReviewExplorer() {
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const filteredReviews = useMemo(() => {
    return data.reviews.filter(review => {
      const matchesSearch = review.cleaned_text.toLowerCase().includes(searchTerm.toLowerCase()) || 
                            review.reviewer_name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesSentiment = sentimentFilter === 'all' || review.sentiment === sentimentFilter;
      return matchesSearch && matchesSentiment;
    });
  }, [searchTerm, sentimentFilter]);

  const totalPages = Math.ceil(filteredReviews.length / itemsPerPage);
  const paginatedReviews = filteredReviews.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div className="space-y-6 animate-in fade-in duration-700">
      <header>
        <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Khám phá Review</h2>
        <p className="text-slate-500 mt-1">Duyệt và lọc chi tiết phản hồi của khách hàng</p>
      </header>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4 bg-white dark:bg-slate-800 p-4 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input 
            type="text"
            placeholder="Tìm kiếm nội dung hoặc tên người dùng..."
            className="w-full pl-10 pr-4 py-2 bg-slate-50 dark:bg-slate-800 border-none rounded-xl focus:ring-2 focus:ring-red-500 transition-all"
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
          />
        </div>
        <div className="flex gap-2">
          {['all', 'positive', 'neutral', 'negative'].map((s) => (
            <button
              key={s}
              onClick={() => { setSentimentFilter(s); setCurrentPage(1); }}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                sentimentFilter === s 
                  ? 'bg-red-600 text-white shadow-lg shadow-red-500/30' 
                  : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-100'
              }`}
            >
              {s === 'all' ? 'Tất cả' : SENTIMENT_LABELS[s].text}
            </button>
          ))}
        </div>
      </div>

      {/* Review List */}
      <div className="space-y-4">
        {paginatedReviews.map((review, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl hover:border-red-200 dark:hover:border-red-500/50 transition-all group">
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center font-bold text-slate-500">
                  {review.reviewer_name[0]}
                </div>
                <div>
                  <h4 className="font-bold text-slate-900 dark:text-white">{review.reviewer_name}</h4>
                  <p className="text-xs text-slate-500 italic">{review.mall_name}</p>
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${SENTIMENT_LABELS[review.sentiment].color}`}>
                {SENTIMENT_LABELS[review.sentiment].text}
              </span>
            </div>
            <p className="text-slate-700 dark:text-slate-300 text-sm leading-relaxed mb-4">
              {review.cleaned_text}
            </p>
            <div className="flex items-center gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1"><MessageSquare size={14} /> {review.review_time}</span>
              <span className="flex items-center gap-1 text-amber-500"><Filter size={14} /> Rating: {review.rating}/5</span>
            </div>
          </div>
        ))}

        {paginatedReviews.length === 0 && (
          <div className="text-center py-20 bg-white dark:bg-slate-800 rounded-2xl border border-dashed border-slate-300 dark:border-slate-600 shadow-xl">
            <p className="text-slate-500">Không tìm thấy review nào phù hợp.</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 py-4">
          <button 
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(prev => prev - 1)}
            className="p-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 disabled:opacity-50 shadow-md hover:bg-slate-700 transition-all"
          >
            <ChevronLeft size={20} />
          </button>
          <span className="text-sm font-medium text-slate-600">Trang {currentPage} / {totalPages}</span>
          <button 
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(prev => prev + 1)}
            className="p-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 disabled:opacity-50 shadow-md hover:bg-slate-700 transition-all"
          >
            <ChevronRight size={20} />
          </button>
        </div>
      )}
    </div>
  );
}
