import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Overview from './pages/Overview';
import MallComparison from './pages/MallComparison';
import ModelBenchmarks from './pages/ModelBenchmarks';
import ReviewExplorer from './pages/ReviewExplorer';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <Overview />;
      case 'malls':
        return <MallComparison />;
      case 'benchmarks':
        return <ModelBenchmarks />;
      case 'reviews':
        return <ReviewExplorer />;
      default:
        return <Overview />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-900 dark:text-slate-100">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
