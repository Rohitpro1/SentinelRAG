import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LandingPage } from './pages/LandingPage';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { MetricsPage } from './pages/MetricsPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { NotFoundPage } from './pages/NotFoundPage';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('playground');

  const renderContent = () => {
    switch (currentTab) {
      case 'landing':
        return <LandingPage onNavigate={setCurrentTab} />;
      case 'playground':
        return <PlaygroundPage />;
      case 'metrics':
        return <MetricsPage />;
      case 'documents':
        return <DocumentsPage />;
      case 'architecture':
        return <ArchitecturePage />;
      default:
        return <NotFoundPage onHome={() => setCurrentTab('playground')} />;
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen flex flex-col bg-[#f8f9ff]">
        <Navbar currentTab={currentTab} setCurrentTab={setCurrentTab} />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-12">
          {renderContent()}
        </main>

        <Footer />
      </div>
    </ErrorBoundary>
  );
};

export default App;
