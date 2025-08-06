import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ResearchWorkflowApp } from './components/ResearchWorkflowApp';
import { AppProviders } from './components/AppProviders';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppProviders>
        <Router>
          <div className="min-h-screen bg-gray-900 text-gray-100">
            <Routes>
              <Route path="/" element={<ResearchWorkflowApp />} />
              <Route path="/research/:workflowId" element={<ResearchWorkflowApp />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </Router>
      </AppProviders>
    </QueryClientProvider>
  );
}

export default App;