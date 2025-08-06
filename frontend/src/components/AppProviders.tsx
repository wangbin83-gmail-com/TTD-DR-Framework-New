import React from 'react';
import { NotificationProvider } from './NotificationSystem';
import { LoadingProvider } from './LoadingSystem';
import { ErrorBoundary } from './ErrorBoundary';

interface AppProvidersProps {
  children: React.ReactNode;
}

export const AppProviders: React.FC<AppProvidersProps> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        // Log error to console in development
        if (process.env.NODE_ENV === 'development') {
          console.error('Application Error:', error, errorInfo);
        }
        
        // In production, you might want to send this to an error reporting service
        // errorReportingService.captureException(error, { extra: errorInfo });
      }}
    >
      <NotificationProvider maxNotifications={5}>
        <LoadingProvider>
          {children}
        </LoadingProvider>
      </NotificationProvider>
    </ErrorBoundary>
  );
};