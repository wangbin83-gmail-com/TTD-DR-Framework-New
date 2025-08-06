import React, { createContext, useContext, useState, useCallback } from 'react';
import { Loader2, RefreshCw, Clock, CheckCircle, XCircle } from 'lucide-react';

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface LoadingItem {
  id: string;
  label: string;
  state: LoadingState;
  progress?: number;
  message?: string;
  startTime?: number;
}

interface LoadingContextType {
  loadingItems: LoadingItem[];
  startLoading: (id: string, label: string, message?: string) => void;
  updateLoading: (id: string, updates: Partial<LoadingItem>) => void;
  finishLoading: (id: string, state: 'success' | 'error', message?: string) => void;
  removeLoading: (id: string) => void;
  isLoading: (id?: string) => boolean;
  getLoadingItem: (id: string) => LoadingItem | undefined;
}

const LoadingContext = createContext<LoadingContextType | undefined>(undefined);

export const useLoading = () => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
};

interface LoadingProviderProps {
  children: React.ReactNode;
}

export const LoadingProvider: React.FC<LoadingProviderProps> = ({ children }) => {
  const [loadingItems, setLoadingItems] = useState<LoadingItem[]>([]);

  const startLoading = useCallback((id: string, label: string, message?: string) => {
    setLoadingItems(prev => {
      const existing = prev.find(item => item.id === id);
      if (existing) {
        return prev.map(item =>
          item.id === id
            ? { ...item, state: 'loading', label, message, startTime: Date.now() }
            : item
        );
      }
      return [...prev, {
        id,
        label,
        state: 'loading',
        message,
        startTime: Date.now(),
      }];
    });
  }, []);

  const updateLoading = useCallback((id: string, updates: Partial<LoadingItem>) => {
    setLoadingItems(prev =>
      prev.map(item =>
        item.id === id ? { ...item, ...updates } : item
      )
    );
  }, []);

  const finishLoading = useCallback((id: string, state: 'success' | 'error', message?: string) => {
    setLoadingItems(prev =>
      prev.map(item =>
        item.id === id
          ? { ...item, state, message, progress: state === 'success' ? 100 : undefined }
          : item
      )
    );

    // Auto-remove after delay
    setTimeout(() => {
      removeLoading(id);
    }, 3000);
  }, []);

  const removeLoading = useCallback((id: string) => {
    setLoadingItems(prev => prev.filter(item => item.id !== id));
  }, []);

  const isLoading = useCallback((id?: string) => {
    if (id) {
      const item = loadingItems.find(item => item.id === id);
      return item?.state === 'loading';
    }
    return loadingItems.some(item => item.state === 'loading');
  }, [loadingItems]);

  const getLoadingItem = useCallback((id: string) => {
    return loadingItems.find(item => item.id === id);
  }, [loadingItems]);

  const value: LoadingContextType = {
    loadingItems,
    startLoading,
    updateLoading,
    finishLoading,
    removeLoading,
    isLoading,
    getLoadingItem,
  };

  return (
    <LoadingContext.Provider value={value}>
      {children}
      <LoadingOverlay />
    </LoadingContext.Provider>
  );
};

const LoadingOverlay: React.FC = () => {
  const { loadingItems } = useLoading();
  const activeItems = loadingItems.filter(item => item.state === 'loading');

  if (activeItems.length === 0) return null;

  return (
    <div className="fixed bottom-4 left-4 z-40 space-y-2 max-w-sm">
      {activeItems.map((item) => (
        <LoadingItem key={item.id} item={item} />
      ))}
    </div>
  );
};

interface LoadingItemProps {
  item: LoadingItem;
}

const LoadingItem: React.FC<LoadingItemProps> = ({ item }) => {
  const getElapsedTime = () => {
    if (!item.startTime) return '';
    const elapsed = Math.floor((Date.now() - item.startTime) / 1000);
    return elapsed > 0 ? `${elapsed}s` : '';
  };

  const getIcon = () => {
    switch (item.state) {
      case 'loading':
        return <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-success-500" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-error-500" />;
      default:
        return <Clock className="w-4 h-4 text-neutral-400" />;
    }
  };

  const getColorClasses = () => {
    switch (item.state) {
      case 'loading':
        return 'bg-white border-primary-200 text-primary-800';
      case 'success':
        return 'bg-success-50 border-success-200 text-success-800';
      case 'error':
        return 'bg-error-50 border-error-200 text-error-800';
      default:
        return 'bg-neutral-50 border-neutral-200 text-neutral-800';
    }
  };

  return (
    <div
      className={`p-3 rounded-lg border-2 shadow-lg backdrop-blur-sm transition-all duration-300 ${getColorClasses()}`}
    >
      <div className="flex items-center space-x-3">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium truncate">{item.label}</p>
            {item.state === 'loading' && (
              <span className="text-xs text-neutral-500 ml-2">
                {getElapsedTime()}
              </span>
            )}
          </div>
          {item.message && (
            <p className="text-xs opacity-75 mt-1">{item.message}</p>
          )}
          {item.progress !== undefined && (
            <div className="mt-2">
              <div className="flex justify-between text-xs mb-1">
                <span>Progress</span>
                <span>{Math.round(item.progress)}%</span>
              </div>
              <div className="w-full bg-neutral-200 rounded-full h-1.5">
                <div
                  className="bg-primary-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${item.progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Skeleton loading components
export const SkeletonLoader: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`animate-pulse bg-neutral-200 rounded ${className}`} />
);

export const SkeletonText: React.FC<{ lines?: number; className?: string }> = ({ 
  lines = 3, 
  className = '' 
}) => (
  <div className={`space-y-2 ${className}`}>
    {Array.from({ length: lines }).map((_, i) => (
      <SkeletonLoader
        key={i}
        className={`h-4 ${i === lines - 1 ? 'w-3/4' : 'w-full'}`}
      />
    ))}
  </div>
);

export const SkeletonCard: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`p-4 border border-neutral-200 rounded-lg ${className}`}>
    <div className="flex items-center space-x-3 mb-3">
      <SkeletonLoader className="w-10 h-10 rounded-full" />
      <div className="flex-1">
        <SkeletonLoader className="h-4 w-1/3 mb-2" />
        <SkeletonLoader className="h-3 w-1/2" />
      </div>
    </div>
    <SkeletonText lines={2} />
  </div>
);

// Button loading states
interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  loading?: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  loading = false,
  loadingText = 'Loading...',
  children,
  disabled,
  className = '',
  ...props
}) => (
  <button
    {...props}
    disabled={disabled || loading}
    className={`relative ${className} ${loading ? 'cursor-not-allowed' : ''}`}
  >
    {loading && (
      <div className="absolute inset-0 flex items-center justify-center">
        <Loader2 className="w-4 h-4 animate-spin mr-2" />
        <span>{loadingText}</span>
      </div>
    )}
    <div className={loading ? 'opacity-0' : 'opacity-100'}>
      {children}
    </div>
  </button>
);