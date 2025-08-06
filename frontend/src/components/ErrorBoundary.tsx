import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      errorId: Date.now().toString(36) + Math.random().toString(36).substr(2),
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      errorInfo,
    });

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // In production, you might want to send this to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: Send to error reporting service
      // errorReportingService.captureException(error, { extra: errorInfo });
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full">
            <div className="bg-white rounded-2xl shadow-xl border border-neutral-200 overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-error-50 to-error-100 p-6 border-b border-error-200">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-error-500 rounded-xl flex items-center justify-center">
                    <AlertTriangle className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-error-900">Something went wrong</h1>
                    <p className="text-error-700 mt-1">
                      An unexpected error occurred in the application
                    </p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-6">
                <div className="space-y-6">
                  {/* Error Summary */}
                  <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-200">
                    <h3 className="font-semibold text-neutral-900 mb-2">Error Details</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-neutral-600">Error ID:</span>
                        <code className="bg-neutral-200 px-2 py-1 rounded text-neutral-800">
                          {this.state.errorId}
                        </code>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-600">Time:</span>
                        <span className="text-neutral-800">
                          {new Date().toLocaleString()}
                        </span>
                      </div>
                      {this.state.error && (
                        <div>
                          <span className="text-neutral-600">Message:</span>
                          <div className="mt-1 p-3 bg-error-50 border border-error-200 rounded-lg">
                            <code className="text-error-800 text-sm break-all">
                              {this.state.error.message}
                            </code>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      onClick={this.handleRetry}
                      className="flex-1 btn-primary space-x-2 justify-center"
                    >
                      <RefreshCw className="w-4 h-4" />
                      <span>Try Again</span>
                    </button>
                    <button
                      onClick={this.handleReload}
                      className="flex-1 btn-secondary space-x-2 justify-center"
                    >
                      <RefreshCw className="w-4 h-4" />
                      <span>Reload Page</span>
                    </button>
                    <button
                      onClick={this.handleGoHome}
                      className="flex-1 btn-secondary space-x-2 justify-center"
                    >
                      <Home className="w-4 h-4" />
                      <span>Go Home</span>
                    </button>
                  </div>

                  {/* Development Info */}
                  {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                    <details className="bg-neutral-900 text-neutral-100 p-4 rounded-xl">
                      <summary className="cursor-pointer font-semibold mb-2 flex items-center space-x-2">
                        <Bug className="w-4 h-4" />
                        <span>Development Details</span>
                      </summary>
                      <div className="mt-3 space-y-3">
                        <div>
                          <h4 className="font-semibold text-neutral-300 mb-1">Stack Trace:</h4>
                          <pre className="text-xs overflow-x-auto whitespace-pre-wrap bg-neutral-800 p-3 rounded border">
                            {this.state.error?.stack}
                          </pre>
                        </div>
                        <div>
                          <h4 className="font-semibold text-neutral-300 mb-1">Component Stack:</h4>
                          <pre className="text-xs overflow-x-auto whitespace-pre-wrap bg-neutral-800 p-3 rounded border">
                            {this.state.errorInfo.componentStack}
                          </pre>
                        </div>
                      </div>
                    </details>
                  )}

                  {/* Help Text */}
                  <div className="bg-primary-50 p-4 rounded-xl border border-primary-200">
                    <h3 className="font-semibold text-primary-900 mb-2">What can you do?</h3>
                    <ul className="text-sm text-primary-800 space-y-1">
                      <li>• Try refreshing the page or clicking "Try Again"</li>
                      <li>• Check your internet connection</li>
                      <li>• Clear your browser cache and cookies</li>
                      <li>• If the problem persists, please contact support with the Error ID</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for easier usage
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};

// Smaller error boundary for specific components
export const InlineErrorBoundary: React.FC<Props> = ({ children, fallback, onError }) => {
  return (
    <ErrorBoundary
      onError={onError}
      fallback={
        fallback || (
          <div className="p-4 bg-error-50 border border-error-200 rounded-lg">
            <div className="flex items-center space-x-2 text-error-800">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm font-medium">Component Error</span>
            </div>
            <p className="text-sm text-error-700 mt-1">
              This component encountered an error and couldn't render properly.
            </p>
          </div>
        )
      }
    >
      {children}
    </ErrorBoundary>
  );
};