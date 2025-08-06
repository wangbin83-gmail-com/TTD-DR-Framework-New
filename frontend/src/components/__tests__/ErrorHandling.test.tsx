import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { NotificationProvider, useNotifications } from '../NotificationSystem';
import { LoadingProvider, useLoading } from '../LoadingSystem';
import { ErrorBoundary } from '../ErrorBoundary';

// Test component that uses notifications
const TestNotificationComponent: React.FC = () => {
  const { showSuccess, showError, showWarning, showInfo } = useNotifications();

  return (
    <div>
      <button onClick={() => showSuccess('Success', 'Operation completed')}>
        Show Success
      </button>
      <button onClick={() => showError('Error', 'Something went wrong')}>
        Show Error
      </button>
      <button onClick={() => showWarning('Warning', 'Please be careful')}>
        Show Warning
      </button>
      <button onClick={() => showInfo('Info', 'Just so you know')}>
        Show Info
      </button>
    </div>
  );
};

// Test component that uses loading
const TestLoadingComponent: React.FC = () => {
  const { startLoading, finishLoading, isLoading } = useLoading();

  const handleStart = () => {
    startLoading('test-operation', 'Testing...', 'Running test operation');
  };

  const handleFinish = () => {
    finishLoading('test-operation', 'success', 'Test completed');
  };

  return (
    <div>
      <button onClick={handleStart}>Start Loading</button>
      <button onClick={handleFinish}>Finish Loading</button>
      <div data-testid="loading-status">
        {isLoading('test-operation') ? 'Loading' : 'Not Loading'}
      </div>
    </div>
  );
};

// Test component that throws an error
const ErrorThrowingComponent: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

describe('Error Handling and User Feedback', () => {
  describe('NotificationSystem', () => {
    it('should display success notification', async () => {
      render(
        <NotificationProvider>
          <TestNotificationComponent />
        </NotificationProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));
      
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
        expect(screen.getByText('Operation completed')).toBeInTheDocument();
      });
    });

    it('should display error notification', async () => {
      render(
        <NotificationProvider>
          <TestNotificationComponent />
        </NotificationProvider>
      );

      fireEvent.click(screen.getByText('Show Error'));
      
      await waitFor(() => {
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
      });
    });

    it('should display warning notification', async () => {
      render(
        <NotificationProvider>
          <TestNotificationComponent />
        </NotificationProvider>
      );

      fireEvent.click(screen.getByText('Show Warning'));
      
      await waitFor(() => {
        expect(screen.getByText('Warning')).toBeInTheDocument();
        expect(screen.getByText('Please be careful')).toBeInTheDocument();
      });
    });

    it('should display info notification', async () => {
      render(
        <NotificationProvider>
          <TestNotificationComponent />
        </NotificationProvider>
      );

      fireEvent.click(screen.getByText('Show Info'));
      
      await waitFor(() => {
        expect(screen.getByText('Info')).toBeInTheDocument();
        expect(screen.getByText('Just so you know')).toBeInTheDocument();
      });
    });

    it('should allow dismissing notifications', async () => {
      render(
        <NotificationProvider>
          <TestNotificationComponent />
        </NotificationProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));
      
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });

      // Find and click the dismiss button
      const dismissButton = screen.getByRole('button', { name: /close/i });
      fireEvent.click(dismissButton);

      await waitFor(() => {
        expect(screen.queryByText('Success')).not.toBeInTheDocument();
      });
    });
  });

  describe('LoadingSystem', () => {
    it('should show loading state', async () => {
      render(
        <LoadingProvider>
          <TestLoadingComponent />
        </LoadingProvider>
      );

      expect(screen.getByTestId('loading-status')).toHaveTextContent('Not Loading');

      fireEvent.click(screen.getByText('Start Loading'));

      await waitFor(() => {
        expect(screen.getByTestId('loading-status')).toHaveTextContent('Loading');
        expect(screen.getByText('Testing...')).toBeInTheDocument();
      });
    });

    it('should finish loading state', async () => {
      render(
        <LoadingProvider>
          <TestLoadingComponent />
        </LoadingProvider>
      );

      fireEvent.click(screen.getByText('Start Loading'));
      
      await waitFor(() => {
        expect(screen.getByTestId('loading-status')).toHaveTextContent('Loading');
      });

      fireEvent.click(screen.getByText('Finish Loading'));

      await waitFor(() => {
        expect(screen.getByTestId('loading-status')).toHaveTextContent('Not Loading');
      });
    });
  });

  describe('ErrorBoundary', () => {
    // Suppress console.error for this test
    const originalError = console.error;
    beforeAll(() => {
      console.error = jest.fn();
    });

    afterAll(() => {
      console.error = originalError;
    });

    it('should catch and display errors', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
      expect(screen.getByText('An unexpected error occurred in the application')).toBeInTheDocument();
    });

    it('should render children when no error', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('No error')).toBeInTheDocument();
    });

    it('should provide retry functionality', () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ErrorThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();

      const retryButton = screen.getByText('Try Again');
      fireEvent.click(retryButton);

      // After retry, the error boundary should reset
      rerender(
        <ErrorBoundary>
          <ErrorThrowingComponent shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('No error')).toBeInTheDocument();
    });
  });

  describe('Integration', () => {
    it('should work together with all providers', async () => {
      render(
        <ErrorBoundary>
          <NotificationProvider>
            <LoadingProvider>
              <TestNotificationComponent />
              <TestLoadingComponent />
            </LoadingProvider>
          </NotificationProvider>
        </ErrorBoundary>
      );

      // Test notifications work
      fireEvent.click(screen.getByText('Show Success'));
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });

      // Test loading works
      fireEvent.click(screen.getByText('Start Loading'));
      await waitFor(() => {
        expect(screen.getByTestId('loading-status')).toHaveTextContent('Loading');
      });
    });
  });
});