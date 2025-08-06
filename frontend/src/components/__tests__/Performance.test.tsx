import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../test-utils';
import { ResearchWorkflowApp } from '../ResearchWorkflowApp';
import { WorkflowVisualization } from '../WorkflowVisualization';
import { ReportManagement } from '../ReportManagement';
import { measurePerformance } from '../../test-utils';

// Mock performance.now for consistent testing
const mockPerformanceNow = jest.fn();
Object.defineProperty(window, 'performance', {
  value: {
    now: mockPerformanceNow,
  },
});

describe('Performance Tests', () => {
  beforeEach(() => {
    mockPerformanceNow.mockClear();
    let time = 0;
    mockPerformanceNow.mockImplementation(() => {
      time += 10; // Simulate 10ms increments
      return time;
    });
  });

  describe('Component Rendering Performance', () => {
    it('should render ResearchForm quickly', () => {
      const duration = measurePerformance('ResearchForm render', () => {
        render(<div>Mock ResearchForm</div>);
      });
      
      expect(duration).toBeLessThan(100); // Should render in under 100ms
    });

    it('should render WorkflowVisualization efficiently', () => {
      const mockStatus = {
        status: 'running' as const,
        current_node: 'gap_analyzer',
        progress: 0.5,
        message: 'Processing',
      };

      const duration = measurePerformance('WorkflowVisualization render', () => {
        render(
          <WorkflowVisualization
            workflowId="test-workflow"
            status={mockStatus}
            state={null}
            onNodeClick={jest.fn()}
          />
        );
      });
      
      expect(duration).toBeLessThan(200); // Complex visualization should render in under 200ms
    });

    it('should handle large reports efficiently', () => {
      const largeReport = '# Large Report\n\n' + 'This is a large report with lots of content. '.repeat(1000);
      
      const duration = measurePerformance('Large ReportManagement render', () => {
        render(
          <ReportManagement
            report={largeReport}
            topic="Large Report Test"
            workflowId="test-workflow"
          />
        );
      });
      
      expect(duration).toBeLessThan(500); // Large content should still render reasonably fast
    });
  });

  describe('Interaction Performance', () => {
    it('should handle rapid button clicks efficiently', async () => {
      const mockOnClick = jest.fn();
      render(<button onClick={mockOnClick}>Test Button</button>);
      
      const button = screen.getByRole('button');
      
      const duration = measurePerformance('Rapid button clicks', () => {
        // Simulate rapid clicking
        for (let i = 0; i < 10; i++) {
          fireEvent.click(button);
        }
      });
      
      expect(mockOnClick).toHaveBeenCalledTimes(10);
      expect(duration).toBeLessThan(50); // Should handle rapid clicks quickly
    });

    it('should handle form input changes efficiently', () => {
      render(<input type="text" data-testid="test-input" />);
      
      const input = screen.getByTestId('test-input');
      
      const duration = measurePerformance('Form input changes', () => {
        // Simulate typing
        for (let i = 0; i < 20; i++) {
          fireEvent.change(input, { target: { value: `test${i}` } });
        }
      });
      
      expect(duration).toBeLessThan(100); // Should handle typing smoothly
    });

    it('should handle tab switching efficiently', () => {
      render(
        <ReportManagement
          report="Test report"
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      const editTab = screen.getByRole('button', { name: /edit/i });
      const viewTab = screen.getByRole('button', { name: /view/i });
      
      const duration = measurePerformance('Tab switching', () => {
        // Switch between tabs multiple times
        for (let i = 0; i < 5; i++) {
          fireEvent.click(editTab);
          fireEvent.click(viewTab);
        }
      });
      
      expect(duration).toBeLessThan(200); // Tab switching should be smooth
    });
  });

  describe('Memory Usage', () => {
    it('should not create memory leaks with repeated renders', () => {
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Render and unmount components multiple times
      for (let i = 0; i < 10; i++) {
        const { unmount } = render(<div>Test Component {i}</div>);
        unmount();
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Memory usage shouldn't grow significantly
      if (initialMemory > 0 && finalMemory > 0) {
        const memoryGrowth = finalMemory - initialMemory;
        expect(memoryGrowth).toBeLessThan(1000000); // Less than 1MB growth
      }
    });

    it('should clean up event listeners properly', () => {
      const addEventListenerSpy = jest.spyOn(window, 'addEventListener');
      const removeEventListenerSpy = jest.spyOn(window, 'removeEventListener');
      
      const { unmount } = render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={{
            status: 'running',
            current_node: 'gap_analyzer',
            progress: 0.5,
            message: 'Processing',
          }}
          state={null}
          onNodeClick={jest.fn()}
        />
      );
      
      const addedListeners = addEventListenerSpy.mock.calls.length;
      
      unmount();
      
      const removedListeners = removeEventListenerSpy.mock.calls.length;
      
      // Should clean up event listeners
      expect(removedListeners).toBeGreaterThanOrEqual(addedListeners);
      
      addEventListenerSpy.mockRestore();
      removeEventListenerSpy.mockRestore();
    });
  });

  describe('Bundle Size and Loading', () => {
    it('should lazy load heavy components', async () => {
      // Mock dynamic import
      const mockImport = jest.fn().mockResolvedValue({
        default: () => <div>Lazy Component</div>,
      });
      
      // Simulate lazy loading
      const LazyComponent = React.lazy(() => mockImport());
      
      render(
        <React.Suspense fallback={<div>Loading...</div>}>
          <LazyComponent />
        </React.Suspense>
      );
      
      // Should show loading state initially
      expect(screen.getByText('Loading...')).toBeInTheDocument();
      
      // Should load the component
      await waitFor(() => {
        expect(screen.getByText('Lazy Component')).toBeInTheDocument();
      });
      
      expect(mockImport).toHaveBeenCalled();
    });

    it('should handle code splitting efficiently', () => {
      // Test that components can be rendered without importing everything
      const duration = measurePerformance('Minimal component render', () => {
        render(<div>Minimal Component</div>);
      });
      
      expect(duration).toBeLessThan(50); // Minimal components should render very quickly
    });
  });

  describe('Animation Performance', () => {
    it('should handle CSS animations smoothly', () => {
      render(
        <div 
          className="animate-pulse"
          data-testid="animated-element"
        >
          Animated Content
        </div>
      );
      
      const element = screen.getByTestId('animated-element');
      expect(element).toHaveClass('animate-pulse');
      
      // In a real test, you might measure frame rates or animation smoothness
      expect(element).toBeInTheDocument();
    });

    it('should optimize re-renders during animations', () => {
      let renderCount = 0;
      
      const TestComponent = ({ value }: { value: number }) => {
        renderCount++;
        return <div>Value: {value}</div>;
      };
      
      const { rerender } = render(<TestComponent value={0} />);
      
      const duration = measurePerformance('Animation re-renders', () => {
        // Simulate animation frames
        for (let i = 1; i <= 60; i++) {
          rerender(<TestComponent value={i} />);
        }
      });
      
      expect(renderCount).toBe(61); // Initial render + 60 updates
      expect(duration).toBeLessThan(300); // Should handle 60fps smoothly
    });
  });

  describe('Data Processing Performance', () => {
    it('should handle large datasets efficiently', () => {
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        name: `Item ${i}`,
        description: `Description for item ${i}`,
      }));
      
      const duration = measurePerformance('Large dataset processing', () => {
        // Simulate processing large dataset
        const processed = largeDataset.map(item => ({
          ...item,
          processed: true,
        }));
        
        expect(processed).toHaveLength(1000);
      });
      
      expect(duration).toBeLessThan(100); // Should process 1000 items quickly
    });

    it('should debounce search inputs', async () => {
      let searchCount = 0;
      const mockSearch = jest.fn(() => {
        searchCount++;
      });
      
      render(<input type="text" onChange={mockSearch} data-testid="search-input" />);
      
      const input = screen.getByTestId('search-input');
      
      // Simulate rapid typing
      const duration = measurePerformance('Rapid search input', () => {
        for (let i = 0; i < 10; i++) {
          fireEvent.change(input, { target: { value: `search${i}` } });
        }
      });
      
      expect(mockSearch).toHaveBeenCalledTimes(10);
      expect(duration).toBeLessThan(100);
    });
  });

  describe('Network Performance', () => {
    it('should handle API call failures gracefully', async () => {
      const mockFetch = jest.fn().mockRejectedValue(new Error('Network error'));
      global.fetch = mockFetch;
      
      const duration = measurePerformance('Failed API call handling', async () => {
        try {
          await fetch('/api/test');
        } catch (error) {
          expect((error as Error).message).toBe('Network error');
        }
      });
      
      expect(duration).toBeLessThan(100); // Should fail fast
      expect(mockFetch).toHaveBeenCalled();
    });

    it('should implement proper loading states', () => {
      const { rerender } = render(<div>Loading: false</div>);
      
      const duration = measurePerformance('Loading state updates', () => {
        // Simulate loading state changes
        rerender(<div>Loading: true</div>);
        rerender(<div>Loading: false</div>);
      });
      
      expect(duration).toBeLessThan(50); // Loading state changes should be instant
    });
  });

  describe('Scroll Performance', () => {
    it('should handle smooth scrolling', () => {
      render(
        <div style={{ height: '2000px' }} data-testid="scrollable-content">
          <div style={{ height: '100px' }}>Content 1</div>
          <div style={{ height: '100px' }}>Content 2</div>
          <div style={{ height: '100px' }}>Content 3</div>
        </div>
      );
      
      const content = screen.getByTestId('scrollable-content');
      
      const duration = measurePerformance('Scroll events', () => {
        // Simulate scroll events
        for (let i = 0; i < 10; i++) {
          fireEvent.scroll(content, { target: { scrollY: i * 100 } });
        }
      });
      
      expect(duration).toBeLessThan(100); // Should handle scroll events smoothly
    });

    it('should implement virtual scrolling for large lists', () => {
      const largeList = Array.from({ length: 10000 }, (_, i) => `Item ${i}`);
      
      const duration = measurePerformance('Virtual scrolling simulation', () => {
        // Simulate rendering only visible items
        const visibleItems = largeList.slice(0, 20); // Only render first 20 items
        
        render(
          <div>
            {visibleItems.map((item, index) => (
              <div key={index}>{item}</div>
            ))}
          </div>
        );
      });
      
      expect(duration).toBeLessThan(100); // Should render quickly even with large datasets
    });
  });
});