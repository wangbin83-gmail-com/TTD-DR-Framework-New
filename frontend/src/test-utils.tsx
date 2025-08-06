import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AppProviders } from './components/AppProviders';

// Custom render function that includes all necessary providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      <AppProviders>
        {children}
      </AppProviders>
    </BrowserRouter>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';
export { customRender as render };

// Mock implementations for common external dependencies
export const mockNotifications = {
  showSuccess: jest.fn(),
  showError: jest.fn(),
  showWarning: jest.fn(),
  showInfo: jest.fn(),
};

export const mockLoading = {
  startLoading: jest.fn(),
  finishLoading: jest.fn(),
  isLoading: false,
};

// Mock ResearchWorkflow hook
export const mockResearchWorkflow = {
  workflowId: null as string | null,
  status: {
    status: 'idle' as 'idle' | 'running' | 'completed' | 'error',
    current_node: null as string | null,
    progress: 0,
    message: 'Ready to start',
  },
  state: null as any,
  finalReport: null as string | null,
  isLoading: false,
  error: null as any,
  wsConnected: true,
  startResearch: jest.fn(),
  stopWorkflow: jest.fn(),
  restartWorkflow: jest.fn(),
  resetWorkflow: jest.fn(),
  getCurrentState: jest.fn(),
  getFinalReport: jest.fn(),
};

// Accessibility testing utilities
export const axeMatchers = {
  toHaveNoViolations: expect.extend({
    toHaveNoViolations(received) {
      const violations = received.violations || [];
      const pass = violations.length === 0;
      
      if (pass) {
        return {
          message: () => `Expected accessibility violations, but found none`,
          pass: true,
        };
      } else {
        const violationMessages = violations.map((violation: any) => 
          `${violation.id}: ${violation.description}`
        ).join('\n');
        
        return {
          message: () => `Expected no accessibility violations, but found:\n${violationMessages}`,
          pass: false,
        };
      }
    },
  }),
};

// Mock external libraries
export const mockExternalLibraries = () => {
  // Mock file-saver
  jest.mock('file-saver', () => ({
    saveAs: jest.fn(),
  }));

  // Mock jsPDF
  jest.mock('jspdf', () => {
    return jest.fn().mockImplementation(() => ({
      addImage: jest.fn(),
      addPage: jest.fn(),
      save: jest.fn(),
    }));
  });

  // Mock html2canvas
  jest.mock('html2canvas', () => {
    return jest.fn().mockResolvedValue({
      toDataURL: jest.fn().mockReturnValue('data:image/png;base64,mock'),
      height: 800,
      width: 600,
    });
  });

  // Mock ReactFlow
  jest.mock('reactflow', () => ({
    __esModule: true,
    default: ({ children, ...props }: any) => (
      <div data-testid="react-flow" {...props}>
        {children}
      </div>
    ),
    useNodesState: () => [[], jest.fn(), jest.fn()],
    useEdgesState: () => [[], jest.fn(), jest.fn()],
    addEdge: jest.fn(),
    Controls: () => <div data-testid="controls" />,
    MiniMap: () => <div data-testid="minimap" />,
    Background: () => <div data-testid="background" />,
    BackgroundVariant: { Dots: 'dots' },
    Panel: ({ children, position, className }: any) => (
      <div data-testid={`panel-${position}`} className={className}>
        {children}
      </div>
    ),
    Handle: ({ type, position }: any) => (
      <div data-testid={`handle-${type}-${position}`} />
    ),
    Position: {
      Top: 'top',
      Bottom: 'bottom',
      Left: 'left',
      Right: 'right',
    },
    MarkerType: {
      ArrowClosed: 'arrowclosed',
    },
  }));
};

// Performance testing utilities
export const measurePerformance = (name: string, fn: () => void) => {
  const start = performance.now();
  fn();
  const end = performance.now();
  const duration = end - start;
  
  // Log performance metrics for analysis
  console.log(`Performance: ${name} took ${duration.toFixed(2)}ms`);
  
  // Assert reasonable performance thresholds
  expect(duration).toBeLessThan(1000); // Should complete within 1 second
  
  return duration;
};

// Keyboard navigation testing utilities
export const testKeyboardNavigation = async (element: HTMLElement) => {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  // Test that all focusable elements can receive focus
  for (const el of Array.from(focusableElements)) {
    (el as HTMLElement).focus();
    expect(document.activeElement).toBe(el);
  }
  
  return focusableElements.length;
};

// Screen reader testing utilities
export const testScreenReaderSupport = (element: HTMLElement) => {
  const issues: string[] = [];
  
  // Check for missing alt text on images
  const images = element.querySelectorAll('img');
  images.forEach((img, index) => {
    if (!img.alt && !img.getAttribute('aria-label')) {
      issues.push(`Image ${index + 1} missing alt text`);
    }
  });
  
  // Check for missing labels on form elements
  const formElements = element.querySelectorAll('input, select, textarea');
  formElements.forEach((el, index) => {
    const hasLabel = el.getAttribute('aria-label') || 
                    el.getAttribute('aria-labelledby') ||
                    element.querySelector(`label[for="${el.id}"]`);
    
    if (!hasLabel) {
      issues.push(`Form element ${index + 1} missing label`);
    }
  });
  
  // Check for proper heading hierarchy
  const headings = element.querySelectorAll('h1, h2, h3, h4, h5, h6');
  let previousLevel = 0;
  headings.forEach((heading, index) => {
    const level = parseInt(heading.tagName.charAt(1));
    if (level > previousLevel + 1) {
      issues.push(`Heading ${index + 1} skips levels (h${previousLevel} to h${level})`);
    }
    previousLevel = level;
  });
  
  return issues;
};

// Color contrast testing utilities
export const testColorContrast = (element: HTMLElement) => {
  // This is a simplified version - in a real implementation,
  // you'd use a library like axe-core for comprehensive testing
  const textElements = element.querySelectorAll('*');
  const issues: string[] = [];
  
  textElements.forEach((el, index) => {
    const styles = window.getComputedStyle(el);
    const color = styles.color;
    const backgroundColor = styles.backgroundColor;
    
    // Basic check for transparent backgrounds on text
    if (color && color !== 'rgba(0, 0, 0, 0)' && 
        backgroundColor === 'rgba(0, 0, 0, 0)') {
      // This might indicate a contrast issue
      const textContent = el.textContent?.trim();
      if (textContent && textContent.length > 0) {
        // Only flag if it's actual text content
        issues.push(`Element ${index + 1} may have contrast issues`);
      }
    }
  });
  
  return issues;
};