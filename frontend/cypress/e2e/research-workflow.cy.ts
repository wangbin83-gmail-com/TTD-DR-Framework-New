describe('Research Workflow E2E Tests', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('should display the welcome page', () => {
    cy.contains('Welcome to TTD-DR Framework').should('be.visible');
    cy.contains('Test-Time Diffusion Deep Researcher').should('be.visible');
    cy.contains('Start your research journey').should('be.visible');
  });

  it('should show research form by default', () => {
    cy.contains('Start New Research').should('be.visible');
    cy.get('textarea[placeholder*="Enter your research topic"]').should('be.visible');
    cy.get('select').should('have.length.at.least', 2);
    cy.get('button[type="submit"]').should('contain', 'Start Research');
  });

  it('should validate required fields', () => {
    cy.get('button[type="submit"]').click();
    cy.contains('Research topic is required').should('be.visible');
  });

  it('should validate minimum topic length', () => {
    cy.get('textarea[placeholder*="Enter your research topic"]').type('AI');
    cy.get('button[type="submit"]').click();
    cy.contains('Topic must be at least 10 characters long').should('be.visible');
  });

  it('should validate iteration range', () => {
    cy.get('input[type="number"]').first().clear().type('0');
    cy.get('button[type="submit"]').click();
    cy.contains('Must be at least 1').should('be.visible');

    cy.get('input[type="number"]').first().clear().type('25');
    cy.get('button[type="submit"]').click();
    cy.contains('Cannot exceed 20').should('be.visible');
  });

  it('should validate source range', () => {
    cy.get('input[type="number"]').eq(1).clear().type('3');
    cy.get('button[type="submit"]').click();
    cy.contains('Must be at least 5').should('be.visible');

    cy.get('input[type="number"]').eq(1).clear().type('150');
    cy.get('button[type="submit"]').click();
    cy.contains('Cannot exceed 100').should('be.visible');
  });

  it('should handle source type selection', () => {
    // Check that some checkboxes are checked by default
    cy.get('input[type="checkbox"]').should('have.length.at.least', 3);
    cy.get('input[type="checkbox"]:checked').should('have.length.at.least', 1);

    // Uncheck and check boxes
    cy.get('input[type="checkbox"]').first().uncheck();
    cy.get('input[type="checkbox"]').first().should('not.be.checked');
    
    cy.get('input[type="checkbox"]').first().check();
    cy.get('input[type="checkbox"]').first().should('be.checked');
  });

  it('should update quality threshold display', () => {
    cy.get('input[type="range"]').invoke('val', '0.9').trigger('input');
    cy.contains('Quality Threshold: 0.9').should('be.visible');
  });

  it('should start research workflow with valid input', () => {
    // Mock the API response
    cy.intercept('POST', '/api/research/start', {
      statusCode: 200,
      body: {
        data: { workflow_id: 'test-workflow-123' },
        message: 'Research started successfully'
      }
    }).as('startResearch');

    // Fill the form
    cy.get('textarea[placeholder*="Enter your research topic"]')
      .type('Artificial Intelligence applications in modern healthcare systems');
    
    cy.get('select').first().select('technology');
    cy.get('select').eq(1).select('advanced');

    // Submit the form
    cy.get('button[type="submit"]').click();

    // Verify API call
    cy.wait('@startResearch').then((interception) => {
      expect(interception.request.body).to.have.property('topic');
      expect(interception.request.body).to.have.property('requirements');
      expect(interception.request.body.topic).to.contain('Artificial Intelligence');
    });

    // Should show loading state
    cy.contains('Starting Research...').should('be.visible');
  });

  it('should navigate to dashboard after starting research', () => {
    // Mock successful research start
    cy.intercept('POST', '/api/research/start', {
      statusCode: 200,
      body: {
        data: { workflow_id: 'test-workflow-123' },
        message: 'Research started successfully'
      }
    }).as('startResearch');

    // Mock WebSocket connection (simplified)
    cy.window().then((win) => {
      // Mock WebSocket
      (win as any).WebSocket = class MockWebSocket {
        constructor(url: string) {
          setTimeout(() => {
            if (this.onopen) this.onopen({} as Event);
          }, 100);
        }
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onclose: (() => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        send() {}
        close() {}
      };
    });

    // Fill and submit form
    cy.get('textarea[placeholder*="Enter your research topic"]')
      .type('Machine learning in medical diagnosis');
    cy.get('button[type="submit"]').click();

    cy.wait('@startResearch');

    // Should navigate to dashboard view
    cy.url().should('include', '/research/test-workflow-123');
    cy.contains('Research Workflow').should('be.visible');
    cy.contains('Workflow ID: test-workflow-123').should('be.visible');
  });

  it('should display workflow stages in dashboard', () => {
    // Navigate directly to a workflow (simulating existing workflow)
    cy.visit('/research/test-workflow-123');

    // Mock workflow status
    cy.intercept('GET', '/api/research/status/test-workflow-123', {
      statusCode: 200,
      body: {
        data: {
          status: 'running',
          current_node: 'gap_analyzer',
          progress: 0.3,
          message: 'Analyzing information gaps...'
        }
      }
    }).as('getStatus');

    // Mock current state
    cy.intercept('GET', '/api/research/state/test-workflow-123', {
      statusCode: 200,
      body: {
        data: {
          topic: 'Machine learning in medical diagnosis',
          iteration_count: 2,
          information_gaps: [{ id: 'gap-1' }],
          quality_metrics: {
            overall_score: 0.72,
            completeness: 0.7,
            coherence: 0.8
          }
        }
      }
    }).as('getState');

    cy.wait(['@getStatus', '@getState']);

    // Verify workflow stages are displayed
    cy.contains('Draft Generation').should('be.visible');
    cy.contains('Gap Analysis').should('be.visible');
    cy.contains('Information Retrieval').should('be.visible');
    cy.contains('Information Integration').should('be.visible');
    cy.contains('Quality Assessment').should('be.visible');
    cy.contains('Self-Evolution').should('be.visible');
    cy.contains('Report Synthesis').should('be.visible');

    // Verify progress information
    cy.contains('30%').should('be.visible');
    cy.contains('Analyzing information gaps...').should('be.visible');
  });

  it('should handle workflow completion and show report', () => {
    const mockReport = `# Research Report: Machine Learning in Medical Diagnosis

## Introduction
Machine learning has revolutionized medical diagnosis...

## Key Findings
- Improved accuracy in image analysis
- Faster diagnosis times
- Reduced human error

## Conclusion
ML continues to transform healthcare...`;

    // Mock completed workflow
    cy.intercept('GET', '/api/research/report/test-workflow-123', {
      statusCode: 200,
      body: {
        data: { report: mockReport }
      }
    }).as('getReport');

    cy.visit('/research/test-workflow-123');
    cy.wait('@getReport');

    // Should show report view
    cy.contains('Research Report').should('be.visible');
    cy.contains('Machine Learning in Medical Diagnosis').should('be.visible');
    
    // Verify export buttons
    cy.contains('Export PDF').should('be.visible');
    cy.contains('Export Word').should('be.visible');
    cy.contains('Export Markdown').should('be.visible');
    cy.contains('Share').should('be.visible');

    // Verify report content
    cy.contains('Introduction').should('be.visible');
    cy.contains('Key Findings').should('be.visible');
    cy.contains('Conclusion').should('be.visible');
  });

  it('should toggle between formatted and raw markdown view', () => {
    const mockReport = '# Test Report\n\nThis is a **test** report.';

    cy.intercept('GET', '/api/research/report/test-workflow-123', {
      statusCode: 200,
      body: {
        data: { report: mockReport }
      }
    }).as('getReport');

    cy.visit('/research/test-workflow-123');
    cy.wait('@getReport');

    // Should show formatted view by default
    cy.contains('Test Report').should('be.visible');
    cy.get('strong').should('contain', 'test');

    // Toggle to raw view
    cy.get('button[title*="Show raw markdown"]').click();
    cy.contains('Raw Markdown').should('be.visible');
    cy.contains('# Test Report').should('be.visible');
    cy.contains('**test**').should('be.visible');

    // Toggle back to formatted view
    cy.get('button[title*="Show formatted view"]').click();
    cy.get('strong').should('contain', 'test');
  });

  it('should handle navigation between views', () => {
    cy.visit('/research/test-workflow-123');

    // Mock API responses
    cy.intercept('GET', '/api/research/status/test-workflow-123', {
      body: { data: { status: 'completed', progress: 1, message: 'Completed' } }
    });
    cy.intercept('GET', '/api/research/report/test-workflow-123', {
      body: { data: { report: '# Test Report' } }
    });

    // Should be able to navigate to new research
    cy.contains('New Research').click();
    cy.url().should('eq', Cypress.config().baseUrl + '/');
    cy.contains('Start New Research').should('be.visible');
  });

  it('should handle errors gracefully', () => {
    // Mock API error
    cy.intercept('POST', '/api/research/start', {
      statusCode: 500,
      body: {
        error: 'Internal server error'
      }
    }).as('startResearchError');

    cy.get('textarea[placeholder*="Enter your research topic"]')
      .type('Test research topic for error handling');
    cy.get('button[type="submit"]').click();

    cy.wait('@startResearchError');

    // Should display error message
    cy.contains('Error').should('be.visible');
    cy.contains('Internal server error').should('be.visible');
  });

  it('should be responsive on mobile devices', () => {
    cy.viewport('iphone-x');
    
    cy.visit('/');
    
    // Form should be responsive
    cy.get('textarea[placeholder*="Enter your research topic"]').should('be.visible');
    cy.get('button[type="submit"]').should('be.visible');
    
    // Navigation should work on mobile
    cy.get('h1').should('contain', 'TTD-DR Framework');
    
    // Form elements should stack properly
    cy.get('select').should('be.visible');
    cy.get('input[type="range"]').should('be.visible');
  });

  it('should maintain state during page refresh', () => {
    // This test would require proper state persistence
    // For now, we'll test that the URL-based navigation works
    cy.visit('/research/test-workflow-123');
    
    cy.reload();
    
    // Should still be on the same workflow page
    cy.url().should('include', '/research/test-workflow-123');
  });
});