/// <reference types="cypress" />

// Custom commands for TTD-DR Framework testing

declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Custom command to fill research form
       * @example cy.fillResearchForm('AI in Healthcare', 'technology', 'advanced')
       */
      fillResearchForm(topic: string, domain?: string, complexity?: string): Chainable<Element>;
      
      /**
       * Custom command to wait for workflow to complete
       * @example cy.waitForWorkflowCompletion()
       */
      waitForWorkflowCompletion(): Chainable<Element>;
      
      /**
       * Custom command to mock API responses
       * @example cy.mockApiResponse('POST', '/api/research/start', { workflow_id: 'test-123' })
       */
      mockApiResponse(method: string, url: string, response: any): Chainable<Element>;
    }
  }
}

Cypress.Commands.add('fillResearchForm', (topic: string, domain = 'general', complexity = 'intermediate') => {
  cy.get('[data-testid="research-topic"]').type(topic);
  cy.get('[data-testid="research-domain"]').select(domain);
  cy.get('[data-testid="complexity-level"]').select(complexity);
});

Cypress.Commands.add('waitForWorkflowCompletion', () => {
  cy.get('[data-testid="workflow-status"]', { timeout: 30000 })
    .should('contain', 'completed')
    .or('contain', 'error');
});

Cypress.Commands.add('mockApiResponse', (method: string, url: string, response: any) => {
  cy.intercept(method, url, response).as('apiCall');
});

// Add data-testid attributes helper
Cypress.Commands.add('getByTestId', (testId: string) => {
  return cy.get(`[data-testid="${testId}"]`);
});

export {};