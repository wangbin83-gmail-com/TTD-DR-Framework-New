import React, { useState, useEffect } from 'react';

function AppMinimal() {
  const [message, setMessage] = useState('Loading...');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    // Test backend connection
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => {
        setBackendStatus('connected');
        setMessage('TTD-DR Framework is ready!');
      })
      .catch(err => {
        setBackendStatus('error');
        setMessage('Backend not available - running in demo mode');
      });
  }, []);

  return (
    <div style={{ 
      padding: '40px', 
      maxWidth: '600px', 
      margin: '0 auto', 
      fontFamily: 'Arial, sans-serif',
      textAlign: 'center'
    }}>
      <h1 style={{ color: '#2563eb', marginBottom: '20px' }}>
        🎯 TTD-DR Framework
      </h1>
      
      <div style={{
        background: '#f8fafc',
        padding: '30px',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        marginBottom: '20px'
      }}>
        <h2 style={{ color: '#1e293b', marginBottom: '15px' }}>
          {message}
        </h2>
        
        <div style={{ marginBottom: '20px' }}>
          <p style={{ 
            color: backendStatus === 'connected' ? '#16a34a' : '#dc2626',
            fontWeight: 'bold'
          }}>
            Backend Status: {backendStatus}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
          <button 
            onClick={() => window.location.href = 'http://localhost:5175/test.html'}
            style={{
              padding: '10px 20px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Test Page
          </button>
          
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 20px',
              background: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Reload
          </button>
        </div>
      </div>

      <div style={{
        background: '#fef3c7',
        padding: '20px',
        borderRadius: '8px',
        borderLeft: '4px solid #f59e0b'
      }}>
        <h3 style={{ color: '#92400e', marginBottom: '10px' }}>Quick Start</h3>
        <ul style={{ textAlign: 'left', color: '#92400e' }}>
          <li>Backend: http://localhost:8000/docs</li>
          <li>Frontend: http://localhost:5175</li>
          <li>Test page: http://localhost:5175/test.html</li>
        </ul>
      </div>
    </div>
  );
}

export default AppMinimal;