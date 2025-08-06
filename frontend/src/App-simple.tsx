import React from 'react';

function AppSimple() {
  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>TTD-DR Framework Test</h1>
      <p>Frontend is working!</p>
      <div style={{ marginTop: '20px' }}>
        <h2>System Status:</h2>
        <ul style={{ textAlign: 'left', maxWidth: '400px', margin: '0 auto' }}>
          <li>✅ React App Loaded</li>
          <li>✅ Vite Development Server</li>
          <li>✅ Tailwind CSS</li>
          <li>✅ Dependencies Installed</li>
        </ul>
      </div>
      <div style={{ marginTop: '30px' }}>
        <button 
          onClick={() => {
            fetch('http://localhost:8000/health')
              .then(res => res.json())
              .then(data => alert('Backend: ' + data.status))
              .catch(err => alert('Backend Error: ' + err.message));
          }}
          style={{ padding: '10px 20px', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Test Backend Connection
        </button>
      </div>
    </div>
  );
}

export default AppSimple;