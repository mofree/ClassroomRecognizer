import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  // React.StrictMode is removed for this specific demo to prevent double-invoking
  // the heavy model loading logic in development, which can cause memory spikes with face-api.js.
  // In production, StrictMode is fine.
  <App />
);
