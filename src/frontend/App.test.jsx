// frontend/App.test.jsx
import { test, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders Header component', () => {
  render(<App />);
  // Check for the header element itself
  const headerElement = document.querySelector('header');
  expect(headerElement).toBeInTheDocument();
});
