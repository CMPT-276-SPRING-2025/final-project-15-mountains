// frontend/PaperDisplay.test.jsx
import { test, beforeEach, afterEach, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import PaperDisplay from './PaperDisplay';

beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        status: "success",
        result: {
          verdict: "Verified",
          accuracy_score: 0.8,
          overall_confidence: 0.8,
          sub_claims: []
        }
      })
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('renders PaperDisplay component', async () => {
  const { container } = render(<PaperDisplay claim="Test Claim" />);
  
  expect(container).toBeTruthy();
  
  expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('displays verification results when claim is provided', async () => {
  render(<PaperDisplay claim="Test Claim" />);
  
  await waitFor(() => {
    expect(screen.getByText(/Highly Supported/i)).toBeInTheDocument();
  }, { timeout: 2000 });

  expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('displays loading state initially', async () => {
  render(<PaperDisplay claim="Another Test Claim" />);
  
  await waitFor(() => {
    const loadingElement = screen.queryByText(/searching through scientific literature/i) || 
                           screen.queryByText(/extracting keywords/i) ||
                           screen.queryByText(/initializing verification/i);
    expect(loadingElement).toBeInTheDocument();
  }, { timeout: 2000 });
});

test('displays error message when fetch fails', async () => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ error: "Internal Server Error" }),
    })
  );

  render(<PaperDisplay claim="Error Test Claim" />);

  await waitFor(() => {
    const errorElement = screen.getByText(/Failed to verify claim/i);
    expect(errorElement).toBeInTheDocument();
  });
});
