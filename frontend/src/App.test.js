import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the app heading', () => {
  render(<App />);
  expect(screen.getByText(/Speech Emotion Recognition/i)).toBeInTheDocument();
});

test('renders file upload area', () => {
  render(<App />);
  expect(screen.getByText(/Choose an audio file/i)).toBeInTheDocument();
});

test('renders record button', () => {
  render(<App />);
  expect(screen.getByText(/Record/i)).toBeInTheDocument();
});
