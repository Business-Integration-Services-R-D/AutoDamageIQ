import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import HistoryPage from './pages/HistoryPage';
import TrainingPage from './pages/TrainingPage';
import ComparePage from './pages/ComparePage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-apple-bg">
        <Header />
        <main className="pt-16">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/result/:id" element={<ResultPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
