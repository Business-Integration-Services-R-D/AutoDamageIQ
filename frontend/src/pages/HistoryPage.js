import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Loader2, Trash2, Eye, Calendar, AlertTriangle, FileText } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const HistoryPage = () => {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(null);

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/analyses`);
      setAnalyses(response.data);
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id, e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!window.confirm('Bu analizi silmek istediğinize emin misiniz?')) return;
    
    setDeleting(id);
    try {
      await axios.delete(`${API_URL}/api/analyses/${id}`);
      setAnalyses(analyses.filter(a => a.id !== id));
    } catch (err) {
      console.error('Delete error:', err);
      alert('Silme işlemi başarısız');
    } finally {
      setDeleting(null);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'Yüksek': return 'text-apple-error bg-red-50';
      case 'Orta': return 'text-apple-warning bg-orange-50';
      default: return 'text-apple-success bg-green-50';
    }
  };

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-apple-secondary" />
          <p className="text-apple-secondary">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-apple-text mb-2">Geçmiş Analizler</h1>
        <p className="text-apple-secondary">
          Tüm araç hasar analizleriniz burada listelenir
        </p>
      </motion.div>

      {analyses.length === 0 ? (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white rounded-2xl shadow-apple p-12 text-center"
        >
          <FileText className="w-16 h-16 mx-auto mb-4 text-apple-border" />
          <h3 className="text-xl font-semibold text-apple-text mb-2">
            Henüz analiz yok
          </h3>
          <p className="text-apple-secondary mb-6">
            İlk araç hasar analizinizi yapmak için başlayın
          </p>
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-6 py-3 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors"
          >
            Yeni Analiz Başlat
          </Link>
        </motion.div>
      ) : (
        <div className="grid gap-4">
          {analyses.map((analysis, index) => (
            <motion.div
              key={analysis.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Link
                to={`/result/${analysis.id}`}
                className="block bg-white rounded-2xl shadow-apple hover:shadow-apple-lg transition-shadow overflow-hidden group"
                data-testid={`analysis-card-${analysis.id}`}
              >
                <div className="flex items-stretch">
                  {/* Thumbnail */}
                  <div className="w-40 h-28 bg-gray-100 flex-shrink-0 overflow-hidden">
                    {analysis.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${analysis.thumbnail}`}
                        alt="Thumbnail"
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-apple-secondary">
                        <FileText className="w-8 h-8" />
                      </div>
                    )}
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1 p-4 flex items-center justify-between">
                    <div>
                      {/* Date & Filename */}
                      <div className="flex items-center gap-3 mb-2">
                        <div className="flex items-center gap-1.5 text-sm text-apple-secondary">
                          <Calendar className="w-4 h-4" />
                          {new Date(analysis.created_at).toLocaleDateString('tr-TR', {
                            day: 'numeric',
                            month: 'long',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      </div>
                      
                      {/* Summary Stats */}
                      <div className="flex items-center gap-4">
                        <div className="flex items-center gap-1.5">
                          <AlertTriangle className="w-4 h-4 text-apple-error" />
                          <span className="font-semibold text-apple-text">
                            {analysis.summary.total_damages}
                          </span>
                          <span className="text-sm text-apple-secondary">hasar</span>
                        </div>
                        
                        <div className="text-apple-secondary">|</div>
                        
                        <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${getRiskColor(analysis.summary.risk_level)}`}>
                          {analysis.summary.risk_level} Risk
                        </span>
                      </div>
                    </div>
                    
                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => handleDelete(analysis.id, e)}
                        disabled={deleting === analysis.id}
                        className="p-2 text-apple-secondary hover:text-apple-error hover:bg-red-50 rounded-lg transition-colors"
                        title="Sil"
                      >
                        {deleting === analysis.id ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <Trash2 className="w-5 h-5" />
                        )}
                      </button>
                      
                      <div className="p-2 text-apple-secondary group-hover:text-apple-text group-hover:bg-gray-100 rounded-lg transition-colors">
                        <Eye className="w-5 h-5" />
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      )}

      {/* New Analysis Button (Fixed) */}
      {analyses.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="fixed bottom-8 right-8"
        >
          <Link
            to="/"
            className="flex items-center gap-2 px-6 py-3 bg-black text-white rounded-full font-medium shadow-lg hover:bg-gray-800 hover:shadow-xl transition-all"
          >
            + Yeni Analiz
          </Link>
        </motion.div>
      )}
    </div>
  );
};

export default HistoryPage;
