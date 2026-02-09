import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Download, Loader2, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

// Severity dots component
const SeverityDots = ({ severity }) => {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          className={`w-2 h-2 rounded-full ${i <= severity ? 'bg-apple-error' : 'bg-apple-border'}`}
        />
      ))}
    </div>
  );
};

// Risk badge component
const RiskBadge = ({ level }) => {
  const colors = {
    'Düşük': 'bg-green-100 text-green-700',
    'Orta': 'bg-orange-100 text-orange-700',
    'Yüksek': 'bg-red-100 text-red-700'
  };
  
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${colors[level] || 'bg-gray-100 text-gray-700'}`}>
      {level}
    </span>
  );
};

const ResultPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    fetchAnalysis();
  }, [id]);

  useEffect(() => {
    if (analysis && imageLoaded) {
      drawAnnotations();
    }
  }, [analysis, imageLoaded]);

  const fetchAnalysis = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/analyses/${id}`);
      setAnalysis(response.data);
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Analiz yüklenemedi');
    } finally {
      setLoading(false);
    }
  };

  const drawAnnotations = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !analysis) return;

    const ctx = canvas.getContext('2d');
    const rect = image.getBoundingClientRect();
    const containerRect = image.parentElement.getBoundingClientRect();
    
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    
    // Calculate image display dimensions
    const imgAspect = analysis.results.image_size.width / analysis.results.image_size.height;
    const containerAspect = containerRect.width / containerRect.height;
    
    let displayWidth, displayHeight, offsetX, offsetY;
    
    if (imgAspect > containerAspect) {
      displayWidth = containerRect.width;
      displayHeight = containerRect.width / imgAspect;
      offsetX = 0;
      offsetY = (containerRect.height - displayHeight) / 2;
    } else {
      displayHeight = containerRect.height;
      displayWidth = containerRect.height * imgAspect;
      offsetX = (containerRect.width - displayWidth) / 2;
      offsetY = 0;
    }
    
    const scaleX = displayWidth / analysis.results.image_size.width;
    const scaleY = displayHeight / analysis.results.image_size.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw damage boxes
    analysis.results.damages.forEach((damage, index) => {
      const [x1, y1, x2, y2] = damage.box;
      const boxX = offsetX + x1 * scaleX;
      const boxY = offsetY + y1 * scaleY;
      const boxW = (x2 - x1) * scaleX;
      const boxH = (y2 - y1) * scaleY;
      
      // Box with rounded corners
      ctx.strokeStyle = '#FF3B30';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      
      const radius = 4;
      ctx.beginPath();
      ctx.roundRect(boxX, boxY, boxW, boxH, radius);
      ctx.stroke();
      
      // Semi-transparent fill
      ctx.fillStyle = 'rgba(255, 59, 48, 0.1)';
      ctx.fill();
      
      // Draw callout line
      const centerX = boxX + boxW / 2;
      const centerY = boxY + boxH / 2;
      const isLeftSide = centerX < canvas.width / 2;
      const lineEndX = isLeftSide ? 10 : canvas.width - 10;
      const lineEndY = 60 + index * 80;
      
      // Bezier curve for callout line
      ctx.strokeStyle = '#FF3B30';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(isLeftSide ? boxX : boxX + boxW, centerY);
      
      const controlX = isLeftSide ? boxX - 30 : boxX + boxW + 30;
      ctx.bezierCurveTo(
        controlX, centerY,
        controlX, lineEndY,
        lineEndX + (isLeftSide ? 150 : -150), lineEndY
      );
      ctx.stroke();
      
      // Draw dot at box connection
      ctx.fillStyle = '#FF3B30';
      ctx.beginPath();
      ctx.arc(isLeftSide ? boxX : boxX + boxW, centerY, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const handleDownloadPDF = async () => {
    setDownloading(true);
    try {
      const response = await axios.get(`${API_URL}/api/analyses/${id}/pdf`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `hasar-raporu-${id.slice(0, 8)}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      alert('PDF indirilemedi');
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-apple-secondary" />
          <p className="text-apple-secondary">Analiz yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center">
        <div className="text-center">
          <XCircle className="w-12 h-12 mx-auto mb-4 text-apple-error" />
          <p className="text-apple-text font-medium mb-2">{error}</p>
          <Link to="/" className="text-apple-secondary hover:text-apple-text">
            Yeni analiz yap
          </Link>
        </div>
      </div>
    );
  }

  const { results } = analysis;
  const { damages, summary } = results;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-8"
      >
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-apple-secondary hover:text-apple-text transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          Yeni Analiz
        </button>
        
        <button
          onClick={handleDownloadPDF}
          disabled={downloading}
          className="flex items-center gap-2 px-5 py-2.5 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50"
          data-testid="download-pdf-button"
        >
          {downloading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Download className="w-5 h-5" />
          )}
          PDF İndir
        </button>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Image with Annotations */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-2"
        >
          <div className="bg-white rounded-2xl shadow-apple overflow-hidden">
            <div className="relative aspect-video bg-gray-100">
              <img
                ref={imageRef}
                src={`data:image/jpeg;base64,${analysis.image_base64}`}
                alt="Analiz edilen araç"
                className="w-full h-full object-contain"
                onLoad={() => setImageLoaded(true)}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 pointer-events-none"
              />
              
              {/* Callout Cards */}
              {damages.map((damage, index) => {
                const [x1, y1, x2, y2] = damage.box;
                const centerX = (x1 + x2) / 2;
                const isLeftSide = centerX < results.image_size.width / 2;
                
                return (
                  <motion.div
                    key={damage.id}
                    initial={{ opacity: 0, x: isLeftSide ? -20 : 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    className={`absolute bg-white rounded-xl p-3 shadow-apple-lg min-w-[140px] ${
                      isLeftSide ? 'left-2' : 'right-2'
                    }`}
                    style={{ top: `${60 + index * 80}px` }}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <AlertTriangle className="w-4 h-4 text-apple-error" />
                      <span className="font-semibold text-apple-text text-sm">
                        {damage.type_tr}
                      </span>
                    </div>
                    {damage.part_tr && (
                      <p className="text-xs text-apple-secondary mb-1">
                        {damage.part_tr}
                      </p>
                    )}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-apple-secondary">
                        %{damage.confidence}
                      </span>
                      <SeverityDots severity={damage.severity} />
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Summary & Details */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          {/* Summary Card */}
          <div className="bg-white rounded-2xl shadow-apple p-6">
            <h2 className="text-lg font-semibold text-apple-text mb-4">Özet</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between py-3 border-b border-apple-border">
                <span className="text-apple-secondary">Toplam Hasar</span>
                <span className="font-semibold text-apple-text text-xl">{summary.total_damages}</span>
              </div>
              
              <div className="flex items-center justify-between py-3 border-b border-apple-border">
                <span className="text-apple-secondary">Etkilenen Parça</span>
                <span className="font-semibold text-apple-text text-xl">{summary.affected_parts}</span>
              </div>
              
              <div className="flex items-center justify-between py-3 border-b border-apple-border">
                <span className="text-apple-secondary">Ortalama Şiddet</span>
                <SeverityDots severity={Math.round(summary.average_severity)} />
              </div>
              
              <div className="flex items-center justify-between py-3">
                <span className="text-apple-secondary">Risk Seviyesi</span>
                <RiskBadge level={summary.risk_level} />
              </div>
            </div>
          </div>

          {/* Damage List */}
          <div className="bg-white rounded-2xl shadow-apple p-6">
            <h2 className="text-lg font-semibold text-apple-text mb-4">Hasar Detayları</h2>
            
            {damages.length > 0 ? (
              <div className="space-y-3">
                {damages.map((damage, index) => (
                  <motion.div
                    key={damage.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + index * 0.05 }}
                    className="p-4 bg-gray-50 rounded-xl"
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium text-apple-text">
                            {damage.type_tr}
                          </span>
                          <span className="text-xs px-2 py-0.5 bg-apple-error/10 text-apple-error rounded-full">
                            %{damage.confidence}
                          </span>
                        </div>
                        <p className="text-sm text-apple-secondary">
                          {damage.part_tr || 'Belirsiz parça'}
                        </p>
                      </div>
                      <SeverityDots severity={damage.severity} />
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 mx-auto mb-3 text-apple-success" />
                <p className="text-apple-text font-medium">Hasar tespit edilmedi</p>
                <p className="text-sm text-apple-secondary">Aracınız iyi durumda görünüyor</p>
              </div>
            )}
          </div>

          {/* Timestamp */}
          <p className="text-center text-sm text-apple-secondary">
            Analiz tarihi: {new Date(analysis.created_at).toLocaleString('tr-TR')}
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default ResultPage;
