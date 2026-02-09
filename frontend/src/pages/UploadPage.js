import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Image, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const UploadPage = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      processFile(droppedFile);
    } else {
      setError('Lütfen geçerli bir resim dosyası seçin');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (file) => {
    setError(null);
    setFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      navigate(`/result/${response.data.id}`);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'Analiz sırasında bir hata oluştu');
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-6 py-12">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-2xl"
      >
        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-apple-text mb-3">
            Araç Hasar Analizi
          </h1>
          <p className="text-apple-secondary text-lg">
            Yapay zeka ile hasarlı aracınızı analiz edin
          </p>
        </div>

        {/* Upload Area */}
        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={`bg-white rounded-3xl border-2 border-dashed transition-all duration-300 ${
                isDragging 
                  ? 'border-black bg-gray-50 scale-[1.02]' 
                  : 'border-apple-border hover:border-gray-400'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="p-12 text-center">
                <div className={`w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center transition-colors ${
                  isDragging ? 'bg-black' : 'bg-gray-100'
                }`}>
                  <Upload className={`w-10 h-10 transition-colors ${
                    isDragging ? 'text-white' : 'text-apple-secondary'
                  }`} />
                </div>
                
                <h3 className="text-xl font-semibold text-apple-text mb-2">
                  Araç fotoğrafını sürükleyin
                </h3>
                <p className="text-apple-secondary mb-6">
                  veya
                </p>
                
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-8 py-3 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors"
                >
                  Dosya Seçin
                </button>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  data-testid="file-input"
                />
                
                <p className="text-sm text-apple-secondary mt-6">
                  PNG, JPG, JPEG • Max 10MB
                </p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white rounded-3xl shadow-apple-lg overflow-hidden"
            >
              {/* Preview Image */}
              <div className="relative aspect-video bg-gray-100">
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="w-full h-full object-contain"
                />
                {uploading && (
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                    <div className="text-center text-white">
                      <Loader2 className="w-12 h-12 mx-auto mb-3 animate-spin" />
                      <p className="font-medium">Analiz ediliyor...</p>
                      <p className="text-sm text-white/70 mt-1">Bu işlem birkaç saniye sürebilir</p>
                    </div>
                  </div>
                )}
              </div>
              
              {/* File Info & Actions */}
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <Image className="w-5 h-5 text-apple-secondary" />
                    </div>
                    <div>
                      <p className="font-medium text-apple-text truncate max-w-[200px]">
                        {file?.name}
                      </p>
                      <p className="text-sm text-apple-secondary">
                        {(file?.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  
                  {!uploading && (
                    <button
                      onClick={resetUpload}
                      className="text-apple-secondary hover:text-apple-text transition-colors"
                    >
                      Değiştir
                    </button>
                  )}
                </div>
                
                {/* Error Message */}
                {error && (
                  <div className="mb-4 p-4 bg-red-50 rounded-xl flex items-center gap-3 text-apple-error">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <p className="text-sm">{error}</p>
                  </div>
                )}
                
                {/* Action Buttons */}
                <div className="flex gap-3">
                  <button
                    onClick={resetUpload}
                    disabled={uploading}
                    className="flex-1 px-6 py-3 border border-apple-border rounded-full font-medium text-apple-text hover:bg-gray-50 transition-colors disabled:opacity-50"
                  >
                    İptal
                  </button>
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="flex-1 px-6 py-3 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                    data-testid="analyze-button"
                  >
                    {uploading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Analiz Ediliyor
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-5 h-5" />
                        Analizi Başlat
                      </>
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Features */}
        <div className="mt-12 grid grid-cols-3 gap-6">
          {[
            { icon: '🔍', title: 'Hasar Tespiti', desc: '6 farklı hasar türü' },
            { icon: '🚗', title: 'Parça Analizi', desc: '23 araç parçası' },
            { icon: '📊', title: 'Detaylı Rapor', desc: 'PDF indirme' },
          ].map((feature, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.1 }}
              className="text-center p-4"
            >
              <div className="text-3xl mb-2">{feature.icon}</div>
              <h4 className="font-medium text-apple-text">{feature.title}</h4>
              <p className="text-sm text-apple-secondary">{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default UploadPage;
