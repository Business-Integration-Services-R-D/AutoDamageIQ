import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Upload, Tag, Play, BarChart3, 
  Plus, Trash2, Save, RefreshCw, CheckCircle, 
  AlertCircle, Loader2, ZoomIn, ZoomOut, Move
} from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

// Hasar sınıfları
const DAMAGE_CLASSES = {
  0: { en: 'crack', tr: 'Çatlak', color: '#FF6B6B' },
  1: { en: 'dent', tr: 'Göçük', color: '#4ECDC4' },
  2: { en: 'glass_shatter', tr: 'Cam Kırığı', color: '#45B7D1' },
  3: { en: 'lamp_broken', tr: 'Lamba Kırığı', color: '#96CEB4' },
  4: { en: 'scratch', tr: 'Çizik', color: '#FFEAA7' },
  5: { en: 'tire_flat', tr: 'Patlak Lastik', color: '#DDA0DD' },
};

const TrainingPage = () => {
  const [activeTab, setActiveTab] = useState('stats');
  const [stats, setStats] = useState(null);
  const [customStats, setCustomStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Annotation state
  const [annotationImage, setAnnotationImage] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [selectedClass, setSelectedClass] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentBox, setCurrentBox] = useState(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Training state
  const [trainingConfig, setTrainingConfig] = useState({
    model_size: 'm',
    epochs: 200,
    batch_size: 16,
    image_size: 640,
    include_custom: true
  });
  const [trainingStatus, setTrainingStatus] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const [unifiedRes, customRes] = await Promise.all([
        axios.get(`${API_URL}/api/training/unified-stats`),
        axios.get(`${API_URL}/api/training/stats`)
      ]);
      setStats(unifiedRes.data);
      setCustomStats(customRes.data);
    } catch (err) {
      setError('İstatistikler yüklenemedi');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Image upload for annotation
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/api/training/upload`, formData);
      
      // Read image for canvas
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setAnnotationImage({
            id: response.data.id,
            src: event.target.result,
            width: img.width,
            height: img.height
          });
          setBoxes([]);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError('Görsel yüklenemedi');
    }
  };

  // Canvas drawing
  const handleCanvasMouseDown = (e) => {
    if (!annotationImage) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = annotationImage.width / rect.width;
    const scaleY = annotationImage.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX / annotationImage.width;
    const y = (e.clientY - rect.top) * scaleY / annotationImage.height;
    
    setIsDrawing(true);
    setCurrentBox({ x, y, width: 0, height: 0, class_id: selectedClass });
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDrawing || !currentBox) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = annotationImage.width / rect.width;
    const scaleY = annotationImage.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX / annotationImage.width;
    const y = (e.clientY - rect.top) * scaleY / annotationImage.height;
    
    setCurrentBox(prev => ({
      ...prev,
      width: x - prev.x,
      height: y - prev.y
    }));
  };

  const handleCanvasMouseUp = () => {
    if (currentBox && Math.abs(currentBox.width) > 0.01 && Math.abs(currentBox.height) > 0.01) {
      // Normalize box (handle negative dimensions)
      const normalizedBox = {
        x: currentBox.width < 0 ? currentBox.x + currentBox.width : currentBox.x,
        y: currentBox.height < 0 ? currentBox.y + currentBox.height : currentBox.y,
        width: Math.abs(currentBox.width),
        height: Math.abs(currentBox.height),
        class_id: currentBox.class_id
      };
      setBoxes(prev => [...prev, normalizedBox]);
    }
    setIsDrawing(false);
    setCurrentBox(null);
  };

  // Draw boxes on canvas
  useEffect(() => {
    if (!canvasRef.current || !annotationImage) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    
    // Clear and draw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    
    // Draw existing boxes
    boxes.forEach((box, idx) => {
      const color = DAMAGE_CLASSES[box.class_id]?.color || '#FF0000';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        box.x * canvas.width,
        box.y * canvas.height,
        box.width * canvas.width,
        box.height * canvas.height
      );
      
      // Label
      ctx.fillStyle = color;
      ctx.font = '12px Arial';
      ctx.fillText(
        DAMAGE_CLASSES[box.class_id]?.tr || 'Bilinmeyen',
        box.x * canvas.width + 2,
        box.y * canvas.height - 4
      );
    });
    
    // Draw current box
    if (currentBox) {
      const color = DAMAGE_CLASSES[currentBox.class_id]?.color || '#FF0000';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        currentBox.x * canvas.width,
        currentBox.y * canvas.height,
        currentBox.width * canvas.width,
        currentBox.height * canvas.height
      );
      ctx.setLineDash([]);
    }
  }, [annotationImage, boxes, currentBox]);

  // Save annotation - Fixed to use correct endpoint
  const handleSaveAnnotation = async () => {
    if (!annotationImage || boxes.length === 0) {
      setError('Lütfen en az bir kutu çizin');
      return;
    }

    try {
      await axios.post(`${API_URL}/api/training/annotate/save`, {
        image_id: annotationImage.id,
        boxes: boxes,
        source: 'user'
      });
      
      setAnnotationImage(null);
      setBoxes([]);
      fetchStats();
      alert('Etiket kaydedildi!');
    } catch (err) {
      console.error('Save error:', err);
      setError('Etiket kaydedilemedi: ' + (err.response?.data?.detail || err.message));
    }
  };

  // Delete last box
  const handleDeleteLastBox = () => {
    setBoxes(prev => prev.slice(0, -1));
  };

  // Start training
  const handleStartTraining = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/training/start`, trainingConfig);
      setTrainingStatus(response.data);
      alert('Eğitim başlatıldı!');
    } catch (err) {
      setError('Eğitim başlatılamadı');
    }
  };

  const tabs = [
    { id: 'stats', label: 'İstatistikler', icon: BarChart3 },
    { id: 'annotate', label: 'Etiketleme', icon: Tag },
    { id: 'train', label: 'Eğitim', icon: Brain },
    { id: 'models', label: 'Modeller', icon: Brain },
  ];

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gray-50 py-8 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-apple-text mb-2">Eğitim Merkezi</h1>
          <p className="text-apple-secondary">Model eğitimi ve veri etiketleme</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-black text-white'
                  : 'bg-white text-apple-secondary hover:bg-gray-100'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 text-red-600 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
            <button onClick={() => setError(null)} className="ml-auto">×</button>
          </div>
        )}

        {/* Content */}
        <AnimatePresence mode="wait">
          {/* Stats Tab */}
          {activeTab === 'stats' && (
            <motion.div
              key="stats"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
              {/* Unified Dataset Stats */}
              <div className="bg-white rounded-2xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Birleşik Veri Seti
                </h3>
                {loading ? (
                  <Loader2 className="w-6 h-6 animate-spin mx-auto" />
                ) : stats ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="text-2xl font-bold">{stats.train_images}</div>
                        <div className="text-sm text-apple-secondary">Eğitim Görseli</div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="text-2xl font-bold">{stats.val_images}</div>
                        <div className="text-sm text-apple-secondary">Doğrulama Görseli</div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-2">Sınıf Dağılımı</h4>
                      {Object.entries(stats.class_distribution).map(([classId, count]) => {
                        const cls = DAMAGE_CLASSES[parseInt(classId)];
                        const maxCount = Math.max(...Object.values(stats.class_distribution));
                        return (
                          <div key={classId} className="flex items-center gap-2 mb-2">
                            <div 
                              className="w-3 h-3 rounded-full" 
                              style={{ backgroundColor: cls?.color }}
                            />
                            <span className="w-24 text-sm">{cls?.tr}</span>
                            <div className="flex-1 bg-gray-100 rounded-full h-2">
                              <div 
                                className="h-2 rounded-full" 
                                style={{ 
                                  width: `${(count / maxCount) * 100}%`,
                                  backgroundColor: cls?.color 
                                }}
                              />
                            </div>
                            <span className="text-sm text-apple-secondary w-12 text-right">{count}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : null}
              </div>

              {/* Custom Dataset Stats */}
              <div className="bg-white rounded-2xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Tag className="w-5 h-5" />
                  Kullanıcı Verileri
                </h3>
                {loading ? (
                  <Loader2 className="w-6 h-6 animate-spin mx-auto" />
                ) : customStats ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="text-2xl font-bold">{customStats.train_images}</div>
                        <div className="text-sm text-apple-secondary">Etiketli</div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="text-2xl font-bold">{customStats.pending_images}</div>
                        <div className="text-sm text-apple-secondary">Bekleyen</div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="text-2xl font-bold">{customStats.total_annotations}</div>
                        <div className="text-sm text-apple-secondary">Toplam Kutu</div>
                      </div>
                    </div>
                    
                    <button
                      onClick={fetchStats}
                      className="w-full py-2 border border-apple-border rounded-lg text-apple-secondary hover:bg-gray-50 flex items-center justify-center gap-2"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Yenile
                    </button>
                  </div>
                ) : null}
              </div>
            </motion.div>
          )}

          {/* Annotation Tab */}
          {activeTab === 'annotate' && (
            <motion.div
              key="annotate"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-white rounded-2xl p-6 shadow-sm"
            >
              <h3 className="text-lg font-semibold mb-4">Görsel Etiketleme</h3>
              
              {!annotationImage ? (
                <div className="border-2 border-dashed border-gray-200 rounded-xl p-12 text-center">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-apple-secondary" />
                  <p className="text-apple-secondary mb-4">Etiketlemek için görsel yükleyin</p>
                  <label className="px-6 py-3 bg-black text-white rounded-full cursor-pointer hover:bg-gray-800">
                    Görsel Seç
                    <input 
                      type="file" 
                      accept="image/*" 
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                  </label>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                  {/* Canvas */}
                  <div className="lg:col-span-3">
                    <div className="relative bg-gray-100 rounded-xl overflow-hidden">
                      <img
                        ref={imageRef}
                        src={annotationImage.src}
                        alt="Annotation"
                        className="hidden"
                        onLoad={() => {
                          const canvas = canvasRef.current;
                          if (canvas) {
                            canvas.width = annotationImage.width;
                            canvas.height = annotationImage.height;
                          }
                        }}
                      />
                      <canvas
                        ref={canvasRef}
                        className="w-full cursor-crosshair"
                        style={{ maxHeight: '500px', objectFit: 'contain' }}
                        onMouseDown={handleCanvasMouseDown}
                        onMouseMove={handleCanvasMouseMove}
                        onMouseUp={handleCanvasMouseUp}
                        onMouseLeave={handleCanvasMouseUp}
                      />
                    </div>
                    
                    {/* Actions */}
                    <div className="flex gap-2 mt-4">
                      <button
                        onClick={handleDeleteLastBox}
                        disabled={boxes.length === 0}
                        className="flex items-center gap-2 px-4 py-2 border border-apple-border rounded-lg hover:bg-gray-50 disabled:opacity-50"
                      >
                        <Trash2 className="w-4 h-4" />
                        Son Kutuyu Sil
                      </button>
                      <button
                        onClick={() => { setAnnotationImage(null); setBoxes([]); }}
                        className="flex items-center gap-2 px-4 py-2 border border-apple-border rounded-lg hover:bg-gray-50"
                      >
                        İptal
                      </button>
                      <button
                        onClick={handleSaveAnnotation}
                        disabled={boxes.length === 0}
                        className="flex items-center gap-2 px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 ml-auto"
                      >
                        <Save className="w-4 h-4" />
                        Kaydet ({boxes.length} kutu)
                      </button>
                    </div>
                  </div>
                  
                  {/* Class Selector */}
                  <div className="space-y-4">
                    <h4 className="font-medium">Hasar Tipi Seç</h4>
                    {Object.entries(DAMAGE_CLASSES).map(([id, cls]) => (
                      <button
                        key={id}
                        onClick={() => setSelectedClass(parseInt(id))}
                        className={`w-full flex items-center gap-3 p-3 rounded-lg border-2 transition-colors ${
                          selectedClass === parseInt(id)
                            ? 'border-black bg-gray-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div 
                          className="w-4 h-4 rounded-full" 
                          style={{ backgroundColor: cls.color }}
                        />
                        <span>{cls.tr}</span>
                      </button>
                    ))}
                    
                    {/* Current boxes */}
                    {boxes.length > 0 && (
                      <div className="mt-6">
                        <h4 className="font-medium mb-2">Çizilen Kutular</h4>
                        <div className="space-y-2">
                          {boxes.map((box, idx) => (
                            <div 
                              key={idx}
                              className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg text-sm"
                            >
                              <div 
                                className="w-3 h-3 rounded-full" 
                                style={{ backgroundColor: DAMAGE_CLASSES[box.class_id]?.color }}
                              />
                              {DAMAGE_CLASSES[box.class_id]?.tr}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Training Tab */}
          {activeTab === 'train' && (
            <motion.div
              key="train"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
              {/* Training Config */}
              <div className="bg-white rounded-2xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Eğitim Ayarları
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Model Boyutu</label>
                    <select 
                      value={trainingConfig.model_size}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, model_size: e.target.value }))}
                      className="w-full p-2 border border-gray-200 rounded-lg"
                    >
                      <option value="n">Nano (En Hızlı)</option>
                      <option value="s">Small</option>
                      <option value="m">Medium</option>
                      <option value="l">Large (Önerilen)</option>
                      <option value="x">XLarge (En İyi)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Epoch Sayısı</label>
                    <input 
                      type="number"
                      value={trainingConfig.epochs}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                      className="w-full p-2 border border-gray-200 rounded-lg"
                      min="10"
                      max="500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Batch Size</label>
                    <input 
                      type="number"
                      value={trainingConfig.batch_size}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                      className="w-full p-2 border border-gray-200 rounded-lg"
                      min="1"
                      max="64"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Görsel Boyutu</label>
                    <select 
                      value={trainingConfig.image_size}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, image_size: parseInt(e.target.value) }))}
                      className="w-full p-2 border border-gray-200 rounded-lg"
                    >
                      <option value="416">416px</option>
                      <option value="640">640px (Önerilen)</option>
                      <option value="800">800px</option>
                      <option value="1024">1024px</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <input 
                      type="checkbox"
                      id="include_custom"
                      checked={trainingConfig.include_custom}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, include_custom: e.target.checked }))}
                      className="w-4 h-4"
                    />
                    <label htmlFor="include_custom" className="text-sm">Kullanıcı verilerini dahil et</label>
                  </div>
                  
                  <button
                    onClick={handleStartTraining}
                    className="w-full py-3 bg-black text-white rounded-lg hover:bg-gray-800 flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    Eğitimi Başlat
                  </button>
                </div>
              </div>

              {/* Training Status */}
              <div className="bg-white rounded-2xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Eğitim Durumu</h3>
                
                {trainingStatus ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      {trainingStatus.status === 'running' ? (
                        <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                      ) : trainingStatus.status === 'completed' ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-yellow-500" />
                      )}
                      <span className="font-medium">{trainingStatus.job_id}</span>
                    </div>
                    <div className="text-sm text-apple-secondary">
                      Durum: {trainingStatus.status}
                    </div>
                    {trainingStatus.message && (
                      <div className="p-3 bg-gray-50 rounded-lg text-sm">
                        {trainingStatus.message}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-apple-secondary py-8">
                    <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Henüz eğitim başlatılmadı</p>
                  </div>
                )}
                
                {/* RunPod Info */}
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-800 mb-2">RunPod GPU Eğitimi</h4>
                  <p className="text-sm text-blue-600">
                    Eğitim RunPod serverless GPU üzerinde çalışacak. 
                    Tahmini süre: ~2-4 saat (veri seti boyutuna göre değişir)
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Models Tab */}
          {activeTab === 'models' && (
            <ModelsTab API_URL={API_URL} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Models Tab Component
const ModelsTab = ({ API_URL }) => {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const [modelsRes, currentRes] = await Promise.all([
        axios.get(`${API_URL}/api/models`),
        axios.get(`${API_URL}/api/models/current`)
      ]);
      setModels(modelsRes.data.models || []);
      setCurrentModel(currentRes.data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleActivate = async (modelId) => {
    try {
      await axios.post(`${API_URL}/api/models/${modelId}/activate`);
      fetchModels();
      alert('Model aktif edildi!');
    } catch (err) {
      alert('Model aktif edilemedi');
    }
  };

  const handleDelete = async (modelId) => {
    if (!window.confirm('Bu modeli silmek istediğinizden emin misiniz?')) return;
    
    try {
      await axios.delete(`${API_URL}/api/models/${modelId}`);
      fetchModels();
    } catch (err) {
      alert('Model silinemedi');
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <motion.div
      key="models"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      {/* Current Model */}
      {currentModel && (
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-200">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="w-6 h-6 text-green-600" />
            <h3 className="text-lg font-semibold text-green-800">Aktif Model</h3>
          </div>
          <p className="text-green-700 font-medium">{currentModel.name}</p>
          <p className="text-sm text-green-600 mt-1">{currentModel.description}</p>
        </div>
      )}

      {/* Model List */}
      <div className="bg-white rounded-2xl p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Tüm Modeller
        </h3>

        <div className="space-y-4">
          {models.length === 0 ? (
            <p className="text-apple-secondary text-center py-8">Henüz model yok</p>
          ) : (
            models.map(model => (
              <div 
                key={model.id}
                className={`p-4 rounded-xl border-2 transition-colors ${
                  model.is_active 
                    ? 'border-green-500 bg-green-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium">{model.name}</h4>
                      <span className={`px-2 py-0.5 text-xs rounded-full ${
                        model.type === 'default' 
                          ? 'bg-blue-100 text-blue-700' 
                          : 'bg-purple-100 text-purple-700'
                      }`}>
                        {model.type === 'default' ? 'Varsayılan' : 'Eğitilmiş'}
                      </span>
                      {model.is_active && (
                        <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-700">
                          Aktif
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-apple-secondary mt-1">{model.description}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Oluşturulma: {new Date(model.created_at).toLocaleDateString('tr-TR')}
                    </p>
                    
                    {/* Metrics */}
                    {model.metrics && Object.keys(model.metrics).length > 0 && (
                      <div className="flex gap-4 mt-2">
                        {model.metrics.mAP50 && (
                          <span className="text-sm">
                            <span className="text-apple-secondary">mAP50:</span>{' '}
                            <span className="font-medium">{(model.metrics.mAP50 * 100).toFixed(1)}%</span>
                          </span>
                        )}
                        {model.metrics.precision && (
                          <span className="text-sm">
                            <span className="text-apple-secondary">Precision:</span>{' '}
                            <span className="font-medium">{(model.metrics.precision * 100).toFixed(1)}%</span>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    {!model.is_active && (
                      <button
                        onClick={() => handleActivate(model.id)}
                        className="px-3 py-1.5 text-sm bg-black text-white rounded-lg hover:bg-gray-800"
                      >
                        Aktif Et
                      </button>
                    )}
                    {model.type === 'trained' && (
                      <button
                        onClick={() => handleDelete(model.id)}
                        className="px-3 py-1.5 text-sm border border-red-200 text-red-600 rounded-lg hover:bg-red-50"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 rounded-xl p-4">
        <h4 className="font-medium text-blue-800 mb-2">Model Seçimi Hakkında</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• <strong>Varsayılan modeller</strong>: Önceden eğitilmiş, hemen kullanılabilir</li>
          <li>• <strong>Eğitilmiş modeller</strong>: Sizin verilerinizle özelleştirilmiş</li>
          <li>• Aktif model, analiz yaparken kullanılacak olan modeldir</li>
          <li>• Yeni eğitim tamamlandığında otomatik olarak listeye eklenir</li>
        </ul>
      </div>
    </motion.div>
  );
};

export default TrainingPage;
