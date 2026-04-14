import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeftRight, Upload, Loader2, AlertTriangle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const ComparePage = () => {
  const [mode, setMode] = useState('upload');
  const [beforeFile, setBeforeFile] = useState(null);
  const [afterFile, setAfterFile] = useState(null);
  const [beforePreview, setBeforePreview] = useState(null);
  const [afterPreview, setAfterPreview] = useState(null);
  const [analyses, setAnalyses] = useState([]);
  const [beforeId, setBeforeId] = useState('');
  const [afterId, setAfterId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fetchAnalyses = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/api/analyses?limit=50`);
      setAnalyses(res.data);
    } catch (_err) {
      // silent
    }
  }, []);

  useEffect(() => {
    if (mode === 'select') {
      fetchAnalyses();
    }
  }, [mode, fetchAnalyses]);

  const handleFileSelect = (type, file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (type === 'before') {
        setBeforeFile(file);
        setBeforePreview(e.target.result);
      } else {
        setAfterFile(file);
        setAfterPreview(e.target.result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;
      if (mode === 'upload') {
        if (!beforeFile || !afterFile) {
          setError('Her iki gorseli de secin');
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append('before_file', beforeFile);
        formData.append('after_file', afterFile);
        response = await axios.post(`${API_URL}/api/compare/upload`, formData, { timeout: 180000 });
      } else {
        if (!beforeId || !afterId) {
          setError('Her iki analizi de secin');
          setLoading(false);
          return;
        }
        response = await axios.post(`${API_URL}/api/compare`, {
          before_id: beforeId,
          after_id: afterId
        }, { timeout: 180000 });
      }
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Karsilastirma yapilamadi');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-apple-text mb-2" data-testid="compare-title">Before / After Karsilastirma</h1>
        <p className="text-apple-secondary mb-8">Teslim oncesi ve sonrasi gorselleri karsilastirarak yeni hasarlari tespit edin</p>
      </motion.div>

      {/* Mode Toggle */}
      <div className="flex gap-3 mb-8" data-testid="mode-toggle">
        <button
          onClick={() => { setMode('upload'); setResult(null); }}
          className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all ${mode === 'upload' ? 'bg-black text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          data-testid="mode-upload"
        >
          Gorsel Yukle
        </button>
        <button
          onClick={() => { setMode('select'); setResult(null); }}
          className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all ${mode === 'select' ? 'bg-black text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          data-testid="mode-select"
        >
          Mevcut Analizden Sec
        </button>
      </div>

      {!result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Before */}
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-apple p-6">
            <h3 className="font-semibold text-apple-text mb-4">Onceki (Before)</h3>
            {mode === 'upload' ? (
              <div>
                <label className="block w-full cursor-pointer">
                  <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${beforePreview ? 'border-green-300 bg-green-50' : 'border-gray-200 hover:border-gray-400'}`}>
                    {beforePreview ? (
                      <img src={beforePreview} alt="Before" className="max-h-48 mx-auto rounded-lg" />
                    ) : (
                      <>
                        <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                        <p className="text-sm text-gray-500">Teslim oncesi gorseli secin</p>
                      </>
                    )}
                  </div>
                  <input type="file" accept="image/*" className="hidden"
                    data-testid="before-file-input"
                    onChange={(e) => handleFileSelect('before', e.target.files[0])} />
                </label>
              </div>
            ) : (
              <select value={beforeId} onChange={(e) => setBeforeId(e.target.value)}
                data-testid="before-select"
                className="w-full p-3 border rounded-xl text-sm">
                <option value="">Analiz secin...</option>
                {analyses.map(a => (
                  <option key={a.id} value={a.id}>
                    {a.filename} - {new Date(a.created_at).toLocaleDateString('tr-TR')} ({a.summary.total_damages} hasar)
                  </option>
                ))}
              </select>
            )}
          </motion.div>

          {/* After */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-apple p-6">
            <h3 className="font-semibold text-apple-text mb-4">Sonraki (After)</h3>
            {mode === 'upload' ? (
              <div>
                <label className="block w-full cursor-pointer">
                  <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${afterPreview ? 'border-green-300 bg-green-50' : 'border-gray-200 hover:border-gray-400'}`}>
                    {afterPreview ? (
                      <img src={afterPreview} alt="After" className="max-h-48 mx-auto rounded-lg" />
                    ) : (
                      <>
                        <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                        <p className="text-sm text-gray-500">Teslim sonrasi gorseli secin</p>
                      </>
                    )}
                  </div>
                  <input type="file" accept="image/*" className="hidden"
                    data-testid="after-file-input"
                    onChange={(e) => handleFileSelect('after', e.target.files[0])} />
                </label>
              </div>
            ) : (
              <select value={afterId} onChange={(e) => setAfterId(e.target.value)}
                data-testid="after-select"
                className="w-full p-3 border rounded-xl text-sm">
                <option value="">Analiz secin...</option>
                {analyses.map(a => (
                  <option key={a.id} value={a.id}>
                    {a.filename} - {new Date(a.created_at).toLocaleDateString('tr-TR')} ({a.summary.total_damages} hasar)
                  </option>
                ))}
              </select>
            )}
          </motion.div>
        </div>
      )}

      {/* Compare Button */}
      {!result && (
        <div className="text-center mb-8">
          <button onClick={handleCompare} disabled={loading}
            data-testid="compare-button"
            className="px-8 py-3 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 inline-flex items-center gap-2">
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Karsilastiriliyor...
              </>
            ) : (
              <>
                <ArrowLeftRight className="w-5 h-5" />
                Karsilastir
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 mb-8" data-testid="compare-error">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {/* Verdict Banner */}
          <div className={`p-6 rounded-2xl border-2 ${
            result.has_new_damage
              ? 'bg-red-50 border-red-200'
              : 'bg-green-50 border-green-200'
          }`} data-testid="verdict-banner">
            <div className="flex items-center gap-3 mb-2">
              {result.has_new_damage ? (
                <AlertTriangle className="w-7 h-7 text-red-600" />
              ) : (
                <CheckCircle className="w-7 h-7 text-green-600" />
              )}
              <h2 className={`text-xl font-bold ${result.has_new_damage ? 'text-red-800' : 'text-green-800'}`}>
                {result.verdict}
              </h2>
            </div>
            <p className={`text-sm ${result.has_new_damage ? 'text-red-600' : 'text-green-600'}`}>
              Karar guveni: {result.verdict_confidence}
            </p>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl shadow-apple p-4 text-center" data-testid="summary-before">
              <p className="text-sm text-apple-secondary">Onceki Hasar</p>
              <p className="text-2xl font-bold text-apple-text">{result.summary?.before_damage_count || 0}</p>
            </div>
            <div className="bg-white rounded-xl shadow-apple p-4 text-center" data-testid="summary-after">
              <p className="text-sm text-apple-secondary">Sonraki Hasar</p>
              <p className="text-2xl font-bold text-apple-text">{result.summary?.after_damage_count || 0}</p>
            </div>
            <div className="bg-white rounded-xl shadow-apple p-4 text-center" data-testid="summary-new">
              <p className="text-sm text-apple-secondary">Yeni Hasar</p>
              <p className={`text-2xl font-bold ${result.new_damage_count > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {result.new_damage_count}
              </p>
            </div>
            <div className="bg-white rounded-xl shadow-apple p-4 text-center" data-testid="summary-change">
              <p className="text-sm text-apple-secondary">Degisim</p>
              <p className="text-2xl font-bold text-apple-text">%{result.difference?.change_percentage || 0}</p>
            </div>
          </div>

          {/* New Damages List */}
          {result.new_damages && result.new_damages.length > 0 && (
            <div className="bg-white rounded-2xl shadow-apple p-6" data-testid="new-damages-list">
              <h3 className="font-semibold text-apple-text mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Yeni Tespit Edilen Hasarlar
              </h3>
              <div className="space-y-3">
                {result.new_damages.map((dmg, i) => (
                  <div key={i} className="p-4 bg-red-50 rounded-xl border border-red-100">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium text-red-800">{dmg.type_tr}</span>
                        {dmg.part_tr && (
                          <span className="text-sm text-red-600 ml-2">{dmg.part_tr}</span>
                        )}
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        dmg.evidence_strength === 'Yuksek'
                          ? 'bg-red-200 text-red-800'
                          : 'bg-amber-200 text-amber-800'
                      }`}>
                        Kanit: {dmg.evidence_strength}
                      </span>
                    </div>
                    <p className="text-xs text-red-500 mt-1">
                      Guven: %{dmg.confidence} | Siddet: {dmg.severity}/5
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Alignment Info */}
          <div className="bg-white rounded-2xl shadow-apple p-6" data-testid="alignment-info">
            <h3 className="font-semibold text-apple-text mb-3">Teknik Detaylar</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-apple-secondary">Hizalama:</span>
                <span className="ml-2 font-medium">{result.alignment?.used ? 'Basarili' : 'Uygulanamadi'}</span>
              </div>
              <div>
                <span className="text-apple-secondary">Eslesen ozellik:</span>
                <span className="ml-2 font-medium">{result.alignment?.good_matches || 'N/A'}</span>
              </div>
              <div>
                <span className="text-apple-secondary">Degisim bolgeleri:</span>
                <span className="ml-2 font-medium">{result.change_region_count || 0}</span>
              </div>
              <div>
                <span className="text-apple-secondary">Degisim orani:</span>
                <span className="ml-2 font-medium">%{result.difference?.change_percentage || 0}</span>
              </div>
            </div>
          </div>

          {/* New Comparison Button */}
          <div className="text-center">
            <button onClick={() => { setResult(null); setError(null); }}
              data-testid="new-compare-button"
              className="px-6 py-2.5 bg-gray-100 text-gray-700 rounded-full font-medium hover:bg-gray-200 transition-colors">
              Yeni Karsilastirma
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ComparePage;
