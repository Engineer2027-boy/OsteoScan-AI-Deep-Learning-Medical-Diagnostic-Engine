import React, { useState } from 'react';
import { Upload, Activity, AlertCircle, CheckCircle, Eye, RefreshCw, FileText } from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Inference API Error');

      const data = await response.json();
      setResults(data);
    } catch (error) {
      alert("Failed to connect to FastAPI backend at http://localhost:8000");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* HEADER */}
        <header className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide">OsteoScan AI</h1>
              <p className="text-xs text-slate-400">Radiological Fracture Diagnostics & Decision Support</p>
            </div>
          </div>
          <span className="text-xs bg-slate-900 border border-slate-800 text-slate-400 px-3 py-1 rounded-full">
            DenseNet-121 + Grad-CAM Engine
          </span>
        </header>

        {/* WORKSPACE GRID */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* UPLOAD PANEL */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <Upload className="w-4 h-4" /> Radiograph Input
              </h2>

              <label className="border-2 border-dashed border-slate-700 hover:border-indigo-500 bg-slate-950/50 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all group">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                <Upload className="w-8 h-8 text-slate-500 group-hover:text-indigo-400 transition-colors mb-2" />
                <span className="text-sm font-medium text-slate-300">Click to upload X-Ray scan</span>
                <span className="text-xs text-slate-500 mt-1">PNG, JPG, DICOM exports supported</span>
              </label>

              {previewUrl && (
                <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-black aspect-square flex items-center justify-center">
                  <img src={previewUrl} alt="X-Ray Preview" className="max-h-full object-contain" />
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 font-semibold text-sm rounded-lg transition-all flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" /> Running PyTorch Tensors...
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" /> Run Diagnostic Inference
                  </>
                )}
              </button>
            </div>
          </div>

          {/* RESULTS PANEL */}
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-5 h-full flex flex-col">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                  <Eye className="w-4 h-4" /> Inference Output & Interpretability
                </h2>
                {results && (
                  <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-colors"
                  >
                    <Eye className="w-3.5 h-3.5" /> {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
                  </button>
                )}
              </div>

              {results ? (
                <div className="space-y-5 flex-1 flex flex-col justify-between">
                  <div className={`p-4 rounded-xl border flex items-center justify-between ${
                    results.fracture_detected 
                      ? 'bg-rose-950/30 border-rose-800 text-rose-200' 
                      : 'bg-emerald-950/30 border-emerald-800 text-emerald-200'
                  }`}>
                    <div className="flex items-center gap-3">
                      {results.fracture_detected ? (
                        <AlertCircle className="w-7 h-7 text-rose-400" />
                      ) : (
                        <CheckCircle className="w-7 h-7 text-emerald-400" />
                      )}
                      <div>
                        <h3 className="font-bold text-lg">
                          {results.fracture_detected ? "Structural Fracture Detected" : "No Abnormality Detected"}
                        </h3>
                        <p className="text-xs opacity-80">Classification score derived via DenseNet-121.</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-black">{results.confidence}%</span>
                      <p className="text-[10px] uppercase tracking-wider opacity-70">Confidence</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 my-auto">
                    <div className="space-y-1.5">
                      <p className="text-xs text-slate-400 font-medium">Original X-Ray</p>
                      <div className="aspect-square bg-black border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center">
                        <img src={previewUrl} alt="Original" className="max-h-full object-contain" />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <p className="text-xs text-slate-400 font-medium">Grad-CAM Heatmap Overlay</p>
                      <div className="aspect-square bg-black border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center">
                        <img 
                          src={showHeatmap ? results.heatmap_base64 : previewUrl} 
                          alt="Grad-CAM Output" 
                          className="max-h-full object-contain" 
                        />
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-center">
                    <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
                      <p className="text-[10px] text-slate-500 uppercase">Latency</p>
                      <p className="text-sm font-semibold text-slate-300">{results.inference_time_ms} ms</p>
                    </div>
                    <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
                      <p className="text-[10px] text-slate-500 uppercase">Pre-processing</p>
                      <p className="text-sm font-semibold text-slate-300">OpenCV CLAHE</p>
                    </div>
                    <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
                      <p className="text-[10px] text-slate-500 uppercase">Architecture</p>
                      <p className="text-sm font-semibold text-slate-300">DenseNet-121</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border border-dashed border-slate-800 rounded-xl bg-slate-950/30">
                  <FileText className="w-10 h-10 text-slate-700 mb-3" />
                  <p className="text-sm text-slate-400 font-medium">Awaiting X-Ray Input</p>
                  <p className="text-xs text-slate-600 max-w-xs mt-1">
                    Upload an X-ray image on the left to trigger deep inference and Grad-CAM visualization.
                  </p>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
