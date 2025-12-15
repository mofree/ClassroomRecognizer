import React, { useState, useRef } from 'react';
import { Student, ImageQualityReport } from '../types';
import { FaceRecognitionService } from '../services/faceRecognitionService';
import { validateStudentImage } from '../services/geminiService';
import { detectFacesFromBase64 } from '../services/apiService';
import { Camera, CheckCircle, AlertTriangle, Upload, X, Loader2, FolderUp, FileCheck, FileWarning, UserPlus, Layers, Wand2, Eye, Users } from 'lucide-react';
interface StudentRegistryProps {
  students: Student[];
  onAddStudent: (student: Student) => void;
  onRemoveStudent: (id: string) => void;
}

interface BatchLog {
  fileName: string;
  status: 'pending' | 'success' | 'error';
  message: string;
}

const StudentRegistry: React.FC<StudentRegistryProps> = ({ students, onAddStudent, onRemoveStudent }) => {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  
  // View Image State
  const [viewImage, setViewImage] = useState<string | null>(null);

  // Single Mode State
  const [name, setName] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [qualityReport, setQualityReport] = useState<ImageQualityReport | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [isStandardized, setIsStandardized] = useState(false);
  
  // Batch Mode State
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchLogs, setBatchLogs] = useState<BatchLog[]>([]);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  
  // Import/Export State
  const [isImporting, setIsImporting] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const importInputRef = useRef<HTMLInputElement>(null);

  /**
   * Helper: Smart Crop & Standardize
   * Uses canvas transforms to perfectly center the face and prevent aspect ratio distortion.
   */
  const createStandardizedImage = (img: HTMLImageElement, detectionBox: any): string => {
    const canvas = document.createElement('canvas');
    const size = 300; // Standard output size
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) throw new Error("Could not get canvas context");

    // Fill background with a nice slate color (handles edges if image is rotated or small)
    ctx.fillStyle = "#1e293b"; 
    ctx.fillRect(0, 0, size, size);

    const { x, y, width, height } = detectionBox;

    // Calculate Face Center
    const cx = x + width / 2;
    const cy = y + height / 2;

    // Determine scale: Make face occupy 90% of the canvas width (Requested by user for accuracy)
    // Previously 55%. 90% removes almost all background.
    const desiredFaceWidth = size * 0.90;
    const scale = desiredFaceWidth / width;

    // --- Transform Logic ---
    // 1. Move canvas origin to center (size/2, size/2)
    // 2. Scale the context
    // 3. Move origin "back" by the face center coordinate
    // Result: The face center ends up at the canvas center
    ctx.translate(size / 2, size / 2);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);

    // REMOVED ALL FILTERS: Keep original color and lighting for best matching accuracy
    ctx.filter = 'none';
    
    // Draw the entire image (transform handles the cropping/positioning)
    ctx.drawImage(img, 0, 0);

    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.95);
  };

  // --- Single Mode Handlers ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setQualityReport(null);
      setIsStandardized(false);
    }
  };

  const handleStandardize = async () => {
    if (!previewUrl) return;

    setIsAnalyzing(true);
    try {
      // 1. Load Image
      const img = document.createElement('img');
      img.src = previewUrl;
      await new Promise((resolve) => { img.onload = resolve });

      // 2. Detect Face to get Box using backend API
      const base64Data = previewUrl.split(',')[1];
      const detectionResponse = await detectFacesFromBase64(base64Data);
      
      if (!detectionResponse.success || detectionResponse.count === 0) {
        alert("未检测到人脸，或人脸过于模糊，无法标准化。请尝试更换照片。");
        setIsAnalyzing(false);
        return;
      }

      // Use the first detected face
      const face = detectionResponse.faces[0];
      const box = {
        x: face.bbox[0],
        y: face.bbox[1],
        width: face.bbox[2] - face.bbox[0],
        height: face.bbox[3] - face.bbox[1]
      };

      // 3. Process Image (Crop/Resize/Enhance)
      const standardizedDataUrl = createStandardizedImage(img, box);
      setPreviewUrl(standardizedDataUrl);
      setIsStandardized(true);

      // 4. Run Quality Check on the NEW Standardized Image
      const report = await validateStudentImage(standardizedDataUrl);
      setQualityReport(report);

    } catch (err: any) {
      console.error(err);
      alert(`标准化处理失败: ${err.message || '未知错误'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };  const handleRegister = async () => {
    if (!name || !previewUrl || (qualityReport && !qualityReport.isValid)) return;
    
    setIsRegistering(true);
    try {
      // Extract face embedding using backend API
      const base64Data = previewUrl.split(',')[1];
      const detectionResponse = await detectFacesFromBase64(base64Data);
      
      if (!detectionResponse.success || detectionResponse.count === 0) {
        alert("注册失败：无法从处理后的图像中提取特征。");
        setIsRegistering(false);
        return;
      }

      // Use the first detected face's embedding
      const face = detectionResponse.faces[0];
      const descriptor = new Float32Array(face.embedding);
      
      if (descriptor && descriptor.length > 0) {
        const newStudent: Student = {
          id: Date.now().toString(),
          name: name.trim(),
          photoUrl: previewUrl, // Save the standardized image URL
          descriptors: [descriptor],
          createdAt: Date.now()
        };
        onAddStudent(newStudent);
        setName('');
        setImageFile(null);
        setPreviewUrl(null);
        setQualityReport(null);
        setIsStandardized(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      } else {
        alert("注册失败：无法从处理后的图像中提取特征。");
      }
    } catch (err: any) {
      console.error(err);
      alert(`注册失败: ${err.message || '未知错误'}`);
    } finally {
      setIsRegistering(false);
    }
  };  const clearImage = () => {
    setImageFile(null);
    setPreviewUrl(null);
    setQualityReport(null);
    setIsStandardized(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // --- Batch Mode Handlers ---
  const handleBatchSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files) as File[];
      setBatchFiles(files);
      // Initialize logs
      setBatchLogs(files.map(f => ({
        fileName: f.name,
        status: 'pending',
        message: '等待中...'
      })));
    }
  };

  const processBatch = async () => {
    if (batchFiles.length === 0) return;
    setIsBatchProcessing(true);

    // Process sequentially
    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i];
      const studentName = file.name.replace(/\.[^/.]+$/, ""); // Remove extension

      // Update log
      setBatchLogs(prev => {
        const newLogs = [...prev];
        newLogs[i] = { ...newLogs[i], message: '正在处理与注册...', status: 'pending' };
        return newLogs;
      });

      // Yield for UI
      await new Promise(r => setTimeout(r, 50));

      try {
        // Convert file to base64
        const reader = new FileReader();
        const base64Data = await new Promise<string>((resolve, reject) => {
          reader.onload = () => {
            if (typeof reader.result === 'string') {
              resolve(reader.result.split(',')[1]); // Remove data URL prefix
            } else {
              reject(new Error("文件读取失败"));
            }
          };
          reader.onerror = () => reject(new Error("文件读取错误"));
          reader.readAsDataURL(file);
        });

        // 1. Detect faces using backend API
        const detectionResponse = await detectFacesFromBase64(base64Data);

        if (detectionResponse.success && detectionResponse.count > 0) {
          // Use the first detected face
          const face = detectionResponse.faces[0];
          
          // 2. Create standardized image
          const img = document.createElement('img');
          img.src = `data:image/jpeg;base64,${base64Data}`;
          await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = () => reject(new Error("图片加载失败"));
          });
          
          const box = {
            x: face.bbox[0],
            y: face.bbox[1],
            width: face.bbox[2] - face.bbox[0],
            height: face.bbox[3] - face.bbox[1]
          };
          
          const standardizedUrl = createStandardizedImage(img, box);
          
          // 3. Extract descriptor from the detection result
          const descriptor = new Float32Array(face.embedding);

          if (descriptor && descriptor.length > 0) {
            const newStudent: Student = {
              id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
              name: studentName,
              photoUrl: standardizedUrl,
              descriptors: [descriptor],
              createdAt: Date.now()
            };
            onAddStudent(newStudent);
            
            setBatchLogs(prev => {
              const newLogs = [...prev];
              newLogs[i] = { ...newLogs[i], status: 'success', message: '已注册 (自动处理)' };
              return newLogs;
            });
          } else {
            throw new Error("特征提取失败");
          }
        } else {
           setBatchLogs(prev => {
            const newLogs = [...prev];
            newLogs[i] = { ...newLogs[i], status: 'error', message: '未检测到人脸' };
            return newLogs;
          });
        }
      } catch (error: any) {
         console.error(`处理文件 ${file.name} 失败:`, error);
         setBatchLogs(prev => {
            const newLogs = [...prev];
            newLogs[i] = { 
              ...newLogs[i], 
              status: 'error', 
              message: `处理失败: ${error.message || '未知错误'}` 
            };
            return newLogs;
          });
      }
    }
    setIsBatchProcessing(false);
  };
  // Export students data to JSON file
  const handleExport = () => {
    try {
      const dataStr = JSON.stringify(students, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `students_backup_${new Date().toISOString().slice(0, 10)}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
      
      console.log('Students data exported successfully');
    } catch (error) {
      console.error('Failed to export students data:', error);
      alert('导出失败，请查看控制台了解详情。');
    }
  };

  // Import students data from JSON file
  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImportFile(file);
      setIsImporting(true);
      
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          if (event.target?.result) {
            const jsonData = event.target.result as string;
            const importedStudents = JSON.parse(jsonData);
            
            // Validate imported data
            if (Array.isArray(importedStudents)) {
              // Convert descriptors from arrays back to Float32Array
              const validStudents = importedStudents.map((student: any) => ({
                ...student,
                descriptors: student.descriptors.map((desc: number[]) => new Float32Array(desc))
              }));
              
              // Add imported students (replace existing ones or merge?)
              validStudents.forEach((student: any) => {
                // Check if student with same ID already exists
                const existingIndex = students.findIndex(s => s.id === student.id);
                if (existingIndex >= 0) {
                  // Optionally update existing student
                  // For now, we'll skip duplicates
                  console.log(`Skipping duplicate student: ${student.name}`);
                } else {
                  // Add new student
                  onAddStudent(student);
                }
              });
              
              alert(`成功导入 ${validStudents.length} 名学生数据！`);
            } else {
              throw new Error('Invalid data format');
            }
          }
        } catch (error) {
          console.error('Failed to import students data:', error);
          alert('导入失败，请确保选择了有效的JSON文件。');
        } finally {
          setIsImporting(false);
          setImportFile(null);
          if (importInputRef.current) importInputRef.current.value = '';
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header with Import/Export buttons */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Users className="w-5 h-5" /> 学生管理
        </h2>
        <div className="flex gap-2">
          <input 
            type="file" 
            ref={importInputRef} 
            onChange={handleImport} 
            accept=".json" 
            className="hidden" 
          />
          <button 
            onClick={() => importInputRef.current?.click()}
            disabled={isImporting}
            className="flex items-center gap-1 bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isImporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                导入中...
              </>
            ) : (
              <>
                <FolderUp className="w-4 h-4" />
                导入
              </>
            )}
          </button>
          <button 
            onClick={handleExport}
            disabled={students.length === 0}
            className="flex items-center gap-1 bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            <FileCheck className="w-4 h-4" />
            导出
          </button>
        </div>
      </div>

      {/* Full Screen Image Viewer */}
      {viewImage && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setViewImage(null)}
        >
          <button className="absolute top-4 right-4 text-white/50 hover:text-white transition-colors">
            <X className="w-8 h-8" />
          </button>
          <img 
            src={viewImage} 
            alt="Full view" 
            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl ring-1 ring-white/10" 
            onClick={(e) => e.stopPropagation()} 
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Registration Column */}
        <div className="lg:col-span-1 bg-slate-800 p-6 rounded-xl border border-slate-700 h-fit flex flex-col">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Camera className="mr-2 text-purple-400" /> 学生注册
          </h2>

          {/* Mode Toggle */}
          <div className="flex bg-slate-900 p-1 rounded-lg mb-6">
            <button
              onClick={() => setMode('single')}
              className={`flex-1 py-2 text-sm font-medium rounded-md flex items-center justify-center gap-2 transition-all ${
                mode === 'single' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-white'
              }`}
            >
              <UserPlus className="w-4 h-4" /> 单人注册
            </button>
            <button
              onClick={() => setMode('batch')}
              className={`flex-1 py-2 text-sm font-medium rounded-md flex items-center justify-center gap-2 transition-all ${
                mode === 'batch' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-4 h-4" /> 批量导入
            </button>
          </div>

          {mode === 'single' ? (
            /* --- SINGLE MODE --- */
            <div className="space-y-4 animate-in fade-in duration-300">
              <div>
                <label className="block text-sm text-slate-400 mb-1">学生姓名</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white focus:border-purple-500 focus:outline-none"
                  placeholder="例如：张三"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1">基准照片</label>
                <div 
                  className={`border-2 border-dashed border-slate-600 rounded-lg p-4 flex flex-col items-center justify-center relative h-48 transition-colors ${!previewUrl ? 'cursor-pointer hover:bg-slate-750 hover:border-slate-500' : ''}`}
                  onClick={() => !previewUrl && fileInputRef.current?.click()}
                >
                  {previewUrl ? (
                    <div className="relative w-full h-full group">
                      <img 
                        src={previewUrl} 
                        alt="Preview" 
                        className="w-full h-full object-contain rounded cursor-zoom-in" 
                        onClick={(e) => { e.stopPropagation(); setViewImage(previewUrl); }}
                      />
                      {/* Change Image Button Overlay */}
                      <button
                        onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                        className="absolute top-2 right-2 p-2 bg-slate-900/80 hover:bg-blue-600 text-white rounded-full opacity-0 group-hover:opacity-100 transition-all border border-slate-600 shadow-xl z-10"
                        title="更换照片"
                      >
                        <Upload className="w-4 h-4" />
                      </button>
                      
                      {isStandardized && <span className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 text-xs rounded text-white font-mono pointer-events-none">已标准化</span>}
                    </div>
                  ) : (
                    <>
                      <Upload className="text-slate-500 mb-2" />
                      <span className="text-sm text-slate-500">点击上传照片</span>
                    </>
                  )}
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="image/*" 
                  className="hidden" 
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleStandardize}
                  disabled={!previewUrl || isAnalyzing}
                  className={`flex-1 text-white py-2 rounded font-medium text-sm transition-colors flex justify-center items-center ${
                    isStandardized ? 'bg-slate-600' : 'bg-blue-600 hover:bg-blue-500'
                  } disabled:bg-slate-700`}
                >
                  {isAnalyzing ? <Loader2 className="animate-spin w-4 h-4 mr-1"/> : (
                    <>
                      <Wand2 className="w-4 h-4 mr-1" />
                      {isStandardized ? "重新处理" : "一键标准化 (90%)"}
                    </>
                  )}
                </button>
                <button
                  onClick={handleRegister}
                  disabled={!previewUrl || !name || isRegistering}
                  className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-2 rounded font-medium text-sm transition-colors flex justify-center items-center"
                >
                  {isRegistering ? <Loader2 className="animate-spin w-4 h-4 mr-1"/> : "确认注册"}
                </button>
              </div>

              {/* Quality Report Card */}
              {qualityReport && (
                <div className={`mt-4 p-3 rounded text-sm border ${qualityReport.isValid ? 'bg-emerald-900/30 border-emerald-800' : 'bg-red-900/30 border-red-800'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">{qualityReport.isValid ? "检测通过" : "检测失败"}</span>
                    <span className="font-mono text-xs px-2 py-0.5 bg-black/40 rounded">{qualityReport.score}/100</span>
                  </div>
                  <ul className="space-y-1 text-xs opacity-90">
                    {qualityReport.issues.length > 0 ? (
                      qualityReport.issues.map((issue, idx) => (
                        <li key={idx} className="flex items-start">
                          <AlertTriangle className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" /> {issue}
                        </li>
                      ))
                    ) : (
                      <li className="flex items-center"><CheckCircle className="w-3 h-3 mr-1" /> 照片符合标准。</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            /* --- BATCH MODE --- */
            <div className="space-y-4 animate-in fade-in duration-300 flex-1 flex flex-col">
              <div 
                  onClick={() => batchInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-600 rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-750 hover:border-slate-500 transition-colors bg-slate-900/50"
                >
                  <FolderUp className="text-slate-400 w-10 h-10 mb-2" />
                  <span className="text-sm font-medium text-slate-300">批量上传图片</span>
                  <span className="text-xs text-slate-500 mt-1">文件名将自动作为学生姓名，并自动裁剪标准化</span>
              </div>
              <input 
                type="file" 
                ref={batchInputRef} 
                onChange={handleBatchSelect} 
                accept="image/*" 
                multiple
                className="hidden" 
              />

              {batchFiles.length > 0 && (
                <div className="flex-1 flex flex-col min-h-[200px]">
                  <div className="flex justify-between items-center mb-2">
                     <span className="text-sm font-semibold text-slate-400">上传队列 ({batchFiles.length})</span>
                     <button 
                       onClick={processBatch}
                       disabled={isBatchProcessing}
                       className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white text-xs px-3 py-1 rounded"
                     >
                       {isBatchProcessing ? "处理中..." : "开始批量处理"}
                     </button>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto bg-slate-900 rounded-lg p-2 space-y-2 border border-slate-700">
                    {batchLogs.map((log, idx) => (
                      <div key={idx} className="flex items-center justify-between text-xs p-2 bg-slate-800 rounded border border-slate-700/50">
                        <span className="truncate max-w-[150px] text-slate-300" title={log.fileName}>{log.fileName}</span>
                        <div className="flex items-center gap-2">
                          <span className={`
                            ${log.status === 'success' ? 'text-emerald-400' : ''}
                            ${log.status === 'error' ? 'text-red-400' : ''}
                            ${log.status === 'pending' ? 'text-slate-500' : ''}
                          `}>
                            {log.status === 'success' && <FileCheck className="w-3 h-3" />}
                            {log.status === 'error' && <FileWarning className="w-3 h-3" />}
                            {log.status === 'pending' && log.message.includes('Processing') && <Loader2 className="w-3 h-3 animate-spin" />}
                          </span>
                          <span className="text-slate-500 w-24 text-right truncate">{log.message}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Roster List */}
        <div className="lg:col-span-2 bg-slate-800 p-6 rounded-xl border border-slate-700 overflow-hidden flex flex-col">
          <h2 className="text-xl font-bold mb-4">已注册学生名单 ({students.length})</h2>
          <div className="overflow-y-auto flex-1 pr-2 space-y-2">
            {students.length === 0 && (
              <p className="text-slate-500 text-center mt-10">暂无已注册学生。</p>
            )}
            {students.map(student => (
              <div key={student.id} className="flex items-center justify-between bg-slate-900 p-3 rounded-lg border border-slate-700">
                <div className="flex items-center gap-3">
                  <div className="relative group/avatar">
                    <img 
                      src={student.photoUrl} 
                      alt={student.name} 
                      className="w-12 h-12 rounded-full object-cover border border-slate-600 cursor-zoom-in transition-all" 
                      onClick={() => setViewImage(student.photoUrl)}
                    />
                    <div className="absolute inset-0 bg-black/30 rounded-full flex items-center justify-center opacity-0 group-hover/avatar:opacity-100 pointer-events-none">
                      <Eye className="w-4 h-4 text-white" />
                    </div>
                  </div>
                  <div>
                    <h3 className="font-medium text-white">{student.name}</h3>
                    <p className="text-xs text-slate-500">ID: {student.id.slice(-6)}</p>
                  </div>
                </div>
                <button 
                  onClick={() => onRemoveStudent(student.id)}
                  className="text-slate-500 hover:text-red-400 p-2"
                  title="移除学生"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentRegistry;