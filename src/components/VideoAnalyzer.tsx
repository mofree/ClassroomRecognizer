import React, { useState, useRef, useEffect } from 'react';
import { RecognitionParams, Student, BehaviorReport, SingleStudentBehaviorReport } from '../types';
import { analyzeClassroomBehavior, analyzeStudentBehavior } from '../services/geminiService';
import { Play, Pause, Upload, Video as VideoIcon, Loader, Timer, Gauge, Camera, Download, XCircle, MousePointer2, Eraser, Hourglass, CheckCheck, BrainCircuit, Activity, X, UserSearch, ScanEye, SquareDashed } from 'lucide-react';

interface VideoAnalyzerProps {
  students: Student[];
  params: RecognitionParams;
  onSnapshotTaken?: (imageData: string) => void; // Callback for snapshot
}

type AnalysisMode = 'none' | 'classroom' | 'student';

const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({ students, params, onSnapshotTaken }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  
  // Playback State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  
  // Behavior Analysis Mode
  const [isAnalyzingBehavior, setIsAnalyzingBehavior] = useState(false);
  const [behaviorReport, setBehaviorReport] = useState<BehaviorReport | null>(null);
  
  // Single Student Analysis
  const [selectStudentMode, setSelectStudentMode] = useState(false);
  const [singleStudentReport, setSingleStudentReport] = useState<SingleStudentBehaviorReport | null>(null);

  // FPS Control
  const [fpsMode, setFpsMode] = useState<'native' | 'low'>('native');
  const [targetFps, setTargetFps] = useState(1); // 1 Frame per second

  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<number>();
  const lastProcessTimeRef = useRef<number>(0);

  // Initial setup
  useEffect(() => {
    // 不需要加载模型，因为我们只做行为分析
    setIsLoadingModels(false);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoSrc(URL.createObjectURL(file));
      
      // Reset State
      setIsPlaying(false);
      setBehaviorReport(null);
      setSingleStudentReport(null);
      setSelectStudentMode(false);
      setCurrentTime(0);
      setDuration(0);
    }
  };

  // Simplified video processing for behavior analysis only
  const processVideoForBehaviorAnalysis = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;

    try {
      // Capture current frame as image
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // For behavior analysis, we don't need to send to backend
      // Just update the UI to show that we're processing
      setIsProcessing(true);
      
      // Simulate some processing time
      await new Promise(resolve => setTimeout(resolve, 100));
      
      setIsProcessing(false);
    } catch (error) {
      console.error('处理视频帧时出错:', error);
      alert('处理视频帧时出错: ' + (error instanceof Error ? error.message : '未知错误'));
      setIsProcessing(false);
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    } else {
      videoRef.current.play();
      setIsPlaying(true);
      processVideoContinuously();
    }
  };

  const processVideoContinuously = () => {
    if (!isPlaying || !videoRef.current) return;
    
    const processFrame = () => {
      if (!isPlaying) return;
      
      const now = performance.now();
      const elapsed = now - lastProcessTimeRef.current;
      const targetInterval = fpsMode === 'low' ? 1000 / targetFps : 16; // ~60fps for native
      
      if (elapsed > targetInterval) {
        processVideoForBehaviorAnalysis();
        lastProcessTimeRef.current = now;
      }
      
      requestRef.current = requestAnimationFrame(processFrame);
    };
    
    requestRef.current = requestAnimationFrame(processFrame);
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const triggerClassroomBehaviorAnalysis = async () => {
    if (!videoRef.current) return;
    
    setIsAnalyzingBehavior(true);
    
    try {
      // Capture current frame as image data URL
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error("无法获取 canvas 上下文");
      
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL('image/jpeg');
      
      // Analyze behavior using Gemini service
      const report = await analyzeClassroomBehavior(imageDataUrl);
      setBehaviorReport(report);
    } catch (error) {
      console.error('行为分析失败:', error);
      alert('行为分析失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setIsAnalyzingBehavior(false);
    }
  };

  const triggerStudentBehaviorAnalysis = async () => {
    if (!videoRef.current) return;
    
    setSelectStudentMode(true);
  };

  const handleStudentSelection = async (student: Student) => {
    if (!videoRef.current) return;
    
    setIsAnalyzingBehavior(true);
    setSelectStudentMode(false);
    
    try {
      // Capture current frame as image data URL
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error("无法获取 canvas 上下文");
      
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL('image/jpeg');
      
      // Analyze single student behavior using Gemini service
      const report = await analyzeStudentBehavior(imageDataUrl);
      setSingleStudentReport(report);
    } catch (error) {
      console.error('学生行为分析失败:', error);
      alert('学生行为分析失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setIsAnalyzingBehavior(false);
    }
  };

  const closeStudentReport = () => {
    setSingleStudentReport(null);
  };

  const closeBehaviorReport = () => {
    setBehaviorReport(null);
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <VideoIcon className="w-5 h-5" />
          行为分析
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          仅进行行为分析，不执行人脸检测
        </p>
      </div>
      
      <div className="p-4">
        {/* Video Upload */}
        {!videoSrc && (
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-slate-500 transition-colors">
            <VideoIcon className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400 mb-4">上传视频文件进行行为分析</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 mx-auto"
            >
              <Upload className="w-4 h-4" />
              选择视频文件
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/*"
              className="hidden"
            />
          </div>
        )}
        
        {/* Video Player */}
        {videoSrc && (
          <div className="space-y-4">
            <div className="relative bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                src={videoSrc}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                className="w-full max-h-[500px]"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ display: 'none' }}
              />
              
              {!isPlaying && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <button
                    onClick={togglePlay}
                    className="p-4 bg-black/50 hover:bg-black/70 rounded-full transition-colors"
                  >
                    <Play className="w-8 h-8 text-white" />
                  </button>
                </div>
              )}
            </div>
            
            {/* Video Controls */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
              
              <input
                type="range"
                min="0"
                max={duration || 100}
                value={currentTime}
                onChange={handleSeek}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={togglePlay}
                    disabled={isProcessing}
                    className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg transition-colors"
                  >
                    {isPlaying ? (
                      <Pause className="w-5 h-5 text-white" />
                    ) : (
                      <Play className="w-5 h-5 text-white" />
                    )}
                  </button>
                  
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <span>FPS模式:</span>
                    <select
                      value={fpsMode}
                      onChange={(e) => setFpsMode(e.target.value as any)}
                      className="bg-slate-700 text-white rounded px-2 py-1"
                    >
                      <option value="native">原生</option>
                      <option value="low">低帧率</option>
                    </select>
                    
                    {fpsMode === 'low' && (
                      <>
                        <span>目标FPS:</span>
                        <input
                          type="number"
                          min="1"
                          max="30"
                          value={targetFps}
                          onChange={(e) => setTargetFps(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
                          className="w-16 bg-slate-700 text-white rounded px-2 py-1"
                        />
                      </>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={triggerClassroomBehaviorAnalysis}
                    disabled={isAnalyzingBehavior || isProcessing}
                    className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                  >
                    {isAnalyzingBehavior ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        分析中...
                      </>
                    ) : (
                      <>
                        <BrainCircuit className="w-4 h-4" />
                        全班行为分析
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={triggerStudentBehaviorAnalysis}
                    disabled={isAnalyzingBehavior || isProcessing}
                    className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                  >
                    <UserSearch className="w-4 h-4" />
                    单学生分析
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Student Selection Overlay */}
      {selectStudentMode && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white">选择学生</h3>
              <button
                onClick={() => setSelectStudentMode(false)}
                className="p-1 hover:bg-slate-700 rounded"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            
            <p className="text-slate-400 text-sm mb-4">
              请选择要分析行为的学生
            </p>
            
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {students.map(student => (
                <button
                  key={student.id}
                  onClick={() => handleStudentSelection(student)}
                  className="w-full text-left p-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <div className="font-medium text-white">{student.name}</div>
                  <div className="text-slate-400 text-sm">{student.id}</div>
                </button>
              ))}
              
              {students.length === 0 && (
                <div className="text-center py-8 text-slate-500">
                  暂无学生数据，请先在学生管理中添加学生
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Behavior Report Modal */}
      {behaviorReport && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white">全班行为分析报告</h3>
              <button
                onClick={closeBehaviorReport}
                className="p-1 hover:bg-slate-700 rounded"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">课堂整体表现</h4>
                <p className="text-slate-300">{behaviorReport.overallPerformance}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">专注度分析</h4>
                <p className="text-slate-300">{behaviorReport.focusAnalysis}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">互动情况</h4>
                <p className="text-slate-300">{behaviorReport.interactionAnalysis}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">建议措施</h4>
                <p className="text-slate-300">{behaviorReport.suggestions}</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Single Student Report Modal */}
      {singleStudentReport && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white">
                {singleStudentReport.studentName} 行为分析报告
              </h3>
              <button
                onClick={closeStudentReport}
                className="p-1 hover:bg-slate-700 rounded"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">个人表现</h4>
                <p className="text-slate-300">{singleStudentReport.personalPerformance}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">专注度评估</h4>
                <p className="text-slate-300">{singleStudentReport.focusAssessment}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">参与度分析</h4>
                <p className="text-slate-300">{singleStudentReport.participationAnalysis}</p>
              </div>
              
              <div className="bg-slate-700/50 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-2">个性化建议</h4>
                <p className="text-slate-300">{singleStudentReport.personalizedSuggestions}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoAnalyzer;