import React, { useState, useRef, useEffect } from 'react';
import { RecognitionParams, Student } from '../types';
import { detectFacesFromFile, detectFacesFromBase64 } from '../services/apiService';
import { Download, Upload, Play, SquareDashedMousePointer, RotateCcw } from 'lucide-react';
import * as XLSX from 'xlsx';

interface ImageAnalyzerProps {
  students: Student[];
  params: RecognitionParams;
  snapshotImageData: string | null;
  onSnapshotFromVideo?: (imageData: string) => void;
}

const ImageAnalyzer: React.FC<ImageAnalyzerProps> = ({ students, params, snapshotImageData, onSnapshotFromVideo }) => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionsCount, setDetectionsCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [lastBase64Image, setLastBase64Image] = useState<string | null>(null); // Store last processed image
  const [detectionResults, setDetectionResults] = useState<any[]>([]); // Store detection results for Excel report
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Handle snapshot from video
  useEffect(() => {
    if (snapshotImageData) {
      setImageUrl(snapshotImageData);
      setImageFile(null);
      setDetectionsCount(0);
      setLastBase64Image(null);
    }
  }, [snapshotImageData]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      setDetectionsCount(0);
      setLastBase64Image(null);
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImageFile(file);
      const imageUrl = URL.createObjectURL(file);
      setImageUrl(imageUrl);
      setDetectionsCount(0);
      setLastBase64Image(null);
      
      try {
        // 读取文件为 Base64 用于后端处理
        const reader = new FileReader();
        reader.onload = async (e) => {
          const base64Image = e.target?.result as string;
          setLastBase64Image(base64Image); // Store for re-detection
          
          await processImageDetection(base64Image);
        };
        reader.readAsDataURL(file);
      } catch (error) {
        console.error('处理图像时出错:', error);
        setDetectionsCount(0);
      }
    }
  };

  // Process image detection with given base64 image
  const processImageDetection = async (base64Image: string) => {
    try {
      setIsProcessing(true);
      
      // 发送到后端进行人脸检测
      console.log('正在发送图像到后端进行人脸检测...');
      const result = await detectFacesFromBase64(base64Image);
      console.log('后端检测结果:', result);
      
      if (result.success) {
        // 转换后端结果为前端格式
        const detectedFaces = result.faces.map(face => ({
          id: face.id,
          detectionScore: face.confidence,
          box: {
            x: face.bbox[0],
            y: face.bbox[1],
            width: face.bbox[2] - face.bbox[0],
            height: face.bbox[3] - face.bbox[1]
          },
          landmarks: face.landmarks.map(([x, y]) => ({ x, y })),
          recognition: face.recognition // 添加识别结果
        }));
        
        setDetectionsCount(detectedFaces.length);
        setDetectionResults(detectedFaces); // 存储检测结果用于Excel报告
        drawFaceOverlays(detectedFaces);
        console.log(`检测到 ${detectedFaces.length} 个人脸`);
      } else {
        console.error('人脸检测失败:', result);
        setDetectionsCount(0);
        setDetectionResults([]); // 清空检测结果
      }
    } catch (error) {
      console.error('人脸检测出错:', error);
      setDetectionsCount(0);
      setDetectionResults([]); // 清空检测结果
    } finally {
      setIsProcessing(false);
    }
  };

  // Re-detect faces with current parameters
  const handleReDetect = async () => {
    if (!lastBase64Image) {
      alert('没有可重新检测的图像，请先上传一张图片');
      return;
    }
    
    await processImageDetection(lastBase64Image);
  };

  // 绘制检测到的人脸
  const drawFaceOverlays = (detections: any[], forDownload: boolean = false) => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 用于跟踪已绘制的学生，确保一个人只绘制一次
    const drawnStudents = new Set<string>();
    
    // Draw face detections
    detections.forEach((detection, index) => {
      const box = detection.box || detection.detection?.box;
      if (!box) return;
      
      // 检查是否已经有识别结果且该学生已经被绘制
      if (detection.recognition) {
        const studentName = detection.recognition.name;
        // 允许显示所有识别结果，但在标签上标注重复
        if (drawnStudents.has(studentName)) {
          // 如果该学生已经被绘制，添加重复标记
          detection.isDuplicate = true;
        } else {
          // 标记该学生已被绘制
          drawnStudents.add(studentName);
        }
      }
      
      // Draw bounding box with gradient color based on confidence
      const confidence = detection.detectionScore || detection.detection?.score || 0;
      let boxColor = '#10b981'; // Green for high confidence
      
      // Color coding based on confidence
      if (confidence < 0.3) {
        boxColor = '#ef4444'; // Red for low confidence
      } else if (confidence < 0.6) {
        boxColor = '#f59e0b'; // Yellow for medium confidence
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      
      // Prepare label text
      let labelText = `人脸 ${index + 1} (${(confidence * 100).toFixed(1)}%)`;
      
      // Add recognition result if available
      if (detection.recognition) {
        const recognition = detection.recognition;
        labelText = `${recognition.name} (${(recognition.confidence * 100).toFixed(1)}%)`;
        // 如果是重复识别，添加标记
        if (detection.isDuplicate) {
          labelText += " [重复]";
        }
      }
      
      // Draw label background
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor + '80'; // Add transparency
      ctx.fillRect(box.x, box.y - 20, textWidth + 10, 20);
      
      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Arial';
      ctx.fillText(labelText, box.x + 5, box.y - 5);
      
      // Draw landmarks if available (only in preview mode, not for download)
      if (!forDownload) {
        const landmarks = detection.landmarks || detection.landmarks?.positions;
        if (landmarks && Array.isArray(landmarks)) {
          ctx.fillStyle = '#f59e0b';
          landmarks.forEach(point => {
            if (point && typeof point.x === 'number' && typeof point.y === 'number') {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
              ctx.fill();
            }
          });
        }
      }
      
      // Draw face dimensions
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Arial';
      ctx.fillText(`${Math.round(box.width)}×${Math.round(box.height)}px`, box.x + 5, box.y + box.height + 15);
    });
  };

  const downloadResults = async () => {
    if (!canvasRef.current || !imageRef.current) return;
    
    // 创建一个新的canvas来合并图像和检测结果
    const combinedCanvas = document.createElement('canvas');
    const ctx = combinedCanvas.getContext('2d');
    if (!ctx) return;
    
    // 设置canvas尺寸与原图一致
    combinedCanvas.width = imageRef.current.naturalWidth;
    combinedCanvas.height = imageRef.current.naturalHeight;
    
    // 绘制原始图像
    ctx.drawImage(imageRef.current, 0, 0, combinedCanvas.width, combinedCanvas.height);
    
    // 创建临时canvas来绘制不包含关键点的检测结果
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasRef.current.width;
    tempCanvas.height = canvasRef.current.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (tempCtx) {
      // 保存当前canvas引用
      const originalCanvasRef = canvasRef.current;
      
      // 临时替换canvas引用以便在临时canvas上绘制
      // 注意：这种方法可能不太理想，让我们采用另一种方法
      
      // 清理临时canvas
      tempCanvas.remove();
    }
    
    // 更好的方法：直接在合并canvas上绘制检测结果（不包含关键点）
    drawFaceOverlaysForDownload(ctx, detectionResults);
    
    // 生成包含参数信息的文件名
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const networkSize = params.networkSize;
    const minConfidence = params.minConfidence;
    
    // 获取图像尺寸信息
    const imageDimensions = `${combinedCanvas.width}x${combinedCanvas.height}`;
    
    const imageFilename = `face-recognition-${timestamp}-net${networkSize}-conf${Math.round(minConfidence * 100)}-img${imageDimensions}.png`;
    
    // 创建图片下载链接
    const imageLink = document.createElement('a');
    imageLink.download = imageFilename;
    imageLink.href = combinedCanvas.toDataURL('image/png');
    imageLink.click();
    
    // 清理canvas元素
    combinedCanvas.remove();
    
    // 创建并下载Excel文件
    downloadExcelReport(timestamp, networkSize, minConfidence, imageDimensions);
  };
  
  // 专门为下载创建的绘制函数，不包含人脸关键点
  const drawFaceOverlaysForDownload = (ctx: CanvasRenderingContext2D, detections: any[]) => {
    // 用于跟踪已绘制的学生，确保一个人只绘制一次
    const drawnStudents = new Set<string>();
    
    // Draw face detections without landmarks
    detections.forEach((detection, index) => {
      const box = detection.box || detection.detection?.box;
      if (!box) return;
      
      // 检查是否已经有识别结果且该学生已经被绘制
      if (detection.recognition) {
        const studentName = detection.recognition.name;
        // 允许显示所有识别结果，但在标签上标注重复
        if (drawnStudents.has(studentName)) {
          // 如果该学生已经被绘制，添加重复标记
          detection.isDuplicate = true;
        } else {
          // 标记该学生已被绘制
          drawnStudents.add(studentName);
        }
      }
      
      // Draw bounding box with gradient color based on confidence
      const confidence = detection.detectionScore || detection.detection?.score || 0;
      let boxColor = '#10b981'; // Green for high confidence
      
      // Color coding based on confidence
      if (confidence < 0.3) {
        boxColor = '#ef4444'; // Red for low confidence
      } else if (confidence < 0.6) {
        boxColor = '#f59e0b'; // Yellow for medium confidence
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      
      // Prepare label text
      let labelText = `人脸 ${index + 1} (${(confidence * 100).toFixed(1)}%)`;
      
      // Add recognition result if available
      if (detection.recognition) {
        const recognition = detection.recognition;
        labelText = `${recognition.name} (${(recognition.confidence * 100).toFixed(1)}%)`;
        // 如果是重复识别，添加标记
        if (detection.isDuplicate) {
          labelText += " [重复]";
        }
      }
      
      // Draw label background
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor + '80'; // Add transparency
      ctx.fillRect(box.x, box.y - 20, textWidth + 10, 20);
      
      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Arial';
      ctx.fillText(labelText, box.x + 5, box.y - 5);
      
      // Draw face dimensions
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Arial';
      ctx.fillText(`${Math.round(box.width)}×${Math.round(box.height)}px`, box.x + 5, box.y + box.height + 15);
    });
  };
  
  const downloadExcelReport = (timestamp: string, networkSize: number, minConfidence: number, imageDimensions: string) => {
    // 准备Excel数据
    const worksheetData = [
      ['人脸识别报告'],
      [`生成时间: ${new Date().toLocaleString('zh-CN')}`],
      [`检测参数: 网络尺寸=${networkSize}, 最小置信度=${minConfidence}`],
      [`图像尺寸: ${imageDimensions}`],
      [], // 空行
      ['学生姓名', '出勤状态', '可信度', '检测时间']
    ];
    
    // 创建一个映射来跟踪检测到的学生
    const detectedStudents = new Map<string, { confidence: number }>();
    
    // 处理检测到的人脸
    detectionResults.forEach(detection => {
      if (detection.recognition) {
        const name = detection.recognition.name;
        const confidence = detection.recognition.confidence;
        detectedStudents.set(name, { confidence });
      }
    });
    
    // 添加所有注册学生的数据
    students.forEach(student => {
      const studentName = student.name;
      if (detectedStudents.has(studentName)) {
        // 学生被检测到
        const detectionInfo = detectedStudents.get(studentName)!;
        worksheetData.push([
          studentName,
          '出勤',
          (detectionInfo.confidence * 100).toFixed(2) + '%',
          new Date().toLocaleString('zh-CN')
        ]);
      } else {
        // 学生未被检测到
        worksheetData.push([
          studentName,
          '缺勤',
          'N/A',
          new Date().toLocaleString('zh-CN')
        ]);
      }
    });
    
    // 添加未识别的人脸（不在学生列表中的检测结果）
    detectionResults.forEach((detection, index) => {
      if (!detection.recognition) {
        worksheetData.push([
          `未知人员-${index + 1}`,
          '出勤（未识别）',
          (detection.detectionScore * 100).toFixed(2) + '%',
          new Date().toLocaleString('zh-CN')
        ]);
      }
    });
    
    // 创建工作表
    const ws = XLSX.utils.aoa_to_sheet(worksheetData);
    
    // 设置列宽
    ws['!cols'] = [
      { wch: 15 }, // 学生姓名
      { wch: 15 }, // 出勤状态
      { wch: 10 }, // 可信度
      { wch: 20 }  // 检测时间
    ];
    
    // 创建工作簿
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '人脸识别结果');
    
    // 生成文件名
    const excelFilename = `face-recognition-report-${timestamp}-net${networkSize}-conf${Math.round(minConfidence * 100)}-img${imageDimensions}.xlsx`;
    
    // 导出Excel文件
    XLSX.writeFile(wb, excelFilename);
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <SquareDashedMousePointer className="w-5 h-5" />
          人脸识别
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          上传图片进行人脸识别检测
        </p>
      </div>
      
      <div className="p-4">
        {/* Image Upload */}
        {!imageUrl && (
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-slate-500 transition-colors">
            <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400 mb-4">上传图片文件进行人脸识别</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 mx-auto"
            >
              <Upload className="w-4 h-4" />
              选择图片文件
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
            />
          </div>
        )}
        
        {/* Image Display and Results */}
        {imageUrl && (
          <div className="space-y-4">
            <div className="relative inline-block max-w-full">
              <img
                ref={imageRef}
                src={imageUrl}
                alt="Uploaded"
                className="max-w-full max-h-[500px] rounded-lg"
                onLoad={() => {
                  // Set canvas dimensions to match image
                  if (imageRef.current && canvasRef.current) {
                    canvasRef.current.width = imageRef.current.naturalWidth;
                    canvasRef.current.height = imageRef.current.naturalHeight;
                  }
                }}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
                style={{ imageRendering: 'pixelated' }}
              />
            </div>
            
            {/* Results Panel */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-medium text-white">检测结果</h3>
                  <p className="text-slate-400 text-sm">
                    检测到 <span className="text-green-400 font-bold">{detectionsCount}</span> 个人脸
                  </p>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleReDetect}
                    disabled={isProcessing || !lastBase64Image}
                    className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                  >
                    {isProcessing ? (
                      <>
                        <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        检测中...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="w-4 h-4" />
                        重新检测
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={() => {
                      setImageUrl(null);
                      setImageFile(null);
                      setDetectionsCount(0);
                      setLastBase64Image(null);
                    }}
                    className="px-3 py-1.5 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    重新上传
                  </button>
                  
                  <button
                    onClick={downloadResults}
                    disabled={detectionsCount === 0}
                    className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                  >
                    <Download className="w-4 h-4" />
                    下载结果
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageAnalyzer;