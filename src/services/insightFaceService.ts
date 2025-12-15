import { RecognitionParams, Student } from '../types';
import { InferenceSession, Tensor } from 'onnxruntime-web';

export class InsightFaceService {
  private static instance: InsightFaceService;
  private isModelsLoaded = false;
  private detectionSession: InferenceSession | null = null;
  private recognitionSession: InferenceSession | null = null;
  private labeledDescriptors: any[] = [];
  private currentParams: RecognitionParams | null = null;

  private constructor() {}

  public static getInstance(): InsightFaceService {
    if (!InsightFaceService.instance) {
      InsightFaceService.instance = new InsightFaceService();
    }
    return InsightFaceService.instance;
  }

  public async loadModels(): Promise<void> {
    if (this.isModelsLoaded) return;
    
    try {
      console.log("正在加载 InsightFace 模型...");
      
      // Check if onnxruntime is available
      if (typeof InferenceSession === 'undefined') {
        throw new Error("ONNX Runtime Web 未正确加载");
      }
      
      // Check browser capabilities
      if (!this.checkBrowserCapabilities()) {
        throw new Error("浏览器不支持必要的功能 (WebAssembly, WebGL)");
      }
      
      // Load detection model (SCRFD)
      console.log("正在加载人脸检测模型...");
      try {
        // Add more detailed logging
        console.log("开始创建检测模型会话...");
        
        this.detectionSession = await InferenceSession.create('/models/insightface/onnx/scrfd_10g_bnkps/scrfd_10g_bnkps.onnx');
        console.log("人脸检测模型会话创建成功");
        
        // Log model info
        if (this.detectionSession) {
          console.log("检测模型输入:", this.detectionSession.inputNames);
          console.log("检测模型输出:", this.detectionSession.outputNames);
        }
      } catch (detectionError: any) {
        console.error("人脸检测模型加载详细错误:", detectionError);
        console.error("错误类型:", typeof detectionError);
        console.error("错误堆栈:", detectionError.stack);
        
        // Try alternative approach - load without options
        try {
          console.log("尝试不带选项加载检测模型...");
          this.detectionSession = await InferenceSession.create('/models/insightface/onnx/scrfd_10g_bnkps/scrfd_10g_bnkps.onnx');
          console.log("人脸检测模型会话创建成功（无选项）");
        } catch (altError: any) {
          console.error("无选项加载也失败:", altError);
          
          // Try to get more specific error information
          let errorMessage = '未知错误';
          if (detectionError instanceof Error) {
            errorMessage = detectionError.message;
          } else if (typeof detectionError === 'string') {
            errorMessage = detectionError;
          } else if (detectionError && typeof detectionError.toString === 'function') {
            errorMessage = detectionError.toString();
          }
          
          throw new Error(`人脸检测模型加载失败: ${errorMessage}`);
        }
      }
      
      // Load recognition model (GLINT100)
      console.log("正在加载人脸识别模型...");
      try {
        // Add more detailed logging
        console.log("开始创建识别模型会话...");
        
        this.recognitionSession = await InferenceSession.create('/models/insightface/onnx/glintr100/glintr100.onnx');
        console.log("人脸识别模型会话创建成功");
        
        // Log model info
        if (this.recognitionSession) {
          console.log("识别模型输入:", this.recognitionSession.inputNames);
          console.log("识别模型输出:", this.recognitionSession.outputNames);
        }
      } catch (recognitionError: any) {
        console.error("人脸识别模型加载详细错误:", recognitionError);
        console.error("错误类型:", typeof recognitionError);
        console.error("错误堆栈:", recognitionError.stack);
        
        // Try alternative approach - load without options
        try {
          console.log("尝试不带选项加载识别模型...");
          this.recognitionSession = await InferenceSession.create('/models/insightface/onnx/glintr100/glintr100.onnx');
          console.log("人脸识别模型会话创建成功（无选项）");
        } catch (altError: any) {
          console.error("无选项加载也失败:", altError);
          
          // Try to get more specific error information
          let errorMessage = '未知错误';
          if (recognitionError instanceof Error) {
            errorMessage = recognitionError.message;
          } else if (typeof recognitionError === 'string') {
            errorMessage = recognitionError;
          } else if (recognitionError && typeof recognitionError.toString === 'function') {
            errorMessage = recognitionError.toString();
          }
          
          throw new Error(`人脸识别模型加载失败: ${errorMessage}`);
        }
      }
      
      this.isModelsLoaded = true;
      console.log("InsightFace 模型加载成功");
    } catch (error) {
      console.error("模型加载失败:", error);
      // 提供更详细的错误信息
      if (error instanceof Error) {
        throw new Error(`InsightFace 模型初始化失败: ${error.message}`);
      } else {
        throw new Error("InsightFace 模型初始化失败。请检查模型文件是否存在且格式正确。");
      }
    }
  }

  private checkBrowserCapabilities(): boolean {
    // Check for WebAssembly support
    if (typeof WebAssembly !== 'object') {
      console.error('WebAssembly not supported');
      return false;
    }
    
    // Check for WebGL support
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        console.error('WebGL not supported');
        return false;
      }
    } catch (e) {
      console.error('WebGL check failed:', e);
      return false;
    }
    
    return true;
  }

  /**
   * Preprocess image for detection model
   */
  private preprocessDetectionImage(imageData: ImageData): Float32Array {
    // SCRFD expects input in BCHW format with specific normalization
    const { width, height, data } = imageData;
    const tensor = new Float32Array(1 * 3 * height * width);
    
    // Normalize and convert to RGB
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        // Convert to RGB and normalize to [0, 1]
        tensor[y * width + x] = data[idx] / 255.0;         // R
        tensor[height * width + y * width + x] = data[idx + 1] / 255.0;  // G
        tensor[2 * height * width + y * width + x] = data[idx + 2] / 255.0;  // B
      }
    }
    
    return tensor;
  }

  /**
   * Preprocess image for recognition model
   */
  private preprocessRecognitionImage(faceImage: ImageData): Float32Array {
    // GLINT100 expects 112x112 RGB image with specific normalization
    const { width, height, data } = faceImage;
    const tensor = new Float32Array(1 * 3 * 112 * 112);
    
    // Resize if needed and normalize
    // Note: In a real implementation, you would use a proper resizing algorithm
    for (let i = 0; i < 112 * 112; i++) {
      const y = Math.floor(i / 112);
      const x = i % 112;
      
      // Simple nearest neighbor for demonstration
      const srcY = Math.floor(y * height / 112);
      const srcX = Math.floor(x * width / 112);
      const srcIdx = (srcY * width + srcX) * 4;
      
      // Normalize to [-1, 1] as expected by GLINT100
      tensor[i] = (data[srcIdx] / 255.0) * 2.0 - 1.0;         // R
      tensor[112 * 112 + i] = (data[srcIdx + 1] / 255.0) * 2.0 - 1.0;  // G
      tensor[2 * 112 * 112 + i] = (data[srcIdx + 2] / 255.0) * 2.0 - 1.0;  // B
    }
    
    return tensor;
  }

  /**
   * Detect faces in an image
   */
  public async detectFaces(imageData: ImageData): Promise<any[]> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    if (!this.detectionSession) {
      throw new Error("检测模型未加载");
    }
    
    try {
      const preprocessed = this.preprocessDetectionImage(imageData);
      const inputTensor = new Tensor('float32', preprocessed, [1, 3, imageData.height, imageData.width]);
      
      const feeds = { input: inputTensor };
      const results = await this.detectionSession.run(feeds);
      
      // Process detection results
      // Note: This is a simplified implementation - actual processing would depend on model outputs
      const detections = this.processDetectionOutput(results, imageData.width, imageData.height);
      
      return detections;
    } catch (error: any) {
      console.error("人脸检测失败:", error);
      throw new Error(`人脸检测失败: ${error.message || '未知错误'}`);
    }
  }

  /**
   * Process detection model output
   */
  private processDetectionOutput(results: any, imageWidth: number, imageHeight: number): any[] {
    // Log the results to understand the structure
    console.log("Detection model output:", results);
    
    // For now, return an empty array to avoid errors
    // In a real implementation, you would parse the outputs to extract bounding boxes
    return [];
  }

  /**
   * Extract face embedding
   */
  public async getFaceEmbedding(faceImage: ImageData): Promise<Float32Array> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    if (!this.recognitionSession) {
      throw new Error("识别模型未加载");
    }
    
    try {
      const preprocessed = this.preprocessRecognitionImage(faceImage);
      const inputTensor = new Tensor('float32', preprocessed, [1, 3, 112, 112]);
      
      const feeds = { input: inputTensor };
      const results = await this.recognitionSession.run(feeds);
      
      // Log the results to understand the structure
      console.log("Recognition model output:", results);
      
      // Extract embedding from results
      // Try different possible output names
      let embedding: Float32Array | null = null;
      
      // Common output names for face recognition models
      const possibleOutputNames = ['output', 'embedding', 'features'];
      
      for (const name of possibleOutputNames) {
        if (results[name] && results[name].data) {
          embedding = results[name].data as Float32Array;
          break;
        }
      }
      
      // If we couldn't find the embedding, throw an error
      if (!embedding) {
        throw new Error("无法从模型输出中提取特征向量");
      }
      
      return embedding;
    } catch (error) {
      console.error("特征提取失败:", error);
      throw error;
    }
  }

  /**
   * Update face matcher with student data
   */
  public updateFaceMatcher(students: Student[], params: RecognitionParams) {
    this.currentParams = params;
    this.labeledDescriptors = students.map(student => ({
      label: student.name,
      descriptors: student.descriptors
    }));
  }

  /**
   * Compute cosine similarity between two vectors
   */
  private computeCosineSimilarity(descriptor1: Float32Array, descriptor2: Float32Array): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < descriptor1.length; i++) {
      dotProduct += descriptor1[i] * descriptor2[i];
      norm1 += descriptor1[i] * descriptor1[i];
      norm2 += descriptor2[i] * descriptor2[i];
    }
    
    if (norm1 === 0 || norm2 === 0) return 0;
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Find best match for a face embedding
   */
  public findBestMatch(queryDescriptor: Float32Array): { label: string; score: number } {
    if (this.labeledDescriptors.length === 0) {
      return { label: 'unknown', score: 0 };
    }

    let bestLabel = 'unknown';
    let maxSimilarity = -1;

    for (const labeledDesc of this.labeledDescriptors) {
      for (const descriptor of labeledDesc.descriptors) {
        const similarity = this.computeCosineSimilarity(queryDescriptor, descriptor);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          bestLabel = labeledDesc.label;
        }
      }
    }

    return { label: bestLabel, score: maxSimilarity };
  }
}