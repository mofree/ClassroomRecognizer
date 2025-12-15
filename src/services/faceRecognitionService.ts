import { RecognitionParams, Student } from '../types';
import { InsightFaceService } from './insightFaceService';

// Use local models instead of CDN
const MODEL_URLS = [
  '/models/insightface',  // Local InsightFace models in public/models/insightface directory
];

// Flag to indicate if we should use backend API instead of browser models
const USE_BACKEND_API = true;
export class FaceRecognitionService {
  private static instance: FaceRecognitionService;
  private isModelsLoaded = false;
  // We store raw labeled descriptors instead of the FaceMatcher because we implement our own Cosine Matcher
  private labeledDescriptors: any[] = []; 
  // Store current parameters
  private currentParams: RecognitionParams | null = null;
  // Flag to indicate if we're using InsightFace models
  private isInsightFaceMode = false;
  // InsightFace service instance
  private insightFaceService: InsightFaceService;

  private constructor() {
    this.insightFaceService = InsightFaceService.getInstance();
  }

  public static getInstance(): FaceRecognitionService {
    if (!FaceRecognitionService.instance) {
      FaceRecognitionService.instance = new FaceRecognitionService();
    }
    return FaceRecognitionService.instance;
  }

  private async waitForFaceApi(): Promise<void> {
    let attempts = 0;
    while (!window.faceapi && attempts < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    if (!window.faceapi) {
      throw new Error("核心组件 (face-api.js) 未加载。请检查网络连接或刷新页面。");
    }
  }

  /**
   * Helper to attempt loading from a specific URL
   */
  private async loadModelsFromUrl(url: string): Promise<void> {
    console.log(`Attempting to load InsightFace models from: ${url}`);
    
    // First, try to load InsightFace specific models
    try {
      // Try to load models using InsightFace service
      await this.insightFaceService.loadModels();
      this.isInsightFaceMode = true;
      console.log(`Successfully loaded InsightFace models from: ${url}`);
      return;
    } catch (insightFaceError) {
      console.warn("Failed to load InsightFace models, falling back to standard models", insightFaceError);
      
      // Fallback to standard models
      await Promise.all([
        window.faceapi.nets.ssdMobilenetv1.loadFromUri(url),
        window.faceapi.nets.faceLandmark68Net.loadFromUri(url),
        // Using FaceRecognitionNet (ResNet-34) which produces 128D vectors compatible with ArcFace-style logic
        window.faceapi.nets.faceRecognitionNet.loadFromUri(url)
      ]);
      
      this.isInsightFaceMode = false;
      console.log(`Successfully loaded standard models from: ${url}`);
    }
  }

  public async loadModels(): Promise<void> {
    // 如果使用后端API，则不需要加载浏览器端模型
    if (USE_BACKEND_API) {
      this.isModelsLoaded = true;
      console.log("使用后端API模式，跳过浏览器端模型加载");
      return;
    }

    if (this.isModelsLoaded) return;
    
    try {
      await this.waitForFaceApi();

      // Iterate through sources until one works
      for (const url of MODEL_URLS) {
        try {
          console.log(`正在尝试加载 AI 模型 (源: ${url})...`);
          await this.loadModelsFromUrl(url);
          
          this.isModelsLoaded = true;
          console.log(`AI 模型加载成功 (来自: ${url})`);
          return; // Exit function on success
        } catch (err) {
          console.warn(`从 ${url} 加载模型失败，尝试下一个源...`, err);
          // Continue to next URL in the loop
        }
      }

      // If loop finishes without returning, all sources failed
      throw new Error("所有模型镜像源均无法连接");
    } catch (error) {
      console.error("Critical Model Load Error:", error);
      throw new Error("AI 模型初始化失败。请检查您的网络连接是否允许访问 GitHub Pages 或 jsDelivr CDN。");
    }
  }
  public async getFaceDetection(imageElement: HTMLImageElement): Promise<any | null> {
    // 如果使用后端API，直接抛出错误提示应该使用后端服务
    if (USE_BACKEND_API) {
      throw new Error("浏览器端模型已禁用，请使用后端API进行人脸检测");
    }

    if (!this.isModelsLoaded) {
      try {
        await this.loadModels();
      } catch (error) {
        console.error("模型加载失败:", error);
        throw new Error(`模型加载失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }    
    if (this.isInsightFaceMode) {
      // Use InsightFace service for detection
      try {
        // Convert image to ImageData
        const canvas = document.createElement('canvas');
        canvas.width = imageElement.naturalWidth;
        canvas.height = imageElement.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error("无法获取 canvas 上下文");
        
        ctx.drawImage(imageElement, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        const detections = await this.insightFaceService.detectFaces(imageData);
        if (detections.length > 0) {
          // Return the first detection with a dummy descriptor for compatibility
          return {
            detection: detections[0],
            descriptor: new Float32Array(512), // InsightFace typically uses 512-dim embeddings
            landmarks: detections[0].landmarks || null
          };
        }
        return null;
      } catch (error: any) {
        console.warn("InsightFace 检测失败，回退到标准模型", error);
        throw new Error(`InsightFace 检测失败: ${error.message || '未知错误'}`);
      }
    }
    
    // Fallback to standard face-api.js detection
    if (!window.faceapi) throw new Error("FaceAPI 未正确加载");
    
    try {
      // Standard pass
      const detection = await window.faceapi
        .detectSingleFace(imageElement)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (detection) return detection;
    } catch (e: any) { 
      console.warn("Standard face detection failed.", e); 
      throw new Error(`标准人脸检测失败: ${e.message || '未知错误'}`);
    }
    
    // Fallback: Aggressive scan
    try {
      const options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: 0.1, 
        maxResults: 10,
      });
      
      const allDetections = await window.faceapi
        .detectAllFaces(imageElement, options)
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      if (allDetections.length > 0) {
        return allDetections.reduce((prev: any, current: any) => {
          return (prev.detection.box.area > current.detection.box.area) ? prev : current;
        });
      }
    } catch (e: any) {
      console.warn("Aggressive scan failed.", e);
      throw new Error(`增强扫描失败: ${e.message || '未知错误'}`);
    }
    
    return null;
  }

  public async getFaceDescriptor(imageElement: HTMLImageElement): Promise<Float32Array | null> {
    // 如果使用后端API，直接抛出错误提示应该使用后端服务
    if (USE_BACKEND_API) {
      throw new Error("浏览器端模型已禁用，请使用后端API进行特征提取");
    }

    if (this.isInsightFaceMode) {
      // Use InsightFace service for recognition
      try {
        // Convert image to ImageData
        const canvas = document.createElement('canvas');
        canvas.width = imageElement.naturalWidth;
        canvas.height = imageElement.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error("无法获取 canvas 上下文");
        
        ctx.drawImage(imageElement, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        const embedding = await this.insightFaceService.getFaceEmbedding(imageData);
        return embedding;
      } catch (error) {
        console.warn("InsightFace 特征提取失败，回退到标准模型", error);
      }
    }    
    // Fallback to standard face-api.js recognition
    const detection = await this.getFaceDetection(imageElement);
    if (!detection) return null;
    return detection.descriptor;
  }

  public updateFaceMatcher(students: Student[], params: RecognitionParams) {
    // Store current parameters
    this.currentParams = params;
    
    // Also update InsightFace service
    this.insightFaceService.updateFaceMatcher(students, params);
    
    if (!window.faceapi || students.length === 0) {
      this.labeledDescriptors = [];
      return;
    }
    // Just store the data; matching is now dynamic
    this.labeledDescriptors = students.map((student) => {
      return new window.faceapi.LabeledFaceDescriptors(
        student.name,
        student.descriptors
      );
    });
  }

  public getIoU(box1: any, box2: any): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;

    if (box1Area + box2Area - intersectionArea === 0) return 0;

    return intersectionArea / (box1Area + box2Area - intersectionArea);
  }

  /**
   * InsightFace Core: Cosine Similarity
   * Calculates the cosine similarity between two vectors.
   * Since face-api descriptors are normalized (L2 norm = 1), dot product equals cosine similarity.
   */
  private computeCosineSimilarity(descriptor1: Float32Array, descriptor2: Float32Array): number {
    let dotProduct = 0;
    for (let i = 0; i < descriptor1.length; i++) {
      dotProduct += descriptor1[i] * descriptor2[i];
    }
    return dotProduct;
  }

  /**
   * Finds the best match using Cosine Similarity (InsightFace Algorithm)
   * Returns { label, score } where score is 0.0 to 1.0 (1.0 is identical)
   */
  private findBestMatchInsightFace(queryDescriptor: Float32Array): { label: string; score: number } {
    if (this.labeledDescriptors.length === 0) {
      return { label: 'unknown', score: 0 };
    }

    let bestLabel = 'unknown';
    let maxSimilarity = -1;

    for (const labeledDesc of this.labeledDescriptors) {
      // A student might have multiple descriptors (multiple registered photos)
      // We find the max similarity among all their photos
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

  /**
   * Improved matching algorithm that ensures one-to-one mapping
   * Each student can only be matched once in a single frame
   */
  public findBestMatchesForFrame(descriptors: Float32Array[], usedLabels: Set<string> = new Set()): Array<{ label: string; score: number; descriptor: Float32Array }> {
    const results: Array<{ label: string; score: number; descriptor: Float32Array }> = [];
    const availableDescriptors = [...descriptors];
    const availableLabeledDescriptors = this.labeledDescriptors.filter(ld => !usedLabels.has(ld.label));

    // For each detected face, find the best match
    for (const descriptor of availableDescriptors) {
      let bestMatch: { label: string; score: number } | null = null;
      let bestLabeledDesc: any = null;
      let maxSimilarity = -1;

      // Find the best match among available students
      for (const labeledDesc of availableLabeledDescriptors) {
        // A student might have multiple descriptors (multiple registered photos)
        // We find the max similarity among all their photos
        for (const studentDescriptor of labeledDesc.descriptors) {
          const similarity = this.computeCosineSimilarity(descriptor, studentDescriptor);
          if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = { label: labeledDesc.label, score: similarity };
            bestLabeledDesc = labeledDesc;
          }
        }
      }

      if (bestMatch && bestLabeledDesc) {
        // Add to results
        results.push({ ...bestMatch, descriptor });
        // Mark this student as used to prevent multiple matches
        usedLabels.add(bestLabeledDesc.label);
        // Remove this student from available pool
        const index = availableLabeledDescriptors.indexOf(bestLabeledDesc);
        if (index > -1) {
          availableLabeledDescriptors.splice(index, 1);
        }
      } else {
        // No good match found
        results.push({ label: 'unknown', score: 0, descriptor });
      }
    }

    return results;
  }

  /**
   * Public method to find best match using Cosine Similarity (InsightFace Algorithm)
   * Returns { label, score } where score is 0.0 to 1.0 (1.0 is identical)
   */
  public findBestMatch(queryDescriptor: Float32Array): { label: string; score: number } {
    if (this.isInsightFaceMode) {
      // Use InsightFace service for matching
      return this.insightFaceService.findBestMatch(queryDescriptor);
    }
    
    // Fallback to standard matching
    return this.findBestMatchInsightFace(queryDescriptor);
  }

  /**
   * Public method to draw face overlays
   */
  public drawFaceOverlaysPublic(ctx: CanvasRenderingContext2D, resizedDetections: any[], params: RecognitionParams) {
    this.drawFaceOverlays(ctx, resizedDetections, params);
  }

  private drawFaceOverlays(ctx: CanvasRenderingContext2D, resizedDetections: any[], params: RecognitionParams) {
    ctx.font = '500 11px "Inter", sans-serif'; 
    ctx.textBaseline = 'middle';

    // Extract all descriptors for batch processing
    const descriptors = resizedDetections.map((d: any) => d.descriptor);
    
    // Use improved matching algorithm to ensure one-to-one mapping
    const usedLabels = new Set<string>();
    const matches = this.findBestMatchesForFrame(descriptors, usedLabels);

    resizedDetections.forEach((d: any, index: number) => {
      const isManual = (d as any).isManual || false;

      // Use the improved matcher
      const matchResult = matches[index];
      
      let effectiveThreshold = params.similarityThreshold;
      
      // Mask mode optimization: Lower the similarity requirement slightly
      if (params.maskMode) {
        effectiveThreshold = Math.max(0.4, effectiveThreshold - 0.1);
      }

      let color = '#ef4444'; // Red (Unknown)
      let statusText = `${matchResult.label} (${(matchResult.score * 100).toFixed(0)}%)`;
      
      // Color coding based on confidence thresholds (ArcFace Logic)
      if (matchResult.score >= effectiveThreshold) {
        color = '#22c55e'; // Green (High Confidence Match)
      } else if (matchResult.score >= effectiveThreshold * 0.8) {
        color = '#eab308'; // Yellow (Low Confidence Match)
      }

      // Draw bounding box
      const box = d.detection.box;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Draw label background
      const textWidth = ctx.measureText(statusText).width;
      ctx.fillStyle = color + '80'; // 50% opacity
      ctx.fillRect(box.x, box.y - 16, textWidth + 8, 16);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(statusText, box.x + 4, box.y - 8);

      // Draw landmark dots if available
      if (d.landmarks) {
        ctx.fillStyle = color;
        const positions = d.landmarks.positions;
        positions.forEach((pos: any) => {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 1.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }

  public async detectAndRecognize(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement,
    params: RecognitionParams,
    manualDetections: any[] = [],
    drawOverlay: boolean = true
  ): Promise<any[]> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) throw new Error("Could not get canvas context");
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Resize canvas to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    // Two-stage detection for better results in blurry classroom videos
    let allDetections: any[] = [];
    
    // Stage 1: Broad detection with very low confidence threshold
    try {
      const stage1Options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: Math.max(0.05, params.minConfidence * 0.5), // Even lower threshold
        maxResults: 100, // Allow more detections
        inputSize: Math.min(800, params.networkSize * 1.2) // Larger input for better detection
      });
      
      const stage1Detections = await window.faceapi
        .detectAllFaces(videoElement, stage1Options)
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      allDetections = [...stage1Detections];
    } catch (stage1Error) {
      console.warn("Stage 1 detection failed:", stage1Error);
    }
    
    // Stage 2: Refinement with original parameters
    try {
      const stage2Options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: params.minConfidence,
        maxResults: 50,
        inputSize: params.networkSize
      });
      
      const stage2Detections = await window.faceapi
        .detectAllFaces(videoElement, stage2Options)
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      // Merge detections, preferring stage 2 results (higher quality)
      allDetections = this.mergeDetections(allDetections, stage2Detections);
    } catch (stage2Error) {
      console.warn("Stage 2 detection failed:", stage2Error);
    }
    
    // Add manual detections
    allDetections = [...allDetections, ...manualDetections];
    
    // Remove duplicates based on IoU
    allDetections = this.removeDuplicateDetections(allDetections, params.iouThreshold);
    
    // Draw overlays if requested
    if (drawOverlay && allDetections.length > 0) {
      this.drawFaceOverlays(ctx, allDetections, params);
    }
    
    return allDetections;
  }

  public async detectAndRecognizeImage(
    imageElement: HTMLImageElement,
    canvasElement: HTMLCanvasElement,
    params: RecognitionParams
  ): Promise<any[]> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) throw new Error("Could not get canvas context");
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Resize canvas to match image
    canvasElement.width = imageElement.naturalWidth;
    canvasElement.height = imageElement.naturalHeight;
    
    // Two-stage detection for better results in blurry classroom images
    let allDetections: any[] = [];
    
    // Stage 1: Broad detection with very low confidence threshold
    try {
      const stage1Options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: Math.max(0.05, params.minConfidence * 0.5), // Even lower threshold
        maxResults: 100, // Allow more detections
        inputSize: Math.min(800, params.networkSize * 1.2) // Larger input for better detection
      });
      
      const stage1Detections = await window.faceapi
        .detectAllFaces(imageElement, stage1Options)
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      allDetections = [...stage1Detections];
    } catch (stage1Error) {
      console.warn("Stage 1 detection failed:", stage1Error);
    }
    
    // Stage 2: Refinement with original parameters
    try {
      const stage2Options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: params.minConfidence,
        maxResults: 50,
        inputSize: params.networkSize
      });
      
      const stage2Detections = await window.faceapi
        .detectAllFaces(imageElement, stage2Options)
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      // Merge detections, preferring stage 2 results (higher quality)
      allDetections = this.mergeDetections(allDetections, stage2Detections);
    } catch (stage2Error) {
      console.warn("Stage 2 detection failed:", stage2Error);
    }
    
    // Remove duplicates based on IoU
    allDetections = this.removeDuplicateDetections(allDetections, params.iouThreshold);
    
    // Draw overlays
    if (allDetections.length > 0) {
      this.drawFaceOverlays(ctx, allDetections, params);
    }
    
    return allDetections;
  }

  /**
   * Merge two sets of detections, preferring higher quality detections
   */
  private mergeDetections(detections1: any[], detections2: any[]): any[] {
    const merged = [...detections1];
    
    // Add stage 2 detections that are not overlapping with stage 1
    for (const det2 of detections2) {
      let isOverlapping = false;
      for (const det1 of detections1) {
        const iou = this.getIoU(det1.detection.box, det2.detection.box);
        if (iou > 0.3) { // If significant overlap
          isOverlapping = true;
          // Replace with stage 2 detection if it has higher confidence
          if (det2.detection.score > det1.detection.score) {
            const index = merged.indexOf(det1);
            if (index > -1) {
              merged[index] = det2;
            }
          }
          break;
        }
      }
      
      if (!isOverlapping) {
        merged.push(det2);
      }
    }
    
    return merged;
  }

  /**
   * Remove duplicate detections based on IoU threshold
   */
  private removeDuplicateDetections(detections: any[], iouThreshold: number): any[] {
    if (detections.length <= 1) return detections;
    
    const filtered: any[] = [];
    
    for (let i = 0; i < detections.length; i++) {
      let keep = true;
      
      for (let j = 0; j < filtered.length; j++) {
        const iou = this.getIoU(detections[i].detection.box, filtered[j].detection.box);
        if (iou > iouThreshold) {
          // Keep the one with higher confidence
          if (detections[i].detection.score > filtered[j].detection.score) {
            filtered[j] = detections[i];
          }
          keep = false;
          break;
        }
      }
      
      if (keep) {
        filtered.push(detections[i]);
      }
    }
    
    return filtered;
  }

}
