import { RecognitionParams, Student } from '../types';

// Multiple model sources ordered by reliability and speed
const MODEL_URLS = [
  'https://vladmandic.github.io/face-api/model',          // Source 1: High-performance mirror (Vlad Mandic)
  'https://justadudewhohacks.github.io/face-api.js/models', // Source 2: Original Author (GitHub Pages)
  'https://cdn.jsdelivr.net/gh/c7x43/face-api.js-models'    // Source 3: jsDelivr CDN Backup
];

export class FaceRecognitionService {
  private static instance: FaceRecognitionService;
  private isModelsLoaded = false;
  // We store raw labeled descriptors instead of the FaceMatcher because we implement our own Cosine Matcher
  private labeledDescriptors: any[] = []; 

  private constructor() {}

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
      // Load specific nets required for the app
      // Promise.all is used to load them in parallel for the specific URL
      await Promise.all([
        window.faceapi.nets.ssdMobilenetv1.loadFromUri(url),
        window.faceapi.nets.faceLandmark68Net.loadFromUri(url),
        // Using FaceRecognitionNet (ResNet-34) which produces 128D vectors compatible with ArcFace-style logic
        window.faceapi.nets.faceRecognitionNet.loadFromUri(url)
      ]);
  }

  public async loadModels(): Promise<void> {
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
    if (!this.isModelsLoaded) await this.loadModels();
    if (!window.faceapi) throw new Error("FaceAPI not loaded");

    try {
      // Standard pass
      const detection = await window.faceapi
        .detectSingleFace(imageElement)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) return detection;
    } catch (e) { console.warn("Standard face detection failed."); }

    // Fallback: Aggressive scan
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
    return null;
  }

  public async getFaceDescriptor(imageElement: HTMLImageElement): Promise<Float32Array | null> {
    const detection = await this.getFaceDetection(imageElement);
    if (!detection) return null;
    return detection.descriptor;
  }

  public updateFaceMatcher(students: Student[], similarityThreshold: number) {
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

  private drawFaceOverlays(ctx: CanvasRenderingContext2D, resizedDetections: any[], params: RecognitionParams) {
    ctx.font = '500 11px "Inter", sans-serif'; 
    ctx.textBaseline = 'middle';

    resizedDetections.forEach((d: any) => {
      const isManual = (d as any).isManual || false;

      // Use the new InsightFace matcher
      const matchResult = this.findBestMatchInsightFace(d.descriptor);
      
      let effectiveThreshold = params.similarityThreshold;
      
      // Mask mode optimization: Lower the similarity requirement slightly
      if (params.maskMode) {
          effectiveThreshold = Math.max(0.4, effectiveThreshold - 0.1); 
      }

      let isUnknown = true;
      let isMaskModeMatch = false;

      if (matchResult.score >= effectiveThreshold) {
        isUnknown = false;
        if (params.maskMode && matchResult.score < params.similarityThreshold) {
           isMaskModeMatch = true;
        }
      }

      const box = d.detection.box;

      // Determine Colors
      let borderColor = 'rgba(239, 68, 68, 0.8)'; // Red
      let bgColor = 'rgba(239, 68, 68, 0.5)';     // Transparent Red
      let displayText = '';

      const scorePercent = Math.round(matchResult.score * 100);

      if (isManual) {
          borderColor = 'rgba(6, 182, 212, 0.9)'; // Cyan
          bgColor = 'rgba(6, 182, 212, 0.5)';
      }

      if (!isUnknown) {
        if (!isManual) {
            // MATCHED (Green)
            borderColor = 'rgba(34, 197, 94, 0.9)';
            bgColor = 'rgba(34, 197, 94, 0.5)';
            
            if (isMaskModeMatch) {
                borderColor = 'rgba(234, 179, 8, 0.9)';
                bgColor = 'rgba(234, 179, 8, 0.5)';
                displayText = `😷 ${matchResult.label} (${scorePercent}%)`;
            } else {
                displayText = `${matchResult.label} (${scorePercent}%)`;
            }
        } else {
            displayText = `人工: ${matchResult.label} (${scorePercent}%)`;
        }
      } else {
        // UNKNOWN
        if (matchResult.label !== 'unknown' && matchResult.label !== '未注册') {
             if(!isManual) {
                borderColor = 'rgba(245, 158, 11, 0.9)';
                bgColor = 'rgba(245, 158, 11, 0.5)';
             }
             displayText = `${isManual?'人工: ':''}? ${matchResult.label} (${scorePercent}%)`;
        } else {
            displayText = `${isManual?'人工: ':''}未识别 (${scorePercent}%)`;
        }
      }

      // --- Custom Drawing ---
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1.5; 
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      const padding = 6;
      const textMetrics = ctx.measureText(displayText);
      const labelHeight = 20; 
      const labelWidth = textMetrics.width + (padding * 2);
      
      ctx.fillStyle = bgColor;
      ctx.fillRect(box.x, box.y - labelHeight, labelWidth, labelHeight);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(displayText, box.x + padding, box.y - (labelHeight / 2) + 1);
    });
  }

  public drawStableResults(ctx: CanvasRenderingContext2D, results: any[]) {
    ctx.font = '600 12px "Inter", sans-serif'; 
    ctx.textBaseline = 'middle';

    results.forEach((r) => {
      const box = r.box;
      const confidencePercent = Math.round(r.confidence * 100);
      const name = r.label;
      
      const borderColor = 'rgba(251, 191, 36, 1)'; 
      const bgColor = 'rgba(251, 191, 36, 0.85)';
      
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 3; 
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Use a cleaner display for stable results
      let displayText = `${name} (置信度 ${confidencePercent}%)`;
      if (name === 'unknown' || name === '未注册') {
          displayText = `未确认 (${confidencePercent}%)`;
          ctx.strokeStyle = 'rgba(148, 163, 184, 1)';
          ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
      } else {
          ctx.fillStyle = bgColor;
      }

      const padding = 8;
      const textMetrics = ctx.measureText(displayText);
      const labelHeight = 24;
      const labelWidth = textMetrics.width + (padding * 2);

      ctx.fillRect(box.x, box.y - labelHeight, labelWidth, labelHeight);

      ctx.fillStyle = '#000000';
      ctx.fillText(displayText, box.x + padding, box.y - (labelHeight / 2) + 1);
    });
  }

  public async detectSpecificFaceAtCoordinate(
    videoEl: HTMLVideoElement,
    x: number,
    y: number,
    params: RecognitionParams
  ): Promise<any | null> {
     if (!this.isModelsLoaded || !window.faceapi) return null;

     const cropSize = 300; 
     const startX = Math.max(0, x - (cropSize / 2));
     const startY = Math.max(0, y - (cropSize / 2));
     
     const tempCanvas = document.createElement('canvas');
     tempCanvas.width = cropSize;
     tempCanvas.height = cropSize;
     const tempCtx = tempCanvas.getContext('2d');
     if (!tempCtx) return null;

     tempCtx.drawImage(videoEl, startX, startY, cropSize, cropSize, 0, 0, cropSize, cropSize);

     const options = new window.faceapi.SsdMobilenetv1Options({
        minConfidence: 0.1, 
        maxResults: 1
     });

     const detection = await window.faceapi
        .detectSingleFace(tempCanvas, options)
        .withFaceLandmarks()
        .withFaceDescriptor();

     if (detection) {
         const relativeBox = detection.detection.box;
         const absoluteBox = new window.faceapi.Box(
             { 
                 x: relativeBox.x + startX, 
                 y: relativeBox.y + startY, 
                 width: relativeBox.width, 
                 height: relativeBox.height 
             }
         );

         return {
             ...detection,
             detection: {
                 ...detection.detection,
                 box: absoluteBox
             },
             isManual: true 
         };
     }

     return null;
  }

  public async detectAndRecognize(
    videoEl: HTMLVideoElement,
    canvasEl: HTMLCanvasElement,
    params: RecognitionParams,
    manualDetections: any[] = [],
    shouldDraw: boolean = true 
  ): Promise<any[]> {
    if (!this.isModelsLoaded || !window.faceapi) return [];

    let effectiveConfidence = params.minConfidence;
    if (params.maskMode && effectiveConfidence > 0.3) {
        effectiveConfidence = 0.2; 
    }

    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: effectiveConfidence,
      maxResults: 100,
      inputSize: params.networkSize || 608 
    });

    const detections = await window.faceapi
      .detectAllFaces(videoEl, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const displaySize = { width: videoEl.videoWidth, height: videoEl.videoHeight };

    if (displaySize.width === 0) return [];

    window.faceapi.matchDimensions(canvasEl, displaySize);
    const resizedAutoDetections = window.faceapi.resizeResults(detections, displaySize);

    const filteredAutoDetections = resizedAutoDetections.filter((auto: any) => {
      const overlap = manualDetections.some((manual: any) => {
        return this.getIoU(auto.detection.box, manual.detection.box) > 0.3;
      });
      return !overlap;
    });

    const finalDetections = [...filteredAutoDetections, ...manualDetections];

    // Attach InsightFace Match Result here for Accumulator
    finalDetections.forEach((d: any) => {
         const match = this.findBestMatchInsightFace(d.descriptor);
         d.bestMatch = match; 
         // Backwards compatibility for code that expects 'distance' property, though we use score
         d.bestMatch.distance = 1 - match.score; 
    });

    if (shouldDraw) {
        const ctx = canvasEl.getContext('2d');
        if (ctx) {
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            this.drawFaceOverlays(ctx, finalDetections, params);
        }
    }

    return finalDetections;
  }

  public async detectAndRecognizeImage(
    imageEl: HTMLImageElement,
    canvasEl: HTMLCanvasElement,
    params: RecognitionParams
  ): Promise<any[]> {
    if (!this.isModelsLoaded || !window.faceapi) return [];

    let effectiveConfidence = params.minConfidence;
    if (params.maskMode && effectiveConfidence > 0.3) {
        effectiveConfidence = 0.2; 
    }

    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: effectiveConfidence,
      maxResults: 100,
      inputSize: params.networkSize || 608 
    });

    const detections = await window.faceapi
      .detectAllFaces(imageEl, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const displaySize = { width: imageEl.width, height: imageEl.height };

    if (displaySize.width === 0 || displaySize.height === 0) return [];

    window.faceapi.matchDimensions(canvasEl, displaySize);
    const resizedDetections = window.faceapi.resizeResults(detections, displaySize);

    // Attach InsightFace Match Result
    resizedDetections.forEach((d: any) => {
         const match = this.findBestMatchInsightFace(d.descriptor);
         d.bestMatch = match; 
    });

    const ctx = canvasEl.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        this.drawFaceOverlays(ctx, resizedDetections, params);
    }

    return resizedDetections;
  }

  public async analyzeFrame(
    videoEl: HTMLVideoElement,
    canvasEl: HTMLCanvasElement,
    params: RecognitionParams,
    manualDetections: any[] = []
  ): Promise<void> {
    if (!this.isModelsLoaded || !window.faceapi) return;

    const displaySize = { width: videoEl.videoWidth, height: videoEl.videoHeight };
    if (displaySize.width === 0) return;

    window.faceapi.matchDimensions(canvasEl, displaySize);
    
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(videoEl, 0, 0, displaySize.width, displaySize.height);

    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: params.minConfidence,
      maxResults: 100, 
      inputSize: params.networkSize || 608
    });

    const detections = await window.faceapi
      .detectAllFaces(videoEl, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedAutoDetections = window.faceapi.resizeResults(detections, displaySize);

    const filteredAutoDetections = resizedAutoDetections.filter((auto: any) => {
      const overlap = manualDetections.some((manual: any) => {
        return this.getIoU(auto.detection.box, manual.detection.box) > 0.3;
      });
      return !overlap;
    });

    const finalDetections = [...filteredAutoDetections, ...manualDetections];
    this.drawFaceOverlays(ctx, finalDetections, params);
  }
}