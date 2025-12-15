# InsightFace 模型集成指南

## 当前状态

我们已经清理了旧的 SSD MobileNet 模型，并为真正的 InsightFace 模型做好了准备。目前代码中保留了兼容性层，但您可以按照以下步骤集成真正的 InsightFace 模型。

## 集成步骤

### 1. 获取 InsightFace 模型文件

您需要获取以下类型的模型文件：

1. **人脸检测模型** (如 SCRFD)
2. **人脸特征提取模型** (如 GLINT100/R100)
3. **人脸对齐模型** (用于关键点检测)

### 2. 模型文件格式要求

模型文件需要转换为 ONNX 格式：
- `.onnx` - ONNX 模型文件

### 3. 文件放置位置

将模型文件按以下结构放置：

```
/public/models/insightface/
├── onnx/
│   ├── scrfd_10g_gnkps/
│   │   └── scrfd_10g_bnkps.onnx
│   └── glintr100/
│       └── glintr100.onnx
```

### 4. 新增 InsightFaceService

我们已经创建了一个新的 `InsightFaceService` 来处理 ONNX 模型的加载和推理：

- **文件位置**: `/src/services/insightFaceService.ts`
- **功能**: 
  - 加载 ONNX 模型
  - 执行人脸检测
  - 执行特征提取
  - 实现人脸匹配

### 5. 修改 FaceRecognitionService

FaceRecognitionService 已经更新以使用新的 InsightFaceService：

#### loadModelsFromUrl 方法
```typescript
private async loadModelsFromUrl(url: string): Promise<void> {
  console.log(`Attempting to load InsightFace models from: ${url}`);
  
  // Try to load models using InsightFace service
  try {
    await this.insightFaceService.loadModels();
    this.isInsightFaceMode = true;
    console.log(`Successfully loaded InsightFace models from: ${url}`);
    return;
  } catch (insightFaceError) {
    console.warn("Failed to load InsightFace models, falling back to standard models", insightFaceError);
    // ... fallback to standard models
  }
}
```

#### getFaceDetection 方法
已更新为人脸检测模型的推理代码：

```typescript
public async getFaceDetection(imageElement: HTMLImageElement): Promise<any | null> {
  if (!this.isModelsLoaded) await this.loadModels();
  
  if (this.isInsightFaceMode) {
    // Use InsightFace service for detection
    try {
      // Convert image to ImageData and detect faces
      const detections = await this.insightFaceService.detectFaces(imageData);
      // ... return detection results
    } catch (error) {
      console.warn("InsightFace 检测失败，回退到标准模型", error);
    }
  }
  
  // ... fallback to standard face-api.js detection
}
```

#### getFaceDescriptor 方法
已更新为特征提取模型的推理代码：

```typescript
public async getFaceDescriptor(imageElement: HTMLImageElement): Promise<Float32Array | null> {
  if (this.isInsightFaceMode) {
    // Use InsightFace service for recognition
    try {
      // Convert image to ImageData and extract features
      const embedding = await this.insightFaceService.getFaceEmbedding(imageData);
      return embedding;
    } catch (error) {
      console.warn("InsightFace 特征提取失败，回退到标准模型", error);
    }
  }
  
  // ... fallback to standard face-api.js recognition
}
```

### 6. HTML 页面更新

在 `index.html` 中已经添加了 ONNX Runtime Web 的引用：

```html
<!-- Load ONNX Runtime for InsightFace models -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"></script>
```

## 注意事项

1. **模型兼容性**：确保您的 ONNX 模型与 ONNX Runtime Web 兼容
2. **性能优化**：InsightFace 模型通常较大，注意内存使用和加载时间
3. **后处理**：可能需要添加 NMS (非极大值抑制) 等后处理步骤
4. **设备适配**：考虑在不同设备上的性能表现

## 测试验证

集成完成后，可以通过以下方式验证：

1. 启动开发服务器：`npm run dev`
2. 访问应用界面
3. 上传测试图片进行人脸检测和识别
4. 检查控制台日志确认模型加载成功

## 技术支持

如果您在集成过程中遇到任何问题，请提供：
1. 模型文件的具体来源和版本
2. 错误日志信息
3. 期望的效果示例

我们会根据您提供的信息进一步协助您完成集成。