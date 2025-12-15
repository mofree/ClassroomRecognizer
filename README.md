# Classroom Face Sentinel (智慧教室人脸考勤系统)

A professional-grade real-time video face recognition application designed for classroom environments. Features AI-powered student registration validation using Google Gemini and customizable recognition parameters for handling blurry or crowded scenes.

## 核心功能 (Key Features)

### 1. 实时视频分析 (Real-time Video Analysis)
*   **专为教室场景优化**: 针对教室监控视角（远距离、多目标、遮挡）进行算法调优。
*   **多模型架构**:
    *   **检测 (Detection)**: SSD Mobilenet V1 (速度快，适合实时视频)。
    *   **识别 (Recognition)**: ResNet-34 (128维特征向量)，生成高精度人脸指纹。
    *   **匹配 (Matching)**: 采用 **InsightFace 同款余弦相似度 (Cosine Similarity)** 算法，比传统的欧氏距离更适合大规模人脸库匹配。
*   **抗模糊处理**: 提供"分析分辨率 (Network Size)"调节功能，支持高达 608px 的输入尺寸，能有效识别后排模糊人脸。

### 2. 智能图片识别与增强 (Smart Image Analysis)
*   **AI 超清修复 (AI Super-Resolution)**: 集成 **Gemini 2.5 Flash Image** 模型。对于模糊的抓拍照片，可一键进行去噪、锐化和超分辨率重建，显著提升识别成功率。
*   **交互式标注**: 提供"清除标注"和"重新识别"功能，方便人工核验细节。

### 3. 学生注册与标准化 (Student Registry)
*   **AI 质量检测**: 注册照片时，调用 **Gemini 2.5 Flash** 自动评估照片质量（光线、角度、清晰度），并返回中文改进建议。
*   **智能标准化 (Auto-Standardization)**: 自动检测人脸并进行 1:1 裁剪和居中校正，确保底库照片的一致性，这是提高识别率的关键。
*   **批量导入**: 支持一次性上传整个班级的照片文件夹，自动以文件名作为姓名进行注册。
*   **数据持久化**: 学生注册数据和识别参数会自动保存到浏览器的localStorage中，即使关闭浏览器也不会丢失。

### 4. 高级辅助工具 (Advanced Tools)
*   **100帧深度确信度分析**: 针对视频流，累计 100 帧的识别结果进行加权投票。有效过滤掉单帧的误报（False Positives），输出高置信度的考勤名单。
*   **人工辅助模式 (Manual Assist)**: 点击视频画面中的任意人脸，强制进行深度检测和特征提取，用于纠正 AI 漏检的目标。
*   **口罩模式 (Mask Mode)**: 专门优化的参数集，降低对口鼻区域的权重依赖，专注于眼部特征匹配。
*   **数据导入导出**: 支持将学生数据导出为JSON文件进行备份，也支持从JSON文件导入学生数据。

## 技术栈 (Tech Stack)

*   **Frontend**: React 19, Tailwind CSS
*   **AI / ML**: 
    *   `face-api.js` (TensorFlow.js) for browser-based face recognition.
    *   `@google/genai` (Gemini API) for image enhancement and quality validation.
*   **Icons**: Lucide React

## 使用说明 (Usage)

### 第一步：环境配置
1.  确保项目根目录下配置了有效的 `API_KEY` (Google Gemini API)。
2.  启动应用。

### 第二步：建立学生底库 (Student Management)
1.  进入 **"学生管理"** 标签页。
2.  **单人注册**: 上传一张学生清晰正面照。点击 **"一键标准化"**，AI 会裁剪出最佳人脸区域并评分。如果评分通过，输入姓名并注册。
3.  **批量导入**: 准备一个文件夹，照片命名为 `姓名.jpg`。点击 **"批量导入"** -> **"开始批量处理"**，系统将自动完成裁剪和注册。
4.  **数据备份**: 点击 **"导出"** 按钮可将所有学生数据保存为JSON文件。点击 **"导入"** 按钮可从JSON文件恢复学生数据。

### 第三步：视频考勤分析 (Real-time Monitor)
1.  进入 **"实时监控"** 标签页。
2.  上传教室监控视频文件 (`.mp4`, `.mov` 等)。
3.  点击 **播放** 按钮，系统开始实时框选和识别人脸。
    *   **绿色框**: 已识别的学生。
    *   **黄色框**: 未知/陌生人。
    *   **红色框**: 嫌疑/警告（可配置）。
4.  **调整参数**: 右侧面板可调整识别阈值。对于模糊视频，建议将 **"分析分辨率"** 调至 `512` 或 `608`。
5.  **生成报告**: 点击 **"100帧 确信度分析"**，系统将自动分析接下来的 100 帧画面，并生成一份稳定的考勤结果截图。

### 第三步：模糊图片处理 (Image Recognition)
1.  进入 **"图片识别"** 标签页。
2.  上传一张模糊的现场照片。
3.  如果识别不出，点击 **"AI 增强"**。等待数秒，图片变清晰后，系统会自动重新进行人脸识别。

## 常见问题 (Troubleshooting)

*   **模型加载失败**: 系统会自动尝试从 VladMandic 镜像、GitHub Pages 和 jsDelivr CDN 加载模型。如果一直转圈，请检查网络是否能访问 GitHub。
*   **识别率低**: 
    1.  检查 **"相似度阈值"**，建议设为 90% (0.90)。
    2.  增大 **"分析分辨率"**。
    3.  确保底库照片质量高（使用标准化功能）。
    4.  如果是侧脸或低头，尝试开启 **"戴口罩增强优化"**（该模式会放宽部分限制）。
*   **数据丢失**: 学生注册数据会自动保存到浏览器的localStorage中。如果数据丢失，可以尝试从备份的JSON文件中导入。

## 数据持久化说明

本系统使用浏览器的localStorage来持久化存储学生注册数据和识别参数：

1. **自动保存**: 每当添加、删除学生或修改识别参数时，数据会自动保存到localStorage。
2. **自动加载**: 应用启动时会自动从localStorage加载之前保存的学生数据和参数。
3. **数据导出**: 可以随时导出学生数据为JSON文件进行备份。
4. **数据导入**: 可以从JSON文件导入学生数据进行恢复。

注意：localStorage的数据仅保存在当前浏览器中，如果更换浏览器或清除浏览器数据，需要重新导入备份文件。