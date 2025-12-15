// 后端 API 服务
const API_BASE_URL = 'http://localhost:5001';

export interface FaceDetectionResult {
  id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  landmarks: [number, number][]; // 关键点
  embedding: number[]; // 特征向量
  recognition?: {
    name: string;
    confidence: number;
  }; // 识别结果
}

export interface DetectionResponse {
  success: boolean;
  faces: FaceDetectionResult[];
  count: number;
  params_used?: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  params?: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
}

export interface ParamsResponse {
  success: boolean;
  params: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
  message?: string;
}

/**
 * 检查后端服务健康状态
 */
export async function checkHealth(): Promise<HealthCheckResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('健康检查失败:', error);
    throw new Error('无法连接到后端服务');
  }
}

/**
 * 上传图像文件进行人脸检测
 * @param imageFile 图像文件
 */
export async function detectFacesFromFile(imageFile: File): Promise<DetectionResponse> {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch (e) {
        // If we can't parse the error response, use the text
        try {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        } catch (e) {
          // If we can't get the text, keep the original message
        }
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error: any) {
    console.error('人脸检测失败:', error);
    const errorMessage = error instanceof Error ? error.message : '未知错误';
    throw new Error(`人脸检测失败: ${errorMessage}`);
  }
}

/**
 * 发送 Base64 图像数据进行人脸检测
 * @param imageData Base64 图像数据
 */
export async function detectFacesFromBase64(imageData: string): Promise<DetectionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/detect-base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('人脸检测失败:', error);
    throw new Error(`人脸检测失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

/**
 * 获取后端服务状态信息
 */
export async function getServiceInfo(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('获取服务信息失败:', error);
    throw new Error('无法连接到后端服务');
  }
}

/**
 * 获取当前检测参数
 */
export async function getCurrentParams(): Promise<ParamsResponse> {
  try {
    console.log('发送获取参数请求');
    
    const response = await fetch(`${API_BASE_URL}/api/params`);
    
    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '获取参数失败');
    }

    return responseData;
  } catch (error) {
    console.error('获取参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`获取参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

/**
 * 设置检测参数
 * @param params 参数对象
 */
export async function setDetectionParams(params: {
  min_confidence?: number;
  network_size?: number;
  min_face_size?: number;
}): Promise<ParamsResponse> {
  try {
    console.log('发送参数更新请求:', params);
    
    const response = await fetch(`${API_BASE_URL}/api/params`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '参数更新失败');
    }

    return responseData;
  } catch (error) {
    console.error('设置参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`设置参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}


/**
 * 更新注册学生数据
 * @param students 学生数据数组
 */
export async function updateRegisteredStudents(students: any[]): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/students`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ students }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('更新学生数据失败:', error);
    throw new Error(`更新学生数据失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}
