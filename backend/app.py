from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import base64
from io import BytesIO
from PIL import Image
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 创建上传目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 全局变量存储模型和参数
face_app = None
current_params = {
    "min_confidence": 0.5,
    "network_size": 640,
    "min_face_size": 20
}

# 存储学生数据用于人脸识别
registered_students = []
def initialize_insightface():
    """初始化 InsightFace 模型"""
    global face_app
    try:
        logger.info("正在初始化 InsightFace...")
        face_app = FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace 初始化成功")
        return True
    except Exception as e:
        logger.error(f"InsightFace 初始化失败: {e}")
        face_app = None
        return False

# 应用启动时初始化模型
initialize_insightface()

@app.route('/')
def index():
    return jsonify({
        "message": "人脸检测后端服务已启动", 
        "status": "running",
        "version": "1.0.0"
    })

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    if face_app is None:
        return jsonify({"error": "人脸识别模型未初始化"}), 500
    
    try:
        # 获取上传的图像文件
        if 'image' not in request.files:
            return jsonify({"error": "没有上传图像文件"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400
        
        # 保存上传的文件
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # 读取图像
        img = cv2.imread(filename)
        if img is None:
            return jsonify({"error": "无法读取图像文件"}), 400
        
        # 设置检测参数
        global current_params
        # 重新准备模型以应用新的检测参数
        face_app.prepare(ctx_id=0, det_thresh=current_params["min_confidence"], det_size=(current_params["network_size"], current_params["network_size"]))
        
        # 设置最小人脸尺寸
        if hasattr(face_app.det_model, 'min_face_size'):
            face_app.det_model.min_face_size = current_params["min_face_size"]
        
        # 人脸检测
        logger.info(f"开始人脸检测，使用参数: {current_params}")
        faces = face_app.get(img)
        logger.info(f"检测到 {len(faces)} 个人脸")
        
        # 过滤低置信度的人脸
        filtered_faces = [face for face in faces if face.det_score >= current_params["min_confidence"]]
        logger.info(f"过滤后剩余 {len(filtered_faces)} 个人脸 (最小置信度: {current_params['min_confidence']})")
        
        # 处理检测结果
        results = []
        for i, face in enumerate(filtered_faces):
            result = {
                "id": i,
                "bbox": face.bbox.tolist(),  # 边界框 [x1, y1, x2, y2]
                "confidence": float(face.det_score),  # 检测置信度
                "landmarks": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else [],  # 关键点
                "embedding": face.embedding.tolist() if hasattr(face, 'embedding') else []  # 特征向量
            }
            results.append(result)
        
        # 删除临时文件
        os.remove(filename)
        
        return jsonify({
            "success": True,
            "faces": results,
            "count": len(results),
            "params_used": current_params
        })
        
    except Exception as e:
        logger.error(f"人脸检测出错: {e}")
        return jsonify({"error": f"人脸检测出错: {str(e)}"}), 500

def smart_deduplication(faces):
    """智能去重：保留相似度高的，去掉相似度低的，并对被去掉的人脸重新匹配"""
    if len(faces) <= 1:
        return faces
    
    # 对人脸按检测置信度排序，置信度高的优先处理
    sorted_faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
    
    # 用于存储去重后的结果
    deduplicated_faces = []
    removed_faces = []  # 存储被移除的人脸
    
    for face in sorted_faces:
        # 检查当前人脸是否与已选择的人脸重叠
        is_duplicate = False
        current_bbox = face.bbox
        
        for selected_face in deduplicated_faces:
            selected_bbox = selected_face.bbox
            
            # 计算IoU (Intersection over Union)
            iou = calculate_iou(current_bbox, selected_bbox)
            
            # 如果IoU超过阈值，认为是重复检测
            if iou > 0.3:  # 可以根据需要调整这个阈值
                is_duplicate = True
                # 保留置信度高的，移除置信度低的
                if face.det_score < selected_face.det_score:
                    # 当前人脸置信度更低，应该被移除
                    removed_faces.append(face)
                else:
                    # 已选择的人脸置信度更低，应该被替换
                    deduplicated_faces.remove(selected_face)
                    deduplicated_faces.append(face)
                    removed_faces.append(selected_face)
                break
        
        # 如果不是重复的，添加到结果中
        if not is_duplicate:
            deduplicated_faces.append(face)
    
    logger.info(f"智能去重：从 {len(faces)} 个人脸中筛选出 {len(deduplicated_faces)} 个，移除 {len(removed_faces)} 个")
    return deduplicated_faces


@app.route('/api/detect-base64', methods=['POST'])
def detect_faces_base64():
    if face_app is None:
        return jsonify({"error": "人脸识别模型未初始化"}), 500
    
    try:
        # 获取 Base64 图像数据
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "没有提供图像数据"}), 400
        
        # 解码 Base64 图像
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 设置检测参数
        global current_params
        # 重新准备模型以应用新的检测参数
        face_app.prepare(ctx_id=0, det_thresh=current_params["min_confidence"], det_size=(current_params["network_size"], current_params["network_size"]))
        
        # 设置最小人脸尺寸
        if hasattr(face_app.det_model, 'min_face_size'):
            face_app.det_model.min_face_size = current_params["min_face_size"]
        
        # 人脸检测
        logger.info(f"开始人脸检测，使用参数: {current_params}")
        faces = face_app.get(img)
        logger.info(f"检测到 {len(faces)} 个人脸")
        
        # 过滤低置信度的人脸
        filtered_faces = [face for face in faces if face.det_score >= current_params["min_confidence"]]
        logger.info(f"过滤后剩余 {len(filtered_faces)} 个人脸 (最小置信度: {current_params['min_confidence']})")
        
        # 智能去重：保留相似度高的，去掉相似度低的
        processed_faces = smart_deduplication(filtered_faces)
        
        # 处理检测结果
        results = []
        for i, face in enumerate(processed_faces):
            # 计算人脸识别结果
            recognition_result = None
            if hasattr(face, 'embedding') and registered_students:
                # 计算与注册学生的相似度
                best_match = find_best_match(face.embedding)
                if best_match:
                    recognition_result = {
                        "name": best_match["name"],
                        "confidence": best_match["confidence"]
                    }
            
            result = {
                "id": i,
                "bbox": [float(x) for x in face.bbox],  # 边界框 [x1, y1, x2, y2]
                "confidence": float(face.det_score),  # 检测置信度
                "landmarks": [[float(x), float(y)] for x, y in face.landmark_2d_106] if hasattr(face, 'landmark_2d_106') else [],  # 关键点
                "embedding": face.embedding.tolist() if hasattr(face, 'embedding') else [],  # 特征向量
                "recognition": recognition_result  # 人脸识别结果
            }
            results.append(result)        
        return jsonify({
            "success": True,
            "faces": results,
            "count": len(results),
            "params_used": current_params
        })
        
    except Exception as e:
        logger.error(f"人脸检测出错: {e}")
        return jsonify({"error": f"人脸检测出错: {str(e)}"}), 500


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的IoU (Intersection over Union)"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集区域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 计算交集面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def find_best_match(query_embedding):
    """查找最佳匹配的学生"""
    global registered_students
    
    logger.info(f"开始人脸识别，注册学生数量: {len(registered_students)}")
    
    if not registered_students:
        logger.info("没有注册学生，跳过人脸识别")
        return None
    
    best_match = None
    highest_similarity = -1
    
    # 将查询嵌入转换为numpy数组
    query_vec = np.array(query_embedding)
    logger.info(f"查询向量维度: {query_vec.shape}")
    
    for i, student in enumerate(registered_students):
        # 学生可能有多个注册照片，检查所有描述符
        # 注意：前端发送的数据结构可能包含额外字段，我们需要提取正确的字段
        descriptors = student.get("descriptors", [])
        if not descriptors:
            # 尝试其他可能的字段名
            descriptors = student.get("descriptor", [])
        
        logger.info(f"学生 {i+1} ({student.get('name', '未知')}) 描述符数量: {len(descriptors)}")
        
        for j, descriptor in enumerate(descriptors):
            # 将学生描述符转换为numpy数组
            # 注意：descriptor可能是列表或numpy数组
            if isinstance(descriptor, list):
                student_vec = np.array(descriptor)
            else:
                student_vec = descriptor
            
            logger.info(f"  描述符 {j+1} 维度: {student_vec.shape}")
            
            # 计算余弦相似度
            dot_product = np.dot(query_vec, student_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_student = np.linalg.norm(student_vec)
            
            logger.info(f"  点积: {dot_product}, 查询范数: {norm_query}, 学生范数: {norm_student}")
            
            if norm_query != 0 and norm_student != 0:
                similarity = dot_product / (norm_query * norm_student)
                logger.info(f"  相似度: {similarity}")
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = {
                        "name": student.get("name", "未知学生"),
                        "confidence": float(similarity)
                    }
                    logger.info(f"  更新最佳匹配: {best_match['name']} ({similarity})")
    
    # 只有当相似度足够高时才返回匹配结果
    # 设置合理的阈值以提高识别率，特别是在教室场景下
    if best_match and best_match["confidence"] >= 0.1:  # 设置阈值为0.1
        logger.info(f"找到最佳匹配: {best_match['name']} ({best_match['confidence']})")
        return best_match
    
    logger.info("未找到匹配度足够的学生")
    return None


@app.route('/api/params', methods=['GET', 'POST'])
def handle_params():
    """处理参数获取和设置"""
    if request.method == 'GET':
        return get_params()
    elif request.method == 'POST':
        return set_params()

@app.route('/api/students', methods=['POST'])
def update_students():
    """更新注册学生数据"""
    global registered_students
    try:
        data = request.get_json()
        if not data or 'students' not in data:
            return jsonify({"error": "没有提供学生数据"}), 400
            
        registered_students = data['students']
        logger.info(f"更新了 {len(registered_students)} 名注册学生")
        
        return jsonify({
            "success": True,
            "message": f"成功更新 {len(registered_students)} 名注册学生",
            "count": len(registered_students)
        })
    except Exception as e:
        logger.error(f"更新学生数据失败: {e}", exc_info=True)
        return jsonify({"error": f"更新学生数据失败: {str(e)}"}), 500

def get_params():
    """获取当前参数"""
    try:
        logger.info(f"返回当前参数: {current_params}")
        return jsonify({
            "success": True,
            "params": current_params
        })
    except Exception as e:
        logger.error(f"获取参数失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"获取参数失败: {str(e)}"
        }), 500

def set_params():
    """设置检测参数"""
    global current_params
    try:
        data = request.get_json()
        if not data:
            logger.warning("没有提供参数数据")
            return jsonify({"error": "没有提供参数数据"}), 400
            
        logger.info(f"收到参数更新请求: {data}")
        
        # 更新参数并进行验证
        updated_params = {}
        errors = []
        
        if 'min_confidence' in data:
            try:
                min_confidence = float(data['min_confidence'])
                if 0.01 <= min_confidence <= 1.0:
                    current_params['min_confidence'] = min_confidence
                    updated_params['min_confidence'] = min_confidence
                else:
                    errors.append(f"min_confidence 必须在 0.01-1.0 之间，当前值: {min_confidence}")
            except (ValueError, TypeError) as e:
                errors.append(f"min_confidence 格式错误: {str(e)}")
        
        if 'network_size' in data:
            try:
                network_size = int(data['network_size'])
                if 320 <= network_size <= 1024:
                    current_params['network_size'] = network_size
                    updated_params['network_size'] = network_size
                else:
                    errors.append(f"network_size 必须在 320-1024 之间，当前值: {network_size}")
            except (ValueError, TypeError) as e:
                errors.append(f"network_size 格式错误: {str(e)}")
        
        if 'min_face_size' in data:
            try:
                min_face_size = int(data['min_face_size'])
                if 10 <= min_face_size <= 100:
                    current_params['min_face_size'] = min_face_size
                    updated_params['min_face_size'] = min_face_size
                else:
                    errors.append(f"min_face_size 必须在 10-100 之间，当前值: {min_face_size}")
            except (ValueError, TypeError) as e:
                errors.append(f"min_face_size 格式错误: {str(e)}")
            
        if errors:
            logger.warning(f"参数验证失败: {errors}")
            return jsonify({"error": "参数验证失败", "details": errors}), 400
            
        logger.info(f"参数已更新: {current_params}")
        
        return jsonify({
            "success": True,
            "message": "参数更新成功",
            "params": current_params,
            "updated_fields": list(updated_params.keys())
        })
    except Exception as e:
        logger.error(f"参数更新失败: {e}", exc_info=True)
        return jsonify({"error": f"参数更新失败: {str(e)}"}), 500

@app.route('/health')
def health_check():
    status = "healthy" if face_app is not None else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": face_app is not None,
        "params": current_params
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)