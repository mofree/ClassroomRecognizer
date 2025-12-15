import requests
import json
import os
import base64

# 测试批量处理功能
BASE_URL = "http://localhost:5001"

def test_batch_processing():
    print("=== 测试批量处理功能 ===")
    
    # 1. 检查健康状态
    print("\n1. 检查后端健康状态...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"健康状态: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return
    
    # 2. 获取当前参数
    print("\n2. 获取当前参数...")
    try:
        response = requests.get(f"{BASE_URL}/api/params", timeout=10)
        print(f"当前参数: {response.json()}")
    except Exception as e:
        print(f"获取参数失败: {e}")
        return
    
    # 3. 测试参数更新
    print("\n3. 测试参数更新...")
    try:
        new_params = {
            "min_confidence": 0.1,
            "network_size": 400,
            "min_face_size": 10
        }
        response = requests.post(f"{BASE_URL}/api/params", json=new_params, timeout=10)
        result = response.json()
        print(f"参数更新结果: {result}")
        if result.get('success'):
            print(f"更新后的参数: {result.get('params')}")
    except Exception as e:
        print(f"参数更新失败: {e}")
    
    # 4. 测试人脸检测API
    print("\n4. 测试人脸检测API...")
    # 查找测试图片
    test_images = []
    test_dir = "./test_images"
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(test_dir, file))
    
    if not test_images:
        print("未找到测试图片，请在项目根目录创建test_images文件夹并放入一些图片")
        # 创建一个简单的测试图片
        try:
            # 创建一个简单的测试图片 (纯色图片)
            import numpy as np
            from PIL import Image
            
            # 创建一个红色的图片
            img = Image.new('RGB', (100, 100), color = 'red')
            test_img_path = "./test_red_image.png"
            img.save(test_img_path)
            test_images.append(test_img_path)
            print(f"创建了一个测试图片: {test_img_path}")
        except Exception as e:
            print(f"创建测试图片失败: {e}")
            return
    
    print(f"找到 {len(test_images)} 张测试图片")
    
    for i, image_path in enumerate(test_images[:3]):  # 只测试前3张图片
        print(f"\n  测试图片 {i+1}: {os.path.basename(image_path)}")
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(f"{BASE_URL}/api/detect", files=files, timeout=30)
                
            if response.status_code == 200:
                result = response.json()
                print(f"    检测结果: {result['count']} 个人脸")
                if 'params_used' in result:
                    print(f"    使用参数: {result['params_used']}")
                if 'faces' in result and result['faces']:
                    for j, face in enumerate(result['faces'][:3]):  # 显示前3个人脸
                        print(f"      人脸 {j+1}: 置信度 {face['confidence']:.4f}")
            else:
                print(f"    检测失败: {response.status_code}")
                try:
                    error_text = response.text
                    print(f"    错误详情: {error_text}")
                except:
                    print(f"    无法获取错误详情")
        except Exception as e:
            print(f"    检测出错: {e}")

if __name__ == "__main__":
    test_batch_processing()