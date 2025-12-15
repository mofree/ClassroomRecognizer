import requests
import json
import base64

# 测试人脸检测功能
BASE_URL = "http://localhost:5001"

def test_face_detection_with_different_params():
    print("=== 测试不同参数下的人脸检测效果 ===")
    
    # 1. 设置高置信度参数
    print("\n1. 设置高置信度参数...")
    high_conf_params = {
        "min_confidence": 0.8,
        "network_size": 640,
        "min_face_size": 20
    }
    response = requests.post(f"{BASE_URL}/api/params", json=high_conf_params)
    print(f"参数设置结果: {response.json()}")
    
    # 2. 设置低置信度参数
    print("\n2. 设置低置信度参数...")
    low_conf_params = {
        "min_confidence": 0.1,
        "network_size": 400,
        "min_face_size": 10
    }
    response = requests.post(f"{BASE_URL}/api/params", json=low_conf_params)
    print(f"参数设置结果: {response.json()}")
    
    # 3. 验证当前参数
    print("\n3. 验证当前参数...")
    response = requests.get(f"{BASE_URL}/api/params")
    current_params = response.json()
    print(f"当前参数: {current_params}")
    
    print("\n=== 测试完成 ===")
    print("现在请在前端上传一张图片进行人脸检测，")
    print("然后调整参数并点击'重新检测'按钮，观察检测结果的变化。")

if __name__ == "__main__":
    test_face_detection_with_different_params()