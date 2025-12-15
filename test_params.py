import requests
import json

# 测试参数更新和人脸检测功能
BASE_URL = "http://localhost:5001"

def test_parameter_update_and_detection():
    print("=== 测试参数更新和人脸检测功能 ===")
    
    # 1. 获取当前参数
    print("\n1. 获取当前参数...")
    response = requests.get(f"{BASE_URL}/api/params")
    print(f"当前参数: {response.json()}")
    
    # 2. 更新参数
    print("\n2. 更新参数...")
    new_params = {
        "min_confidence": 0.1,
        "network_size": 400,
        "min_face_size": 10
    }
    response = requests.post(f"{BASE_URL}/api/params", json=new_params)
    print(f"参数更新结果: {response.json()}")
    
    # 3. 再次获取参数确认更新
    print("\n3. 确认参数更新...")
    response = requests.get(f"{BASE_URL}/api/params")
    print(f"更新后的参数: {response.json()}")
    
    # 4. 进行健康检查
    print("\n4. 健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"健康状态: {response.json()}")

if __name__ == "__main__":
    test_parameter_update_and_detection()