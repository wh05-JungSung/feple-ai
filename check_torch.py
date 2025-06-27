import torch

print("--- PyTorch/CUDA 진단 시작 ---")
print(f"PyTorch 버전: {torch.__version__}")

is_available = torch.cuda.is_available()
print(f"CUDA 사용 가능 여부 (is_available): {is_available}")

if is_available:
    print(f"CUDA 버전 (PyTorch 빌드): {torch.version.cuda}")
    print(f"연결된 GPU 개수: {torch.cuda.device_count()}")
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("\n>>> 진단 결과: 현재 설치된 PyTorch는 'CPU 전용' 버전입니다.")
    print(">>> 해결 방법: GPU를 지원하는 PyTorch로 재설치해야 합니다. (아래 2단계 참고)")

print("----------------------------")