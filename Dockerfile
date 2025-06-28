# 1. 베이스 이미지: CUDA 12.3 + "cuDNN 9"이 포함된, 실제로 존재하는 이미지를 사용합니다.
# PyTorch의 cu121 휠은 상위 버전의 CUDA와 호환되므로, 이 이미지가 모든 호환성 문제를 해결하는 핵심입니다.
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# 2. 시스템 환경 설정 및 필수 패키지 설치
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PIP_ROOT_USER_ACTION=ignore

# 3. PPA 추가 및 Python 3.12와 필수 개발 도구 설치
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        ffmpeg \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 4. python3.12를 기본 python 및 python3으로 설정
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# 5. pip 설치 및 업그레이드 (가장 안정적인 방식)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python
RUN python -m pip install --upgrade pip setuptools wheel

# 6. 작업 디렉토리 설정
WORKDIR /app

# 7. requirements.txt 복사
COPY requirements.txt .

# 8. 모든 라이브러리 설치 (PyTorch 공식 추가 인덱스 사용)
RUN python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 9. 프로젝트의 나머지 파일들을 작업 디렉토리로 복사
COPY . .

# 10. API 서버가 사용할 포트 지정
EXPOSE 8000

# 11. 컨테이너가 시작될 때 실행할 명령어
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]