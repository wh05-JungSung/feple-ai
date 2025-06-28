# feple-ai: AI 기반 음성 통화 품질 분석 시스템

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Feple은 음성 통화 내용을 AI 모델로 분석하여, 정중함, 긍/부정 표현, 완곡어 사용 등 다양한 품질 지표를 정량적으로 제공하는 풀스택 프로젝트입니다. STT(Speech-to-Text), 화자 분리, 자연어 처리 기술을 통합하여 통화 품질에 대한 객관적이고 데이터 기반의 인사이트를 제공하는 것을 목표로 합니다.

## ✨ 주요 기능 (Key Features)

* **🔊 STT (Speech-to-Text)**: `faster-whisper`를 사용하여 음성 파일을 텍스트로 정확하게 변환합니다.
* **👥 화자 분리 (Speaker Diarization)**: `pyannote.audio`를 통해 대화 참여자들의 발화를 분리하여 누가 언제 말했는지 식별합니다.
* **📊 정량적 품질 평가 (Quantitative Quality Assessment)**: 변환된 텍스트를 기반으로 다음과 같은 핵심 지표를 계산합니다.
    * **존댓말 사용 비율**: 문장 단위로 종결 어미와 선어말 어미 '-시-'를 분석하여 계산합니다.
    * **긍정/부정 표현 비율**: `KiwiPy` 형태소 분석기로 단어의 기본형을 추출하고, KNU 한국어 감성사전과 비교하여 텍스트의 감성을 분석합니다.
    * **완곡어/쿠션어 사용 비율**: 정해진 키워드 및 정규식 패턴을 통해 고객 응대의 부드러움을 측정합니다.

## 🛠️ 기술 스택 및 아키텍처 (Tech Stack & Architecture)

이 프로젝트는 현대적인 웹 기술 스택을 기반으로 구축되었습니다.

### Backend

* **Language**: `Python 3.12`
* **AI/ML**: `PyTorch (+cu121)`, `faster-whisper`, `pyannote.audio`, `kiwipiepy`, `kss`
* **API Framework**: `FastAPI`
* **Containerization**: `Docker` (NVIDIA CUDA 12.1 Devel Image 기반)

### Architecture

```
사용자 (Browser)
     |
     v
Vercel (React Frontend)
     |
     v
Vercel Serverless Function (/api/analyze)
     |
     v
Cloud Service (e.g., Google Cloud Run)
     |
     v
Docker Container (FastAPI Backend on GPU)
     |
     v
분석 결과 (JSON) -> 프론트엔드로 역순 전달
```

## 🚀 시작하기 (Getting Started)

프로젝트를 로컬 환경에서 실행하는 방법입니다.

### 사전 요구사항

* [Git](https://git-scm.com/)
* [Docker](https://www.docker.com/products/docker-desktop/)
* [Node.js](https://nodejs.org/) (v18 이상)
* NVIDIA 그래픽 카드 및 최신 드라이버

### Backend 설정 (Docker)

백엔드 API 서버를 Docker 컨테이너로 실행합니다.

1.  **레포지토리 클론**
    ```bash
    git clone [https://github.com/your-username/feple.git](https://github.com/your-username/feple.git)
    cd feple
    ```

2.  **환경 변수 파일 생성**
    프로젝트 루트에 `.env` 파일을 생성하고, Hugging Face에서 발급받은 Access Token을 입력합니다.
    ```env
    # .env
    HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
    ```

3.  **Docker 이미지 빌드**
    프로젝트 루트에서 아래 명령어를 실행하여 Docker 이미지를 빌드합니다. 약간의 시간이 소요될 수 있습니다.
    ```bash
    docker build -t feple-backend .
    ```

4.  **Docker 컨테이너 실행**
    빌드가 완료되면, GPU를 사용하는 컨테이너를 실행합니다.
    ```bash
    docker run -p 8000:8000 --gpus all --env-file .env --name feple_api feple-backend
    ```

5.  **로컬 테스트**
    웹 브라우저에서 `http://localhost:8000/docs` 로 접속하여 API 문서를 확인하고, 음성 파일을 업로드하여 직접 테스트할 수 있습니다.

## 📖 API 사용법

### `POST /analyze/`

음성 파일을 업로드하여 통화 품질 분석 결과를 요청합니다.

* **Request Body**:
    * `Content-Type`: `multipart/form-data`
    * `file`: 분석할 오디오 파일 (예: `.wav`, `.mp3`)

* **Success Response (200 OK)**:
    ```json
    {
      "metrics": {
        "honorific_ratio": 85.71,
        "positive_word_ratio": 11.15,
        "negative_word_ratio": 6.61,
        "euphonious_word_ratio": 14.28
      },
      "transcript": [
        {
          "text": "네 고객님 무엇을 도와드릴까요",
          "speaker": "SPEAKER_00"
        },
        {
          "text": "제가 어제 주문한 상품이 아직 안 와서요",
          "speaker": "SPEAKER_01"
        }
      ]
    }
    ```
