# feple-ai: AI 기반 통화 품질 분석 시스템

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**feple-ai**는 음성 통화 내용을 AI 모델로 분석하여, 상담 품질을 다각도로 평가하는 AI 분석 시스템입니다. STT(Speech-to-Text), 화자 분리, 자연어 처리(NLP), LLM 기술을 통합하여 통화 내용에 대한 객관적이고 데이터 기반의 인사이트를 제공하는 것을 목표로 합니다.

## ✨ 주요 기능 (Key Features)

-   **🔊 STT (Speech-to-Text)**: `faster-whisper`를 사용하여 음성 파일을 텍스트로 정확하게 변환합니다.
-   **👥 화자 분리 (Speaker Diarization)**: `simple-diarizer`를 통해 대화 참여자들의 발화를 분리하고, 후처리 로직을 통해 '상담사(Agent)'와 '고객(Customer)' 역할을 명확히 구분합니다.
-   **📊 다차원 품질 평가 (Multi-dimensional Quality Assessment)**: 정량적 지표와 정성적 지표를 모두 활용하여 통화 내용을 종합적으로 분석합니다.
    -   **메타데이터**:
        -   `session_id`: 각 분석 세션의 고유 ID
        -   `mid_category` (주제 분류): LLM을 통해 통화의 핵심 주제를 분류
        -   `result_label` (상담 결과): LLM을 통해 상담의 마무리 상태를 분류
        -   `profane` (비속어 사용): LLM을 통해 고객의 비속어 사용 여부를 감지
    -   **상담 태도 지표**:
        -   `honorific_ratio` (존댓말): 형태소 분석 기반의 정확한 존댓말 사용 비율
        -   `positive/negative_word_ratio` (긍정/부정어): KNU 감성사전 기반의 감성 분석
        -   `euphonious_word_ratio` (완곡어/쿠션어): 고객 부담을 줄여주는 표현 사용 비율
        -   `empathy_ratio` (공감 표현): 형태소 및 패턴 분석 기반의 공감 표현 비율
        -   `apology_ratio` (사과 표현): 형태소 분석 기반의 사과 표현 비율
    -   **대화 흐름 지표**:
        -   `avg_response_latency` (평균 응답 속도): 고객 발화 후 상담사 응답까지의 평균 시간
        -   `interruption_count` (가로채기 횟수): 상담사가 고객의 말을 끊은 횟수
        -   `silence_ratio` (침묵 비율): 전체 대화에서 침묵이 차지하는 비율
        -   `talk_ratio` (발화 비율): 상담사 대비 고객의 발화량 비율
    -   **LLM 기반 정성 평가**:
        -   `suggestions` (문제 해결력): OpenAI API를 통해 문제 해결 과정을 단계별로 점수화
        -   `customer_sentiment_trend` (고객 감정 변화): 상담 전후 고객의 감정 변화 추이 분석

## 🛠️ 기술 스택 및 아키텍처 (Tech Stack & Architecture)

### Backend

-   **Language**: `Python 3.12`
-   **AI/ML**: `PyTorch (+cu121)`, `faster-whisper`, `simple-diarizer`, `kiwipiepy`, `kss`, `OpenAI API`
-   **API Framework**: `FastAPI`
-   **Containerization**: `Docker` (NVIDIA CUDA 12.3 `cudnn9` Devel Image 기반)

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

-   [Git](https://git-scm.com/)
-   [Docker](https://www.docker.com/products/docker-desktop/)
-   NVIDIA 그래픽 카드 및 최신 드라이버

### Backend 설정 (Docker)

백엔드 API 서버를 Docker 컨테이너로 실행합니다.

1.  **레포지토리 클론**
    ```bash
    git clone [https://github.com/your-username/feple-ai.git](https://github.com/your-username/feple-ai.git)
    cd feple-ai
    ```

2.  **환경 변수 파일 생성**
    프로젝트 루트에 `.env` 파일을 생성하고, OpenAI에서 발급받은 API 키를 입력합니다.
    ```env
    # .env
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

3.  **Docker 이미지 빌드**
    프로젝트 루트에서 아래 명령어를 실행하여 Docker 이미지를 빌드합니다. 약간의 시간이 소요될 수 있습니다.
    ```bash
    docker build -t feple-ai .
    ```

4.  **Docker 컨테이너 실행**
    빌드가 완료되면, GPU를 사용하는 컨테이너를 실행합니다.
    ```bash
    docker run -p 8000:8000 --gpus all --env-file .env --name feple_api feple-ai
    ```

5.  **로컬 테스트**
    웹 브라우저에서 `http://localhost:8000/docs` 로 접속하여 API 문서를 확인하고, 음성 파일을 업로드하여 직접 테스트할 수 있습니다.

## 📖 API 사용법

### `POST /analyze/`

음성 파일을 업로드하여 통화 품질 분석 결과를 요청합니다.

-   **Request Body**:
    -   `Content-Type`: `multipart/form-data`
    -   `file`: 분석할 오디오 파일 (예: `.wav`, `.mp3`)

-   **Success Response (200 OK)**:
    ```json
    {
        "processing_times": {
            "diarization": "25.31s",
            "stt": "15.78s",
            "merge": "0.01s",
            "post_processing": "0.02s",
            "metrics_calculation": "8.54s",
            "total": "49.66s"
        },
        "transcript": [
            {
                "text": "상담사 답변",
                "speaker": "Agent",
                "start_time": 0.5,
                "end_time": 4.2
            },
            {
                "text": "고객 질문",
                "speaker": "Customer",
                "start_time": 4.5,
                "end_time": 5.1
            }
        ],
        "metrics": {
            "session_id": "1751357467928",
            "mid_category": "주문/결제/입금 확인",
            "result_label": "만족",
            "profane": 0,
            "honorific_ratio": 95.83,
            "positive_word_ratio": 10.15,
            "negative_word_ratio": 1.61,
            "euphonious_word_ratio": 20.83,
            "empathy_ratio": 4.17,
            "apology_ratio": 0.0,
            "suggestions": 0.6,
            "customer_sentiment_early": 0.25,
            "customer_sentiment_late": 0.5,
            "customer_sentiment_trend": 0.25,
            "avg_response_latency": 0.25,
            "interruption_count": 1,
            "silence_ratio": 0.11,
            "talk_ratio": 0.75
        }
    }
    ```
