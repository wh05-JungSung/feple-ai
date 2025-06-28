import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.pipeline import VoiceAnalysisPipeline

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 파이프라인 인스턴스는 앱이 시작될 때 한 번만 생성합니다.
pipeline = VoiceAnalysisPipeline()

@app.get("/")
def read_root():
    return {"message": "Feple Voice Analysis API"}

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    # 임시로 파일을 저장할 경로
    temp_file_path = f"temp_{file.filename}"

    try:
        # 업로드된 파일을 임시 파일로 저장
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 저장된 임시 파일을 파이프라인으로 분석
        print(f"Analyzing {temp_file_path}...")
        results = pipeline.run(temp_file_path)

        # 분석 결과가 에러 메시지를 포함하는지 확인
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return results

    except Exception as e:
        # 모든 종류의 에러를 처리
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 분석이 끝나면 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)