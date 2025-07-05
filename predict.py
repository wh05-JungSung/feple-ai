from cog import BasePredictor, Input, Path
import tempfile
import requests
from src.pipeline import VoiceAnalysisPipeline

class Predictor(BasePredictor):
    def setup(self):
        """
        Replicate 서버가 시작될 때 한 번만 실행되어,
        모델과 분석 파이프라인을 미리 로드합니다.
        """
        print("Initializing analysis pipeline...")
        self.pipeline = VoiceAnalysisPipeline()
        print("Pipeline initialization complete.")

    def predict(
        self,
        audio: str = Input(description="분석할 오디오 파일의 URL")
    ) -> dict:
        """
        API 요청이 올 때마다 실행되어, 전달된 URL의 음성 파일을 분석합니다.
        """
        print(f"Starting analysis for audio URL: {audio}")

        # --- 파일 다운로드 로직 추가 ---
        try:
            # stream=True로 대용량 파일도 처리 가능하게, timeout을 300초(5분)로 넉넉하게 설정
            with requests.get(audio, stream=True, timeout=300) as r:
                r.raise_for_status() # HTTP 오류가 있으면 예외 발생
                
                # 다운로드한 오디오를 저장할 임시 파일 생성
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    for chunk in r.iter_content(chunk_size=8192): 
                        temp_audio_file.write(chunk)
                    
                    local_audio_path = temp_audio_file.name
            
            print(f"Audio downloaded to temporary path: {local_audio_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download audio file: {e}")
            raise Exception(f"오디오 파일 다운로드 실패: {e}")
        # -----------------------------

        # 이제 원격 URL 대신, 로컬에 저장된 임시 파일 경로로 파이프라인 실행
        results = self.pipeline.run(local_audio_path)
        
        print("Analysis finished.")
        return results