from cog import BasePredictor, Input, Path
import tempfile
from src.pipeline import VoiceAnalysisPipeline

class Predictor(BasePredictor):
    def setup(self):
        """
        Replicate 서버가 시작될 때 한 번만 실행되어,
        모델과 분석 파이프라인을 미리 로드합니다.
        """
        print("파이프라인 초기화...")
        self.pipeline = VoiceAnalysisPipeline()
        print("파이프라인 초기화 완료.")

    def predict(
        self,
        audio: Path = Input(description="분석할 음성 파일")
    ) -> dict:
        """
        API 요청이 올 때마다 실행되어, 업로드된 음성 파일을 분석합니다.
        """
        print(f"분석 파일: {audio}")

        # Replicate의 Path 객체를 임시 파일로 저장하여 파이프라인에 전달
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
            with open(audio, "rb") as source_file:
                temp_audio_file.write(source_file.read())
            
            # 파이프라인 실행
            results = self.pipeline.run(temp_audio_file.name)
        
        print("분석 완료.")
        return results