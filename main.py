import click
import json
import logging
from src.pipeline import VoiceAnalysisPipeline

# WARNING 레벨 이상의 로그만 출력하도록 설정
logging.basicConfig(level=logging.WARNING)
logging.getLogger('Kss').setLevel(logging.ERROR) # Kss 로그는 에러만 출력

@click.command()
@click.option('--audio', required=True, type=click.Path(exists=True), help='분석할 음성 파일의 경로')
def main(audio):
    """
    음성 파일을 입력받아 전체 분석 과정을 총괄하는 메인 파이프라인 클래스
    """
    print(f"분석 파일: {audio}")
    pipeline = VoiceAnalysisPipeline()
    
    try:
        results = pipeline.run(audio)
        
        print("\n---  최종 분석 결과 ---")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("-----------------------\n")
        
    except Exception as e:
        print(f"{type(e).__name__}: {e} 발생")

if __name__ == '__main__':
    main()