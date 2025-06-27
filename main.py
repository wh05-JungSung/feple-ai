import click
import json
from src.pipeline import VoiceAnalysisPipeline

@click.command()
@click.option('--audio', required=True, type=click.Path(exists=True), help='Path to the audio file to analyze.')
def main(audio):
    """
    Analyzes a given audio file and extracts various call quality metrics.
    """
    print(f"Starting analysis for: {audio}")
    pipeline = VoiceAnalysisPipeline()
    
    try:
        results = pipeline.run(audio)
        
        # 결과를 예쁘게 출력
        print("\n---  최종 분석 결과 ---")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("-----------------------\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()