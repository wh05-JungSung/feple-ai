import torch
import librosa
import gc
from dotenv import load_dotenv
import os

from src.utils import load_model_config, load_keyword_config
from src.metrics_calculator import MetricsCalculator

class VoiceAnalysisPipeline:
    def __init__(self):
        # 문자열 "cuda" 대신 torch.device("cuda") 객체로 저장합니다.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model_config = load_model_config()
        self.keyword_config = load_keyword_config()
        self.metrics_calculator = MetricsCalculator(self.keyword_config)
        
        load_dotenv()
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

    def run(self, audio_path):
        audio_waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = librosa.get_duration(y=audio_waveform, sr=sr)

        # 1. 화자 분리
        speaker_turns = self._run_diarization(audio_waveform, sr)
        
        # 2. 음성 인식
        word_segments = self._run_stt(audio_waveform)
        
        # 3. 결과 종합
        structured_transcript = self._merge_results(speaker_turns, word_segments)
        
        # 4. 지표 계산
        final_metrics = self.metrics_calculator.calculate_all_metrics(structured_transcript, total_duration)
        
        return final_metrics

    def _run_diarization(self, waveform, sr):
        from pyannote.audio import Pipeline
        print("[1/4] Running speaker diarization...")
        diarization_pipeline = Pipeline.from_pretrained(
            self.model_config['pyannote'],
            use_auth_token=self.hf_token
        ).to(torch.device("cpu"))
        
        audio_for_diarization = {"waveform": torch.from_numpy(waveform[None, :]), "sample_rate": sr}
        diarization = diarization_pipeline(audio_for_diarization, num_speakers=2)
        
        turns = [{'start': turn.start, 'end': turn.end, 'speaker': speaker} 
                 for turn, _, speaker in diarization.itertracks(yield_label=True)]
        
        del diarization_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        print("Diarization complete. Model unloaded.")
        return turns

    def _run_stt(self, waveform):
        from faster_whisper import WhisperModel
        print("[2/4] Running Speech-to-Text...")
        stt_model = WhisperModel(self.model_config['whisper'], device=str(self.device), compute_type="int8")
        
        segments, _ = stt_model.transcribe(waveform, word_timestamps=True)
        
        words = []
        for segment in segments:
            for word in segment.words:
                words.append({'start': word.start, 'end': word.end, 'text': word.word})

        del stt_model
        gc.collect()
        torch.cuda.empty_cache()
        print("STT complete. Model unloaded.")
        return words

    def _merge_results(self, speaker_turns, word_segments):
        print("[3/4] Merging results...")
        for word in word_segments:
            word['speaker'] = 'UNKNOWN'
            for turn in speaker_turns:
                if word['start'] >= turn['start'] and word['end'] <= turn['end']:
                    word['speaker'] = turn['speaker']
                    break
        
        if not word_segments: return []
        
        merged_transcript = []
        current_segment = {'text': word_segments[0]['text'], 'speaker': word_segments[0]['speaker']}

        for i in range(1, len(word_segments)):
            if word_segments[i]['speaker'] == current_segment['speaker']:
                current_segment['text'] += ' ' + word_segments[i]['text']
            else:
                merged_transcript.append(current_segment)
                current_segment = {'text': word_segments[i]['text'], 'speaker': word_segments[i]['speaker']}
        merged_transcript.append(current_segment)
        
        return merged_transcript