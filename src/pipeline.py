import torch
import librosa
import gc
from dotenv import load_dotenv
import os
import logging
import time
import re
from collections import Counter
from multiprocessing import Process, Queue, set_start_method
import random

# --- CUDA 멀티프로세싱 오류 해결 ---
# 'spawn' 시작 방식을 설정하여 CUDA 초기화 충돌을 방지합니다.
# 프로그램이 시작될 때 한 번만 호출되어야 합니다.
try:
    set_start_method('spawn', force=True)
    print("멀티프로세싱 오류 해결")
except RuntimeError:
    pass

from simple_diarizer.diarizer import Diarizer
from src.utils import load_model_config, load_keyword_config
from src.metrics_calculator import MetricsCalculator

def run_diarization_in_process(audio_path, result_queue):
    logging.basicConfig(level=logging.WARNING)
    try:
        print("[1/4] 화자 분리 (simple_diarizer)")
        diarizer = Diarizer(
            embed_model='xvec',
            cluster_method='sc',
            window=0.7,
            period=0.35
        )
        segments = diarizer.diarize(audio_path, num_speakers=2)

        turns = [
            {'start': seg['start'], 'end': seg['end'], 'speaker': f"SPEAKER_{seg['label']:02d}"}
            for seg in segments
        ]
        result_queue.put(turns)
    except Exception as e:
        result_queue.put(e)

class VoiceAnalysisPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용할 하드웨어: {self.device}")

        self.model_config = load_model_config()
        self.keyword_config = load_keyword_config()
        self.metrics_calculator = MetricsCalculator(self.keyword_config)

        load_dotenv()
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

    def _preprocess_text(self, text):
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _run_stt(self, waveform):
        from faster_whisper import WhisperModel
        print("[2/4] 음성 인식(STT) 시작")
        stt_model = WhisperModel(self.model_config['whisper'], device=str(self.device), compute_type="int8")
        
        segments, _ = stt_model.transcribe(waveform, word_timestamps=True)
        
        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    clean_text = self._preprocess_text(word.word)
                    if clean_text:
                        words.append({'start': word.start, 'end': word.end, 'text': clean_text})

        del stt_model
        gc.collect()
        torch.cuda.empty_cache()
        return words

    def run(self, audio_path):
        session_id = f"{int(time.time())}{random.randint(100, 999)}"
        print(f"새로운 분석 세션을 시작합니다. (ID: {session_id})")

        total_start_time = time.time()
        processing_times = {}

        # 1. 화자 분리
        diarization_start_time = time.time()
        result_queue = Queue()
        diarization_process = Process(target=run_diarization_in_process, args=(audio_path, result_queue))
        diarization_process.start()
        speaker_turns = result_queue.get()
        diarization_process.join()

        if isinstance(speaker_turns, Exception):
            raise speaker_turns
        
        processing_times['diarization'] = time.time() - diarization_start_time
        print(f"화자 분리 완료. (소요 시간: {processing_times['diarization']:.2f}초)")

        # 2. 음성 인식 (STT)
        stt_start_time = time.time()
        audio_waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(audio_waveform) / sr
        word_segments = self._run_stt(audio_waveform)
        processing_times['stt'] = time.time() - stt_start_time
        print(f"음성 인식(STT) 완료. (소요 시간: {processing_times['stt']:.2f}초)")
        
        # 3. 결과 종합
        merge_start_time = time.time()
        structured_transcript = self._merge_results(speaker_turns, word_segments)
        processing_times['merge'] = time.time() - merge_start_time
        print(f"결과 종합 완료. (소요 시간: {processing_times['merge']:.2f}초)")

        # 4. 후처리
        postprocess_start_time = time.time()
        final_transcript = self._postprocess_transcript(structured_transcript)
        processing_times['post_processing'] = time.time() - postprocess_start_time
        print(f"후처리 완료. (소요 시간: {processing_times['post_processing']:.2f}초)")

        # 5. 지표 계산
        metrics_start_time = time.time()
        final_metrics = self.metrics_calculator.calculate_all_metrics(
            final_transcript, 
            speaker_turns, 
            total_duration,
            session_id
        )
        processing_times['metrics_calculation'] = time.time() - metrics_start_time
        print(f"지표 계산 완료. (소요 시간: {processing_times['metrics_calculation']:.2f}초)")

        processing_times['total'] = time.time() - total_start_time
        
        final_results = {
            "processing_times": {k: f"{v:.2f}s" for k, v in processing_times.items()},
            "transcript": final_transcript,
            "metrics": final_metrics
        }
        
        return final_results

    def _merge_results(self, speaker_turns, word_segments):
        print("[3/4] 결과 종합 시작")
        if not word_segments: return []

        for word in word_segments:
            word['speaker'] = 'UNKNOWN'
            word_mid_point = word['start'] + (word['end'] - word['start']) / 2
            for turn in speaker_turns:
                if word_mid_point >= turn['start'] and word_mid_point <= turn['end']:
                    word['speaker'] = turn['speaker']
                    break
        
        merged_transcript = []
        if not word_segments:
            return merged_transcript
            
        current_segment = {
            'text': word_segments[0]['text'], 
            'speaker': word_segments[0]['speaker'],
            'start_time': word_segments[0]['start'],
            'end_time': word_segments[0]['end']
        }

        for i in range(1, len(word_segments)):
            word = word_segments[i]
            if word['speaker'] == current_segment['speaker'] and \
               word['start'] - word_segments[i-1]['end'] < 1.0:
                current_segment['text'] += ' ' + word['text']
                current_segment['end_time'] = word['end']
            else:
                merged_transcript.append(current_segment)
                current_segment = {
                    'text': word['text'], 
                    'speaker': word['speaker'],
                    'start_time': word['start'],
                    'end_time': word['end']
                }
        merged_transcript.append(current_segment)
        
        return merged_transcript

    def _postprocess_transcript(self, transcript):
        print("[후처리] 대본 정리 시작")
        if not transcript: return []

        for i, segment in enumerate(transcript):
            if segment['speaker'] == 'UNKNOWN':
                if i > 0:
                    segment['speaker'] = transcript[i-1]['speaker']
                else:
                    for next_segment in transcript[i+1:]:
                        if next_segment['speaker'] != 'UNKNOWN':
                            segment['speaker'] = next_segment['speaker']
                            break

        remerged_transcript = []
        if transcript:
            remerged_transcript.append(transcript[0])
            for segment in transcript[1:]:
                if segment['speaker'] == remerged_transcript[-1]['speaker']:
                    remerged_transcript[-1]['text'] += ' ' + segment['text']
                    remerged_transcript[-1]['end_time'] = segment['end_time']
                else:
                    remerged_transcript.append(segment)
        
        speaker_counts = Counter(seg['speaker'] for seg in remerged_transcript)
        
        if len(speaker_counts) > 0:
            agent_label = speaker_counts.most_common(1)[0][0]
            
            for seg in remerged_transcript:
                if seg['speaker'] == agent_label:
                    seg['speaker'] = 'Agent'
                else:
                    seg['speaker'] = 'Customer'

        print("[후처리] 대본 정리 완료")
        return remerged_transcript