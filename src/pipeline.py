import torch
import librosa
import gc
from dotenv import load_dotenv
import os
import logging
import time

from multiprocessing import Process, Queue

from simple_diarizer.diarizer import Diarizer
from src.utils import load_model_config, load_keyword_config
from src.metrics_calculator import MetricsCalculator

def run_diarization_in_process(audio_path, result_queue):
    logging.basicConfig(level=logging.WARNING)
    try:
        print("[1/4] 화자 분리 (simple_diarizer)")
        diarizer = Diarizer(embed_model='xvec', cluster_method='sc')
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

    def run(self, audio_path):
        total_start_time = time.time()
        
        processing_times = {}

        # 1. 화자 분리
        diarization_start_time = time.time()
        result_queue = Queue()
        diarization_process = Process(target=run_diarization_in_process, args=(audio_path, result_queue))
        diarization_process.start()
        result = result_queue.get()
        diarization_process.join()

        if isinstance(result, Exception):
            raise result
        
        speaker_turns = result
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

        # 4. 지표 계산
        metrics_start_time = time.time()
        final_metrics = self.metrics_calculator.calculate_all_metrics(structured_transcript, total_duration)
        processing_times['metrics_calculation'] = time.time() - metrics_start_time
        print(f"지표 계산 완료. (소요 시간: {processing_times['metrics_calculation']:.2f}초)")

        processing_times['total'] = time.time() - total_start_time
        
        final_results = {
            "processing_times": {k: f"{v:.2f}s" for k, v in processing_times.items()},
            "diarization_result": speaker_turns,
            "transcript": structured_transcript,
            "metrics": final_metrics
        }
        
        return final_results

    def _run_stt(self, waveform):
        from faster_whisper import WhisperModel
        print("[2/4] 음성 인식(STT) 시작")
        stt_model = WhisperModel(self.model_config['whisper'], device=str(self.device), compute_type="int8")
        
        segments, _ = stt_model.transcribe(waveform, word_timestamps=True)
        
        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words.append({'start': word.start, 'end': word.end, 'text': word.word})

        del stt_model
        gc.collect()
        torch.cuda.empty_cache()
        return words

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
            
        current_segment = {'text': word_segments[0]['text'], 'speaker': word_segments[0]['speaker']}

        for i in range(1, len(word_segments)):
            if word_segments[i]['speaker'] == current_segment['speaker'] and \
               word_segments[i]['start'] - word_segments[i-1]['end'] < 1.0:
                current_segment['text'] += ' ' + word_segments[i]['text']
            else:
                merged_transcript.append(current_segment)
                current_segment = {'text': word_segments[i]['text'], 'speaker': word_segments[i]['speaker']}
        merged_transcript.append(current_segment)
        
        return merged_transcript