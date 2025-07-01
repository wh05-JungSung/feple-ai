import json
import re
import kss
import numpy as np
from collections import defaultdict
from kiwipiepy import Kiwi
from src.llm_evaluator import LLMEvaluator

class MetricsCalculator:
    kiwi = None
    senti_dict = None
    llm_evaluator = None

    def __init__(self, keywords):
        self.keywords = keywords
        
        if MetricsCalculator.kiwi is None:
            print("KiwiPy 형태소 분석기 초기화 중...")
            MetricsCalculator.kiwi = Kiwi()
            for word in self._get_senti_words():
                MetricsCalculator.kiwi.add_user_word(word, 'NNP', 0)
            print("KiwiPy 초기화 및 사용자 사전 추가 완료.")

        if MetricsCalculator.senti_dict is None:
            MetricsCalculator.senti_dict = self._load_knu_senti_lexicon()
        
        if MetricsCalculator.llm_evaluator is None:
            print("LLM 평가기 초기화 중...")
            MetricsCalculator.llm_evaluator = LLMEvaluator()

        self.euphonious_patterns = [
            re.compile(p) for p in [
                r'인 것 같습니다', r'ㄹ 것 같습니다', r'일 듯합니다', r'ㄹ지도 모릅니다',
                r'하기는 어렵습니다', r'ㄹ 수 있을까요\??', r'해 주시겠어요\??',
                r'드릴까요\??', r'해드릴까요\??', r'부탁드립니다'
            ]
        ]
    
    def _get_senti_words(self):
        try:
            with open('data/SentiWord_info.json', 'r', encoding='utf-8') as f:
                senti_data = json.load(f)
            return [item['word_root'] for item in senti_data]
        except:
            return []

    def _load_knu_senti_lexicon(self):
        senti_dict = {}
        try:
            with open('data/SentiWord_info.json', 'r', encoding='utf-8') as f:
                senti_data = json.load(f)
            for item in senti_data:
                word = item['word_root']
                polarity = int(item['polarity'])
                senti_dict[word] = polarity
            print("KNU 감성사전 로드 완료.")
            return senti_dict
        except FileNotFoundError:
            print("[경고] 'data/SentiWord_info.json' 파일을 찾을 수 없습니다.")
            return {}
            
    def calculate_all_metrics(self, structured_transcript, total_duration_seconds):
        agent_words, agent_sentences = self._extract_agent_data(structured_transcript)
        customer_sentences = self._extract_customer_sentences(structured_transcript)
        
        if not agent_words:
            return {"error": "상담사 발화가 없습니다."}

        # --- 고객 감정 추세 분석 ---
        print("LLM 기반 고객 감정 추세 분석 시작...")
        customer_sentiment_scores = [self.llm_evaluator.get_sentiment_score(sent) for sent in customer_sentences]
        
        sentiment_early = 0
        sentiment_late = 0
        if len(customer_sentiment_scores) >= 3:
            n_sentences = len(customer_sentiment_scores)
            split_point = n_sentences // 3
            
            early_scores = customer_sentiment_scores[:split_point]
            late_scores = customer_sentiment_scores[-split_point:]
            
            sentiment_early = np.mean(early_scores) if early_scores else 0
            sentiment_late = np.mean(late_scores) if late_scores else 0
        
        sentiment_trend = sentiment_late - sentiment_early
        print(f"고객 감정 추세 분석 완료. (초반: {sentiment_early:.2f}, 후반: {sentiment_late:.2f})")
        
        total_sentence_count = len(agent_sentences)
        if total_sentence_count == 0:
            return {
                "honorific_ratio": 0, "positive_word_ratio": 0, "negative_word_ratio": 0,
                "euphonious_word_ratio": 0, "empathy_ratio": 0, "apology_ratio": 0,
                "suggestions": 0.0, "customer_sentiment_early": float(sentiment_early),
                "customer_sentiment_late": float(sentiment_late), "customer_sentiment_trend": float(sentiment_trend)
            }

        positive_morph_count, negative_morph_count, total_morph_count = self._count_sentiment_morphemes(agent_words)
        honorific_sentence_count = self._count_honorific_sentences(agent_sentences)
        euphonious_sentence_count = self._count_euphonious_sentences(agent_sentences)
        empathy_sentence_count = self._count_empathy_sentences(agent_sentences)
        apology_sentence_count = self._count_apology_sentences(agent_sentences)
        
        print("LLM 기반 정성 평가 시작...")
        suggestions = self.llm_evaluator.get_suggestion_score(structured_transcript)
        print(f"LLM 기반 정성 평가 완료. (점수: {suggestions})")

        # --- 요청하신 순서대로 최종 지표 딕셔너리 구성 ---
        final_metrics = {
            "honorific_ratio": (honorific_sentence_count / total_sentence_count) * 100,
            "positive_word_ratio": (positive_morph_count / total_morph_count) * 100 if total_morph_count > 0 else 0,
            "negative_word_ratio": (negative_morph_count / total_morph_count) * 100 if total_morph_count > 0 else 0,
            "euphonious_word_ratio": (euphonious_sentence_count / total_sentence_count) * 100,
            "empathy_ratio": (empathy_sentence_count / total_sentence_count) * 100,
            "apology_ratio": (apology_sentence_count / total_sentence_count) * 100,
            "suggestions": suggestions,
            "customer_sentiment_early": float(sentiment_early),
            "customer_sentiment_late": float(sentiment_late),
            "customer_sentiment_trend": float(sentiment_trend)
        }
        return final_metrics

    def _extract_agent_data(self, transcript):
        agent_turns = [seg for seg in transcript if seg.get('speaker') == 'Agent']
        
        if not agent_turns:
            return [], []

        agent_full_text = ' '.join([turn['text'] for turn in agent_turns])
        agent_words = agent_full_text.split()
        agent_sentences = kss.split_sentences(agent_full_text)
        
        return agent_words, agent_sentences

    def _extract_customer_sentences(self, transcript):
        customer_turns = [seg for seg in transcript if seg.get('speaker') == 'Customer']
        if not customer_turns:
            return []
        customer_full_text = ' '.join([turn['text'] for turn in customer_turns])
        return kss.split_sentences(customer_full_text)

    def _count_sentiment_morphemes(self, words):
        if not self.senti_dict or not self.kiwi:
            return 0, 0, 0
            
        positive_count = 0
        negative_count = 0
        total_morph_count = 0
        
        full_text = " ".join(words)
        result = self.kiwi.tokenize(full_text)
        
        for token in result:
            total_morph_count += 1
            if token.tag in ['VA', 'VV', 'XSA', 'XSV']:
                stem = token.form + '다'
            else:
                stem = token.form

            if stem in self.senti_dict:
                polarity = self.senti_dict[stem]
                if polarity > 0:
                    positive_count += 1
                elif polarity < 0:
                    negative_count += 1
                    
        return positive_count, negative_count, total_morph_count

    def _count_honorific_sentences(self, sentences):
        count = 0
        honorific_ending = re.compile(r'(습니다|ㅂ니다|세요|셔요|까요\?)$')

        for sent in sentences:
            if honorific_ending.search(sent):
                count += 1
                continue

            tokens = self.kiwi.tokenize(sent)
            for token in tokens:
                if token.tag == 'EP' and token.form == '시':
                    count += 1
                    break
        return count

    def _count_euphonious_sentences(self, sentences):
        count = 0
        cushion_words = self.keywords.get('cushion', [])
        for sent in sentences:
            found = False
            for word in cushion_words:
                if word in sent:
                    count += 1
                    found = True
                    break
            if found:
                continue

            for pattern in self.euphonious_patterns:
                if pattern.search(sent):
                    count += 1
                    break
        return count

    def _count_empathy_sentences(self, sentences):
        count = 0
        empathy_roots = self.keywords.get('empathy_roots', [])
        empathy_patterns = self.keywords.get('empathy_patterns', [])
        
        for sent in sentences:
            found_by_pattern = False
            for pattern in empathy_patterns:
                if pattern in sent:
                    count += 1
                    found_by_pattern = True
                    break
            if found_by_pattern:
                continue

            tokens = self.kiwi.tokenize(sent)
            for token in tokens:
                if token.form in empathy_roots and token.tag in ['NNG', 'XR']:
                    count += 1
                    break 
        return count
        
    def _count_apology_sentences(self, sentences):
        count = 0
        apology_roots = self.keywords.get('apology_roots', [])
        for sent in sentences:
            tokens = self.kiwi.tokenize(sent)
            for token in tokens:
                if token.form in apology_roots and token.tag in ['NNG', 'XR']:
                    count += 1
                    break
        return count