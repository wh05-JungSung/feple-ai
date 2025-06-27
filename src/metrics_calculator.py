import json
import re
import kss
from collections import defaultdict
from kiwipiepy import Kiwi

class MetricsCalculator:
    senti_dict = None
    kiwi = None

    def __init__(self, keywords):
        self.keywords = keywords
        
        # Kiwi 형태소 분석기 초기화 (최초 1회)
        if MetricsCalculator.kiwi is None:
            print("KiwiPy 형태소 분석기 초기화 중...")
            MetricsCalculator.kiwi = Kiwi()
            # 사용자 사전에 감성 사전의 단어들을 추가하여 분석 정확도 향상
            for word in self._get_senti_words():
                # 사용자 사전 추가 (단어, 품사, 점수). NNP는 고유명사를 의미
                MetricsCalculator.kiwi.add_user_word(word, 'NNP', 0)
            print("KiwiPy 초기화 및 사용자 사전 추가 완료.")

        # KNU 감성 사전 로드 (최초 1회)
        if MetricsCalculator.senti_dict is None:
            MetricsCalculator.senti_dict = self._load_knu_senti_lexicon()
        
        self.euphonious_patterns = [
            re.compile(p) for p in [
                r'인 것 같습니다', r'ㄹ 것 같습니다', r'일 듯합니다', r'ㄹ지도 모릅니다',
                r'하기는 어렵습니다', r'ㄹ 수 있을까요\??', r'해 주시겠어요\??',
                r'드릴까요\??', r'해드릴까요\??', r'부탁드립니다'
            ]
        ]
    
    def _get_senti_words(self):
        """감성 사전에 있는 단어 목록만 간단히 불러옵니다."""
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
        
        if not agent_words:
            return {"error": "상담사 발화가 없습니다."}

        positive_morph_count, negative_morph_count, total_morph_count = self._count_sentiment_morphemes(agent_words)
        total_sentence_count = len(agent_sentences)
        honorific_sentence_count = self._count_honorific_sentences(agent_sentences)
        euphonious_sentence_count = self._count_euphonious_sentences(agent_sentences)

        final_metrics = {
            "honorific_ratio": (honorific_sentence_count / total_sentence_count) * 100 if total_sentence_count > 0 else 0,
            "positive_word_ratio": (positive_morph_count / total_morph_count) * 100 if total_morph_count > 0 else 0,
            "negative_word_ratio": (negative_morph_count / total_morph_count) * 100 if total_morph_count > 0 else 0,
            "euphonious_word_ratio": (euphonious_sentence_count / total_sentence_count) * 100 if total_sentence_count > 0 else 0,
        }
        return final_metrics

    def _extract_agent_data(self, transcript):
        speaker_counts = defaultdict(int)
        for segment in transcript:
            speaker_counts[segment['speaker']] += len(segment['text'].split())
        
        if not speaker_counts: return [], []
        
        agent_label = max(speaker_counts, key=speaker_counts.get)
        agent_turns = [seg for seg in transcript if seg['speaker'] == agent_label]
        
        agent_words = []
        for turn in agent_turns:
            agent_words.extend(turn['text'].strip().split())
            
        agent_full_text = ' '.join([turn['text'] for turn in agent_turns])
        agent_sentences = kss.split_sentences(agent_full_text)
        
        return agent_words, agent_sentences

    def _count_sentiment_morphemes(self, words):
        """KiwiPy를 사용해 형태소 분석 후 긍정/부정 형태소의 개수를 셉니다."""
        if not self.senti_dict or not self.kiwi:
            return 0, 0, 0
            
        positive_count = 0
        negative_count = 0
        total_morph_count = 0
        
        full_text = " ".join(words)
        
        # kiwi.tokenize()로 텍스트를 분석합니다.
        result = self.kiwi.tokenize(full_text)
        
        for token in result:
            total_morph_count += 1
            # 형태소의 품사(tag)를 확인하여 용언(동사/형용사)일 경우 기본형을 만듭니다.
            # KiwiPy의 품사 태그: VA(형용사), VV(동사), XSA(형용사 파생 접미사), XSV(동사 파생 접미사)
            if token.tag in ['VA', 'VV', 'XSA', 'XSV']:
                # '좋' + '다' -> '좋다' 형태로 기본형 복원
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
        honorific_infix = re.compile(r'시')
        for sent in sentences:
            if honorific_ending.search(sent) or honorific_infix.search(sent):
                count += 1
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