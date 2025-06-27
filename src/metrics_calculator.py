from collections import defaultdict

class MetricsCalculator:
    def __init__(self, keywords):
        self.keywords = keywords

    def calculate_all_metrics(self, structured_transcript, total_duration_seconds):
        # 이 함수에서 14개 지표를 모두 계산하여 딕셔너리로 반환합니다.
        # 예시로 몇 가지만 구현합니다.
        
        agent_words, customer_words, agent_turns, customer_turns = self._extract_speaker_data(structured_transcript)
        
        if not agent_words:
            return { "error": "상담사 발화가 없습니다." }

        # 1. 정중함 및 언어 품질
        politeness = self._calculate_politeness(agent_words)
        
        # 2. 공감적 소통
        # empathy = self._calculate_empathy(agent_words)

        # 3. 감정 안정성
        # emotional_stability = self._calculate_emotional_stability(customer_turns)
        
        # 4. 대화 흐름
        # flow = self._calculate_flow(structured_transcript, total_duration_seconds)
        
        # 모든 결과를 합쳐서 반환
        final_metrics = {**politeness} #, **empathy, **emotional_stability, **flow}
        return final_metrics

    def _extract_speaker_data(self, transcript):
        # 상담사/고객 데이터 분리 로직 (가정: 발화량이 많은 쪽이 상담사)
        speaker_counts = defaultdict(int)
        for segment in transcript:
            speaker_counts[segment['speaker']] += len(segment['text'].split())
        
        if not speaker_counts: return [], [], [], []
        
        agent_label = max(speaker_counts, key=speaker_counts.get)
        
        agent_words = []
        customer_words = []
        agent_turns = []
        customer_turns = []
        
        for segment in transcript:
            words = segment['text'].strip().split()
            if segment['speaker'] == agent_label:
                agent_words.extend(words)
                agent_turns.append(segment)
            else:
                customer_words.extend(words)
                customer_turns.append(segment)
        
        return agent_words, customer_words, agent_turns, customer_turns

    def _calculate_politeness(self, agent_words):
        total_word_count = len(agent_words)
        if total_word_count == 0:
            return {
                "honorific_ratio": 0, "positive_word_ratio": 0,
                "negative_word_ratio": 0, "euphonious_word_ratio": 0
            }

        counts = defaultdict(int)
        for word in agent_words:
            for key, keyword_list in self.keywords.items():
                if key in ['honorifics', 'positive', 'negative', 'euphonious']:
                    for keyword in keyword_list:
                        if keyword in word:
                            counts[key] += 1
                            break
        
        return {
            "honorific_ratio": (counts["honorifics"] / total_word_count) * 100,
            "positive_word_ratio": (counts["positive"] / total_word_count) * 100,
            "negative_word_ratio": (counts["negative"] / total_word_count) * 100,
            "euphonious_word_ratio": (counts["euphonious"] / total_word_count) * 100,
        }