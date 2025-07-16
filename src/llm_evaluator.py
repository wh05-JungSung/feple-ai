import os
import openai
from dotenv import load_dotenv
import re
import json

class LLMEvaluator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.client = None
            print("[경고] OPENAI_API_KEY가 설정되지 않았습니다. LLM 기반 평가는 건너뜁니다.")
        else:
            self.client = openai.OpenAI(api_key=api_key)
            
    def get_conversation_analysis(self, transcript):
        """
        대화 전체를 분석하여 주제, 결과, 비속어 사용 여부를 JSON으로 반환합니다.
        """
        if not self.client:
            return {"mid_category": "기타", "result_label": "분석 불가", "profane": 0}

        conversation = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in transcript])

        system_instructions = (
            "당신은 고객 상담 내용을 듣고, 대화의 주제, 결과, 고객의 비속어 사용 여부를 정확하게 분석하는 AI입니다."
        )

        user_input = f"""
        [상담 대화 내용]
        {conversation}

        [분석 항목 및 선택 옵션]
        1.  `mid_category`: 대화의 핵심 주제를 아래 목록에서 하나만 선택해줘.
            - ["요금 안내", "요금 납부", "요금제 변경", "선택약정 할인", "납부 방법 변경", "부가서비스 안내", "소액 결제", "휴대폰 정지/분실/파손", "기기변경", "명의/번호/유심 해지", "기타"]
        2.  `result_label`: 상담의 최종 결과를 아래 목록에서 하나만 선택해줘. (궁극적인 해결이 아닌, 상담 자체의 마무리 상태)
            - ["만족", "미흡", "해결 불가", "추가 상담 필요"]
        3.  `profane`: 고객이 비속어(욕설, 공격적인 언어)를 사용했는지 여부를 판단해줘. (사용했으면 1, 아니면 0)

        [출력 지시]
        분석 결과를 반드시 아래의 JSON 형식에 맞춰서 제공해줘.
        {{
            "mid_category": "...",
            "result_label": "...",
            "profane": ...
        }}
        """

        try:
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=user_input,
                instructions=system_instructions,
                text={"format": {"type": "json_object"}},
                temperature=0
            )
            
            analysis_result = json.loads(response.output[0].content[0].text)
            return analysis_result
            
        except Exception as e:
            print(f"[LLM 대화분석 오류] API 호출 또는 JSON 파싱에 실패했습니다: {e}")
            return {"mid_category": "기타", "result_label": "분석 불가", "profane": 0}

    def get_suggestion_score(self, transcript):
        """
        상담사가 고객의 문제를 해결하기 위해 구체적인 해결 방안을 제시했는지 여부를 평가합니다.
        """
        if not self.client:
            return 0.0

        conversation = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in transcript])

        system_instructions = (
            "당신은 고객 상담 대화를 분석하여, 상담사가 문제 해결을 위한 '구체적인 행동 제안'을 했는지 판단하는 AI 분석가입니다. "
            "답변은 반드시 'Yes' 또는 'No'로만 해야 합니다."
        )

        user_input = f"""
        [분석 지시]
        아래 [상담 대화 내용]을 읽고, 상담사(Agent)가 고객(Customer)의 문제를 해결하기 위해 '구체적인 행동 방안'을 제시했는지 판단해주세요.
        '구체적인 행동 방안'이란, 고객이 직접 수행하거나 상담사가 시스템을 통해 조치할 수 있는 명확한 해결책을 의미합니다.
        
        - 예시 (Yes): "휴대폰을 재부팅 해보시겠어요?", "제가 전산 시스템에서 바로 변경해 드리겠습니다.", "가까운 A/S 센터를 방문해주세요."
        - 예시 (No): "알아보겠습니다.", "확인해 보겠습니다.", "어렵습니다." (단순 응대나 부정적 답변은 제안이 아님)

        [상담 대화 내용]
        {conversation}

        [질문]
        상담사가 위 대화에서 구체적인 행동 방안을 제시했습니까? (Yes/No)
        """

        try:
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=user_input,
                instructions=system_instructions,
                temperature=0,
                max_output_tokens=16
            )
            
            response_text = response.output[0].content[0].text.strip().lower()
            
            if 'yes' in response_text:
                return 1.0
            else:
                return 0.0
            
        except Exception as e:
            print(f"[LLM 제안 평가 오류] OpenAI API 호출에 실패했습니다: {e}")
            return 0.0
        
    def get_sentiment_score(self, text):
        """
        LLM을 사용하여 한 문장의 감정을 분석하고 점수를 반환합니다.
        - Positive: 1, Neutral: 0, Negative: -1
        """
        if not self.client:
            return 0 # 클라이언트가 없으면 0점 반환

        system_instructions = (
            "당신은 문장의 감정을 분석하는 AI입니다. 문장을 읽고 'Positive', 'Neutral', 'Negative' 중 하나로만 답변해야 합니다."
        )
        
        try:
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=text,
                instructions=system_instructions,
                temperature=0,
                max_output_tokens=16
            )
            
            sentiment_text = response.output[0].content[0].text.strip().lower()

            if 'positive' in sentiment_text:
                return 1
            elif 'negative' in sentiment_text:
                return -1
            else:
                return 0
        except Exception as e:
            print(f"[LLM 감정분석 오류] API 호출에 실패했습니다: {e}")
            return 0 # 오류 발생 시 중립으로 처리

    def verify_sentence_intentions(self, sentences, intention):
        """
        주어진 문장 리스트가 특정 의도(공감, 사과 등)에 맞는지 LLM으로 일괄 검증합니다.
        명확한 예시와 단순한 프롬프트를 사용하여 모델의 안정성을 높입니다.
        """
        if not self.client or not sentences:
            return [False] * len(sentences)

        formatted_sentences = "\n".join([f"문장 {i+1}: \"{sent}\"" for i, sent in enumerate(sentences)])
        
        if intention == "공감":
            intention_desc = "고객의 감정이나 상황에 동조하며 이해와 위로를 표현하는 '진심 어린 공��'의 의도가 담겨 있는지 판단합니다."
            example_input = '문장 1: "많이 불편하셨겠습니다."\n문장 2: "네, 알겠습니다."'
            example_output = '[true, false]'
            judgment_criteria = "단순 동의나 사실 확인이 아닌, 감정적인 지지를 표현해야 합니다."
        elif intention == "사과":
            intention_desc = "자신의 과실이나 서비스의 문제로 인해 발생한 불편에 대해 용서를 구하는 '진정한 사과'의 의도가 담겨 있는지 판단합니다."
            example_input = '문장 1: "정말 죄송합니다."\n문장 2: "죄송하지만 그건 규정상 어렵습니다."'
            example_output = '[true, false]'
            judgment_criteria = "조건부 사과나 변명이 아닌, 직접적인 사과의 표현이어야 합니다."
        else:
            return [False] * len(sentences)

        system_instructions = (
            "당신은 한국어 문장의 의도를 정확히 파악하는 AI 분석가입니다. "
            "각 문장이 주어진 '판단 의도'와 일치하는지 개별적으로 판단하고, 결과를 반드시 JSON 형식의 boolean 리스트로만 반환해야 합니다."
        )
        
        user_input = f"""
        [판단 의도]
        {intention_desc}

        [판단 기준]
        {judgment_criteria}

        [분석 대상 문장 목록]
        {formatted_sentences}

        [출력 지시]
        위 [분석 대상 문장 목록]의 각 문장이 [판단 의도]에 부합하면 true, 아니면 false로 판단하여, 아래 예시와 같이 JSON 형식의 리스트로만 결과를 반환해주세요.
        다른 어떤 설명도 추가하지 마세요.

        [출력 예시]
        - 분석 대상:
        {example_input}
        - 당신의 출력:
        {{"results": {example_output}}}

        [분석 결과 출력 (JSON 형식)]
        """

        try:
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=user_input,
                instructions=system_instructions,
                text={"format": {"type": "json_object"}},
                temperature=0
            )
            
            result_json = json.loads(response.output[0].content[0].text)
            verified_results = result_json.get("results", [])

            if len(verified_results) == len(sentences):
                return verified_results
            else:
                print(f"[LLM 검증 오류] 입력과 출력의 개수가 다릅니다. (입력: {len(sentences)}, 출력: {len(verified_results)})")
                return [False] * len(sentences)

        except Exception as e:
            print(f"[LLM 검증 오류] API 호출 또는 JSON 파싱에 실패했습니다: {e}")
            return [False] * len(sentences)