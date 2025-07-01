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
            - ["상품 및 서비스 일반", "주문/결제/입금 확인", "취소/반품/교환/환불/AS", "회원 관리", "배송 문의", "이벤트/할인", "콘텐츠", "제휴", "기타"]
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
        OpenAI Responses API를 사용하여 문제 해결 제안 점수를 평가합니다.
        """
        if not self.client:
            return 0.0

        # LLM에게 전달할 대화록을 간단한 텍스트로 변환
        conversation = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in transcript])

        # System Prompt (또는 instructions)
        system_instructions = (
            "당신은 고객 상담 내용을 분석하고 문제 해결 과정을 평가하는 전문 평가자입니다. "
            "주어진 상담 대화 내용을 분석하여, 문제 해결 과정이 아래 규칙 중 어디에 해당하는지 판단하고, "
            "반드시 '1.0', '0.6', '0.2', '0.0' 중 하나의 숫자로만 답변해야 합니다."
        )

        # User-facing Prompt
        user_input = f"""
        [상담 대화 내용]
        {conversation}

        [평가 규칙]
        - 1.0점: 최초로 제시한 아이디어로 문제가 해결됨.
        - 0.6점: 첫 번째 아이디어는 실패했지만, 두 번째로 제시한 아이디어로 해결됨.
        - 0.2점: 세 번 이상의 아이디어를 제시하여 문제를 해결함.
        - 0.0점: 대화가 끝날 때까지 문제가 해결되지 못함.
        """

        try:
            # 최신 'responses.create' API 사용
            response = self.client.responses.create(
                model="gpt-4.1-nano",          # 최신, 비용 효율적인 모델
                input=user_input,            # 사용자 입력
                instructions=system_instructions, # 시스템 지시사항
                temperature=0,               # 일관된 답변을 위해 0으로 설정
                max_output_tokens=100         # 점수만 받으므로 토큰 수 제한
            )
            
            full_response_text = response.output[0].content[0].text.strip()
            
            match = re.search(r'(\d\.\d)$', full_response_text)
            if match:
                return float(match.group(1))
            else:
                print(f"[LLM 파싱 오류] 모델이 예상치 못한 답변을 반환했습니다: {full_response_text}")
                return 0.0
            
        except Exception as e:
            print(f"[LLM 평가 오류] OpenAI API 호출에 실패했습니다: {e}")
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