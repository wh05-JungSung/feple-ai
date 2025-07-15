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
        고도화된 프롬프트를 사용하여 문제 해결 제안 점수를 일관되고 정확하게 평가합니다.
        LLM이 단계적으로 사고하도록 유도하여 최종 점수만 반환받습니다.
        """
        if not self.client:
            return 0.0

        conversation = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in transcript])

        system_instructions = (
            "당신은 고객 상담 내용을 분석하여 문제 해결 과정을 평가하는 매우 꼼꼼한 QA 분석가입니다. "
            "주어진 대화 내용과 평가 규칙에 따라, 문제 해결 점수를 '1.0', '0.6', '0.2', '0.0' 중 하나의 숫자로만 반환해야 합니다."
        )

        user_input = f"""
        [분석 지시]
        아래의 [상담 대화 내용]을 읽고, [평가 규칙]에 따라 점수를 매겨주세요.
        당신은 점수를 결정하기 위해 내부적으로 다음 단계를 거쳐야 합니다:
        1. 상담사가 고객의 문제를 해결하기 위해 제시한 '구체적인 행동 방안'을 순서대로 식별합니다.
        2. 대화가 끝났을 때 고객의 문제가 해결되었는지 최종적으로 판단합니다.
        3. 위 분석 내용을 바탕으로 아래 [평가 규칙]에 가장 적합한 점수를 선택합니다.
        4. 최종적으로 결정된 숫자 점수 하나만 출력합니다. 다른 어떤 설명도 붙이지 마세요.

        [평가 규칙]
        - 1.0점: 상담사가 제시한 첫 번째 해결 방안으로 문제가 해결됨.
        - 0.6점: 첫 번째 방안은 실패했지만, 두 번째 해결 방안으로 문제가 해결됨.
        - 0.2점: 세 번 이상의 해결 방안을 제시한 끝에 문제가 해결됨.
        - 0.0점: 제시된 방안들로 문제가 해결되지 못했거나, 해결 여부가 불분명함.

        [평가 예시]
        - 예시 대화 1: "고객님, 휴대폰을 재부팅해보세요." "네. 어, 이제 되네요. 감사합니다."
        - 이 경우 첫 제안으로 해결되었으므로 당신의 최종 출력은 '1.0'이어야 합니다.
        
        - 예시 대화 2: "재부팅 해보세요." "안되네요." "그럼 유심을 뺐다 껴보세요." "아, 이제 됩니다!"
        - 이 경우 두 번째 제안으로 해결되었으므로 당신의 최종 출력은 '0.6'이어야 합니다.

        - 예시 대화 3: "재부팅 해보세요." "안돼요." "유심도 다시 껴봤어요?" "네, 그래도 안돼요." "그럼 서비스센터 방문하셔야겠네요."
        - 이 경우 문제가 해결되지 못했으므로 당신의 최종 출력은 '0.0'이어야 합니다.

        [상담 대화 내용]
        {conversation}

        [최종 점수 출력 (숫자만)]
        """

        try:
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=user_input,
                instructions=system_instructions,
                temperature=0,
                max_output_tokens=5  # 점수만 받으므로 토큰 수를 줄여 효율성 증대
            )
            
            # 응답 텍스트에서 숫자만 정확히 파싱
            response_text = response.output[0].content[0].text.strip()
            match = re.search(r"(\d\.\d)", response_text)
            
            if match:
                return float(match.group(1))
            else:
                print(f"[LLM 파싱 오류] 모델이 예상치 못한 답변을 반환했습니다: {response_text}")
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

    def verify_sentence_intentions(self, sentences, intention):
        """
        주어진 문장 리스트가 특정 의도(공감, 사과 등)에 맞는지 LLM으로 일괄 검증합니다.
        """
        if not self.client or not sentences:
            return [False] * len(sentences)

        # LLM에게 전달할 프롬프트용으로 문장 리스트를 변환
        formatted_sentences = "\n".join([f"{i+1}. \"{sent}\"" for i, sent in enumerate(sentences)])
        
        # 의도에 따라 프롬프트 내용 변경
        if intention == "공감":
            intention_desc = "고객의 감정이나 상황에 동조하며 마음을 알아주는 '진심 어린 공감'"
            example_input = "1. \"많이 힘드셨겠어요.\"\n2. \"네, 공감합니다.\""
            example_output = "[true, true]"
        elif intention == "사과":
            intention_desc = "자신의 과실을 인정하고 용서를 구하는 '진정한 사과'"
            example_input = "1. \"정말 죄송합니다.\"\n2. \"죄송하지만 그건 규정상 어렵습니다.\""
            example_output = "[true, false]"
        else:
            return [False] * len(sentences)

        system_instructions = (
            "당신은 한국어 문장의 숨은 의도를 정확히 파악하는 AI 분석가입니다. "
            "주어진 문장 목록을 보고, 각 문장이 제시된 '판단 의도'와 일치하는지 개별적으로 판단해야 합니다."
        )
        
        user_input = f"""
        [판단 의도]
        {intention_desc}

        [판단 방법]
        아래 [분석 대상 문장] 목록의 각 문장이 위의 [판단 의도]에 부합하면 true, 아니면 false를 반환합니다.
        결과는 반드시 [true, false, ...] 형태의 불리언 리스트(boolean list)를 포함하는 JSON 형식으로만 출력해야 합니다.

        [출력 예시]
        - 분석 대상:
        {example_input}
        - 당신의 출력:
        {{"results": {example_output}}}

        [분석 대상 문장]
        {formatted_sentences}

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

            # 결과 리스트의 길이가 입력과 다를 경우를 대비한 ��어 코드
            if len(verified_results) == len(sentences):
                return verified_results
            else:
                print(f"[LLM 검증 오류] 입력과 출력의 개수가 다릅니다. (입력: {len(sentences)}, 출력: {len(verified_results)})")
                return [False] * len(sentences)

        except Exception as e:
            print(f"[LLM 검증 오류] API 호출 또는 JSON 파싱에 실패했습니다: {e}")
            return [False] * len(sentences)