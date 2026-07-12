# SNU AI CHALLENGE HOMEPAGE

## ABOUT
Overview
서울대학교 데이터사이언스 대학원에서는 최신 인공지능 연구의 최전선에서 기존 모델의 한계를 극복하고 다양한 상황에서의 일반화 성능 향상을 위해 활발히 연구를 진행하고 있습니다. 뿐만 아니라, 이러한 연구 성과가 좀 더 확산되고, 보다 많은 사람들이 인공지능 연구에 관심을 가질 수 있도록 하는 데에도 많은 노력을 기울이고 있습니다. 이러한 노력의 일환으로 저희 대학원에서 자체적으로 가공한 데이터를 공개하여 국내외 대학 학부생들이 최신 인공지능 모델을 직접 개발해보고 성능을 겨루어 볼 수 있는 경연의 장을 마련하였습니다.


Challenge Description
1. 과제 정의: 텍스트로 풀어보는 장면의 재구성
주어진 스토리라인 (캡션)에 맞게 4개의 이미지 프레임을 올바른 순서로 재배열하는 문제를 해결해야 합니다.

introduction

2. 문제 배경 및 중요성
이 과제는 이미지와 캡션을 개별적으로 인식하는 수준을 넘어, 여러 장면 (frames)을 스토리 라인 (캡션)의 맥락 속에서 재구성하여 올바른 시간적 전개 순서를 복원하는 멀티모달 이해 능력을 평가합니다.

3. 입력 및 출력 형식 
입출력 형식에 관한 간단한 설명은 다음과 같습니다.

입력: 자연어 문장과 여러 장의 프레임으로 구성된 데이터 (예: { “text”: “자연어 문장”, “frames”: [image_3, image_1, image_4, image_2] })
출력: 정답 순서대로 다시 배열하였을 때 각 프레임의 위치. (예: [3, 4, 1, 2], 정답 순서대로 다시 배열하였을 때 첫번째 프레임은 3번째에 위치, 두번째 프레임은 4번째에 위치,…)
4. 대회 일정 
사전 참가 신청 시작: 2026년 6월 15일
예선 (온라인 경진대회): 2026년 6월 29일 ~ 2026년 7월 24일
본선 (공개 발표 평가): 2026년 8월 7일
※ 참가 신청은 7/17에 마감되니, 참여를 희망하시는 모든 팀들은 그 전에 신청하시기 바랍니다.

상세 일정은 Timeline 탭을 참고하여 주시기 바랍니다.

5. 대회 진행 절차 
본 경진대회는 예선과 본선으로 나누어 진행됩니다.

예선 기간 동안 참가자들은 제공된 학습 데이터를 바탕으로 모델을 개발하고, 테스트 데이터에 대한 모델의 추론 결과를 제출하여 모델 성능을 겨루게 됩니다. 예선 기간 중 리더보드는 전체 테스트 데이터의 70% (Public data)만을 이용하여 업데이트됩니다.

예선이 종료되면, 테스트 데이터 전체에 대한 모델 성능 기준 상위 순위팀을 대상으로 코드 검증 및 보고서 검토를 수행합니다. 코드 검증 및 보고서 검토를 거쳐 예선 점수가 확정되고, 상위 10개 내외 팀이 본선에 진출합니다.
참가팀들의 성능 달성 수준에 따라 예선 종료 후 외부 데이터셋을 기반으로 한 성능 평가가 별도로 진행될 수 있습니다.
본선은 오프라인 발표평가로 진행됩니다. 참가자는 예선 기간 중 모델을 개발한 과정을 정리하여 심사위원께 발표하며, 예선에서 달성한 모델의 정량/정성 평가 점수와 본선 발표 점수를 종합적으로 고려하여 최종 우승자를 결정합니다. 본선에서의 점수 배정은 다음과 같습니다.
항목	설명	점수
예선 점수	예선 테스트 데이터에 대한 추론 정확도
과제 해결 전략의 논리적 타당성, 효율성 및 독창성 (보고서 기반 서면 평가)	40
데이터 활용	데이터 전처리 및 활용 전략의 적절성	15
모델 설계 및 학습 방법론	문제 특성에 부합하는 모델 구조 선택 및 학습 방법론의 선택과 적용	15
최적화 방법론	추론 환경 제약에 맞춘 모델 성능 최적화 수준	10
자원 효율성	태스크 수행 시 필요한 메모리 사용량 및 응답 속도(지연 시간)	10
구축 비용	시스템 구축 시 요구되는 연산량(학습) 및 데이터 전처리 비용(외부 API 사용 등)	10
총합		100


6. 시상 및 혜택 
총 상금: 2,300만원
시상 훈격	구분	시상팀 수	상금
서울대학교 총장상	대상	1	1,000만원
금상	1	500만원
은상	1	300만원
동상	1	200만원
장려상	3	100만원


7. 대회 규칙 
Rules 탭을 참고하여 주시기 바랍니다. 규칙을 준수하지 않는 경우 실격처리될 수 있습니다.

8. 참가 자격 
국내외 대학 학부 재학생 또는 휴학생
※ 전공 및 학년 제한 없음
※ 단, 상금은 국내 은행 계좌로만 입금 가능하며, 참가 자격 관련 추가 문의는 담당자에게 문의 바랍니다.

## TIMELINE
Timeline
날짜 (한국시간 기준)	내용
6월 22일 - 7월 17일	참가 신청 (링크)
6월 29일 10:00 AM	경진대회 시작
6월 29일 10:00 AM - 7월 24일 11:59 PM	경진대회 예선 진행 (제출 가능 기간)
※ 참가신청 종료: 7월 17일
7월 25일	최종 리더보드 공개 (Public + Private data)
7월 25일 - 7월 28일	리더보드 상위팀 대상: 검증용 코드및 보고서 제출
8월 3일	본선 진출 대상자 발표
8월 6일	본선 발표 자료 제출 마감
8월 7일 (추후 변동 가능)	본선 발표 평가
8월 말	최종 순위 발표 및 시상
예선 추론 결과 제출 및 검증용 코드 제출 등 모든 일정의 마감 시간은 마지막 날의 11:59 PM (한국 시간 기준) 입니다.

## DATA
Data
1. 데이터셋 개요
본 대회는 비디오 프레임의 시간적 순서를 예측하는 과제입니다. 참가자는 뒤섞인 4개의 비디오 프레임 이미지와 해당 비디오에 대한 텍스트 설명(Sentence)을 바탕으로, 원본 비디오의 올바른 시간 순서를 복원해야 합니다.

데이터셋은 학습용(train)과 평가용(test)으로 구분되어 제공됩니다. 각 샘플은 고유한 Id를 가지며, 이에 대응하는 이미지 폴더와 메타데이터가 CSV 파일로 제공됩니다. 별도의 검증용(Validation) 데이터셋은 제공되지 않으며, 참가자들은 학습용(train) 데이터셋에서 일부를 검증용으로 사용할 수 있습니다.

비디오 프레임을 추출하기 위한 원본 비디오는 다양한 원천으로부터 수집되었습니다. 일부 프레임은 생성형 인공지능을 이용하여 생성되었습니다. 비디오 프레임에 대응하는 텍스트는 사람이 직접 작성하거나, 생성형 인공지능을 이용하여 생성되었습니다. 학습용 데이터에는 정제되지 않은 데이터가 포함되어 있습니다. 즉, 비디오 프레임만으로는 정답을 하나로 특정할 수 없거나, 텍스트와 관련없는 프레임 (검은색 프레임 등)이 포함되어 있을 수 있습니다. 이러한 데이터에 대한 전처리를 수행하는 것이 모델의 성능을 향상시킬 수 있을 것이며, 효과적인 전처리 적용 여부가 본선 평가 기준에 포함됩니다.

정답은 각 프레임에 대해서 원본에서의 위치를 나열한 것으로 설정됩니다. 예를 들어, a, b, c, d가 원본이고 a, d, b, c가 뒤섞인 프레임이라면 정답은 [1, 4, 2, 3]로 결정됩니다.

2. 문제 예시
경진대회 문제 중 일부를 예시로 보여드리면 다음과 같습니다.

data

data

data

3. 데이터 액세스 및 사용
본 데이터는 경진대회 참여 목적 외 다른 목적으로의 활용을 제한하며, 본 데이터의 원본 또는 수정본을 재배포하는 행위 또한 금지됩니다.

## RULES
Rules
1. 참가 자격
국내외 대학 학부 재학생 또는 휴학생 (전공 및 학년 제한 없음).
상금은 대표자 명의로 된 국내 은행 계좌로만 지급 가능.
참가 신청시 재학증명서를 같이 제출해야 함. (대회 시작일로부터 1개월 이내 발급본)
참가 자격 관련 추가 문의는 담당자에게 문의 바람.
2. 참여 규칙
개인 또는 팀(최대 4인) 으로 참여 가능.
참가 신청시 팀을 결성하여 신청하여야 하며, 신청 후 팀을 합치거나 나누는 것은 불가능함 (적발시 실격).
동일인의 다계정 참가 등록은 금지되며, 적발 시 팀 전체 실격 처리.
3. 모델 학습·추론 및 외부 자원 사용 규칙
3.1 사용 환경
사용 언어: Python
모든 모델은 인터넷이 차단된 로컬 환경(CPU 또는 GPU)에서 실행가능해야 함.
3.2 외부 API 사용
학습 및 추론 과정에서 외부 상용 API(예: ChatGPT, Gemini, Grok 등) 사용을 금지함.
데이터 전처리 목적에 한해 외부 상용 API 사용을 허용하되, 총 사용 비용은 3만 원을 초과할 수 없음.
외부 API를 활용한 경우, 발표 자료에 해당 API의 사용 방식, 실험 결과 및 발생 비용을 명시해야 함.
평가 과정에서 총 구축 비용이 3만 원을 초과한 것으로 판단될 경우 실격 처리될 수 있음.
3.3 데이터 및 모델
모델 학습 시 외부 데이터 사용을 금지하며, 제공된 학습 데이터만 사용할 수 있음.
오픈소스 모델(예: LLaMA 등)은 활용 가능하나, 2026년 5월 31일 이전에 모델 가중치가 공개된 경우에 한함.
허용 여부가 불확실한 경우 Kaggle Discussion 탭을 통해 운영진에게 사전 문의할 것.
모델 앙상블(여러 모델의 추론 결과를 조합하는 행위)은 허용하지 않음. (동일 모델에 대해 데이터셋을 여러 개로 나누어 파인튜닝한 후 추론 결과를 조합하는 방식도 허용하지 않음)
제공된 데이터를 활용하여 데이터 증강 (Data augmentation)은 가능하지만, 생성형 모델 (Generative Model)을 이용한 데이터 생성/변형은 허용하지 않음.
모델 경량화 기법(예: Quantization, LoRA 등)은 허용함.
추론 전략 및 중간 추론 표현 방식(예: Chain-of-Thought, Multi-turn Chat, Test time augmentation 등)은 아래 추론 시간 제한에 위배되지 않는 한에서 활용 가능함.
테스트 셋 전체에 대해 검증 환경 (“4. 제출 규칙” 참조) 에서 24시간 이내에 추론이 완료되어야 함.
3.4 데이터 누수
평가 데이터셋 정보를 학습에 활용하는 행위(데이터 누수, Data Leakage)는 금지함.
예: 평가 데이터에 대한 수작업 라벨링 후 학습에 활용
예: 평가 데이터의 특성을 분석하여 학습 데이터 전처리 또는 모델 설계에 활용
위반 시 실격 처리함.
3.5 규칙 위반
규칙을 준수하지 않을 경우 실격 처리될 수 있음.
대회 진행 중에도 리더보드 상위 참가자를 대상으로 코드 및 결과에 대한 재현성 검증을 요청할 수 있음.
4. 제출 규칙
제출은 1일 2회로 제한.
최종 제출 모델은 NVIDIA RTX 3090 GPU(VRAM 24GB) 1개를 이용하여 실행 가능해야 함.
서버 사양 상세
CPU: AMD EPYC 7502 32-Core Processor x 2
Memory: 512 GB
GPU: GeForce RTX 3090
NVIDIA driver version: 550.54.15
CUDA version: 12.4
5. 예선 평가 규칙
예선기간 중 리더보드는 전체 테스트 데이터의 70% (Public data)만을 이용하여 업데이트됨. 순위는 Exact Match Accuracy를 기준으로 결정.
제출한 이미지 순서가 정답 순서와 완전히 동일한 경우에만 정답으로 인정되며, 순서가 하나라도 다를 경우 오답으로 처리.
예) 정답이 [1, 4, 2, 3]일 때, [1, 4, 2, 3]만 정답으로 처리되며 그 외의 모든 순서는 오답으로 간주.
예선 종료 후 테스트 데이터 전체 (Public + Private data)에 대한 모델 성능 기준 상위 순위팀을 대상으로 코드 검증 및 보고서 검토를 수행.
참가팀들의 성능 달성 수준에 따라 예선 종료 후 외부 데이터셋을 기반으로 한 성능 평가가 별도로 진행될 수 있습니다.
6. 코드 재현성 검증 및 보고서 제출 관련 사항
예선 종료 후 상위 16위 이내 참가자는 지정된 기한 내에 아래 자료를 제출해야 함.
학습 코드 및 추론 코드 (.py 형태)
최종 모델 가중치 파일
과제 해결을 위해 사용한 방법론을 보고서 형태로 정리 (A4 5쪽 이내 분량, MS 워드 파일로 제출하고 별도 양식은 없음.)
제출 코드 관련 준수 사항:
README.md 파일을 상세히 작성할 것
데이터 입·출력 경로는 상대 경로로 표기할 것
코드 및 주석 인코딩은 UTF-8로 통일할 것
모든 코드는 오류 없이 실행되어야 함
개발 환경(OS, 하드웨어 사양) 및 라이브러리 버전을 명시할 것
추론(Inference) 코드는 별도로 작성하고, 사용된 모델 가중치 파일을 함께 제출할 것.
모델 구동 코드 및 가중치를 포함한 전체 모델의 크기는 80 GB를 초과하여서는 안 됨.
상세한 절차는 추후 별도 안내
7. 본선 발표 자료 제출
코드 재현성 검증을 통과한 예선 점수 상위팀들을 대상으로 오프라인 발표 평가를 진행함.
발표 자료는 발표 전날까지 PPT 또는 PDF 형식으로 제출할 것.
발표는 10분 이내 분량으로 준비.
8. 본선 발표 규칙
본 발표 10분, 질의응답 5분으로 진행됨.
발표는 기 제출된 발표자료를 이용하며, 진행하며 주최측에서 발표자료를 세팅할 예정.
팀별 대표자가 발표하는 것을 원칙으로 함.
팀원 전체가 본선 발표 행사에 참석하여야 함.
발표는 공개 발표로 진행함.
※ 규칙이 추가될 수 있으니 수시로 확인하시어 불이익을 받는 일이 없도록 해주시기 바랍니다. (최종 업데이트: 7월 1일 12:00)

## PARTICIPATION
본 대회는 Kaggle 플랫폼을 이용하여 진행됩니다.

참가를 원하시는 분들께서는 아래 링크로 들어가서 구글 폼을 작성해주시기 바랍니다.

참가 신청 링크

참가신청을 해 주시면 자격요건 확인 후 아래 Kaggle 경진대회 접근 권한을 열어드리도록 하겠습니다. (평일 오전 11시 30분 일괄 등록, 금요일 오후 6시 추가 등록하며, 토/일요일은 등록 진행하지 않고 월요일 일괄 진행.)

Kaggle 플랫폼에서 1회 답안 제출 후 팀 결성을 허용하고 있습니다. 따라서 접근 권한을 얻으시게 되면 1회 답안 제출한 후 “Team” 탭으로 들어가 팀 결성을 우선적으로 해주시기 바랍니다.

Kaggle 경진대회 링크

## FAQ
FAQ
아래에 자주나오는 질문과 답변을 정리하였습니다. 추가적으로 궁금하신 부분은 편하게 연락주시면 됩니다. contact us.

Q. 해외 대학 재학 중인 학부생인데 참여 가능한가요?
네, 해외 대학 재학 중인 학부생도 참여 가능합니다. 다만, 상금은 국내 계좌로만 지급 가능하다는 점을 유의해 주세요.

Q. 고등학생이나 대학원생도 참여가 가능한가요?
아니요. 본 대회는 학부생을 대상으로 진행되며, 고등학생 및 대학원생은 참여할 수 없습니다. 팀원 중 1명이라도 학부생이 아닌 인원이 섞여있을 경우 팀원 전체가 실격처리 됩니다.

Q. 학부생이지만 수료했습니다. 참여가 가능할까요?
재적증명서 또는 재학증명서 등 학부 졸업 상태가 아님을 입증할 수 있는 경우에 한해 허용됩니다.

Q. 팀 구성은 어떻게 이루어지나요?
개인(1인) 또는 팀(최대 4인)으로 참가할 수 있습니다. 참가 신청 후 팀 병햡은 불가능하며, 신청할 때 팀을 구성해서 신청하여야 합니다. 또한 동일인의 다계정 참가(중복 등록)는 금지되며, 적발 시 팀원 전체가 실격 처리됩니다.

Q. 주어진 학습 데이터 외에 외부 데이터 사용이 가능한가요?
아니요. 대회에서 제공하는 데이터 외의 외부 데이터 사용은 금지됩니다.

Q. 사용 가능한 모델에 제한이 있나요?
네, 아래 기준을 만족해야 합니다.

2026년 5월 31일 이전에 오픈소스로 공개된 모델만 사용 가능
로컬 환경에서 실행 가능해야 함 (인터넷 연결 없이도 동작해야 함)
학습 및 추론 과정에서 외부 상용 API(예: ChatGPT, Gemini, Grok 등) 사용 금지
단, 데이터 전처리 목적에 한해 외부 상용 API 사용을 예외적으로 허용하며, 이 경우 총 사용 비용은 3만 원을 초과할 수 없습니다.
해석이 애매한 경우 대회 게시판을 통해 문의해 주세요. 자세한 내용은 Rules 탭을 참고 바랍니다.

Q. 제출 횟수 제한이 있나요?
네. 1일 최대 2회까지 제출할 수 있습니다. (UTC 기준 00:00 - 23:59 동안 2회 제출 가능.)

Q. 제출 파일 형식이 꼭 CSV여야 하나요?
네. 제출물은 지정된 파일 형식(CSV)과 인코딩 방식을 반드시 따라야 합니다. 자세한 제출 규격(컬럼명, 행 수, 정렬 기준, 인코딩 등)은 Rules 페이지를 참고해 주세요.

Q. 본선은 어디에서 진행하나요?
본선은 서울대학교에서 진행되며 상세 일정 및 장소는 추후 공지될 예정입니다.

Q. 본선 발표는 누가 해야 하나요?
각 팀별 대표자가 하는 것을 원칙으로 합니다.

Q. 본선 발표에 팀원 전체가 참가해야 하나요?
팀원 전체가 참석해야 합니다.

Q. 본선에서 발표 평가는 누가 하나요?
본선 발표 평가는 서울대학교 데이터사이언스대학원 소속 교수진이 심사합니다.

Q. 최종 제출 모델은 NVIDIA RTX 3090 GPU(VRAM 24GB) 환경에서 실행 가능해야 한다고 들었습니다. 해당 GPU가 없는데 실행 가능 여부는 어떻게 확인하나요?
예선 종료 후 순위권에 포함된 참가자에게는 NVIDIA RTX 3090 GPU 환경을 제공할 예정이며, 해당 환경에서 모델 세팅 및 실행 가능 여부를 직접 검증하실 수 있습니다. 서버의 상세 스펙은 Rules 탭의 4.제출 규칙 항목을 참조하시기 바랍니다.

Q. 대회에서 사용한 방법론으로 논문을 쓰고 싶은데 가능한가요?
대회에서 공개·제출되는 산출물의 지식재산권 및 사용 범위는 대회 규정(Rules)에 따르며, 대회 운영 측의 권리가 포함될 수 있습니다. 논문/대외 공개를 고려하시는 경우, 사전에 운영진에 문의해 주시기 바랍니다.

Q. 팀별 상금은 어떻게 분배되나요?
상금은 팀의 대표자로 등록한 인원에게 일괄 지급할 예정입니다. 팀원간 분배는 내부적으로 결정해 주시면 됩니다.

Q. 상금에 대한 세금이 발생하나요?
상금을 받게 되는 경우 이에 대한 세금이 발생할 수 있습니다. 납세 의무는 수상자에게 있음을 알려드립니다.



# KAGGLE PAGE 
## OVERVIEW

Overview
서울대학교 데이터사이언스 대학원에서는 최신 인공지능 연구의 최전선에서 기존 모델의 한계를 극복하고 다양한 상황에서의 일반화 성능 향상을 위해 활발히 연구를 진행하고 있습니다. 뿐만 아니라, 이러한 연구 성과가 좀 더 확산되고, 보다 많은 사람들이 인공지능 연구에 관심을 가질 수 있도록 하는 데에도 많은 노력을 기울이고 있습니다. 이러한 노력의 일환으로 저희 대학원에서 자체적으로 가공한 데이터를 공개하여 국내외 대학 학부생들이 최신 인공지능 모델을 직접 개발해보고 성능을 겨루어 볼 수 있는 경연의 장을 마련하였습니다.

Start

24 days ago
Close

14 days to go
Description
1. 과제 정의: 텍스트로 풀어보는 장면의 재구성
주어진 스토리라인 (캡션)에 맞게 4개의 이미지 프레임을 올바른 순서로 재배열하는 문제를 해결해야 합니다. 

2. 문제 배경 및 중요성
이 과제는 이미지와 캡션을 개별적으로 인식하는 수준을 넘어, 여러 장면 (frames)을 스토리 라인 (캡션)의 맥락 속에서 재구성하여 올바른 시간적 전개 순서를 복원하는 멀티모달 이해 능력을 평가합니다.

3. 입력(Input) 및 출력(Output) 형식:
입출력 형식에 관한 간단한 설명은 다음과 같습니다.

입력: 자연어 문장과 여러 장의 프레임으로 구성된 데이터 (예: { "text": "자연어 문장", "frames": [image_3, image_1, image_4, image_2] })
출력: 정답 순서대로 다시 배열하였을 때 각 프레임의 위치. (예: [3, 4, 1, 2], 정답 순서대로 다시 배열하였을 때 첫번째 프레임은 3번째에 위치, 두번째 프레임은 4번째에 위치,…)
4. 대회 일정 
사전 참가 신청 시작: 2026년 6월 15일
예선 (온라인 경진대회): 2026년 6월 29일 ~ 2026년 7월 24일
본선 (공개 발표 평가): 2026년 8월 7일
※ 참가 신청은 7/17에 마감되니, 참여를 희망하시는 모든 팀들은 그 전에 신청하시기 바랍니다.

5. 대회 진행 절차 
본 경진대회는 예선과 본선으로 나누어 진행됩니다.

예선 기간 동안 참가자들은 제공된 학습 데이터를 바탕으로 모델을 개발하고, 테스트 데이터에 대한 모델의 추론 결과를 제출하여 모델 성능을 겨루게 됩니다. 예선 기간 중 리더보드는 전체 테스트 데이터의 70% (Public data)만을 이용하여 업데이트됩니다.

예선이 종료되면, 테스트 데이터 전체에 대한 모델 성능 기준 상위 순위팀을 대상으로 코드 검증 및 보고서 검토를 수행합니다. 코드 검증 및 보고서 검토를 거쳐 예선 점수가 확정되고, 상위 10개 내외 팀이 본선에 진출합니다.

참가팀들의 성능 달성 수준에 따라 예선 종료 후 외부 데이터셋을 기반으로 한 성능 평가가 별도로 진행될 수 있습니다.
본선은 오프라인 발표평가로 진행됩니다. 참가자는 예선 기간 중 모델을 개발한 과정을 정리하여 심사위원께 발표하며, 예선에서 달성한 모델의 정량/정성 평가 점수와 본선 발표 점수를 종합적으로 고려하여 최종 우승자를 결정합니다. 본선에서의 점수 배정은 다음과 같습니다.

항목	설명	점수
모델 성능	예선 테스트 데이터에 대한 추론 정확도
과제 해결 전략의 논리적 타당성, 효율성 및 독창성 (보고서 기반 서면 평가)	40
데이터 활용	데이터 전처리 및 활용 전략의 적절성	15
모델 설계 및 학습 방법론	문제 특성에 부합하는 모델 구조 선택 및 학습 방법론의 선택과 적용	15
최적화 방법론	추론 환경 제약에 맞춘 모델 성능 최적화 수준	10
자원 효율성	태스크 수행 시 필요한 메모리 사용량 및 응답 속도(지연 시간)	10
구축 비용	시스템 구축 시 요구되는 연산량(학습) 및 데이터 전처리 비용(외부 API 사용 등)	10
총합		100
6. 대회 규칙
Rules 탭을 참고하여 주시기 바랍니다. 규칙을 준수하지 않는 경우 실격처리될 수 있습니다.

7. 참가 자격
국내외 대학 학부 재학생 또는 휴학생 ※ 전공 및 학년 제한 없음

[주최] 서울대학교

[주관] 서울대학교 데이터사이언스 대학원

[후원] (주) 모레, (주) 모티프테크놀로지스, BK 21

Evaluation
예선 순위는 Exact Match Accuracy를 기준으로 결정됩니다. 제출한 이미지 순서가 정답 순서와 완전히 동일한 경우에만 정답으로 인정되며, 순서가 하나라도 다를 경우 오답으로 처리되며 별도의 부분 점수는 부여하지 않습니다. 예를 들어 정답이 [1,4,2,3]일 때, [1,4,2,3]만 정답으로 처리되며 그 외의 모든 순서는 오답으로 간주됩니다.

## DATA

Dataset Description
본 대회는 비디오 프레임의 시간적 순서를 예측하는 과제입니다. 참가자는 뒤섞인 4개의 비디오 프레임 이미지와 해당 비디오에 대한 텍스트 설명(Sentence)을 바탕으로, 원본 비디오의 올바른 시간 순서를 복원해야 합니다.

데이터셋은 학습용(train)과 평가용(test)으로 구분되어 제공됩니다. 각 샘플은 고유한 Id를 가지며, 이에 대응하는 이미지 폴더와 메타데이터가 CSV 파일로 제공됩니다.

파일
train.csv: 모델 학습을 위한 데이터입니다. 비디오 프레임의 파일명, 텍스트 설명, 그리고 정답(원래 순서)이 포함되어 있습니다.
test.csv: 순서 예측을 수행해야 하는 평가용 데이터입니다. 정답은 제공되지 않습니다.
sample_submission.csv: 제출 파일의 예시 포맷입니다. 올바른 형식의 제출을 위해 참고하시기 바랍니다.
train/: 학습용 이미지 폴더입니다. 각 폴더명은 train.csv의 Id와 일치합니다.
test/: 평가용 이미지 폴더입니다. 각 폴더명은 test.csv의 Id와 일치합니다.
CSV 컬럼
train.csv
학습 데이터셋에는 다음과 같은 컬럼이 포함됩니다.

Id: 각 비디오 샘플의 고유 식별자입니다. (영문 대소문자 및 숫자 조합 6자리)
Sentence: 해당 비디오 클립을 묘사하는 텍스트 캡션입니다. 프레임의 순서를 추론하는 데 중요한 단서가 됩니다.
Input_1 ~ Input_4: 모델에 입력될 4개 프레임의 이미지 파일명입니다.
주의: 각 파일명은 랜덤한 3자리 영문 코드로 암호화되어 있으며, 파일명 자체에는 순서 정보가 포함되어 있지 않습니다.
이미지는 알파벳 순서로 정렬되어 제공되지만, 이는 실제 시간 순서와 무관합니다.
No_ordering: 프레임 재배열이 불필요한 데이터인지 여부입니다. (True일 경우 이미지는 셔플링되지 않았으며, 정답은 [1, 2, 3, 4]로 고정됩니다.)
Answer: 프레임의 실제 시간적 순서를 나타내는 정답 라벨입니다.
형식: [n, n, n, n]형태의 리스트 문자열, n은 1~4의 정수 값.
예시: [2, 4, 3, 1]인 경우, 제공된 이미지 중 첫 번째(Input_1)가 실제로는 2번째 프레임, 두 번째(Input_2)가 4번째 프레임임을 의미합니다.
test.csv
평가 데이터셋은 학습 데이터와 유사하지만, 정답과 관련된 정보가 제외됩니다.

Id: 비디오 샘플 고유 식별자.
Sentence: 비디오 텍스트 캡션.
Input_1 ~ Input_4: 암호화된 이미지 파일명.
데이터 디렉토리 구조
train/
├── 00aB12/
│   ├── 00aB12_aek.jpg  (Input_1)
│   ├── 00aB12_bmw.jpg  (Input_2)
│   ├── 00aB12_cyd.jpg  (Input_3)
│   └── 00aB12_dqa.jpg  (Input_4)
├── 01cD34/
│   ├── ...
...

test/
├── 02eF56/
│   ├── ...
...
본 데이터는 경진대회 참여 목적 외 다른 목적으로의 활용을 제한하며, 본 데이터의 원본 또는 수정본을 재배포하는 행위 또한 금지됩니다.


## RULES

Competition Rules
1. 참가 자격
국내외 대학 학부 재학생 또는 휴학생 (전공 및 학년 제한 없음).
상금은 대표자 명의로 된 국내 은행 계좌로만 지급 가능.
참가 신청시 재학증명서를 같이 제출해야 함. (대회 시작일로부터 1개월 이내 발급본)
참가 자격 관련 추가 문의는 담당자에게 문의 바람.
2. 참여 규칙
개인 또는 팀(최대 4인) 으로 참여 가능.
참가 신청시 팀을 결성하여 신청하여야 하며, 신청 후 팀을 합치거나 나누는 것은 불가능함 (적발시 실격).
동일인의 다계정 참가 등록은 금지되며, 적발 시 팀 전체 실격 처리.
3. 모델 학습·추론 및 외부 자원 사용 규칙
3.1 사용 환경
사용 언어: Python
모든 모델은 인터넷이 차단된 로컬 환경(CPU 또는 GPU)에서 실행가능해야 함.
3.2 외부 API 사용
학습 및 추론 과정에서 외부 상용 API(예: ChatGPT, Gemini, Grok 등) 사용을 금지함.
데이터 전처리 목적에 한해 외부 상용 API 사용을 허용하되, 총 사용 비용은 3만 원을 초과할 수 없음.
외부 API를 활용한 경우, 발표 자료에 해당 API의 사용 방식, 실험 결과 및 발생 비용을 명시해야 함.
평가 과정에서 총 구축 비용이 3만 원을 초과한 것으로 판단될 경우 실격 처리될 수 있음.
3.3 데이터 및 모델
모델 학습 시 외부 데이터 사용을 금지하며, 제공된 학습 데이터만 사용할 수 있음.
오픈소스 모델(예: LLaMA 등)은 활용 가능하나, 2026년 5월 31일 이전에 모델 가중치가 공개된 경우에 한함.
허용 여부가 불확실한 경우 Kaggle Discussion 탭을 통해 운영진에게 사전 문의할 것.
모델 앙상블(여러 모델의 추론 결과를 조합하는 행위)은 허용하지 않음. (동일 모델에 대해 데이터셋을 여러 개로 나누어 파인튜닝한 후 추론 결과를 조합하는 방식도 허용하지 않음)
모델 경량화 기법(예: Quantization, LoRA 등)은 허용함.
추론 전략 및 중간 추론 표현 방식(예: Chain-of-Thought, Multi-turn Chat, Test time augmentation 등)은 아래 추론 시간 제한에 위배되지 않는 한에서 활용 가능함.
테스트 셋 전체에 대해 검증 환경 ("4. 제출 규칙" 참조) 에서 24시간 이내에 추론이 완료되어야 함.
3.4 데이터 누수
평가 데이터셋 정보를 학습에 활용하는 행위(데이터 누수, Data Leakage)는 금지함.
예: 평가 데이터에 대한 수작업 라벨링 후 학습에 활용
예: 평가 데이터의 특성을 분석하여 학습 데이터 전처리 또는 모델 설계에 활용
위반 시 실격 처리함.
3.5 규칙 위반
규칙을 준수하지 않을 경우 실격 처리될 수 있음.
대회 진행 중에도 리더보드 상위 참가자를 대상으로 코드 및 결과에 대한 재현성 검증을 요청할 수 있음.
4. 제출 규칙
제출은 1일 2회로 제한.
최종 제출 모델은 NVIDIA RTX 3090 GPU(VRAM 24GB) 1개를 이용하여 실행 가능해야 함.
서버 사양 상세
CPU: AMD EPYC 7502 32-Core Processor x 2
Memory: 512 GB
GPU: GeForce RTX 3090
NVIDIA driver version: 550.54.15
CUDA version: 12.4
5. 예선 평가 규칙
예선기간 중 리더보드는 전체 테스트 데이터의 70% (Public data)만을 이용하여 업데이트됨. 순위는 Exact Match Accuracy를 기준으로 결정.
제출한 이미지 순서가 정답 순서와 완전히 동일한 경우에만 정답으로 인정되며, 순서가 하나라도 다를 경우 오답으로 처리.
예) 정답이 [1, 4, 2, 3]일 때, [1, 4, 2, 3]만 정답으로 처리되며 그 외의 모든 순서는 오답으로 간주.
예선 종료 후 테스트 데이터 전체 (Public + Private data)에 대한 모델 성능 기준 상위 순위팀을 대상으로 코드 검증 및 보고서 검토를 수행.
참가팀들의 성능 달성 수준에 따라 예선 종료 후 외부 데이터셋을 기반으로 한 성능 평가가 별도로 진행될 수 있습니다.
6. 코드 재현성 검증 및 보고서 제출 관련 사항
예선 종료 후 상위 16위 이내 참가자는 지정된 기한 내에 아래 자료를 제출해야 함.
학습 코드 및 추론 코드 (.py 형태)
최종 모델 가중치 파일
과제 해결을 위해 사용한 방법론을 보고서 형태로 정리 (A4 5쪽 이내 분량, MS 워드 파일로 제출하고 별도 양식은 없음.)
제출 코드 관련 준수 사항:
README.md 파일을 상세히 작성할 것
데이터 입·출력 경로는 상대 경로로 표기할 것
코드 및 주석 인코딩은 UTF-8로 통일할 것
모든 코드는 오류 없이 실행되어야 함
개발 환경(OS, 하드웨어 사양) 및 라이브러리 버전을 명시할 것
추론(Inference) 코드는 별도로 작성하고, 사용된 모델 가중치 파일을 함께 제출할 것
모델 구동 코드 및 가중치를 포함한 전체 모델의 크기는 80 GB를 초과하여서는 안 됨.
상세한 절차는 추후 별도 안내
7. 본선 발표 자료 제출
코드 재현성 검증을 통과한 예선 점수 상위팀들을 대상으로 오프라인 발표 평가를 진행함.
발표 자료는 발표 전날까지 PPT 또는 PDF 형식으로 제출할 것.
발표는 10분 이내 분량으로 준비.
8. 본선 발표 규칙
본 발표 10분, 질의응답 5분으로 진행됨.
발표는 기 제출된 발표자료를 이용하며, 진행하며 주최측에서 발표자료를 세팅할 예정.
팀별 대표자가 발표하는 것을 원칙으로 함.
팀원 전체가 본선 발표 행사에 참석하여야 함.
발표는 공개 발표로 진행함.
※ 규칙이 추가될 수 있으니 수시로 확인하시어 불이익을 받는 일이 없도록 해주시기 바랍니다. (최종 업데이트: 6월 29일 19:00)

Kaggle Competition Foundational Rules
(Non-editable)

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.
The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize. b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records. c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible. d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website. b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website. c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules. d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification. b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code. c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email. b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria. c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph. d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle. e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded. f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions). c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8. d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above. f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>. b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law. b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity. c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission. e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

