# kor-lyrics-to-story

CUAI Winter Conference 2024 NLP TEAM 2 (25.01.05~)

## 🪽 노래 가사로 스토리 생성 AI

이는 노래 가사를 입력하면 이야기를 생성하는 AI 프로젝트로, CUAI 7기 동계 컨퍼런스 NLP 2팀에서 진행했습니다.

- CUAI 레포지토리: \_

## 🌀 진행 현황

- [x] 멜론 차트 TOP100 10년치 분량 가사 데이터 크롤링 (id, title, singer, genre, lyrics)
- [x] koGPT2 활용해 태깅 없이 이야기 생성 모델 학습
- [x] KLUE/roberta-base 활용해 감성 분류 모델 생성
- [] 감성, 테마 태깅 후 이야기 생성 모델 학습 후 성능 확인

## 🛠️

- Python(^3.8.20)
- BeautifulSoup4(^4.12.2)
- Selenium(^4.19.0)
- torch(^2.4.1)
- transformers(^4.46.3)
