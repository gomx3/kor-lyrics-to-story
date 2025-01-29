from transformers import pipeline
import pandas as pd

# 가사 데이터 로드
data = pd.read_csv("./datasets/lyrics_splitted_by_sentence.csv")
data = pd.DataFrame(data).head(50)

# 2. Hugging Face 분류 파이프라인 설정
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 3. 분류할 주제 정의
# candidate_labels = ["사랑", "우정", "일상", "자신감", "파티", "추억"]
labels = ['기쁨', '슬픔', '당황', '분노', '상처', '불안']

# 4. 가사별 주제 분석
def analyze_themes(lyrics_list, labels):
    results = []
    for index, item in lyrics_list.iterrows():  # 각 행을 순회
        lyric = item["lyrics"]  # 가사 컬럼
        index = item["index"]  # 인덱스 컬럼
        print(lyric, index)
        result = classifier(lyric, candidate_labels=labels)
        
        # 가장 높은 점수를 받은 주제 선택
        theme = result['labels'][0]
        results.append({"index": index, "lyrics": lyric, "theme": theme})
    return results

# 5. 분석 수행
theme_results = analyze_themes(data, candidate_labels)

# 6. 결과를 DataFrame으로 변환 및 저장
df = pd.DataFrame(theme_results)
print(df)

# 결과를 CSV 파일로 저장
df.to_csv("lyrics_test.csv", index=False, encoding="utf-8-sig")
