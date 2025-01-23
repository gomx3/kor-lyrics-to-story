import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt_tab')

# 가사 데이터 로드
data = pd.read_csv("./datasets/lyrics_splitted_by_sentence.csv")
df = pd.DataFrame(data)

# 전처리
def preprocess(text):
    text = re.sub(r"[^가-힣a-zA-Z\s]", "", text)  # 특수문자 제거
    text = text.lower()  # 소문자 변환
    stop_words = set(stopwords.words('english'))  # 영어 불용어
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  # 불용어 제거 및 짧은 단어 필터링
    
    return " ".join(tokens)

data['cleaned_lyrics'] = data['lyrics'].apply(preprocess)

print(data[['lyrics', 'cleaned_lyrics']])

# 1. 단어 빈도 계산
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_lyrics'])
word_freq = pd.DataFrame({
    "word": vectorizer.get_feature_names_out(),
    "frequency": X.toarray().sum(axis=0)
}).sort_values(by="frequency", ascending=False)

print(word_freq.head())

# 2. WordCloud 시각화
wordcloud = WordCloud(
    font_path="malgun.ttf", background_color="white"
).generate(" ".join(data['cleaned_lyrics']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# ---

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_lyrics'])

# 2. LDA 모델 학습
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)  # 주제 개수 3개
lda_model.fit(tfidf_matrix)

# 3. 주요 단어 출력 (주제별)
terms = tfidf_vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda_model.components_):
    print(f"주제 {idx + 1}: ", [terms[i] for i in topic.argsort()[-10:]])

# 1. LDA 결과로 각 가사에 대한 주제 분포 추출
topic_distribution = lda_model.transform(tfidf_matrix)
data['dominant_topic'] = topic_distribution.argmax(axis=1)

# 2. 주제 레이블 할당
topic_labels = {
    0: "그리움",
    1: "고백",
    2: "감정 혼란"
}
data['theme'] = data['dominant_topic'].map(topic_labels)

print(data[['lyrics', 'cleaned_lyrics', 'theme']])