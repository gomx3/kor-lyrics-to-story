import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from konlpy.tag import Okt

# csv 읽기
data = pd.read_csv("./datasets/lyrics_splitted_by_sentence.csv")
lyrics = data['lyrics']

# 전처리 (명사 추출)
okt = Okt()
def preprocess(text):
    tokens = okt.nouns(text)
    return ' '.join(tokens)

lyrics_processed = lyrics.apply(preprocess)

# 불용어 리스트 (gpt)
korean_stop_words = [
    "그리고", "그러나", "그런데", "그래서", "하지만", "그러므로", "또한",
    "저는", "나는", "우리는", "당신은", "그는", "그녀는", "그들", 
    "이", "그", "저", "이것", "그것", "저것", 
    "어디", "언제", "무엇", "누구", "왜", "어떻게",
    "있다", "없다", "이다", "아니다", "한다", "했다",
    "안", "못", "좀", "아주", "매우", "너무",
    "같이", "보다", "처럼", "위해", "때문에", "그러니",
    "수", "것", "거", "등", "들", "중", "번", "개"
]

# 벡터화
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=korean_stop_words)
lyrics_matrix = vectorizer.fit_transform(lyrics_processed)

# LDA 토픽 모델링
num_topics = 1
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(lyrics_matrix)

# 토픽별 주요 키워드 추출
words = vectorizer.get_feature_names_out()
topics = {}
for idx, topic in enumerate(lda.components_):
    top_keywords = [words[i] for i in topic.argsort()[:-31:-1]]  # 상위 30개 단어
    topics[f"Topic {idx + 1}"] = top_keywords

print(top_keywords)

# 6. 워드 클라우드 생성 및 시각화
plt.figure(figsize=(15, 10))
for i, (topic, keywords) in enumerate(topics.items(), 1):
    wordcloud = WordCloud(font_path="malgun.ttf", background_color="white").generate(' '.join(keywords))
    plt.subplot(2, 3, i)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(topic)

plt.tight_layout()
plt.show()