import os
import pandas as pd
import json
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 감정 카테고리
emotions = ['상처', '불안', '기쁨', '슬픔', '분노', '당황']

# 라벨 인코더
le = LabelEncoder()



### 데이터 로드
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 중인 파일 위치
lyrics1_path = os.path.join(script_dir, "dataset/label_result_song_short.csv")
lyrics2_path = os.path.join(script_dir, "dataset/translated_lyrics.csv")

lyrics1 = pd.read_csv(lyrics1_path) # 감성 라벨링 가사 데이터
lyrics2 = pd.read_csv(lyrics2_path) # 가사 한국어 번역 데이터

lyrics = lyrics2.merge(lyrics1, on=["index", "id", "title", "singer", "genres", "lyrics"])

novel_final_path = os.path.join(script_dir, "dataset/novel_final.csv")
novel_final = pd.read_csv(novel_final_path)
data = novel_final.copy()



### Preprocessing

# 빈도가 높은 감성 두 개를 얻는 함수
def get_top_two_sentiments(sentiments):
    sentiment_counts = sentiments.value_counts()
    top_two = sentiment_counts.head(2).index.tolist()

    # 2개 미만일 경우 빈 값 처리
    return top_two + [None] * (2 - len(top_two))

#
def get_lyrics_final(lyrics):
    # groupby와 agg를 활용해 변환
    lyrics = lyrics.groupby("id").agg({
        "title": "first",   # 첫 번째 값 유지
        "singer": "first",  # 첫 번째 값 유지
        "genres": "first",  # 첫 번째 값 유지
        "lyrics": "first",  # 첫 번째 값 유지
        "translated_lyrics": list,  # 리스트로 묶음
        "sentiment_label": lambda x: get_top_two_sentiments(x)  # 빈도가 높은 2개의 감정 추출
    }).reset_index()

    # sentiment_label에서 두 개의 감정을 각각의 칼럼으로 분리
    lyrics[['top_sentiment', 'second_sentiment']] = pd.DataFrame(lyrics['sentiment_label'].tolist(), index=lyrics.index)

    # 불필요한 열 삭제
    lyrics = lyrics.drop(columns=["sentiment_label"])

    # 감성 태그 인코딩
    le.fit(emotions)

    lyrics['감정1_encoded'] = le.transform(lyrics['top_sentiment'])
    lyrics['감정2_encoded'] = le.transform(lyrics['second_sentiment'])

    return lyrics

#
def get_novel_final(data):
    le.fit(emotions)

    # 감정1과 감정2를 숫자로 변환
    data['감정1_encoded'] = le.transform(data['top_sentiment'])
    data['감정2_encoded'] = le.transform(data['second_sentiment'])

    # 수치형 감정1, 감정2와 텍스트 결합
    data['input_text'] = data['감정1_encoded'].astype(str) + " | " + data['감정2_encoded'].astype(str) + " | " + data['text']

    # 불필요한 문자 제거
    data['input_text'] = data['input_text'].str.replace(r"[\[\],']", "", regex=True)

    data[['top_sentiment', 'second_sentiment', 'input_text']].sample(n=5)

    return data

# 최종 데이터 생성
lyrics_final = get_lyrics_final(lyrics)
data = get_novel_final(data)



### 데이터셋 클래스 정의
class NovelsDataset(Dataset):
    def __init__(self, data, tokenizer, text_column="input_text", max_length=128):
        self.data = data.reset_index(drop=True)  # 인덱스 리셋 (중복 방지)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, self.text_column] # input_text 가져오기

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # labels 생성 (input_ids 복사 및 패딩 토큰을 -100으로 설정)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }



### Modeling

# KOGPT2 모델과 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

# 모델 초기화
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

dataset = NovelsDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    eval_strategy="epoch",
)

train_novels, eval_novels = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = NovelsDataset(train_novels, tokenizer)
eval_dataset = NovelsDataset(eval_novels, tokenizer)



### Training

# Trainer에 데이터셋 전달
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 검증 데이터셋
)

# 샘플 출력
sample = train_dataset[0]
print("Input IDs:", sample['input_ids'])
print("Attention Mask:", sample['attention_mask'])
print("Labels:", sample['labels'])

# 모델 학습
trainer.train()


### validation
device = torch.device("cuda")

# 이야기 생성 함수
def generate_story(lyrics_input):
    # 입력 토큰화
    input_ids = tokenizer.encode(lyrics_input, return_tensors='pt').to(device)

    # 출력 생성
    output = model.generate(
        input_ids, 
        max_new_tokens=150, # 새로 생성할 토큰의 개수를 제한
        num_return_sequences=1, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.9, 
        repetition_penalty=1.2, 
        do_sample=True  # 샘플링 활성화
    )

    # 입력 길이 추적
    input_length = input_ids.shape[1]

    # 생성된 토큰 중 입력 토큰 이후의 부분만 디코딩
    generated_tokens = output[0][input_length:]
    generated_story = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_story


i = 732
print(lyrics_final["title"][i])

lyrics = " ".join(lyrics_final["translated_lyrics"][i])

prompt = lyrics_final['감정1_encoded'][i].astype(str) + " | " + lyrics_final['감정2_encoded'][i].astype(str) + " | " + lyrics
print(prompt)

# 가사로 이야기 생성
generated_story = generate_story(prompt)
print("생성된 이야기:", generated_story)