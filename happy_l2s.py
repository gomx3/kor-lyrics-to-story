import os
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader 
import json

totalNovels = 15

# 소설 데이터 불러오기 함수
def getNovelsFromCSV():
    json_file_path = './datasets/novels_data.json' # JSON 파일 경로

    novels_list = [] # 소설 content 값을 저장할 리스트

    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as f:
        novels = json.load(f)

        # content 값만 추출하여 리스트에 추가 (totalNovels만큼만 반복)
        for i, novel in enumerate(novels):
            if i >= totalNovels: break  # totalNovels 만큼만 처리

            # 행복 태그를 여러 번 추가 (예: 3번 반복)
            emotion_tag = "[행복] " * 3  # 행복 태그 3번 반복
            novels_list.append(emotion_tag + novel['content'])

    return novels_list

# 데이터셋 클래스 정의
class NovelsDataset(Dataset):
    def __init__(self, lyrics, tokenizer, max_length=512):
        self.lyrics = lyrics
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.lyrics[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # labels 생성 (input_ids 복사 및 패딩 토큰을 -100으로 설정)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

# 데이터 준비
novels = getNovelsFromCSV()

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', 
                                                    eos_token='</s>', 
                                                    unk_token='<unk>',
                                                    pad_token='<pad>', 
                                                    mask_token='<mask>')
dataset = NovelsDataset(novels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 모델 초기화
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# >>> 모델 토큰 지원 추가
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id


# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 모델 학습
trainer.train()

# 이야기 생성 함수
def generate_story(lyrics_input, emotion_tag="행복"):
    # 감정 태그를 앞에 추가하여 모델에 전달
    emotion_input = f"[{emotion_tag}] {lyrics_input}"

    # 모델에 감정 태그와 함께 텍스트 입력
    input_ids = tokenizer.encode(emotion_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, num_return_sequences=1)
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_story

# 예시 가사로 이야기 생성
lyrics_input = "떴다 떴다 비행기 날아라 날아라 높이 높이 날아라 우리 비행기"
generated_story = generate_story(lyrics_input, emotion_tag="행복")
print("생성된 행복한 이야기:", generated_story)
