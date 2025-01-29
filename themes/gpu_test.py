import os
import pandas as pd
import json
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

# 소설 데이터 불러오기 함수
def getNovelsFromCSV():
    json_file_path = './datasets/gpt_story.json' # JSON 파일 경로
    novels_list = [] # 소설 content 값을 저장할 리스트
    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as f:
        novels = json.load(f)
        # content 값만 추출하여 리스트에 추가 (totalNovels만큼만 반복)
        for i, novel in enumerate(novels):
            # content에 감정 태그 추가
            emotion_tag = "[슬픔] "
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
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>', 
    eos_token='</s>', 
    unk_token='<unk>',
    pad_token='<pad>', 
    mask_token='<mask>'
)

dataset = NovelsDataset(novels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 모델 초기화
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

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
    eval_strategy="epoch",  # 수정된 부분: eval_strategy -> evaluation_strategy
)

train_novels, eval_novels = train_test_split(novels, test_size=0.2, random_state=42)
train_dataset = NovelsDataset(train_novels, tokenizer)
eval_dataset = NovelsDataset(eval_novels, tokenizer)

# Trainer에 데이터셋 전달
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 검증 데이터셋
)

# 모델 학습
trainer.train()

# 이야기 생성 함수
def generate_story(lyrics_input, emotion_tag="슬픔"):
    # 감정 태그를 앞에 추가하여 모델에 전달
    emotion_input = f"[{emotion_tag}] {lyrics_input}"
    # 입력 토큰화
    input_ids = tokenizer.encode(emotion_input, return_tensors='pt')
    
    # GPU로 입력 텐서를 이동
    input_ids = input_ids.to(device)

    # 출력 생성
    output = model.generate(
        input_ids, 
        max_new_tokens=150,  # 새로 생성할 토큰의 개수를 제한
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

# 가사로 이야기 생성
lyrics_input = "귀를 기울일 때마다 너의 목소리가 들리니까 That's why I love when tears are falling I don't wanna give it up 하나만을 바라왔던 Days 괴로워진 맘에 하늘만 바라볼 때 혼자가 아니야 마음속에 떠오르는 Voice yeah 넌 노래하는 듯이 내 맘을 어루만져와 이젠 말할 수 있어 Cause I'll be with you 그 어떤 불안도 지금 너와 Listen to my sweet heartbeat 바꿔나갈 테니까 귀를 기울일 때마다 나의 목소리가 들리니까 That's why you laugh when tears are falling 다가올 그 어떤 날도 사랑할 수 있는 우리니까 That's why I love when tears are falling Story in my dream 드디어 너를 만났던 One day 손에 가득한 꽃다발을 건넨 나 넘칠 듯한 고마웠던 맘 떨어진 눈물의 수만큼 가득 피어나 이젠 말할 수 있어 Cause I'll be with you 그 어떤 불안도 지금 너와 Listen to my sweet heartbeat 바꿔나갈 테니까 귀를 기울일 때마다 나의 목소리가 들리니까 That's why you laugh when tears are falling 다가올 그 어떤 날도 사랑할 수 있는 우리니까 That's why I love when tears are falling"
generated_story = generate_story(lyrics_input, emotion_tag="슬픔")
print("생성된 이야기:", generated_story)

# 생성된 이야기 저장
output_file_path = './tears_are_falling.txt'  # 저장할 파일 경로
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(generated_story)
