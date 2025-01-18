import os
import pandas as pd
import json
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

totalNovels = 100

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
    eval_strategy="epoch",
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

    # 출력 생성
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, 
                            temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2, do_sample=True  # 샘플링 활성화
    )

    # 입력 토큰 개수 계산
    input_length = input_ids.shape[1]

    # 생성된 토큰 중 입력 토큰 이후의 부분만 디코딩
    generated_tokens = output[0][input_length:]
    generated_story = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_story

# 가사로 이야기 생성
lyrics_input = "생각이 많은 건 말이야 당연히 해야 할 일이야 나에겐 우리가 지금 1순위야 안전한 유리병을 핑계로 바람을 가둬 둔 것 같지만  기억나? 그날의 우리가 잡았던 그 손엔 말이야 설레임보다 커다란 믿음이 담겨서 난 함박웃음을 지었지만 울음이 날 것도 같았어 소중한 건 언제나 두려움이니까  문을 열면 들리던 목소리 너로 인해 변해있던 따뜻한 공기 여전히 자신 없지만 안녕히  저기, 사라진 별의 자리 아스라이 하얀 빛 한동안은 꺼내 볼 수 있을 거야 아낌없이 반짝인 시간은 조금씩 옅어져 가더라도 너와 내 맘에 살아 숨 쉴 테니   여긴, 서로의 끝이 아닌 새로운 길 모퉁이 익숙함에 진심을 속이지 말자 하나 둘 추억이 떠오르면 많이 많이 그리워할 거야 고마웠어요 그래도 이제는 사건의 지평선 너머로  솔직히 두렵기도 하지만 노력은 우리에게 정답이 아니라서 마지막 선물은 산뜻한 안녕  저기, 사라진 별의 자리 아스라이 하얀 빛 한동안은 꺼내 볼 수 있을 거야 아낌없이 반짝인 시간은 조금씩 옅어져 가더라도 너와 내 맘에 살아 숨 쉴 테니  여긴, 서로의 끝이 아닌 새로운 길 모퉁이 익숙함에 진심을 속이지 말자  하나 둘 추억이 떠오르면 많이 많이 그리워할 거야 고마웠어요 그래도 이제는 사건의 지평선 너머로  저기, 사라진 별의 자리 아스라이 하얀 빛 한동안은 꺼내 볼 수 있을 거야 아낌없이 반짝인 시간은 조금씩 옅어져 가더라도 너와 내 맘에 살아 숨 쉴 테니  여긴, 서로의 끝이 아닌 새로운 길 모퉁이 익숙함에 진심을 속이지 말자  하나 둘 추억이 떠오르면 많이 많이 그리워할 거야 고마웠어요 그래도 이제는 사건의 지평선 너머로  사건의 지평선 너머로"
generated_story = generate_story(lyrics_input, emotion_tag="슬픔")
print("생성된 이야기:", generated_story)
