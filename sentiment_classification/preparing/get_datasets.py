import pandas as pd

train_src = pd.read_csv('./datasets/sentiment_conversation/origin_train.csv')
val_src = pd.read_csv('./datasets/sentiment_conversation/origin_validation.csv')

# 필요한 열 선택
train = train_src[['사람문장1', '사람문장2', '사람문장3', '감정_대분류']]
val = val_src[['사람문장1', '사람문장2', '사람문장3', '감정_대분류']]

# 데이터 변환: 사람문장 열 하나로 묶기
train = train.melt(id_vars=['감정_대분류'], value_vars=['사람문장1', '사람문장2', '사람문장3'], 
                          var_name='문장_종류', value_name='text').drop(columns=['문장_종류'])
val = val.melt(id_vars=['감정_대분류'], value_vars=['사람문장1', '사람문장2', '사람문장3'], 
                      var_name='문장_종류', value_name='text').drop(columns=['문장_종류'])

# NaN 값 제거 (문장이 없는 경우)
train.dropna(inplace=True)
val.dropna(inplace=True)

train.rename(columns={'감정_대분류':'sentiment'}, inplace=True)
val.rename(columns={'감정_대분류':'sentiment'}, inplace=True)



# 감성 종류 확인
print('train:', train['sentiment'].unique())
print('validation: ', val['sentiment'].unique())



# 감성 레이블 정수 변환 (0~5)
sentiment_mapping = {
    '기쁨': 0,
    '슬픔': 1,
    '분노': 2,
    '상처': 3,
    '불안': 4,
    '당황': 5,
}

train['sentiment'] = train['sentiment'].map(sentiment_mapping)
val['sentiment'] = val['sentiment'].map(sentiment_mapping)

# CSV 파일 저장
train.to_csv('./datasets/sentiment_conversation/train.csv', index=False, encoding='utf-8-sig')
val.to_csv('./datasets/sentiment_conversation/val.csv', index=False, encoding='utf-8-sig')
