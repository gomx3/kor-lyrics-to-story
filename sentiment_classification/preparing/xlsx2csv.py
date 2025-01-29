import os
import pandas as pd

train = pd.read_excel('./datasets/sentiment_conversation/origin_train.xlsx')
train.to_csv('./datasets/sentiment_conversation/origin_train.csv')

validation = pd.read_excel('./datasets/sentiment_conversation0/origin_validation.xlsx')
validation.to_csv('./datasets/sentiment_conversation/origin_validation.csv')
