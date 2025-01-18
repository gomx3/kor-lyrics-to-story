import os 
import json

# 소설 txt 파일의 디렉토리 경로
novels_path = './datasets/novels'

# JSON 데이터를 저장할 리스트
novels = []

# 폴더 내의 모든 .txt 파일을 처리
for filename in os.listdir(novels_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(novels_path, filename)

        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # JSON 형식으로 추가
        novel_entry = {
            'file_name': filename.replace('.txt', ''),
            'content': content,
        }
        novels.append(novel_entry)

# JSON 파일로 저장
output_file = './datasets/novels_data.json'
with open(output_file, 'w', encoding='utf-8') as json_f:
    json.dump(novels, json_f, ensure_ascii=False, indent=4)

print(f"모든 .txt 파일 속 소설이 [{output_file}] 파일로 저장되었습니다.")
