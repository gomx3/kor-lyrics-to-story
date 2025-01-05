import os
from bs4 import BeautifulSoup
from time import sleep
import csv

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# 데이터 저장 파일 경로
lyrics_data_path = './dataset'
if not os.path.exists(lyrics_data_path):
    os.makedirs(lyrics_data_path)

# CSV 파일 초기화
csv_file_path = os.path.join(lyrics_data_path, 'top100_chart_2023_2014.csv')
with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'title', 'singer', 'genre', 'lyrics']) # header

# 셀리니움 크롤링 함수 설정
options = Options() # 자동화 도구 접근 제한 우회
options.add_argument("--disable-blink-features=AutomationControlled") # Automation Info Bar 비활성화 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

driver = webdriver.Chrome(options=options) # 드라이버 설정

url = 'https://www.melon.com/chart/age/index.htm' # 멜론 차트 페이지 주소
driver.get(url) # 드라이버가 해당 url 접속
sleep(2)



# 멜론 로그인
driver.find_element(By.CLASS_NAME, 'menu_bg.menu09').click()
sleep(1)

driver.find_element(By.CLASS_NAME, 'btn_gate.melon').click()
sleep(1)

user_id = input("id: ")
user_pw = input("pw: ")

driver.find_element(By.CLASS_NAME, 'text51').click()
sleep(1)
driver.find_element(By.CLASS_NAME, 'text51').send_keys(user_id)
sleep(1)

driver.find_element(By.CLASS_NAME, 'text51.text_password01').click()
sleep(1)
driver.find_element(By.CLASS_NAME, 'text51.text_password01').send_keys(user_pw)
sleep(1)

driver.find_element(By.CLASS_NAME, 'btn_login03').click()
sleep(1)



# 연도 순회
try:
    driver.find_element(By.CLASS_NAME, 'cur_menu.mlog').click()
    sleep(1)

    ### 차트 파인더 클릭
    driver.find_element(By.CLASS_NAME, 'btn_chart_f').click()
    sleep(1)

    ### 연대 차트 선택
    driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/h4[3]/a').click()
    sleep(1)

    for decade_index in [1, 2]: # 연대 2020, 2010
        ### 연대 선택
        driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[1]/div[1]/ul/li[{decade_index}]/span/label').click()
        sleep(1)

        for year_index in (range(1, 5) if decade_index == 1 else range(1, 7)):
            ### 연도 선택
            driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[2]/div[1]/ul/li[{year_index}]/span/label').click()
            sleep(1)

            ### 장르/스타일 선택
            driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/div/div[5]/div[1]/ul/li[2]/span/label').click()
            sleep(1)

            ### 검색 버튼 클릭
            driver.find_element(By.XPATH, '//*[@id="d_srch_form"]/div[2]/button/span/span').click()
            sleep(1)

            # TOP100 순회 및 데이터 수집
            for song_index in range(1, 101):
                
                # 51-100위 일 경우 처리
                if song_index == 51:
                    try:
                        driver.find_element(By.XPATH, '//*[@id="frm"]/div[2]/span/a').click()
                        sleep(1)
                    except Exception as page_error:
                        print(f"Error while navigating to top 51-100: {page_error}")
                        break
                
                # 음악 상세 페이지 이동
                driver.find_element(By.XPATH, f'//*[@id="chartListObj"]/tr[{song_index}]/td[4]/div/a').click()
                
                # html 정보 가져오기
                try:
                    # song_name 요소가 로드될 때까지 대기
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'song_name'))
                    )
                    html = driver.page_source  # 현재 페이지의 HTML 정보 가져오기
                    soup = BeautifulSoup(html, 'lxml')

                    # 데이터 수집
                    song_detail_url = soup.select_one('head > meta:nth-child(12)').get('content')
                    if 'songId=' in song_detail_url:
                        id = song_detail_url.split('songId=')[-1] # url에서 songId 추출
                    else:
                        print(f'songId is not found in {song_detail_url}')

                    title = soup.select_one('#downloadfrm > div > div > div.entry > div.info > div.song_name')
                    title.strong.extract()
                    title = title.text.strip()
                    singer = soup.select_one('#downloadfrm > div > div > div.entry > div.info > div.artist > a').span.extract().text.strip()
                    genre = soup.select_one('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(6)').text.strip()
                    lyrics = soup.select_one('#d_video_summary')

                    # 가사 줄바꿈을 공백으로 변환
                    for br in lyrics.find_all('br'):
                        br.replace_with(' ')
                    lyrics = lyrics.text.strip()

                    # 데이터 저장
                    with open(csv_file_path, mode='a', newline='', encoding='utf-8-sig') as file:
                        writer = csv.writer(file)
                        writer.writerow([id, title, singer, genre, lyrics])
                    print(f'Saved: {title} - {singer}')
                    
                except Exception as inner_e:
                    print(f"Error while fetching song title: {inner_e}")
                
                # 이전 페이지로 돌아감
                driver.back()
                sleep(1)

except Exception as e:
    print('Error:', e)



finally:
    driver.quit()  # 드라이버 종료
    print("Crawling finished")
