import os
from bs4 import BeautifulSoup, Comment
from time import sleep
import csv
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# 데이터 저장 파일 경로
lyrics_data_path = './datasets'
if not os.path.exists(lyrics_data_path):
    os.makedirs(lyrics_data_path)

# CSV 파일 초기화
csv_file_path = os.path.join(lyrics_data_path, 'lyrics_splitted_by_sentence.csv')
# with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
#     writer = csv.writer(file)
#     writer.writerow(['index', 'id', 'title', 'singer', 'genres', 'lyrics']) # header

# 셀리니움 크롤링 함수 설정
options = Options() # 자동화 도구 접근 제한 우회
options.add_argument("--disable-blink-features=AutomationControlled") # Automation Info Bar 비활성화 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Whale/4.29.282.14 Safari/537.36")

driver = webdriver.Chrome(options=options) # 드라이버 설정

url = 'https://www.melon.com/chart/age/index.htm' # 멜론 차트 페이지 주소
driver.get(url) # 드라이버가 해당 url 접속
sleep(2)


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

    index = 0 # 가사의 index

    for decade_index in [1, 2]: # 연대 2020, 2010
        ### 연대 선택
        driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[1]/div[1]/ul/li[{decade_index}]/span/label').click()
        sleep(1)
        
        if decade_index == 1: r = range(5, 6)
        else: r = range(1, 6)

        for year_index in r:
            ### 연도 선택
            driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[2]/div[1]/ul/li[{year_index}]/span/label').click()
            sleep(1)

            ### 장르/스타일 선택
            driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/div/div[5]/div[1]/ul/li[2]/span/label').click()
            sleep(1)

            ### 검색 버튼 클릭
            driver.find_element(By.XPATH, '//*[@id="d_srch_form"]/div[2]/button/span/span').click()
            sleep(1)

            if decade_index == 1: 
                ra = range(92, 101)
                driver.find_element(By.XPATH, '//*[@id="frm"]/div[2]/span/a').click()
                sleep(1)
            else: ra = range(1, 101)

            # TOP100 순회 및 데이터 수집
            for song_index in ra:
                
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
                sleep(random.uniform(1, 5)) # 크롤링 속도 불규칙 조정
                try:
                    # song_name 요소가 로드될 때까지 대기
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'song_name'))
                    )
                    html = driver.page_source  # 현재 페이지의 HTML 정보 가져오기
                    soup = BeautifulSoup(html, 'lxml')

                    if soup.select_one('#lst100 > td:nth-child(4) > div > div > div.ellipsis.rank01 > span > span'):
                        continue

                    ### 데이터 수집
                    song_detail_url = soup.select_one('head > meta:nth-child(12)').get('content')
                    if 'songId=' in song_detail_url:
                        id = song_detail_url.split('songId=')[-1] # url에서 songId 추출
                    else:
                        print(f'songId is not found in {song_detail_url}')

                    title = soup.select_one('#downloadfrm > div > div > div.entry > div.info > div.song_name')
                    title.strong.extract()
                    title = title.text.strip()
                    singer = soup.select_one('#downloadfrm > div > div > div.entry > div.info > div.artist > a > span:nth-child(1)').text.strip()
                    genres = soup.select_one('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(6)').text.strip()
                    lyrics = soup.select_one('#d_video_summary')

                    if lyrics:
                        for element in lyrics(text=lambda text: isinstance(text, Comment)):
                            element.extract() # 주석 제거

                        # <br> 태그를 줄바꿈으로 변환하여 텍스트로 추출
                        lyrics = lyrics.get_text(separator='\n').strip()

                        ### 데이터 저장
                        for line in lyrics.split('\n'):
                            if line.strip():  # 빈 줄은 제외
                                index += 1
                                with open(csv_file_path, mode='a', newline='', encoding='utf-8-sig') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([index, id, title, singer, genres, line])
                    else:
                        print("Lyrics not found")

                    print(f'Saved: {singer} - {title}')

                except Exception as inner_e:
                    print(f"Error: {inner_e}")
                
                # 이전 페이지로 돌아감
                driver.back()
                sleep(1)

except Exception as e:
    print('Error:', e)



finally:
    driver.quit()  # 드라이버 종료
    print("Crawling finished")
