import os
import requests
from bs4 import BeautifulSoup
from time import sleep
import csv

import selenium
from selenium import webdriver
# from selenium.webdriver import Chrome
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By # for find_element
 
# 데이터 저장 파일 경로
lyrics_data_path = './dataset'
if not os.path.exists(lyrics_data_path):
    os.makedirs(lyrics_data_path)

# 셀리니움 크롤링 함수 설정
driver = webdriver.Chrome() # 드라이버 설정

url = 'https://www.melon.com/chart/age/index.htm' # 멜론 차트 페이지 주소
driver.get(url) # 드라이버가 해당 url 접속
sleep(2)

# 차트 순회
try:
    driver.find_element(By.CLASS_NAME, 'cur_menu.mlog').click()
    sleep(1)

    ### 차트 파인더 클릭
    driver.find_element(By.CLASS_NAME, 'btn_chart_f').click()
    sleep(1)

    ## 연대 차트 선택
    driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/h4[3]/a').click()
    sleep(1)

    for decade_index in [1, 2]: # 연대 2020, 2010
        ### 연대 선택
        driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[1]/div[1]/ul/li[{decade_index}]/span/label').click()
        sleep(1)

        for year_index in (range(1, 5) if decade_index == 1 else range(1, 8)):
            ### 연도 선택
            driver.find_element(By.XPATH, f'//*[@id="d_chart_search"]/div/div/div[2]/div[1]/ul/li[{year_index}]/span/label').click()
            sleep(1)

            ### 장르/스타일 선택
            driver.find_element(By.XPATH, '//*[@id="d_chart_search"]/div/div/div[5]/div[1]/ul/li[2]/span/label').click()
            sleep(1)

            ### 검색 버튼
            driver.find_element(By.XPATH, '//*[@id="d_srch_form"]/div[2]/button/span/span').click()
            sleep(1)

except Exception as e:
    print('Error:', e)


sleep(10)
