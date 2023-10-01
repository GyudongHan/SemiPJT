import requests
import pandas as pd

def apidata(service, df_name, apikey): # 서비스명, 저장할 이름, 인증키
    years = [2020, 2021, 2022] # 3개년 데이터

    data_dict = {}

    for year in years:
        page = 1  # 초기식

        while True:
            end_num = page * 1000
            start_num = end_num - 999
            url = f'http://openapi.seoul.go.kr:8088/{apikey}/json/{service}/{start_num}/{end_num}/{year}'
            res = requests.get(url)
            sales_data = res.json()

            # 데이터가 있는지 확인
            if f'{service}' in sales_data and 'row' in sales_data[f'{service}']:
                if year not in data_dict:
                    data_dict[year] = []
                data_dict[year].append(pd.DataFrame(sales_data[f'{service}']['row']))
                page += 1
            else:
                break  # 데이터가 없으면 종료
