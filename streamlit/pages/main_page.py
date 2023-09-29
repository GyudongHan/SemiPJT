import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import folium
import base64
import pyproj
import time


# 한글 폰트 설정
plt.rcParams['font.family'] = "AppleGothic"
# Windows, 리눅스 사용자
# plt.rcParams['font.family'] = "NanumGothic"
plt.rcParams['axes.unicode_minus'] = False


# 모델 불러오기
model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

# 데이터 프레임 불러오기
df = pd.read_csv('data/전체_수정_streamlit용.csv')
df1 = pd.read_csv('data/골목_streamlit용.csv')
df2 = pd.read_csv('data/비골목_streamlit용.csv')

## ----------------------------------------------side bar ----------------------------------------------
with st.sidebar:
    # Select market
    unique_market = df['상권_코드_명'].unique().tolist()
    selected_feature1 = st.selectbox("상권을 선택하세요", unique_market)
    
    # Select quarter
    unique_quarter = ['1분기', '2분기', '3분기', '4분기']
    selected_feature3 = st.selectbox("분기를 선택하세요!", unique_quarter)

    
# 상권 타입 지정 
type = df[df['상권_코드_명'] == selected_feature1]['상권_구분_코드_명'].tolist()[0]
if type == '골목상권' :
    type_code = 0 # 골목상권
else:
    type_code = 1 # 비골목상권

## --------------------------------------------- 메인 텍스트 영역 -------------------------------------
st.subheader('📊  강남구 편의점 매출 예측 서비스')
st.markdown('###### 좌측 사이드바에서 상권과 분기를 선택하면, 시간대별 예상 매출을 확인하실 수 있습니다')
st.caption('하단 지도에서 상권의 영역을 확인해보세요!👀')
##------------------------------------------------ 지도 영역 -------------------------------------------
# HTML 파일을 읽어 Base64로 변환
# with open('map.html', 'r') as f:
# html = f.read()
# b64 = base64.b64encode(html.encode()).decode()
with open('map.html', 'r', encoding='utf-8') as f:
    html = f.read()
    b64 = base64.b64encode(html.encode()).decode()

# Base64로 인코딩된 HTML을 출력
st.markdown(f'<iframe src="data:text/html;base64,{b64}" width=750 height=500></iframe>', unsafe_allow_html=True)
st.markdown('---')
# ----------------------------------------------------------------------------------------------------------------------------
## 변수 영역
feature_names_gol = df1.iloc[:, 7:].columns.tolist() 
feature_names_ngol = df2.iloc[:, 7:].columns.tolist() 
# st.write(feature_names_gol)
# st.write(feature_names_ngol )

#시간대, 분기 값 리스트의 앞에 넣기
time1, time2, time3, time4, time5, quarter1, quarter2, quarter3 = 0, 0, 0, 0, 0, 0, 0, 0

if selected_feature3 == unique_quarter[0]:
    quarter1 = 1
    quarter2 = 0
    quarter3 = 0
    quarter_type = 1
elif selected_feature3 == unique_quarter[1]:
    quarter1 = 0
    quarter2 = 1
    quarter3 = 0
    quarter_type = 2
elif selected_feature3 == unique_quarter[2]:
    quarter1 = 0
    quarter2 = 0
    quarter3 = 1
    quarter_type = 3
else :
    quarter1 = 0
    quarter2 = 0
    quarter3 = 0
    quarter_type = 4

numeric_user_inputs = []
for i in range (6) :
    user_input_i = [time1, time2, time3, time4, time5, quarter1, quarter2, quarter3]
    numeric_user_input_i=[]
    if i <= 4 :
        user_input_i[i] = 1
    if type_code == 0 : 
        filter_df = df1[(df1['상권_코드_명'] == selected_feature1) & (df1['분기'] == quarter_type) & (df1['기준_년_코드'] == 2022)]
    else :
        filter_df = df2[(df2['상권_코드_명'] == selected_feature1) & (df2['분기'] == quarter_type) & (df2['기준_년_코드'] == 2022)]
    filter_list_i = filter_df.iloc[i, 7:].tolist()
    user_input_i.extend(filter_list_i)
    for value in user_input_i :
        try:
            numeric_value = float(value)
            numeric_user_input_i.append(numeric_value)
        # 예외처리
        except ValueError:
            st.error(f"입력값 '{value}'은(는) 숫자로 변환할 수 없습니다.")
    numeric_user_inputs.append(numeric_user_input_i)



# 배열 생성    
feature_array = np.array(numeric_user_inputs)
#st.write(feature_array)

    ## 예측
if type_code == 0:
    predictions = model1.predict(feature_array)
else :
    predictions = model2.predict(feature_array)
    #st.write(predictions)

## 시각화
# 데이터 프레임으로 변경
df_predictions = pd.DataFrame({'예상 매출': predictions})
df_predictions.insert(0, '시간대', ['00 ~ 06', '06 ~ 11', '11 ~ 14', '14 ~ 17', '17 ~ 21', '21 ~ 24'])

# 정수로 변환
df_predictions['예상 매출'] = df_predictions['예상 매출'].astype(int)

    # plotly 시각화
st.markdown(f"### {selected_feature1} {selected_feature3} 편의점 시간대별 예상 매출")
st.caption('👉 어느 시간대에 매출이 가장 높은지 확인해보세요!')
bar_trace = go.Bar(
    x=df_predictions['시간대'],
    y=df_predictions['예상 매출'],
    text=[f'{val:,}' for val in df_predictions['예상 매출']],
    textposition='inside',
    texttemplate='%{text}',
)

layout = go.Layout(
    xaxis_title='시간대',
    yaxis_title='예상 매출'
)

bar_fig = go.Figure(data=[bar_trace], layout=layout)
st.plotly_chart(bar_fig) 
max_type = df_predictions.loc[df_predictions['예상 매출'].idxmax()]['시간대']
st.markdown(f'####  👉 시간대 {max_type} 의 매출이 가장 높습니다!')

st.markdown('---')
st.write('')


# 합치기
# feature_array와 predictions를 수평으로 연결
predictions = predictions[:, np.newaxis] # 2D 배열로 만들기 
merged_array = np.hstack((feature_array, predictions))


# 결과 출력
# st.write("Merged Array:")
# st.write(merged_array)