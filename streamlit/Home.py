# -*- coding:utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu


# 텍스트
st.header('🏪' + ' 강남구 편의점 매출 예측 서비스')
# st.markdown('---')

# 이미지
# st.image("data/편의점 아이콘.png", use_column_width=True, width = 300, height = 200) 


# st.subheader(':sunglasses: HomePage')
# st.markdown('**소개**')
st.markdown('본 서비스는 **공공데이터**를 기반으로 **강남구의 편의점 매출**에 영향을 미치는 다양한 지표를 분석하여 **시간대별 매출 예측**을 제공합니다.')
st.markdown('---') 


col1, col2 = st.columns(2)

# MainPage
with col1:
    st.subheader('📊 MainPage')
    st.image('data/newplot.png',use_column_width=True)
    st.caption('**비고** : 매출 예측에 사용되는 데이터는 모두 2022년 분기별 최신 데이터를 활용하였습니다')
    st.markdown('   ')
    st.markdown('   ')
    st.markdown('   ')
    st.markdown('   ')
    st.markdown('   ')
    st.markdown('##### [종합] 시간대별 예상 매출')
    st.markdown('**소개** : 상권별 편의점 매출 예측 데이터에 대한 시각화 자료를 확인할 수 있습니다.')
    st.markdown('**사용법** : Side bar에서 상권과 분기를 선택 후 Main에서 결과 확인합니다.')
    # if st.button('Mainpage'):
    #     # 서브페이지 함수 호출
    #     main_page()
# SubPage
with col2:
    st.subheader('🔎 SubPage')
    st.image('data/subpage_home.png', use_column_width=True)
    st.caption('**비고** : 슬라이더의 기본값은 각 상권 및 분기별 최신 데이터로 설정되어 있습니다.')
    st.markdown('##### [세부] 매출 요인 조절 & 예측')
    st.markdown('**소개** : 매출과 관련된 변수를 직접 조절하여 예상 매출을 확인할 수 있습니다.')
    st.markdown('**예상 사용 시나리오** : 이번에 예측하고자 하는 편의점 근처에 새로운 지하철 라인이 개통한다면, 지하철 승하차 승객수를 조절한 후 그 결과를 확인해 볼 수 있습니다.')
    st.markdown('**소개** : 매출과 관련된 변수를 직접 조절하여 예상 매출을 확인할 수 있습니다.')
    st.markdown('**사용법**')
    st.markdown('☝️ Side bar에서 상권과 분기를 선택하면, 상권별로 조절할 수 있는 변수가 Main에 표시됩니다.')
    st.markdown('✌️ 변수들을 슬라이더로 조절한 후에 "예측 버튼"을 누르면 선택한 변수로 예상 매출액을 확인할 수 있습니다.')

    # if st.button('Subpage'):
    #     # 서브페이지 함수 호출
    #     sub_page()

# 페이지 연결
def main_page():
    st.header("Main")
def sub_page():
    st.subheader("Sub")


