import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import scipy.interpolate
from datetime import datetime 
import numpy  as np
import scipy as sp

# functions
def make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(go.Scatter(x=categories, y=traceY, name=traceName, line_shape='spline',mode="lines+markers+text",              
        text=['<br>{num}</br>'.format(num = i) for i in traceY],
        textposition=textPos,
        marker_color='#DDA0DD',                
        textfont=dict(
            family="Corbel",
            size=14,
            color="#cb91ce "
        )),
        secondary_y=True,
    )
    fig.add_trace(go.Scatter(x=categories, y=[50]*len(categories) , name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
        secondary_y=True,
    )


    fig.add_trace(
        go.Bar(x=categories, y=barY, name=barName ,text=["{:.1%}".format(i) for i in barY],
            textposition='auto',
            marker_color='#8B008B',
              ),
        secondary_y=False, 
    )

    # Add figure title
    fig.update_layout(
        title_text="SASB dimensions / Volume & Intelligence Score",
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=legendY,
        xanchor="center",
        x=legendX
    ))

    fig.update_layout(
        yaxis = dict(
            tickformat = ',.0%',
            range = yRange        
        )
    )
    fig.update_layout(
        font=dict(
            family="Corbel",
            size=14,
            color="#000000"
        )
    )
    fig.update_layout(plot_bgcolor="#FFFFFF")



    fig.update_layout(
        title={
            'y':titleY,
            'x':titleX,
            'xanchor': 'right',
            'yanchor': 'top'})

    fig.update_xaxes(
            tickangle = 0,
            title_font = {"size": 14},
            title_standoff = 100,
            tickmode= "auto")

    fig.update_layout(
        autosize=False,
        width=570,
        height=450,
        margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
        ),
    )
    col.plotly_chart(fig)
    

def make_button(col, text, col1, col2, col3, text1, text2, image):
    style = ("text-align:left; padding: 0px; font-family: arial black;, "
         "font-size: 300%")
    if col.button(text):
        categories = [ 'Environment',
            'Social <br>Capital',
            'Human <br>Capital',
            'Leadership & <br>Governance',
            'Business Model & <br>Innovation']
        traceY = [50.38,49.84,47.82,48.37,53.73]
        traceName = "Intelligence Score"
        barName = "Volume / Contribution of<br>each SASB dimension"
        barY = [0.03,0.13,0.22,0.52,0.09]
        legendX = 0.5
        legendY = -0.44
        titleX = 0.7
        titleY = 0.9
        textPos = 'top center'
        yRange = [0,0.6]
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col1)
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col2)
        col1.markdown(text1)# text under 1st plot
        col1.markdown("---")
        col2.markdown(text2)# text under 2nd plot
        col2.markdown("---")
        col3.image(image, width = 900)

def make_button_df(col, text, df1, df2, col1, col2):
    if col.button(text):
        col1.dataframe(df1)
        col2.dataframe(df2)
        
        
def make_pie(size, colors, column):
    labels = ['','']
    sizes =[100-size, size]
    efont = {'fontname':'Comic Sans MS'}
    color = [[i/255. for i in c] for c in colors]
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.35
    kwargs = dict(colors=color, startangle=90)
    outside, _ = ax.pie(sizes, radius=1, pctdistance=1-width/2,labels=labels,**kwargs)
    plt.setp( outside, width=width, edgecolor='white')

    kwargs = dict(size=60, fontweight='bold', va='center', color = '#616161', **efont)
    ax.text(0, 0, size, ha='center', **kwargs)
    column.pyplot(fig)
    
def make_rect(text, value, col):
    hfont = {'fontname':'Comic Sans MS'}
    efont = {'fontname':'Corbel'}
    plt1 = plt.figure()
    currentAxis = plt1.gca()
    currentAxis.set_axis_off()
    currentAxis.add_patch(Rectangle((0.2, 0.3), 0.7, 0.4, fill=None, alpha=10))
    plt1.text(0.3, 0.39, text, fontsize=15, **hfont)
    plt1.text(0.3, 0.51, value , fontsize=40, **hfont)
    col.pyplot(plt1)

# page config
st.set_page_config(page_title="ESG AI", layout='wide', initial_sidebar_state="expanded")
style = ("text-align:center; padding: 0px; font-family: arial black;, "
         "font-size: 400%")
title = f"<h1 style='{style}'>DataESG<sup>AI</sup></h1><br><br>"
st.write(title, unsafe_allow_html=True)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1500}px;
    }}
    .reportview-container .main {{

    }}
</style>
""",
        unsafe_allow_html=True,
    )

companies = ['Select a Company', 'Airbnb', 'Airbnb'] # companies list
col0_1, col0_2, col0_3 =st.beta_columns((0.7, 1, 0.7))
company = col0_2.selectbox('Select a Company to Analyze', companies)
col0_2.write('---')
# current directory
dataPath = os.path.join(".", "airbnb") # change while using companies list

if company and company != "Select a Company":
    
    
# 1st line
    style = ("text-align:left; padding: 0px; font-family: arial black;, "
         "font-size: 300%")
    
    title = f"<h1 style='{style}'>Airbnb</h1>"
    col1_T, col1_1, col1_2 = st.beta_columns((1, 2.7, 30))
    imagePath = os.path.join(dataPath, "logo_airbnb.jpg")
    image = Image.open(imagePath)
    col1_1.image(image, width = 70)
    col1_2.write(title, unsafe_allow_html=True)
# 2nd line
    col2_1, col2_2, col2_3, col2_4 = st.beta_columns((1, 1, 1, 1))
    text1 = 'Stock price, as of 08-Feb-2021' 
    text2 = 'ABNB Mkt cap, 03-Feb-2021'
    text3 = 'Airbnb Revenue Q3, 2020'
    value1 = '$197.24'
    value2 = '$111 B'
    value3 = '$1.3 B'
    percent = 50
    c1 = (232, 232, 232)
    c2 = (139, 0, 139)
    make_pie(percent,[c1,c2], col2_1)
    make_rect(text1, value1, col2_2)
    make_rect(text2, value2, col2_3)
    make_rect(text3, value3, col2_4)
# 3rd line
    col3_T, col3_1, col3_2 =st.beta_columns((0.4, 3, 3))
    
    # 1st plot
    
    company = 'Airbnb'
    df = pd.read_csv(f'Companies/{company}/events.csv',parse_dates=['Date'],index_col='Date')

    env_lables = [25,21,24,19,20,13,23]
    social_lables = [10,2,6,1,22,16]
    hum_labels = [11,8,9]
    gov_labels = [18,5,12,4,0,7,17]
    bus_labels =  [15,14,3]

    envoirmental_df = df.loc[(df['N_label']==25)|
                             (df['N_label']==21)|
                             (df['N_label']==24)|
                             (df['N_label']==19)|
                             (df['N_label']==20)|
                             (df['N_label']==13)|
                             (df['N_label']==23)]


    social_df = df.loc[(df['N_label']==10)|
                             (df['N_label']==2)|
                             (df['N_label']==6)|
                             (df['N_label']==1)|
                             (df['N_label']==22)|
                             (df['N_label']==16)]

    human_capital_df = df.loc[(df['N_label']==11)|
                             (df['N_label']==8)|
                             (df['N_label']==9)]


    gov_df = df.loc[(df['N_label']==18)|
                             (df['N_label']==5)|
                             (df['N_label']==12)|
                             (df['N_label']==4)|
                             (df['N_label']==0)|
                             (df['N_label']==7)|
                             (df['N_label']==17)]

    business_df = df.loc[(df['N_label']==15)|
                             (df['N_label']==14)|
                             (df['N_label']==3)]

    dates = df.resample('M').mean().index

    env_month_avg = envoirmental_df.resample('M').mean()
    env_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in env_month_avg.index:
            env_month_avg.loc[date] = np.nan

    env_month_avg.sort_index(inplace=True)

    env_month_avg.fillna(method='ffill',inplace=True)
    env_month_avg.fillna(method='backfill',inplace=True)


    social_month_avg = social_df.resample('M').mean()
    social_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in social_month_avg.index:
            social_month_avg.loc[date] = np.nan

    social_month_avg.sort_index(inplace=True)
    social_month_avg.fillna(method='ffill',inplace=True)
    social_month_avg.fillna(method='backfill',inplace=True)



    human_month_avg = human_capital_df.resample('M').mean()
    human_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in human_month_avg.index:
            human_month_avg.loc[date] = np.nan

    human_month_avg.sort_index(inplace=True)
    human_month_avg.fillna(method='ffill',inplace=True)
    human_month_avg.fillna(method='backfill',inplace=True)



    gov_month_avg = gov_df.resample('M').mean()
    gov_month_avg.drop('N_label',axis=1,inplace=True)


    for date in dates:
        if date not in gov_month_avg.index:
            gov_month_avg.loc[date] = np.nan

    gov_month_avg.sort_index(inplace=True)
    gov_month_avg.fillna(method='ffill',inplace=True)
    gov_month_avg.fillna(method='backfill',inplace=True)



    bus_month_avg = business_df.resample('M').mean()
    bus_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in bus_month_avg.index:
            bus_month_avg.loc[date] = np.nan

    bus_month_avg.sort_index(inplace=True)
    bus_month_avg.fillna(method='ffill',inplace=True)
    bus_month_avg.fillna(method='backfill',inplace=True)


    if len(envoirmental_df)==0:
        month_avgs = social_month_avg*0.23+human_month_avg*0.3+gov_month_avg*0.21+bus_month_avg*0.26
    else:
        month_avgs = env_month_avg.values*0.05+social_month_avg*0.23+human_month_avg*0.3+gov_month_avg*0.21+bus_month_avg*0.21


    month_avgs = month_avgs.ewm(span=10).mean()

    envoirmental_df = df.loc[(df['N_label']==24)]


    social_df = df.loc[(df['N_label']==1)|
                       (df['N_label']==22)]

    human_capital_df = df.loc[(df['N_label']==8)]

    gov_df = df.loc[(df['N_label']==4)]

    env_month_avg = envoirmental_df.resample('M').mean()
    env_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in env_month_avg.index:
            env_month_avg.loc[date] = np.nan

    env_month_avg.sort_index(inplace=True)

    env_month_avg.fillna(method='ffill',inplace=True)
    env_month_avg.fillna(method='backfill',inplace=True)


    social_month_avg = social_df.resample('M').mean()
    social_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in social_month_avg.index:
            social_month_avg.loc[date] = np.nan

    social_month_avg.sort_index(inplace=True)
    social_month_avg.fillna(method='ffill',inplace=True)
    social_month_avg.fillna(method='backfill',inplace=True)



    human_month_avg = human_capital_df.resample('M').mean()
    human_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in human_month_avg.index:
            human_month_avg.loc[date] = np.nan

    human_month_avg.sort_index(inplace=True)
    human_month_avg.fillna(method='ffill',inplace=True)
    human_month_avg.fillna(method='backfill',inplace=True)



    gov_month_avg = gov_df.resample('M').mean()
    gov_month_avg.drop('N_label',axis=1,inplace=True)


    for date in dates:
        if date not in gov_month_avg.index:
            gov_month_avg.loc[date] = np.nan

    gov_month_avg.sort_index(inplace=True)
    gov_month_avg.fillna(method='ffill',inplace=True)
    gov_month_avg.fillna(method='backfill',inplace=True)



    bus_month_avg = business_df.resample('M').mean()
    bus_month_avg.drop('N_label',axis=1,inplace=True)

    for date in dates:
        if date not in bus_month_avg.index:
            bus_month_avg.loc[date] = np.nan

    bus_month_avg.sort_index(inplace=True)
    bus_month_avg.fillna(method='ffill',inplace=True)
    bus_month_avg.fillna(method='backfill',inplace=True)


    if len(envoirmental_df)==0:
        month_avgs_material = social_month_avg*0.23+human_month_avg*0.3+gov_month_avg*0.47
    else:
        month_avgs_material = env_month_avg.values*0.05+social_month_avg*0.23+human_month_avg*0.3+gov_month_avg*0.42

    month_avgs_material = month_avgs_material.ewm(span=10).mean()



    x = [str(x).split('T')[0] for x in month_avgs.index.values]
    x[-1] = '2021-01-22'
    y1 = month_avgs.values.reshape(-1)
    y2 = month_avgs_material.values.reshape(-1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y1, name="Intelligence Score", line = dict(color='#DDA0DD', width = 3), line_shape='spline', mode="lines"),
    )

    fig.add_trace(go.Scatter(x=x, y=y2, name="Material Intelligence Score", line = dict(color='#8B008B', width = 3), line_shape='spline',mode="lines"),
    )

    fig.add_trace(go.Scatter(x=x, y=[50]*len(x), name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
    )

    fig.update_layout(
        title_text="<b>Intelligence Score / drivers by SASB dimension</b>"
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
        )
    )

    fig.update_layout(
        font=dict(
            family="Corbel",
            size=14,
            color="#000000"
        )
    )

    fig.update_layout(
        title={

            'xanchor': 'left',
            'yanchor': 'top'}
    )

    fig.update_xaxes(
            tickangle = -30,
            title_font = {"size": 14},
            title_standoff = 100,
            tickmode= "auto"
    )


    fig.update_layout(plot_bgcolor="#FFFFFF")
    col3_1.plotly_chart(fig)
    
    # 2nd plot
    
    company = 'Airbnb'
    df = pd.read_csv(f'Companies/{company}/events.csv',parse_dates=['Date'])
    df2 = pd.read_csv(f'Companies/{company}/adj_close.csv',parse_dates=['dt'])
    df2.columns = ['Date','close']
    df2.head()

    for i,row in df2.iterrows():
        rows = df.loc[df['Date']==row['Date']]
        avg_day = np.mean(rows['AvgTone'].values)
        df2.loc[i,'AvgTone'] = avg_day

    df2.set_index(df2['Date'],inplace=True)
    df2.drop('Date',axis=1,inplace=True)
    df2.dropna(inplace=True)

    df2 = pd.read_csv(f'Companies/{company}/averages.csv',parse_dates=['Date'],index_col='Date')

    df2.drop('AvgTone',axis=1).to_csv('Adj_Close.csv',index=False)



    x = [str(x).split('T')[0] for x in df2.index.values]
    y1 = df2['AvgTone'].values
    y2 = df2['close'].values
    x_new = np.linspace(0,len(x)-1, 100)
    a_BSpline = sp.interpolate.make_interp_spline(np.arange(0,len(x)), y1)
    y_new1 = a_BSpline(x_new)

    a_BSpline = sp.interpolate.make_interp_spline(np.arange(0,len(x)), y2)
    y_new2 = a_BSpline(x_new)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=x, y=y_new1 , name="Intelligence Score", line = dict(color='#DDA0DD', width = 1), mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(go.Scatter(x=x, y=y_new2, name="Adj. Closing Price", line = dict(color='#8B008B', width = 1), mode="lines"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="<b>SASB Intelligence Score / IPO Price<b>"
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    ))


    fig.update_layout(
        font=dict(
            family="Corbel",
            size=14,
            color="#000000"
        )
    )
    fig.update_layout(plot_bgcolor="#FFFFFF")



    fig.update_layout(
        title={
            'y':0.9,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'})

    fig.update_xaxes(
            tickangle = -30,
            title_font = {"size": 14},
            title_standoff = 100,
            tickmode= "auto")

    col3_2.plotly_chart(fig)

    


# 4th line
    col4_1, col4_2, col4_3, col4_4, col4_5 = st.beta_columns((1, 1, 1, 1, 1))
    percent = 30
    c1 = (232, 232, 232)
    c2 = (139, 0, 139)
    style = ("text-align:center; padding: 0px; font-family: comic sans ms;, "
         "font-size: 100%")
    
    title = f"<h1 style='{style}'>Environment</h1>"
    col4_1.write(title, unsafe_allow_html=True)
    make_pie(50,[c1,c2], col4_1)
    
    title = f"<h1 style='{style}'>Social Capital</h1>"
    col4_2.write(title, unsafe_allow_html=True)
    make_pie(50,[c1,c2], col4_2)

    title = f"<h1 style='{style}'>Human Capital</h1>"
    col4_3.write(title, unsafe_allow_html=True)
    make_pie(47,[c1,c2], col4_3)
    
    title = f"<h1 style='{style}'>Business Model & Innovation</h1>"
    col4_4.write(title, unsafe_allow_html=True)
    make_pie(48,[c1,c2], col4_4)
    
    title = f"<h1 style='{style}'>Leadership & Governance</h1>"
    col4_5.write(title, unsafe_allow_html=True)
    make_pie(54,[c1,c2], col4_5)

# text line
    style = ("text-align:center; padding: 0px; font-family: comic sans ms;, "
         "font-size: 250%")
    
    title = f"<h1 style='{style}'>Expanding for in-depth Insights</h1><br>"
    st.write(title, unsafe_allow_html=True)
    st.write('---')

# th line # change plot names
#   col3, col4, col5 = st.beta_columns((3, 3, 2))
#     col3.dataframe(df1)
#     col4.dataframe(df1)
#     hfont = {'fontname':'Comic Sans MS'}
#     efont = {'fontname':'Corbel'}
#     value1 = '111.1 mln $'
#     value2 = '22.2 mln $'
#     plt1 = plt.figure()
#     currentAxis = plt1.gca()
#     currentAxis.set_axis_off()
#     currentAxis.add_patch(Rectangle((0.1, 0.01), 0.7, 0.4, fill=None, alpha=10))
#     plt1.text(0.25, 0.7, 'Brackmard minimum', fontsize=20, **hfont)
#     plt1.text(0.25, 0.55, value1 , fontsize=30, **efont)
#     currentAxis.add_patch(Rectangle((0.1, 0.49), 0.7, 0.4, fill=None, alpha=10))
#     plt1.text(0.25, 0.33, 'Brackmard minimum', fontsize=20, **hfont)
#     plt1.text(0.25, 0.175, value2 , fontsize=30, **efont)
#     col5.pyplot(plt1)
    
# 5th line
    col_T, col5_1, col5_2, col5_3, col5_4, col5_5 = st.beta_columns((0.25, 0.1, 0.1, 0.1, 0.1, 0.1))
# 6th line
    col6_T, col6_1, col6_2 = st.beta_columns((0.4, 3, 3))
    col7_T, col7_1 = st.beta_columns((1, 5))
# 7th line
    text1 = """
     Put some text here to see result
    """ # text under 1st plot
    text2 = """
     Put some text here to see result
    """ # text under 2nd plot
    imagePath = os.path.join(dataPath, "screen_airbnb.png")
    image1 = Image.open(imagePath) 
    make_button(col5_1, "SASB Env", col6_1, col6_2, col7_1, text1, text2, image1)
    imagePath = os.path.join(dataPath, "screen_airbnb.png")
    image2 = Image.open(imagePath)
    make_button(col5_2, "SASB SOC", col6_1, col6_2, col7_1, text1, text2, image2)
    imagePath = os.path.join(dataPath, "screen_airbnb.png")
    image3 = Image.open(imagePath)
    make_button(col5_3, "SASB HUM", col6_1, col6_2, col7_1, text1, text2, image3)
    imagePath = os.path.join(dataPath, "screen_airbnb.png")
    image4 = Image.open(imagePath)
    make_button(col5_4, "SASB BUS", col6_1, col6_2, col7_1, text1, text2, image4)
    imagePath = os.path.join(dataPath, "screen_airbnb.png")
    image5 = Image.open(imagePath)
    make_button(col5_5, "SASB LEAD", col6_1, col6_2, col7_1, text1, text2, image5)
    
# text line
    st.write('---')
    style = ("text-align:center; padding: 0px; font-family: comic sans ms;, "
         "font-size: 250%")
    
    title = f"<h1 style='{style}'>Analysis of Airbnb's Performance</h1>"
    st.write(title, unsafe_allow_html=True)
    st.write('---')

# 8th line
    col_T, col8_1, col8_2 = st.beta_columns((0.27, 0.31, 0.3))
# 9th line
    col_T, col9 = st.beta_columns((1.3, 3))
# 10th line
    col_T, col10 = st.beta_columns((1.3, 3))
# 11th line
    col_T, col11 = st.beta_columns((1.3, 3))
    
    dataPath1 = os.path.join(dataPath, "Financials--Balance Sheet.csv")
    dataPath2 = os.path.join(dataPath, "Operating Performance Metrics.csv")
    dataPath3 = os.path.join(dataPath, "Financials--Income Statement.csv")
    dataPath4 = os.path.join(dataPath, "Financials--Cash Flow.csv")
    df1 = pd.read_csv(dataPath1)
    df2 = pd.read_csv(dataPath2)
    df3 = pd.read_csv(dataPath3)
    df4 = pd.read_csv(dataPath4)
    if col8_1.button("Financial Performance"):
        style = ("text-align:left; padding: 0px; font-family: corbel;, "
             "font-size: 150%")
    
        title = f"<h1 style='{style}'>Balance Sheet</h1>"
        col9.write(title, unsafe_allow_html=True)
        col9.dataframe(df1.fillna(""))
        col9.write('---')
        title = f"<h1 style='{style}'>Income Statement</h1>"
        col10.write(title, unsafe_allow_html=True)
        df3 = df3.loc[:, df3.columns.notnull()]
        col10.dataframe(df3.fillna(""))
        col10.write('---')
        title = f"<h1 style='{style}'>Cash Flow</h1>"
        col11.write(title, unsafe_allow_html=True)
        col11.dataframe(df4.fillna(""))
        col10.write('---')
    if col8_2.button("Operating Performance"):
        style = ("text-align:left; padding: 0px; font-family: corbel;, "
             "font-size: 150%")
        title = f"<h1 style='{style}'>Operating Metrics</h1>"
        col9.write(title, unsafe_allow_html=True)
        col9.dataframe(df2.fillna(""))
        col9.write('---')
#   make_button_df(col7_1, "1st button", df1, df2, df col8_1, col8_2)
#   make_button_df(col7_2, "2nd button", df3, df4, col8_1, col8_2)
    
# th line
#   col1, col2= st.beta_columns((1, 4))
#   metric_options = ['SASB dimensions', 'SASB Social Capital',
#                         'SASB Human Capital', 'SASB Leadership & Governance',
#                         'SASB Business Model & Innovation']
#   line_metric = col1.radio("Choose Option", options=metric_options)
    
    # 1st plot
#   if line_metric == 'SASB dimensions':
#         categories = [ 'Environment',
#         'Social <br>Capital',
#         'Human <br>Capital',
#         'Leadership & <br>Governance',
#         'Business Model & <br>Innovation']
#         traceY = [50.38,49.84,47.82,48.37,53.73]
#         traceName = "Intelligence Score"
#         barName = "Volume / Contribution of<br>each SASB dimension"
#         barY = [0.03,0.13,0.22,0.52,0.09]
#         legendX = 0.5
#         legendY = -0.44
#         titleX = 0.7
#         titleY = 0.9
#         textPos = 'top center'
#         yRange = [0,0.6]
#         col = col2
#         make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
       
#     # 2nd plot
#     elif line_metric == 'SASB Social Capital':
#         categories = [ 'Access &<br>Affordability',
#             'Customer <br>Privacy',
#             'Customer <br>Welfare',
#             'Data <br>Security',
#             'Human <br>Rights &<br>Community <br>Relations',
#             'Selling <br>Practices &<br>Product <br>Labeling']
#         traceY = [52.43, 47.86, 53.82, 48.3, 48.03, 48.6]
#         traceName = "Intelligence Score"
#         barName = "Volume"
#         barY = [0.21,0.28,0.07,0.34,0.10,0.0012]
#         legendX = 0.5
#         legendY = -0.5
#         titleX = 0.56
#         titleY = 0.9
#         textPos = 'top center'
#         yRange = [0,0.6]
#         col = col2
#         make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)

#     # 3rd plot
#     elif line_metric == 'SASB Human Capital':
#         categories = [ 'Employee Engagement<br>Inclusion & Diversity',
#             'Employee Health & Safety',
#             'Labor Practices']
#         traceY = [49.64, 46.28 , 47.54]
#         traceName = "Intelligence Score"
#         barName = "Volume"
#         barY = [0.30,0.37,0.33]
#         legendX = 0.5
#         legendY = -0.35
#         titleX = 0.58
#         titleY = 0.9
#         textPos = 'top center'
#         yRange = [0,0.6]
#         col = col2
#         make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
      
#     # 4th plot
#     elif line_metric == 'SASB Leadership & Governance':
#         categories = [ 'Business<br>Ethics',
#                       'Competitive<br>Behavior',
#                       'Critical<br>Incident Risk<br>Management',
#                       'Director<br>Removal',
#                       'Management<br>Of Legal &<br>Regulatory<br>Framework',
#                       'Supply<br>Chain<br>Manage<br>ment',
#                       'Systemic<br>Risk<br>Manage<br>ment'
#                      ]
#         traceY = [43.28, 49.99 , 45.08, 48.68, 48.03, 51.14, 49.13]
#         traceName = "Intelligence Score"
#         barName = "Volume"
#         barY = [0.09, 0.10, 0.11, 0.0009, 0.53, 0.01, 0.16]
#         legendX = 0.42
#         legendY = -0.6
#         titleX = 0.74
#         titleY = 0.9
#         textPos = 'top center'
#         yRange = [0,0.6]
#         col = col2
#         make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)

#     # 5th plot
#     elif line_metric == 'SASB Business Model & Innovation':
#         categories = [ 'Business Model<br>Resilience',
#                       'Product Design &<br>Lifecycle Management',
#                       'Product Quality &<br>Safety',
#                      ]
#         traceY = [54.03, 55.55 , 51.61]
#         traceName = "Intelligence Score"
#         barName = "Volume"
#         barY = [0.64,0.33,0.03] 
#         legendX = 0.5
#         legendY = -0.31
#         titleX = 0.78
#         titleY = 0.9
#         textPos = 'top center'
#         yRange = [0,0.6]
#         col = col2
#         make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
        
