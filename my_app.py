import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

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
    

def make_button(col, text, col1, col2):
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
    value1 = '111.1 mln $'
    value2 = '22.2 mln $'
    plt1 = plt.figure()
    currentAxis = plt1.gca()
    currentAxis.set_axis_off()
    currentAxis.add_patch(Rectangle((0.2, 0.3), 0.7, 0.4, fill=None, alpha=10))
    plt1.text(0.30, 0.55, text, fontsize=20, **hfont)
    plt1.text(0.30, 0.39, value , fontsize=30, **efont)
    col.pyplot(plt1)


currentDirectory = os.path.abspath(os.getcwd())
dataPath = os.path.join(currentDirectory, "MainDataNotFull.csv")
df1 = pd.read_csv(dataPath)

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
company = st.selectbox('Select a Company to Analyze', companies)
if company and company != "Select a Company":
    
    
# 1st line
    style = ("text-align:left; padding: 15px; font-family: arial black;, "
         "font-size: 300%")
    
    title = f"<h1 style='{style}'>Airbnb</h1>"
    col1_T, col1_1, col1_2 = st.beta_columns((1, 3, 30))
    image = Image.open(r'C:\Users\khach\logo_airbnb.jpg')
    col1_1.image(image, width = 100)
    col1_2.write(title, unsafe_allow_html=True)
# 2nd line
    col2_1, col2_2, col2_3, col2_4 = st.beta_columns((1, 1, 1, 1))
    text = 'Some text'
    value = '111.1 mln $'
    percent = 30
    c1 = (232, 232, 232)
    c2 = (139, 0, 139)
    make_pie(percent,[c1,c2], col2_1)
    make_rect(text, value, col2_2)
    make_rect(text, value, col2_3)
    make_rect(text, value, col2_4)
# 3rd line
    col3_1, col3_2 =st.beta_columns((1, 1))
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
    col = col3_1
    make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
    col = col3_2
    make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)

# 4th line
    col4_1, col4_2, col4_3, col4_4, col4_5 = st.beta_columns((1, 1, 1, 1, 1))
    percent = 30
    c1 = (232, 232, 232)
    c2 = (139, 0, 139)
    style = ("text-align:center; padding: 0px; font-family: comic sans ms;, "
         "font-size: 100%")
    
    title = f"<h1 style='{style}'>Some Text</h1>"
    col4_1.write(title, unsafe_allow_html=True)
    make_pie(52,[c1,c2], col4_1)
    
    title = f"<h1 style='{style}'>Some Text</h1>"
    col4_2.write(title, unsafe_allow_html=True)
    make_pie(60,[c1,c2], col4_2)

    title = f"<h1 style='{style}'>Some Text</h1>"
    col4_3.write(title, unsafe_allow_html=True)
    make_pie(14,[c1,c2], col4_3)
    
    title = f"<h1 style='{style}'>Some Text</h1>"
    col4_4.write(title, unsafe_allow_html=True)
    make_pie(73,[c1,c2], col4_4)
    
    title = f"<h1 style='{style}'>Some Text</h1>"
    col4_5.write(title, unsafe_allow_html=True)
    make_pie(97,[c1,c2], col4_5)

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
    col6_1, col6_2 = st.beta_columns((3, 3))
    
    make_button(col5_1, "SASB Env", col6_1, col6_2)
    make_button(col5_2, "SASB SOC", col6_1, col6_2)
    make_button(col5_3, "SASB HUM", col6_1, col6_2)
    make_button(col5_4, "SASB BUS", col6_1, col6_2)
    make_button(col5_5, "SASB LEAD", col6_1, col6_2)
    
# text line
    st.write('---')
    style = ("text-align:center; padding: 0px; font-family: comic sans ms;, "
         "font-size: 250%")
    
    title = f"<h1 style='{style}'>Analysis of Airbnb's Performance</h1>"
    st.write(title, unsafe_allow_html=True)
    st.write('---')

# 7th line
    col_T, col7_1, col7_2 = st.beta_columns((0.41, 0.1, 0.1))
# 8th line
    col8_1, col8_2 = st.beta_columns((3, 3))
    
    dataPath1 = os.path.join(currentDirectory, "Financials--Balance Sheet.csv")
    dataPath2 = os.path.join(currentDirectory, "Operating Performance Metrics.csv")
    dataPath3 = os.path.join(currentDirectory, "Financials--Income Statement.csv")
    dataPath4 = os.path.join(currentDirectory, "Financials--Cash Flow.csv")
    df1 = pd.read_csv(dataPath1)
    df2 = pd.read_csv(dataPath2)
    df3 = pd.read_csv(dataPath3)
    df4 = pd.read_csv(dataPath4)
    make_button_df(col7_1, "1st button", df1, df2, col8_1, col8_2)
    make_button_df(col7_2, "2nd button", df3, df4, col8_1, col8_2)

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
        
