import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    

def buttons(col, text):
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
        col = col9
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
        col = col10
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
        
        
def make_pie(size, colors, column):
    labels = ['','']
    sizes =[size, 100-size]
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
    col11, col12, col13, col14, col15 = st.beta_columns((1, 1, 1, 1, 1))
    percent = 30
    c1 = (232, 232, 232)
    c2 = (139, 0, 139)

    make_pie(percent,[c1,c2], col11)
    make_pie(percent,[c1,c2], col12)
    make_pie(percent,[c1,c2], col13)
    make_pie(percent,[c1,c2], col14)
    make_pie(percent,[c1,c2], col15)
    # 2nd line
    col3, col4, col5 = st.beta_columns((3, 3, 2))
    col3.dataframe(df1)
    col4.dataframe(df1)
    hfont = {'fontname':'Comic Sans MS'}
    efont = {'fontname':'Corbel'}
    value1 = '111.1 mln $'
    value2 = '22.2 mln $'
    plt1 = plt.figure()
    currentAxis = plt1.gca()
    currentAxis.set_axis_off()
    currentAxis.add_patch(Rectangle((0.1, 0.01), 0.7, 0.4, fill=None, alpha=10))
    plt1.text(0.25, 0.7, 'Brackmard minimum', fontsize=20, **hfont)
    plt1.text(0.25, 0.55, value1 , fontsize=30, **efont)
    currentAxis.add_patch(Rectangle((0.1, 0.49), 0.7, 0.4, fill=None, alpha=10))
    plt1.text(0.25, 0.33, 'Brackmard minimum', fontsize=20, **hfont)
    plt1.text(0.25, 0.175, value2 , fontsize=30, **efont)
    col5.pyplot(plt1)
    
    # 3rd line
    colT, col6, col7, col8 = st.beta_columns((0.35, 0.1, 0.1, 0.1))
    col9, col10 = st.beta_columns((3, 3))
    
    buttons(col6, "1st button")
    buttons(col7, "2nd button")
    buttons(col8, "3rd button")

    # 4th line
    col1, col2= st.beta_columns((1, 4))
    metric_options = ['SASB dimensions', 'SASB Social Capital',
                          'SASB Human Capital', 'SASB Leadership & Governance',
                          'SASB Business Model & Innovation']
    line_metric = col1.radio("Choose Option", options=metric_options)
    
    # 1st plot
    if line_metric == 'SASB dimensions':
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
        col = col2
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
       
    # 2nd plot
    elif line_metric == 'SASB Social Capital':
        categories = [ 'Access &<br>Affordability',
            'Customer <br>Privacy',
            'Customer <br>Welfare',
            'Data <br>Security',
            'Human <br>Rights &<br>Community <br>Relations',
            'Selling <br>Practices &<br>Product <br>Labeling']
        traceY = [52.43, 47.86, 53.82, 48.3, 48.03, 48.6]
        traceName = "Intelligence Score"
        barName = "Volume"
        barY = [0.21,0.28,0.07,0.34,0.10,0.0012]
        legendX = 0.5
        legendY = -0.5
        titleX = 0.56
        titleY = 0.9
        textPos = 'top center'
        yRange = [0,0.6]
        col = col2
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)

    # 3rd plot
    elif line_metric == 'SASB Human Capital':
        categories = [ 'Employee Engagement<br>Inclusion & Diversity',
            'Employee Health & Safety',
            'Labor Practices']
        traceY = [49.64, 46.28 , 47.54]
        traceName = "Intelligence Score"
        barName = "Volume"
        barY = [0.30,0.37,0.33]
        legendX = 0.5
        legendY = -0.35
        titleX = 0.58
        titleY = 0.9
        textPos = 'top center'
        yRange = [0,0.6]
        col = col2
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
      
    # 4th plot
    elif line_metric == 'SASB Leadership & Governance':
        categories = [ 'Business<br>Ethics',
                      'Competitive<br>Behavior',
                      'Critical<br>Incident Risk<br>Management',
                      'Director<br>Removal',
                      'Management<br>Of Legal &<br>Regulatory<br>Framework',
                      'Supply<br>Chain<br>Manage<br>ment',
                      'Systemic<br>Risk<br>Manage<br>ment'
                     ]
        traceY = [43.28, 49.99 , 45.08, 48.68, 48.03, 51.14, 49.13]
        traceName = "Intelligence Score"
        barName = "Volume"
        barY = [0.09, 0.10, 0.11, 0.0009, 0.53, 0.01, 0.16]
        legendX = 0.42
        legendY = -0.6
        titleX = 0.74
        titleY = 0.9
        textPos = 'top center'
        yRange = [0,0.6]
        col = col2
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)

    # 5th plot
    elif line_metric == 'SASB Business Model & Innovation':
        categories = [ 'Business Model<br>Resilience',
                      'Product Design &<br>Lifecycle Management',
                      'Product Quality &<br>Safety',
                     ]
        traceY = [54.03, 55.55 , 51.61]
        traceName = "Intelligence Score"
        barName = "Volume"
        barY = [0.64,0.33,0.03] 
        legendX = 0.5
        legendY = -0.31
        titleX = 0.78
        titleY = 0.9
        textPos = 'top center'
        yRange = [0,0.6]
        col = col2
        make_bar_plot(categories, traceY, traceName, barY, barName, legendX, legendY, titleX, titleY, textPos, yRange, col)
        
