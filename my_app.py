import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ESG AI", layout='centered', initial_sidebar_state="collapsed")
style = ("text-align:center; padding: 0px; font-family: arial black;, "
         "font-size: 400%")
title = f"<h1 style='{style}'>DataESG<sup>AI</sup></h1><br><br>"
st.write(title, unsafe_allow_html=True)


companies = ['Select a Company', 'Airbnb', 'Airbnb'] 
company = st.selectbox('Select a Company to Analyze', companies)
if company and company != "Select a Company":
    # 1st plot
    col1, col2 = st.beta_columns((1, 3))

    metric_options = ['SASB dimensions / Volume & Intelligence Score', 'Social Capital SASB Intelligence Score',
                          'Human Capital SASB Intelligence Score', 'Leadership & Governance SASB Intelligence Score',
                          'Business Model & Innovation SASB Intelligence Score']
    line_metric = col1.radio("Choose Option", options=metric_options)

    if line_metric == 'SASB dimensions / Volume & Intelligence Score':
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        categories = [ 'Environment',
            'Social <br>Capital',
            'Human <br>Capital',
            'Leadership & <br>Governance',
            'Business Model & <br>Innovation']

        # Add traces
        fig.add_trace(go.Scatter(x=categories, y=[50.38,49.84,47.82,48.37,53.73], name="Intelligence Score", line_shape='spline',mode="lines+markers+text",              
            text=['<b>50.38</b>','<b>49.84</b>','<b>47.82</b>','<b>48.37</b>','<b>53.73</b>'], # Bold version <b> </b>
            textposition="top center",
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
            go.Bar(x=categories, y=[0.03,0.13,0.22,0.52,0.09], name="Volume / Contribution of<br>each SASB dimension",text=['3%','13%','22%','52%','9%'],
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
            y=-0.44,
            xanchor="center",
            x=0.5
        ))

        fig.update_layout(
            yaxis = dict(
                tickformat = ',.0%',
                range = [0,0.6]        
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
                'y':0.9,
                'x':0.7,
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
        col2.plotly_chart(fig)
    # 2nd plot
    elif line_metric == 'Social Capital SASB Intelligence Score':

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        categories = [ 'Access &<br>Affordability',
            'Customer <br>Privacy',
            'Customer <br>Welfare',
            'Data <br>Security',
            'Human <br>Rights &<br>Community <br>Relations',
            'Selling <br>Practices &<br>Product <br>Labeling']

        # Add traces
        fig.add_trace(go.Scatter(x=categories, y=[52.43, 47.86, 53.82, 48.3, 48.03, 48.6] , name="Intelligence Score", line_shape='spline',mode="lines+markers+text",              
            text=[52.43, 47.86, 53.82, 48.3, 48.03, 48.6],
            textposition="top center",
            marker_color='#DDA0DD',
            textfont=dict(
                family="Corbel",
                size=14,
                color="#cb91ce ")),
            secondary_y=True,
        )
        fig.add_trace(go.Scatter(x=categories, y=[50]*len(categories) , name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
            secondary_y=True,
        )


        fig.add_trace(
            go.Bar(x=categories, y=[0.21,0.28,0.07,0.34,0.10,0.0012], name="Volume",text=['21%','28%','7%','34%','10%', '0.12%'],
                textposition='outside',
                marker_color='#8B008B'),                
            secondary_y=False
        )

        # Add figure title
        fig.update_layout(
            title_text="Social Capital SASB Intelligence Score"
        )

        fig.update_layout(
            yaxis = dict(
                tickformat = ',.0%',
                range = [0,0.4]        
            )
        )
        fig.update_layout(
            font=dict(
                family="Corbel",
                size=14,
                color="#000000"
            )
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        ))

        fig.update_layout(plot_bgcolor="#FFFFFF")

        fig.update_layout(
            title={
                'y':0.9,
                'x':0.56,
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

        col2.plotly_chart(fig)
    # 3rd plot
    elif line_metric == 'Human Capital SASB Intelligence Score':
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        categories = [ 'Employee Engagement<br>Inclusion & Diversity',
            'Employee Health & Safety',
            'Labor Practices']

        # Add traces
        fig.add_trace(go.Scatter(x=categories, y=[49.64, 46.28 , 47.54] , name="Intelligence Score", line_shape='spline',mode="lines+markers+text",              
            text=[49.64, 46.28 , 47.54],
            textposition="top center",
            marker_color='#DDA0DD',    
                textfont=dict(
                    family="Corbel",
                    size=14,
                    color="#cb91ce")), 
                      secondary_y=True,
        )


        fig.add_trace(go.Scatter(x=categories, y=[50]*len(categories) , name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
            secondary_y=True,
        )


        fig.add_trace(
            go.Bar(x=categories, y=[0.30,0.37,0.33], name="Volume",text=['30%','37%','33%'],
                textposition='inside',
                marker_color='#8B008B'),
            secondary_y=False, 
        )

        # Add figure title
        fig.update_layout(
            title_text="Human Capital SASB Intelligence Score"
        )

        fig.update_layout(
            yaxis = dict(
                tickformat = ',.0%',
                range = [0,0.5]        
            )
        )
        fig.update_layout(
            font=dict(
                family="Corbel",
                size=14,
                color="#000000"
            )
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5
        ))

        fig.update_layout(plot_bgcolor="#FFFFFF")

        fig.update_layout(
            title={
                'y':0.9,
                'x':0.58,
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
        col2.plotly_chart(fig)
    # 4th plot
    elif line_metric == 'Leadership & Governance SASB Intelligence Score':

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        categories = [ 'Business<br>Ethics',
                      'Competitive<br>Behavior',
                      'Critical<br>Incident Risk<br>Management',
                      'Director<br>Removal',
                      'Management<br>Of Legal &<br>Regulatory<br>Framework',
                      'Supply<br>Chain<br>Manage<br>ment',
                      'Systemic<br>Risk<br>Manage<br>ment'
                     ]


        # Add traces
        fig.add_trace(go.Scatter(x=categories, y=[43.28, 49.99 , 45.08, 48.68, 48.03, 51.14, 49.13] , name="Intelligence Score", line_shape='spline',mode="lines+markers+text",              
            text=[43.28, 49.99 , 45.08, 48.68, 48.03, 51.14, 49.13],
            textposition="top center",
            marker_color='#DDA0DD',  
                textfont=dict(
                    family="Corbel",
                    size=14,
                    color="#cb91ce ")),
                secondary_y=True,
        )

        fig.add_trace(go.Scatter(x=categories, y=[50]*len(categories) , name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
            secondary_y=True,
        )


        fig.add_trace(
            go.Bar(x=categories, y=[0.09, 0.10, 0.11, 0.0009, 0.53, 0.01, 0.16], name="Volume",text=['9%','10%','11%', '0.09%', '53%', '1%', '16%'],
                textposition='outside',
                marker_color='#8B008B'),
            secondary_y=False, 
        )

        # Add figure title
        fig.update_layout(
            title_text="Leadership & Governance SASB Intelligence Score"
        )

        fig.update_layout(
            yaxis = dict(
                tickformat = ',.0%',
                range = [0,0.6]        
            )
        )
        fig.update_layout(
            font=dict(
                family="Corbel",
                size=14,
                color="#000000"
            )
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
            xanchor="center",
            x=0.42
        ))

        fig.update_layout(plot_bgcolor="#FFFFFF")

        fig.update_layout(
            title={
                'y':0.9,
                'x':0.74,
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
        
        col2.plotly_chart(fig)
    # 5th plot
    elif line_metric == 'Business Model & Innovation SASB Intelligence Score':
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        categories = [ 'Business Model<br>Resilience',
                      'Product Design &<br>Lifecycle Management',
                      'Product Quality &<br>Safety',
                     ]

        # Add traces
        fig.add_trace(go.Scatter(x=categories, y=[54.03, 55.55 , 51.61] , name="Intelligence Score", line_shape='spline',mode="lines+markers+text",              
            text=[54.03, 55.55 , 51.61],
            textposition="top center",
            marker_color='#DDA0DD',
                textfont=dict(
                    family="Corbel",
                    size=14,
                    color="#cb91ce ")),
            secondary_y=True,
        )
        fig.add_trace(go.Scatter(x=categories, y=[50]*len(categories) , name="Neutral", line = dict(color='#000000', width = 0.6, dash='dot'),mode="lines"),
            secondary_y=True,
        )


        fig.add_trace(
            go.Bar(x=categories, y=[0.64,0.33,0.03], name="Volume",text=['64%','33%','3%'],
                textposition='outside',
                marker_color='#8B008B'),
            secondary_y=False, 
        )

        # Add figure title
        fig.update_layout(
            title_text="Business Model & Innovation SASB Intelligence Score"
        )

        fig.update_layout(
            yaxis = dict(
                tickformat = ',.0%',
                range = [0,0.8]        
            )
        )
        fig.update_layout(
            font=dict(
                family="Corbel",
                size=14,
                color="#000000"
            )
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.31,
            xanchor="center",
            x=0.5
        ))

        fig.update_layout(plot_bgcolor="#FFFFFF")

        fig.update_layout(
            title={
                'y':0.9,
                'x':0.78,
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
        
        col2.plotly_chart(fig)
