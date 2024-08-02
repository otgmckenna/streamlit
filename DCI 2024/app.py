import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data
df = pd.read_csv('scores.csv')

# Converting the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Creating the Streamlit app
# Setting wide mode
st.set_page_config(layout='wide')

# Adding a title and description
st.title('Drum Corps International 2024 Scores')
st.write('This app contains data from the 2024 Drum Corps International season.')
st.write('Data courtesy of [DCI.org](https://www.dci.org)')
st.write('---')

# Creating a sidebar for the user to select a corps, location or date
st.sidebar.title('Filter Data')
corps = st.sidebar.multiselect('Select Corps', df['Corps'].unique())
location = st.sidebar.multiselect('Select Location', df['Location'].unique())
st.sidebar.write('---')

# Adding a dropdown to selection visualization type
visualization = st.sidebar.selectbox('Select Visualization', ['None', 'Scores by Corps at Selected Location', 'Scores by Date for Selected Corps'])

# If the user selects a corps AND location, display an error message
if corps and location:
    st.error('Please select either a corps or a location, not both. The stats are boring otherwise. Trust me.')
# If the user selects a corps, display the stats for that corps
elif corps:
    corps_df = df[df['Corps'].isin(corps)]
    st.write('### Corps Stats')
    st.write(corps_df)
# If the user selects a location, display the stats for that location
elif location:
    location_df = df[df['Location'].isin(location)]
    st.write('### Location Stats')
    st.write(location_df)
else:
    st.write('### All Scores')
    st.write(df)

# Only display a visualization if the user selects one
# Filter the data based on the user's selection
# If the user selects a corps, only allow: Line plot, scatter plot
# If the user selects a location, only allow: Bar chart
# Filter results based on user selection
if visualization == 'Scores by Corps at Selected Location':
    if len(location) > 1:
        st.error('Please select only one location for the bar chart.')
    # If more than one location is selected, display an error message
    elif location:
        fig = px.bar(location_df, x='Corps', y='Total', color='Corps', title='Scores by Corps at Selected Location')
        fig.update_traces(texttemplate='%{y}', textposition='inside')
        fig.update_layout(uniformtext_minsize=15, uniformtext_mode='hide')
        st.plotly_chart(fig)
    else:
        st.error('Please select a location for the bar chart.')
elif visualization == 'Scores by Date for Selected Corps':
    if corps:
        fig = px.line(corps_df, x='Date', y='Total', color='Corps', title='Scores by Date for Selected Corps')
        st.plotly_chart(fig)
    else:
        st.error('Please select one or more corps for the line chart.')