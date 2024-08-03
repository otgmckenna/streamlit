import streamlit as st

st.set_page_config(page_title='2024 DCI Data', page_icon='🎵')

def intro():
    import streamlit as st
    
    st.markdown(
        """
        ## 2024 Drum Corps International Data
        
        This app contains data from the 2024 Drum Corps International season.
        
        Select a page from the dropdown menu on the sidebar to the left to explore the data.
        
        *Data courtesy of [dci.org](https://www.dci.org)*
        
        *Data also available on [Kaggle](https://www.kaggle.com/datasets/olivermckenna/drum-corps-international-2024-scores-and-captions)*
        
        ### About the Data
        The data in this app is from the 2024 Drum Corps International season. Drum Corps International (DCI) is a non-profit organization that governs the junior drum corps activity. It is made up of member corps who have earned their membership through competition. The corps perform in competitions across the United States and are judged in various categories, such as brass, percussion, and color guard.
        
        ### Data Exploration Pages
        - **Data Exploration**: An overview of the scores for the season, organized in a filterable table.
        - **Corps Comparison**: Compare the scores of different corps.
        - **Score Distribution**: View the distribution of scores for a specific score or caption.
        - **Overall Rankings**: View the overall rankings of the corps, based on each corps' most recent score.
        - **More to come!**
        
        """
    )

def data_exploration():
    import pandas as pd
    from pandas.api.types import (
        is_categorical_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_object_dtype,
    )
    import streamlit as st
    import plotly.express as px
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from urllib.error import URLError
    
    df = pd.read_csv('scores.csv')
    
    # Dropping the CTot column as it is not needed
    df.drop(columns='CTot', inplace=True)
    
    # Converting the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    # Adding a UI to the dataframe that will allow the user to filter the data
    # Original code from: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI to filter the dataframe.

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filter dataframe
        """
        modify = st.checkbox('Add filters')
        if not modify:
            return df

        df = df.copy()

        # Donvert datetimes to standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
            
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Choose columns to filter", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"{column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

        return df
    st.write('## Filter the data')
    st.write('Check the box below to add filters to the data')
    st.dataframe(filter_dataframe(df))

def corps_comparison():
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from urllib.error import URLError
    
    df = pd.read_csv('scores.csv')
    
    # Dropping the CTot column as it is not needed
    df.drop(columns='CTot', inplace=True)
    
    # Converting the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filtering the data to only include certain scores: GETot, VP_Tot, VA_Tot, CG_Tot, Vis_Tot, MB_Tot, MA_Tot, MP_Tot, Mus_Tot
    df = df[['Location', 'Date', 'Corps', 'Total', 'GE_Tot', 'VP_Tot', 'VA_Tot', 'CG_Tot', 'Vis_Tot', 'MB_Tot', 'MA_Tot', 'MP_Tot', 'Mus_Tot']]
    
    # Renaming the columns to make them more readable
    df.columns = ['Location', 'Date', 'Corps', 'Total', 'General Effect', 'Visual Proficiency', 'Visual Analysis', 'Color Guard', 'Visual Total', 'Brass', 'Music Analysis', 'Percussion', 'Music Total']
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[3]}')
    
    st.write('### Compare the scores of different corps')
    st.write('Select the corps and the score/caption to compare.')
    
    # Choosing the corps to compare
    corps = st.multiselect('Select corps to compare', df['Corps'].unique())
    
    # Letting user choose only one numerical column to compare
    columns = st.selectbox('Select a score or caption to compare', df.select_dtypes('number').columns)
    
    # Creating the comparison plot using the chosen corps and the chosen numrical columns
    # Showing evolution of scores over the season
    fig = px.line(df[df['Corps'].isin(corps)], x='Date', y=columns, color='Corps', title=f'Comparison of corps - {columns}', width=1600, height=800)
    # Adding dot to the line plot that indicate a corp's performance
    fig.update_traces(mode='markers+lines')
    # Updating hover text to be cleaner
    fig.update_traces(hovertemplate='<br>Score: %{y}<br>Date: %{x}')
    # Changing the y-axis label
    fig.update_yaxes(title_text='Score')
    st.plotly_chart(fig)

def score_dist():
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import plotly.figure_factory as ff
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from urllib.error import URLError
    
    df = pd.read_csv('scores.csv')
    
    # Dropping the CTot column as it is not needed
    df.drop(columns='CTot', inplace=True)
    
    # Converting the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filtering the data to only include certain scores: GETot, VP_Tot, VA_Tot, CG_Tot, Vis_Tot, MB_Tot, MA_Tot, MP_Tot, Mus_Tot
    df = df[['Location', 'Date', 'Corps', 'Class', 'Total', 'GE_Tot', 'VP_Tot', 'VA_Tot', 'CG_Tot', 'Vis_Tot', 'MB_Tot', 'MA_Tot', 'MP_Tot', 'Mus_Tot']]
    
    # Renaming the columns to make them more readable
    df.columns = ['Location', 'Date', 'Corps', 'Class', 'Total', 'General Effect', 'Visual Proficiency', 'Visual Analysis', 'Color Guard', 'Visual Total', 'Brass', 'Music Analysis', 'Percussion', 'Music Total']
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[4]}')
    
    st.write('### Distribution of scores')
    st.write('Select the score or caption to view the distribution.')
    
    # Letting user choose only one numerical column to compare
    columns = st.selectbox('Select a score or caption to view the distribution', df.select_dtypes('number').columns)
    
    # Adding a filter to show only a certain class, if desired
    class_filter = st.selectbox('Filter by class', ['All', 'World', 'Open', 'All-Age'])
    
    if class_filter != 'All':
        df = df[df['Class'] == class_filter]
    elif class_filter == 'All':
        df = df
    
    # Creating a histogram of the chosen score/caption using Plotly Figure Factory, showing each class in a different color
    fig = ff.create_distplot([df[columns][df['Class'] == c].dropna() for c in df['Class'].unique()], df['Class'].unique(), colors=['#c8363d', '#1761af', '#6aa338'], bin_size=1)
    # Adjusting the hover text
    fig.update_traces(hovertemplate='Score: %{x}<br>Density: %{y}')
    # Adding a vertical line to indicate the mean score and showing the mean value
    fig.add_vline(x=df[columns].mean(), line_dash='dash', line_color='gray', annotation_text='Mean', annotation_position='top left')
    # Changing the x-axis label
    fig.update_xaxes(title_text='Score')
    # Changing the y-axis label
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig)


def overall_scores():
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from urllib.error import URLError
    
    df = pd.read_csv('scores.csv')
    
    # Dropping the CTot column as it is not needed
    df.drop(columns='CTot', inplace=True)
    
    # Converting the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filtering the data to only include certain scores: GETot, VP_Tot, VA_Tot, CG_Tot, Vis_Tot, MB_Tot, MA_Tot, MP_Tot, Mus_Tot
    df = df[['Location', 'Date', 'Corps', 'Class', 'Total', 'GE_Tot', 'VP_Tot', 'VA_Tot', 'CG_Tot', 'Vis_Tot', 'MB_Tot', 'MA_Tot', 'MP_Tot', 'Mus_Tot']]
    
    # Renaming the columns to make them more readable
    df.columns = ['Location', 'Date', 'Corps', 'Class', 'Total', 'General Effect', 'Visual Proficiency', 'Visual Analysis', 'Color Guard', 'Visual Total', 'Brass', 'Music Analysis', 'Percussion', 'Music Total']
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    
    st.write('### Overall Rankings')
    st.write('View the overall rankings of the corps, based on each corps\' most recent score.')
    
    # Getting the most recent scores for each corps
    most_recent_scores = df.loc[df.groupby('Corps')['Date'].idxmax()]
    
    # Sorting the data by the Total column
    most_recent_scores = most_recent_scores.sort_values('Total', ascending=True)
    
    # Ading a filter to show only a certain class, if desired
    class_filter = st.selectbox('Filter by class', ['All', 'World', 'Open', 'All-Age'])
    
    if class_filter != 'All':
        most_recent_scores = most_recent_scores[most_recent_scores['Class'] == class_filter]
    
    # Creating a horizontal bar plot of the most recent scores
    fig = px.bar(most_recent_scores, x='Total', y='Corps', orientation='h', title='Overall Rankings', width=1600, height=1000)
    # Adding score to the bars
    fig.update_traces(text=most_recent_scores['Total'], textposition='inside')
    # Adding a vertical line to indicate the mean score
    fig.add_vline(x=most_recent_scores['Total'].mean(), line_dash='dash', line_color='white', annotation_text='Average', annotation_position='bottom left')
    # Coloring the bars based on class
    fig.update_traces(marker_color=most_recent_scores['Class'].map({'World': '#c8363d', 'Open': '#1761af', 'All-Age': '#6aa338'}))
    # Cleaning up the hover text
    fig.update_traces(hovertemplate='Corps: %{y}<br>Score: %{x}<br>')
    # Changing the x-axis label
    fig.update_xaxes(title_text='Score')
    # Changing the y-axis label
    fig.update_yaxes(title_text='Corps')
    st.plotly_chart(fig)

page_names_to_funcs = {
    'Home': intro,
    'Overall Rankings': overall_scores,
    'Data Exploration': data_exploration,
    'Corps Comparison': corps_comparison,
    'Score Distribution': score_dist
}

page_name = st.sidebar.selectbox('Select a page', list(page_names_to_funcs.keys()))
page_names_to_funcs[page_name]()

st.sidebar.success('Select a page from the dropdown menu to explore the data.')