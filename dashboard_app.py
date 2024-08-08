from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from urllib.parse import unquote



external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']




def init_dash(server):
    dash_app = dash.Dash(server=server, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)
    
    def load_data(department, filename):
        decoded_filename = unquote(filename)
        file_path = os.path.join('C:/Users/Administrator/Desktop/MCB_PROJECT/Dashboard',department, decoded_filename)
        return pd.read_excel(file_path)

    def transform_linkedin_data(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def create_management_plots(df, columns):
        plots = []
        for col in columns:
            # Combine low percentage shareholders into "Participants"
            df['Shareholder Type'] = df['Shareholders'].apply(lambda x: x if x in ['Gemeente Maastricht (excl. index)', 'Provincie Limburg (excl. index)', 'Stuurgroep Congressen(excl. index)'] else 'Participants')
            df_grouped = df.groupby('Shareholder Type')[col].sum().reset_index()
            fig = px.pie(df_grouped, names='Shareholder Type', values=col, title=f'Distribution of {col}')
            plots.append(dcc.Graph(figure=fig))
            
            # Accumulative amount for each category
            fig_accumulative = px.bar(df_grouped, x='Shareholder Type', y=col, title=f'Accumulative Amount for {col}')
            plots.append(dcc.Graph(figure=fig_accumulative))
            
            # Categorize shareholders based on contribution size
            df_sorted = df.sort_values(by=col, ascending=False)
            df_sorted['Category'] = pd.cut(df_sorted[col], bins=[-1, 1000, 1250, 1500, 2000, 10000, 20000, float('inf')],
                                        labels=['XSmall', 'Small', 'Medium', 'Large', 'XLarge', 'XXLarge', 'Mega'])
            fig2 = px.bar(df_sorted, x='Shareholders', y=col, color='Category', title=f'{col} by Shareholder Category')
            plots.append(dcc.Graph(figure=fig2))

            # Percentage of each subcategory for Participants
            df_participants = df_sorted[df_sorted['Shareholder Type'] == 'Participants']
            if not df_participants.empty:
                df_participants_grouped = df_participants.groupby('Category')[col].sum().reset_index()
                fig4 = px.pie(df_participants_grouped, names='Category', values=col, title=f'Participants  Distribution of {col}')                
                # Add amounts to the pie chart for Participants subcategories
                fig4.update_traces(textinfo='label+percent+value')
                plots.append(dcc.Graph(figure=fig4))            

            # Create pie chart for each category
            for category in df_sorted['Category'].unique():
                df_category = df_sorted[df_sorted['Category'] == category]
                fig3 = px.pie(df_category, names='Shareholders', values=col, title=f'{category} Contributors - Distribution of {col}')
                plots.append(dcc.Graph(figure=fig3))
        
        return plots

    def create_linkedin_plots(df):
        metrics = [
            'Impressions (total)', 
            'Reactions (total)', 
            'Comments (total)', 
            'Reposts (total)', 
            'Engagement rate (organic)', 
            'Engagement rate (sponsored)', 
            'Engagement rate (total)'
        ]
        
        plots = []
        for metric in metrics:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[metric], mode='lines+markers', name=metric))
            fig.update_layout(title=f'Time Series of {metric}', xaxis_title='Date', yaxis_title=metric)
            plots.append(dcc.Graph(figure=fig))
        
        return plots

    def transform_data_yearly(df, year):
        df = df.reset_index()
        metric_names = df[df.columns[1]].values  # Save the metric names
        print("metric_names:", metric_names)
        print("Data after resetting index:")
        print(df.head())
        
        df = df.melt(id_vars=[df.columns[0]], var_name='Period', value_name='Value')
        df.rename(columns={df.columns[0]: 'Metric'}, inplace=True)
        print("Data after melting and renaming columns:")
        print(df.head())
        
        df['Date'] = df.apply(lambda row: parse_date(row['Period'], year), axis=1)
        print("Data after creating 'Date' column:")
        print(df.head())
        
        df.dropna(subset=['Date'], inplace=True)
        print("Data after dropping rows with NaN 'Date':")
        print(df.head())
        
        df['Metric'] = df['Metric'].apply(lambda x: metric_names[int(x)])  # Replace metric index with names
        return df

    def transform_data_overall(df):
        df = df.reset_index()
        metric_names = df[df.columns[1]].values  # Save the metric names
        print("metric_names:", metric_names)
        
        print("Data after resetting index:")
        print(df.head())
        
        df = df.melt(id_vars=[df.columns[0]], var_name='Period', value_name='Value')
        df.rename(columns={df.columns[0]: 'Metric'}, inplace=True)
        print("Data after melting and renaming columns:")
        print(df.head())
        
        df['Date'] = df.apply(lambda row: parse_date(row['Period']), axis=1)
        print("Data after creating 'Date' column:")
        print(df.head())
        
        df.dropna(subset=['Date'], inplace=True)
        print("Data after dropping rows with NaN 'Date':")
        print(df.head())
        
        df['Metric'] = df['Metric'].apply(lambda x: metric_names[int(x)])  # Replace metric index with names
        return df

    def parse_date(period, year=None):
        print(f"Parsing date for period: {period}")
        if year is None:
            try:
                period_parts = period.split('(')
                period_name = period_parts[0].strip()
                year = int(period_parts[1].replace(')', ''))
            except (ValueError, IndexError) as e:
                print(f"Failed to parse date for period: {period} with error: {e}")
                return None
        else:
            period_name = period

        # Determine the month based on the period name
        if '1st Quarter' in period_name:
            month = 3
        elif '2nd Quarter' in period_name:
            month = 6
        elif '3rd Quarter' in period_name:
            month = 9
        elif '4th Quarter' in period_name:
            month = 12
        elif '1st Half' in period_name:
            month = 6
        elif '2nd Half' in period_name:
            month = 12
        elif 'Overall' in period_name:
            month = 12
        else:
            return None
        
        return pd.Timestamp(year=year, month=month, day=1)

    def create_marketing_plots(df):
        average_scores_by_hashtag = df.groupby('Hashtags')['Weighted_Engagement_Score'].mean().sort_values(ascending=False)
        top_10_hashtags = average_scores_by_hashtag.head(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_10_hashtags.values,
            y=top_10_hashtags.index,
            orientation='h',
            marker=dict(color='rgba(50, 171, 96, 0.6)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1))
        ))

        fig.update_layout(title='Top 10 Hashtags by Weighted Engagement Score',
                          xaxis_title='Average Weighted Engagement Score',
                          yaxis_title='Hashtags')
        return dcc.Graph(figure=fig)

    def create_plots(df, metrics):
        color_map = {
            2022: "red",
            2023: "green",
            2024: "blue",
            2025: "orange",
            2026: "purple",
            2027: "brown",
            2028: "pink",
            2029: "gray",
            2030: "cyan"
            # Add more years if needed
        }
        plots = []
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            
            fig = go.Figure()
            
            # Time series plot for quarterly progression
            quarter_df = metric_df[metric_df['Period'].str.contains('Quarter')]
            fig.add_trace(go.Scatter(x=quarter_df['Date'], y=quarter_df['Value'], mode='lines+markers', name='Quarterly Progression'))
            
            # Add horizontal lines for overall values
            overall_df = metric_df[metric_df['Period'].str.contains('Overall')]
            for index, row in overall_df.iterrows():
                year = row['Date'].year
                color = color_map.get(year, "black")
                fig.add_shape(
                    type="line",
                    x0=quarter_df['Date'].min(),
                    y0=row['Value'],
                    x1=quarter_df['Date'].max(),
                    y1=row['Value'],
                    line=dict(
                        color=color,
                        width=2,
                        dash="dashdot",
                    ),
                )
                fig.add_trace(go.Scatter(x=[quarter_df['Date'].min()], y=[row['Value']], mode='lines', name=f'Overall {row["Date"].year}'))

            fig.update_layout(title=f'Time Series of {metric}', xaxis_title='Date', yaxis_title='Value')
            plots.append(dcc.Graph(figure=fig))
        
        return plots
    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),    
        html.H1(id='dashboard-title', style={'textAlign': 'center'}),
        html.Div(id='graph-container'),
        html.Hr(),
        html.H4("Filter Options", style={'textAlign': 'center'}),
        html.Div([
            dcc.Dropdown(id='status-filter', multi=True, placeholder="Select Status", style={'marginBottom': '10px'}),
            dcc.Checklist(
                id='shortlist-filter',
                options=[
                    {'label': 'Shortlist: True', 'value': 'true'},
                    {'label': 'Shortlist: False', 'value': 'false'},
                    {'label': 'Both', 'value': 'both'}
                ],
                value=['true', 'false'],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Button('Filter', id='filter-button', n_clicks=0, className='btn btn-primary')
        ], id='filter-options', style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}),
        html.Div([
            dcc.Dropdown(id='column-filter', multi=True, placeholder="Select Columns", style={'marginBottom': '10px'}),
            html.Button('Show Plots', id='show-plots-button', n_clicks=0, className='btn btn-primary')
        ], id='column-options', style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}),
    ])

    @dash_app.callback(
        [Output('dashboard-title', 'children'), Output('graph-container', 'children'), Output('filter-options', 'style'), Output('column-options', 'style'), Output('column-filter', 'options')],
        [Input('url', 'pathname'), Input('filter-button', 'n_clicks'), Input('show-plots-button', 'n_clicks')],
        [State('status-filter', 'value'), State('shortlist-filter', 'value'), State('column-filter', 'value')]
    )
    def display_page(pathname, n_clicks_filter, n_clicks_show, selected_status, selected_shortlist, selected_columns):
        parts = pathname.split('/')
        department = parts[2]
        filename = parts[3]
        df = load_data(department, filename)
        
        df.fillna(0, inplace=True)
        
        title = f'{department} Department Dashboard'
        plots = []
        filter_options_style = {'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}
        column_options_style = {'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}
        column_filter_options = []

        if department == 'BDM':
            filter_options_style = {'textAlign': 'center', 'marginBottom': '20px', 'display': 'block'}
            if n_clicks_filter > 0:
                if selected_status:
                    df = df[df['Status'].isin(selected_status)]
                if selected_shortlist:
                    if 'both' in selected_shortlist:
                        df = df[df['Shortlist'].astype(str).str.lower().isin(['true', 'false'])]
                    else:
                        df = df[df['Shortlist'].astype(str).str.lower().isin(selected_shortlist)]     
            shortlist_counts = df['Shortlist'].value_counts().reset_index()
            shortlist_counts.columns = ['Shortlist', 'count']
            shortlist_fig = px.bar(shortlist_counts, x='Shortlist', y='count', title='Shortlist Distribution')
            status_fig = px.pie(df, names='Status', title='Status Distribution')
            spinoff_fig = px.histogram(df, x='Spinoff', title='Spinoff Value Distribution')
            similarity_fig = px.scatter(df, x='Similarity(campus with association)', y='Name of Association', title='Similarity Scores')
            combined_fig = px.scatter(df, x='Similarity(campus with association)', y='Spinoff', color='Name of Association', title='Combined Similarity and Spinoff Scores')
            plots = [
                dcc.Graph(figure=shortlist_fig),
                dcc.Graph(figure=status_fig),
                dcc.Graph(figure=spinoff_fig),
                dcc.Graph(figure=similarity_fig),
                dcc.Graph(figure=combined_fig)
            ]
        elif department == 'Marketing':
            if 'Overall' in filename:
                df_transformed = transform_data_overall(df)
            else:
                if '2022' in filename:
                    year = 2022
                elif '2023' in filename:
                    year = 2023
                else:
                    year = None
                
                df_transformed = transform_data_yearly(df, year)
            
            metrics = df_transformed['Metric'].unique()
            print("Unique metrics found:", metrics)
            plots = create_plots(df_transformed, metrics)
        
        elif department == 'Linkedin':
            df_transformed = transform_linkedin_data(df)
            plots = create_linkedin_plots(df_transformed)
        
        elif department == 'MarketingUpload':
            plots = create_marketing_plots(df)

        elif department == 'Management':
            column_options_style = {'textAlign': 'center', 'marginBottom': '20px', 'display': 'block'}
            column_filter_options = [{'label': col, 'value': col} for col in df.columns[1:]]  # Exclude 'Shareholders'
            if selected_columns and n_clicks_show > 0:
                plots = create_management_plots(df, selected_columns)

        return title, plots, filter_options_style, column_options_style, column_filter_options


    @dash_app.callback(
        Output('status-filter', 'options'),
        [Input('url', 'pathname')]
    )
    def set_status_options(pathname):
        parts = pathname.split('/')
        department = parts[2]
        filename = parts[3]
        df = load_data(department, filename)
        if department == 'BDM':
            return [{'label': status, 'value': status} for status in df['Status'].unique()]
        return []

    return dash_app.server