import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from dash import Dash, dcc, html
# import dash_mantine_components as dmc
from dash.dependencies import Input, Output

def get_demography_df():
    files = os.listdir("experiment/demography/")
    entries = []
    for file in tqdm(files):
        with open(f"experiment/demography/{file}","r") as f:
            entry = json.load(f)
            entry["frame_id"] = int(file.split("_")[1])
            entries.append(entry)
    demography_df = pd.DataFrame(entries)
    return demography_df



# Divide into age-categories
def map_age(age):
    age_list = [0, 18, 25, 45, 65]
    for idx, upper in enumerate(age_list):
        if age < upper:
            return f"{age_list[idx-1]}-{upper-1}"
    return "65+"



def generate_visualizations():
    print("Generating visualizations..")

    demo_df = get_demography_df()

    # Map raw age into age groups 
    demo_df["age_group"] = demo_df.age.apply(map_age)

    app = Dash()

    def get_filtered_df(selected_range):
        # Compute min and max frame_id
        min_frame_id = demo_df['frame_id'].min()
        max_frame_id = demo_df['frame_id'].max()
        range_frame_id = max_frame_id - min_frame_id
        
        # Convert percentage slider values to frame_id range
        start_id = min_frame_id + (selected_range[0] / 100.0) * range_frame_id
        end_id = min_frame_id + (selected_range[1] / 100.0) * range_frame_id
        
        # Filter the DataFrame based on the calculated frame_id range
        filtered_df = demo_df[(demo_df['frame_id'] >= start_id) & (demo_df['frame_id'] <= end_id)]
        return filtered_df
    

    # Callback to update the sankey diagram based on the slider
    @app.callback(
        Output('sankey-emotion-race-age-graph', 'figure'),
        Input('frame-slider', 'value'),
        Input('race-emotion-age-radio-btn', 'value'))
    def update_emotion_race_sankey_graph_slider(selected_range, col_radio):
        filtered_df = get_filtered_df(selected_range)
        if col_radio == 'dominant_race':
            group_order = ['dominant_race', 'dominant_emotion', 'age_group']
        elif col_radio == 'dominant_emotion':
            group_order = ['dominant_emotion', 'age_group', 'dominant_race']
        else:  # 'age-race-emotion'
            group_order = ['age_group', 'dominant_race', 'dominant_emotion']

        # Group and count transitions
        transition_count = filtered_df.groupby(group_order).size().reset_index(name='count')
        first_transition = filtered_df.groupby(group_order[:2]).size().reset_index(name='count1')
        second_transition = filtered_df.groupby(group_order[1:]).size().reset_index(name='count2')

        # Calculate total counts for normalization of each segment
        total_first = first_transition['count1'].sum()
        total_second = second_transition['count2'].sum()

        first_transition['percentage'] = (first_transition['count1'] / total_first) * 100
        second_transition['percentage'] = (second_transition['count2'] / total_second) * 100

        # Get unique labels for nodes and create mappings
        labels = [filtered_df[col].unique() for col in group_order]
        labels_flat = [item for sublist in labels for item in sublist]
        index_mapping = {label: idx for idx, label in enumerate(labels_flat)}

        # Map the labels to indices for both transitions
        first_transition['source'] = first_transition[group_order[0]].map(index_mapping)
        first_transition['target'] = first_transition[group_order[1]].map(index_mapping)

        second_transition['source'] = second_transition[group_order[1]].map(index_mapping)
        second_transition['target'] = second_transition[group_order[2]].map(index_mapping)

        # Create the Sankey diagram data structure
        node_labels = labels_flat
        links = {
            'source': first_transition['source'].tolist() + second_transition['source'].tolist(),
            'target': first_transition['target'].tolist() + second_transition['target'].tolist(),
            'value': first_transition['percentage'].tolist() + second_transition['percentage'].tolist(),
        }

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="blue"
            ),
            link=links
        )])

        fig.update_layout(title_text="Sankey Diagram: Dynamic Transitions", font_size=10)
        return fig
    
    # Callback to update the gender race/emotion/age bar graph based on the slider
    @app.callback(
        Output('gender-emotion-race-age-bar-graph', 'figure'),
        Input('race-emotion-age-radio-btn', 'value'),
        Input('frame-slider', 'value'))
    def update_gender_emotion_race_age_graph_slider(col_radio, selected_range):
        filtered_df = get_filtered_df(selected_range)
        gender_df = filtered_df.groupby([col_radio,"dominant_gender"]).size().unstack(fill_value=0)
        grouped_percent = gender_df.div(gender_df.sum(axis=1), axis=0) * 100
        title_ref = {"dominant_race": "Race", "dominant_emotion": "Emotion", "age_group": "Age"}
        title_val = title_ref[col_radio]
        # Plotting the stacked bar chart
        fig = go.Figure()
        
        # Add one trace for each gender
        for gender in grouped_percent.columns:
            fig.add_trace(go.Bar(
                name=gender,
                x=grouped_percent[gender],  # percentage values
                y=grouped_percent.index,  # race/emotion categories
                orientation='h'
            ))
        
        # Update layout for a stacked bar chart
        fig.update_layout(
            barmode='stack',
            title=f'Distribution of Gender by {title_val}',
            xaxis_title=f'Percentage of {title_val}',
            yaxis_title=f'{title_val}',
        )
        
        return fig
    
    # Callback to update the spider graph based on the slider
    @app.callback(
        Output('spider-graph', 'figure'),
        Input('frame-slider', 'value'),
        Input('race-emotion-age-radio-btn', 'value'))
    def update_radial_graph_slider(selected_range, col_radio):
        filtered_df = get_filtered_df(selected_range)
        
        specific_df = pd.DataFrame(filtered_df[col_radio].value_counts(normalize=True)*100).reset_index()
        # Create the figure
        fig = go.Figure(data=go.Scatterpolar(
            r=specific_df['proportion'],
            theta=specific_df[col_radio],
            fill='toself'
        ))

        fig.update_layout(
            title="Percentage representation in movie segment",
            polar=dict(
                radialaxis=dict(
                    visible=True
                ),
            ),
            showlegend=False
        )
        
        return fig
    
    # Callback to update the line graph based on the slider
    @app.callback(
        Output('evolution-line-graph', 'figure'),
        Input('frame-slider', 'value'),
        Input('race-emotion-age-radio-btn', 'value'))
    def update_line_graph_slider(selected_range, col_radio):
        filtered_df = get_filtered_df(selected_range)
        filtered_df.loc[:,"count"] = 1
        pivot_df = filtered_df.pivot_table(index="frame_id", columns=col_radio, values="count", aggfunc="sum", fill_value=0).cumsum()
        # Create the figure
        fig = go.Figure()

        # Make sure the columns are treated as a list even if there's only one
        columns_list = pivot_df.columns.tolist()
        
        # Adding traces for each race/emotion/age
        for column in columns_list:
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[column],
                mode='lines',
                name=str(column),
                hoverinfo='name+x+y',
                stackgroup='one' # Stacking
            ))
        
        # updating the layout
        fig.update_layout(
            title= "Representation over time in movie segment",
            xaxis_title="Frame ID",
            yaxis_title="Cumulative count of appearances",
            hovermode="x"
        )
        
        return fig
    
    app.layout = html.Div([
        html.H1(children="Movie Visualizations"),
        dcc.RangeSlider(
            id='frame-slider',
            min=0,
            max=100,
            value=[0, 100],
            step=5,  # Percentage step
            marks={i: f'{i}%' for i in range(0, 101, 10)}
        ),
        html.H2(children="Distribution of Race, Emotion and Age"),
        html.Div([
            dcc.RadioItems(
                id='race-emotion-age-radio-btn',
                options=[
                    {'label': 'Race', 'value': 'dominant_race'},
                    {'label': 'Emotion', 'value': 'dominant_emotion'},
                    {'label': 'Age', 'value': 'age_group'}
                ],
                value='dominant_race',  # Default value
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}  # Space out the buttons
            )
        ], style={'margin-bottom': '20px'}),  # Space below the radio buttons
        # Container for graphs
        html.Div([
            dcc.Graph(id='spider-graph', style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='evolution-line-graph', style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex'}),
        html.Div([
            dcc.Graph(id='gender-emotion-race-age-bar-graph', style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='sankey-emotion-race-age-graph', style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex'})
    ])

    app.run()



    
    