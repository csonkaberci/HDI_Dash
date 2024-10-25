# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:36:44 2024

@author: Berci
"""

import warnings
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from plotly.subplots import make_subplots

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
file_path = '1_emberi_fejlettseg.csv'
data = pd.read_csv(file_path)

# Select human development groups and filter out null values
human_development_groups = data['Human Development Groups'].dropna().unique()
countries_list = data['Country'].dropna().unique()

# Filter data columns
years = [str(year) for year in range(1990, 2021)]
indicator_names = [
    "Human Development Index",
    "Life Expectancy at Birth",
    "Expected Years of Schooling",
    "Mean Years of Schooling",
    "Gross National Income Per Capita",
    "Gender Development Index",
    "HDI female",
    "Life Expectancy at Birth, female",
    "Expected Years of Schooling, female",
    "Mean Years of Schooling, female",
    "Gross National Income Per Capita, female",
    "HDI male",
    "Life Expectancy at Birth, male",
    "Expected Years of Schooling, male",
    "Mean Years of Schooling, male",
    "Gross National Income Per Capita, male",
    "Inequality-adjusted Human Development Index",
    "Coefficient of human inequality",
    "Overall loss (%)",
    "Inequality in life expectancy",
    "Inequality in education",
    "Gender Inequality Index",
    "Maternal Mortality Ratio (deaths per 100,000 live births)",
    "Adolescent Birth Rate (births per 1,000 women ages 15-19)",
    "Labour force participation rate, female (% ages 15 and older)",
    "Planetary pressures-adjusted Human Development Index",
    "Difference from HDI value (%)",
    "Carbon dioxide emissions per capita (production) (tonnes)",
    "Material footprint per capita (tonnes)"
]

# Initialize Dash application
app = dash.Dash(__name__)


# Layout structure
app.layout = html.Div(style={'backgroundColor': '#2c455e', 'padding' : '90px'},
                      children=[
                      
    
    html.H1('Emberi Fejlettségi Szintek - Dash Alkalmazás', style={'textAlign': 'center', 'color':'white'}),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Az Ön adatai', value='tab-1', style={'backgroundColor': '#2c455e', 'color': 'white'}, selected_style={'backgroundColor': '#485e75', 'color': 'white'}),
        dcc.Tab(label='A projekt adatai', value='tab-2', style={'backgroundColor': '#2c455e', 'color': 'white'}, selected_style={'backgroundColor': '#485e75', 'color': 'white'}),
    ]),

    html.Div(id='tabs-content'),

    html.H2('Országok helye a 2021-es HDI rangsorban', style={'color': 'white', 'textAlign': 'center'}),
    html.H3('Fejlettségi szint kiválasztása', style={'color': 'white'}),

    dcc.Dropdown(
        id='dropdown',
        options=[{'label': group, 'value': group} for group in human_development_groups if group],
        placeholder='Válasszon fejlettségi szintet...',
        style={'backgroundColor': 'white'}
    ),

    html.Div(id='output-country-list'),

    html.H2('A változók alakulása az évek múlásával', style={'color': 'white', 'textAlign': 'center'}),
    html.H3('Ország kiválasztása', style={'color': 'white'}),

    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in countries_list if country],
        placeholder='Válasszon országot...',
        style={'backgroundColor': 'white'},
        multi=False  # Allow only single selection for the table
    ),

    dash_table.DataTable(
        id='country-data-table',
        columns=[{"name": "Indicator", "id": "Indicator"}] + [{"name": year, "id": year} for year in years],  # Columns
        data=[],  # Initially empty
        style_header={'backgroundColor': '#2c455e', 'color':'white'},
        style_table={'overflowX': 'auto'}, 
        style_data={  # Apply the background color to table data
        'backgroundColor': '#2c455e',  # Change to your preferred color
        'color': 'white',  # Text color
    },
        style_cell={  # Set default style for all cells
        'textAlign': 'center',  # Optional: Center align text
        'padding': '5px'
    }
    ),

    html.Br(),
    html.Hr(),
    html.Br(),

    html.H2('Fejlettségi Index Idősor Diagram', style={'textAlign': 'center', 'color':'white'}),
    html.H3('Válassz országot/országokat!', style={'color':'white'}),
    dcc.Dropdown(
        id='hdi-country-dropdown',
        options=[{'label': country, 'value': country} for country in countries_list if country],
        placeholder='Válasszon országokat...',
        style={'backgroundColor': 'white'},
        multi=True, # Allow multiple selection for the graph
        value=['Hungary']
    ),

    dcc.Graph(id='hdi-graph'),  # Graph to display HDI over years

    # New components for frequency distribution
    html.H2('Gyakorisági eloszlás diagram', style={'textAlign': 'center', 'color':'white'}),
    html.H3('Válassz évet!', style={'color': 'white'}),
    dbc.Row([
        dbc.Col(
            dcc.Slider(
                id='year-slider',
                min=1990,
                max=2020,
                value=2020,  # Default value
                marks={year: str(year) for year in range(1990, 2021, 5)},
                step=1  # Only one year at a time
            ),
            width=10  # Left column larger
        ),
        html.H3('Válassz változót!', style={'color': 'white'}),
        dbc.Col(
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': indicator, 'value': indicator} for indicator in indicator_names],
                placeholder='Válasszon változót...',
                style={'backgroundColor': 'white'},
                value='Human Development Index'  # Default value
            ),
            width=2  # Right column smaller
        )
    ]),
html.H3('Válaszd ki a klaszterek számát!', style={'color': 'white'}),
    # New slider for number of bins
    dcc.Slider(
        id='bins-slider',
        min=5,
        max=50,
        value=20,  # Default number of bins
        marks={i: str(i) for i in range(5, 51, 5)},
        step=1
    ),

    dcc.Graph(id='frequency-graph'),  # Graph to display frequency distribution

    # Thematic map for Task 7
    html.H2('Tematikus Térkép', style={'textAlign': 'center', 'color':'white'}),
    html.H3('Válassz változót!', style={'color': 'white'}),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='choropleth-variable-dropdown',
                options=[{'label': indicator, 'value': indicator} for indicator in indicator_names],
                placeholder='Válasszon változót...',
                style={'backgroundColor': 'white'},
                value='Human Development Index'  # Default value
            ),
            width=4
        ),
    ]),

    dcc.Graph(id='choropleth-map' , style={'width': '100%', 'height': '800px'}),  # Graph for choropleth map
    
    # New Section for GDP and Variable Regression
    html.H2('GDP/Fő és a Kiválasztott Változó szerinti Idősor Diagram', style={'textAlign': 'center', 'color':'white'}),

    # Dropdown for selecting country
    html.H3('Válassz országot!', style={'color': 'white'}),
    dcc.Dropdown(
        id='regression-country-dropdown',
        options=[{'label': country, 'value': country} for country in countries_list if country],
        placeholder='Válasszon országot...',
        style={'backgroundColor': 'white'},
        multi=False,
        value='Hungary'
    ),
    html.H3('Válassz változót!', style={'color': 'white'}),
    # Dropdown for selecting variable
    dcc.Dropdown(
        id='regression-variable-dropdown',
        options=[{'label': indicator, 'value': indicator} for indicator in indicator_names],
        placeholder='Válasszon változót...',
        style={'backgroundColor': 'white'},
        value='Gross National Income Per Capita'  # Default value
    ),
    html.H3('Válaszd ki a regresszió fokát!', style={'color': 'white'}),
    # Slider for regression degree
    dcc.Slider(
        id='regression-degree-slider',
        min=1,
        max=3,
        value=2,
        marks={i: str(i) for i in range(1, 6)},
        step=1
    ),

    # Graph for GDP per Capita and Variable
    dcc.Graph(id='dual-axis-graph'),
],)


# Update tabs content
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Markdown('''### Az Én adataim
                Név: Csonka Bertalan
                Neptun-kód: D93OKE  
                E-mail: bertalan2002@gmail.com  
            '''), 
        ], style={
        'backgroundColor': '#2c455e',  # Light blue background color
        'padding': '10px',  # Add some padding around the text
        'borderRadius': '10px',
        'color':'white',
    })
    elif tab == 'tab-2':
        return html.Div([
            dcc.Markdown('''### A projekt adatai
                Cél: Az emberi fejlettségi szintek elemzése.  
                Megvalósítás módja: Python, Dash és Plotly használatával.  
                Adathalmaz információk: Az adatok a Human Development Report alapján készültek.  
            ''')
        ], style={
        'backgroundColor': '#2c455e',  # Light blue background color
        'padding': '10px',  # Add some padding around the text
        'borderRadius': '10px',
        'color':'white',
    })

# Handle dropdown selection for development levels
@app.callback(
    Output('output-country-list', 'children'),
    Input('dropdown', 'value')
)
def update_country_list(selected_group):
    if selected_group:
        filtered_data = data[data['Human Development Groups'] == selected_group]
        # Az 'HDI Rank (2021)' oszlop értékeinek átalakítása egész számra
        filtered_data['HDI Rank (2021)'] = filtered_data['HDI Rank (2021)'].astype(int)
        countries = filtered_data[['Country', 'HDI Rank (2021)']].sort_values('Country')
        return html.Ul(
            [html.Li(f"{row['Country']} - {row['HDI Rank (2021)']}", style={'textAlign': 'center'}) for _, row in countries.iterrows()],
            style={'backgroundColor': '#2c455e', 'padding': '20px', 'color': 'white', 'listStyleType': 'none'}
        )
    return html.Div()

# Update country data table
@app.callback(
    Output('country-data-table', 'data'),
    Input('country-dropdown', 'value')
)
def update_country_table(selected_country):
    if selected_country:
        country_data = data[data['Country'] == selected_country]
        reshaped_data = []
        for indicator in indicator_names:
            row = {"Indicator": indicator}
            for year in years:
                value = country_data.get(f"{indicator} ({year})", pd.Series([None])).values[0]
                row[year] = round(value, 2) if pd.notnull(value) else None
            reshaped_data.append(row)
        return reshaped_data
    return []

# Color palette for the countries
color_palette = px.colors.qualitative.Set1  # You can choose a different palette if you want

# Update graph based on selected countries
@app.callback(
    Output('hdi-graph', 'figure'),
    Input('hdi-country-dropdown', 'value')
)
def update_graph(selected_countries):
    if selected_countries:
        filtered_data = data[data['Country'].isin(selected_countries)]

        # Create a figure
        fig = go.Figure()

        # Plot each country with a different color
        for i, country in enumerate(selected_countries):
            country_data = filtered_data[filtered_data['Country'] == country]
            fig.add_trace(go.Scatter(
                x=years,
                y=country_data[[f'Human Development Index ({year})' for year in years]].values.flatten(),
                mode='lines+markers',
                name=country,
                line=dict(color=color_palette[i % len(color_palette)])  # Assign a color from the palette
            ))

        # Update the layout
        fig.update_layout(
            title='Human Development Index alakulása az évek alatt',
            xaxis_title='Év',
            yaxis_title='Human Development Index',
            legend_title='Ország',
            hovermode='x unified',
            paper_bgcolor='#2c455e',  # Change background color for the figure
            plot_bgcolor='#2c455e',  # Change plot area background color
            font=dict(color='white')
        )
        return fig
    return go.Figure()  # Return an empty figure if no countries are selected

# Update frequency distribution graph based on year and variable
@app.callback(
    Output('frequency-graph', 'figure'),
    Input('year-slider', 'value'),
    Input('variable-dropdown', 'value'),
    Input('bins-slider', 'value')  # Add bins-slider input
)
def update_frequency_graph(selected_year, selected_variable, bins):
    if selected_variable:
        # Filter data for the selected year
        year_data = data[f"{selected_variable} ({selected_year})"].dropna()
        
        # Create a histogram with specified number of bins
        fig = go.Figure(data=go.Histogram(
            x=year_data,
            xbins=dict(
                start=year_data.min(),
                end=year_data.max(),
                size=(year_data.max() - year_data.min()) / bins,  # Dynamically set bin size
            ),
            name=selected_variable
        ))

        # Update layout for the graph
        fig.update_layout(
            title=f'{selected_variable} gyakorisági eloszlása {selected_year}-ban/ben',
            xaxis_title=selected_variable,
            yaxis_title='Gyakoriság',
            bargap=0.1,
            paper_bgcolor='#2c455e',  # Change background color for the figure
            plot_bgcolor='#2c455e',  # Change plot area background color
            font=dict(color='white')
        )

        return fig
    return go.Figure()  # Return an empty figure if no variable is 

# Update thematic map based on selected variable and animate over years
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('choropleth-variable-dropdown', 'value')]
)
def update_choropleth_map(selected_variable):
    # Filter data for the selected variable across all years
    relevant_columns = [f"{selected_variable} ({year})" for year in range(1990, 2021)]
    filtered_data = data[['Country'] + relevant_columns].dropna()

    # Reshape the data for animation, creating a year column
    reshaped_data = pd.melt(
        filtered_data, 
        id_vars='Country', 
        value_vars=relevant_columns, 
        var_name='Year', 
        value_name=selected_variable
    )

    # Extract the year number from the 'Year' column
    reshaped_data['Year'] = reshaped_data['Year'].str.extract('(\d+)').astype(int)

    # Create the animated choropleth map
    fig = px.choropleth(
        reshaped_data,
        locations='Country',
        locationmode='country names',
        color=selected_variable,
        hover_name='Country',
        animation_frame='Year',  # Enable animation by year
        title=f"{selected_variable} változása az évek során, térképen ábrázolva",
        color_continuous_scale=px.colors.sequential.Plasma,
    
    )
    fig.update_layout(geo_bgcolor='#2c455e',  paper_bgcolor='#2c455e', font=dict(color='white'))  # Change background color for the figure
    return fig

# Callback to update the dual-axis graph with regression lines
@app.callback(
    Output('dual-axis-graph', 'figure'),
    [Input('regression-country-dropdown', 'value'),
     Input('regression-variable-dropdown', 'value'),
     Input('regression-degree-slider', 'value')]
)
def update_dual_axis_graph(selected_country, selected_variable, degree):
    if selected_country and selected_variable:
        # Filter data for the selected country
        country_data = data[data['Country'] == selected_country]

        # Prepare data for GDP per Capita
        years = np.array([str(year) for year in range(1990, 2021)])  # Ensure years are string type
        gdp_years = np.array(years).astype(int)
        gdp_values = country_data[[f'Gross National Income Per Capita ({year})' for year in years]].values.flatten()

        # Prepare data for the selected variable
        variable_values = country_data[[f'{selected_variable} ({year})' for year in years]].values.flatten()

        # Ensure data is valid
        if len(gdp_values) == 0 or len(variable_values) == 0:
            return go.Figure()  # Return empty figure if no data

        # Fit regression model for GDP
        gdp_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        gdp_model.fit(gdp_years.reshape(-1, 1), gdp_values)
        gdp_trend = gdp_model.predict(gdp_years.reshape(-1, 1))

        # Fit regression model for the selected variable
        variable_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        variable_model.fit(gdp_years.reshape(-1, 1), variable_values)
        variable_trend = variable_model.predict(gdp_years.reshape(-1, 1))

        # Create a dual-axis figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add GDP trace
        fig.add_trace(go.Scatter(
            x=gdp_years,
            y=gdp_values,
            mode='lines+markers',
            name='GDP/Fő',
            line=dict(color='lightskyblue')
        ), secondary_y=False)

        # Add regression line for GDP
        fig.add_trace(go.Scatter(
            x=gdp_years,
            y=gdp_trend,
            mode='lines',
            name='GDP Regressziós egyenes',
            line=dict(color='tomato', dash='dash')
        ), secondary_y=False)

        # Add variable trace
        fig.add_trace(go.Scatter(
            x=gdp_years,
            y=variable_values,
            mode='lines+markers',
            name=selected_variable,
            line=dict(color='lightgreen')
        ), secondary_y=True)

        # Add regression line for the selected variable
        fig.add_trace(go.Scatter(
            x=gdp_years,
            y=variable_trend,
            mode='lines',
            name='Változó regressziós egyenese',
            line=dict(color='orange', dash='dash')
        ), secondary_y=True)

        # Update layout for the graph
        fig.update_layout(
            title=f'{selected_country}: GDP/Fő és {selected_variable} alakulása az évek alatt',
            xaxis_title='Év',
            yaxis_title='GDP/Fő',
            legend_title='Legend',
            hovermode='x unified',
            paper_bgcolor='#2c455e',  # Change background color for the figure
            plot_bgcolor='#2c455e',  # Change plot area background color
            font=dict(color='lightgray')
            
        )
        # Set y-axis titles
        fig.update_yaxes(title_text='GDP/Fő', secondary_y=False)
        fig.update_yaxes(title_text=selected_variable, secondary_y=True)

        return fig

    return go.Figure()  # Return an empty figure if no selections are made


if __name__ == '__main__':
    app.run_server(debug=True)
