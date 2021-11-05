import plotly.graph_objs as go
import dash
from dash.dcc.Graph import Graph
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
# from scripts.database import DataBase
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# DataFrame containing the values of the positive and negative reviews of each country
df = pd.DataFrame({'Country': ["Germany", "England", "Spain"],
                   'Positive_reviews': [8000, 12000, 9000],
                   'Negative_reviews': [3000, 4000, 2000]})

# DataFrame bar figures
#fig = go.Figure([go.Scatter(x=df['Country'], y=df['Positive_reviews'])
#                 ])
ger = df.query("Country == 'Germany'")
eng = df.query("Country == 'England'")
sp = df.query("Country == 'Spain'")
#query = df.iloc.columns("Country == 'Germany'")
#print(query.columns['Positive_reviews'])
allCntry = {'data': [{'x': df['Country'], 'y': df['Positive_reviews'], 'type': 'bar', 'name': 'Positive reviews per Country'}]}
#allCntry = px.bar(df, x="Country", y="Positive_reviews", barmode="group")


# Initialise the app# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Define the app
app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                     html.Div(className='four columns div-user-controls',
                              children=[
                                  html.H2('DASH - Reviews per country'),
                                  html.P('Visualising time series with Plotly - Dash.'),
                                  html.P('Pick a country from the dropdown below.'),
                                  html.Div(
                                      className='countryselector',
                                      children=[dcc.Dropdown(
                                          id="Country",
                                          options=[
                                              {'label': 'All Countries', 'value': 'allCntry'},
                                              {'label': 'Germany', 'value': 'Germany'},
                                              {'label': 'England', 'value': 'England'},
                                              {'label': 'Spain', 'value': 'Spain'}
                                          ],
                                          style={'color': '#1E1E1E'},
                                          optionHeight=35,
                                          value='allCntry',
                                          multi=False,
                                          searchable=False,
                                          placeholder='Select a country',
                                          clearable=False, )
                                      ]
                                  ),
                                  html.Div(className='eight columns div-for-charts bg-grey',
                                           children=[
                                               dcc.Graph(id='graph', figure=allCntry)
                                           ])
                              ])
                 ]

                 )
    ]

)


@app.callback(dash.dependencies.Output('graph', 'figure'),
              [dash.dependencies.Input('Country', 'value')])

def graph_update(selected_value):
    print(selected_value)
    #columns = ['Positive_reviews', 'Negative_reviews']
    if selected_value == 'allCntry':
        fig = {'data': [{'x': df['Country'], 'y': df['Positive_reviews'], 'type': 'bar',
                         'name': 'Positive reviews per Country'}]}
        return fig
    else:
        fig = {'data': [
            {'x': selected_value, 'y': df.loc[df['Country'] == selected_value, 'Positive_reviews'],
                'type': 'bar', 'name': 'Positive reviews per Country'}]}
        # fig = px.bar(df.iloc['Country'] == selected_value, x=selected_value, y='Positive and Negative reviews', barmode='group')
        # fig = {'data': [
        #     {'x': df['Country'], 'y': df['Positive_reviews', 'Negative_reviews'],
        #         'type': 'bar', 'name': 'Positive reviews per Country'}]}
        return fig






    # if selected_value == allCntry:
    #     #fig = px.bar(df, x="Country", y="Positive_reviews", barmode='group')
    #     fig = {'data': [
    #         {'x': df['Country'], 'y': df['Positive_reviews'], 'type': 'bar', 'name': 'Positive reviews per Country'}],         'layout': {
    #         'title': 'Dash Data Visualization'
    #     }}
    #     # fig = go.Figure([go.Bar(x=df['Country'] == selected_value, y=df['Positive_reviews'])])
    #     # fig.update_layout(title='Positive Reviews per country', xaxis_title=selected_value, yaxis_title='Positive Reviews',
    #     #                   barmode='group')
    #     return fig
    # else:
    #     fig = go.Figure([go.Bar(x=df['Country'] == selected_value, y=df['Positive_reviews'])])
    #     fig.update_layout(title='Positive Reviews per country', xaxis_title=selected_value, yaxis_title='Positive Reviews',
    #                       barmode='group')
    #     return fig



    # elif selected_value == 'England':
    #     fig = go.Figure([go.Bar(x=df['Country'] == 'England', y=df['Positive_reviews'])])
    #
    #     fig.update_layout(title='Positive Reviews per country',
    #                       xaxis_title='Country',
    #                       yaxis_title='Positive Reviews'
    #                       )
    #     return fig
    #
    # elif selected_value == 'Spain':
    #     fig = go.Figure([go.Bar(x=df['Country'] == 'Spain', y=df['Positive_reviews'])])
    #
    #     fig.update_layout(title='Positive Reviews per country',
    #                       xaxis_title='Country',
    #                       yaxis_title='Positive Reviews'
    #                       )
    #    return fig

# def change_graph(selected_value):
#     Germany = plt.plot(ger.Country, ger.Positive_reviews, label='Germany')
#     England = plt.plot(eng.Country, eng.Positive_reviews, label='England')
#     Spain = plt.plot(sp.Country, sp.Positive_reviws, label='Spain')
#     if selected_value == 'Germany':
#         updated_fig = Germany
#         return updated_fig
#     elif selected_value == 'England':
#         updated_fig = England
#         return updated_fig
#     elif selected_value == 'Spain':
#         updated_fig = Spain
#         return updated_fig


# Callback for timeseries price
# @app.callback(Output('timeseries', 'figure'),
#               [Input('countryselector', 'value')])
# def update_graph(selected_dropdown_value):
#     trace1 = []
#     df_sub = df
#     for stock in selected_dropdown_value:
#         trace1.append(go.Scatter(x=df_sub[df_sub['stock'] == stock].index,
#                                  y=df_sub[df_sub['stock'] == stock]['value'],
#                                  mode='lines',
#                                  opacity=0.7,
#                                  name=stock,
#                                  textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(
#                   colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
#                   template='plotly_dark',
#                   paper_bgcolor='rgba(0, 0, 0, 0)',
#                   plot_bgcolor='rgba(0, 0, 0, 0)',
#                   margin={'b': 15},
#                   hovermode='x',
#                   autosize=True,
#                   title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
#                   xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
#               ),
#
#               }
#
#     return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
