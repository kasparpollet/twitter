import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from scripts.database import DataBase
import pandas

data = DataBase.get_tweets()


# dash.layout = html.Div([
#     html.Div([
#         html.H1('Price Optimization Dashboard'),
#         html.H2('Choose a product name'),
#         dcc.Dropdown(
#             id='product-dropdown',
#             options='dict_products',
#             multi=True,
#             value = ["Ben & Jerry's Wake and No Bake Cookie Dough Core Ice Cream","Brewdog Punk IPA"]
#         ),
#         dcc.Graph(
#             id='product-like-bar'
#         )
#     ], style={'width': '40%', 'display': 'inline-block'}),
#     html.Div([
#         html.H2('All product info'),
#         html.Table(id='my-table'),
#         html.P(''),
#     ], style={'width': '55%', 'float': 'right', 'display': 'inline-block'}),
#     html.Div([
#         html.H2('price graph'),
#         dcc.Graph(id='product-trend-graph'),
#         html.P('')
#     ], style={'width': '100%',  'display': 'inline-block'})
#
# ])
#
# app = dash.Dash(__name__)
# server = app.server
# if __name__ == '__main__':
#     app.run_server(debug=True)