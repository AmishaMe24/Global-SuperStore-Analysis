import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import warnings
import plotly.io as pio
from matplotlib import pyplot as plt
import io
import base64
pio.templates.default = "plotly_white"
import scipy.stats as stats
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.tools as tls

warnings.filterwarnings('ignore')

# Initialize app
app = dash.Dash(
    'My app',
    external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ]
)

# Color scheme
#PRIMARY_COLOR = '#CFDBF2'  # Light Blue-Gray
PRIMARY_COLOR = '#b5c7eb'
ACCENT_COLOR = '#F2CFEC'  # Light Pink
SECONDARY_COLOR = '#F2E6CF'  # Light Beige
SUCCESS_COLOR = '#CFF2D5'  # Light Green
TEXT_COLOR = '#2C3E50'  # Dark blue-gray for text

super_store_df = pd.read_csv('Global_Superstore2.csv', encoding='iso-8859-1')
super_store_df['Order Date'] = pd.to_datetime(super_store_df['Order Date'])
super_store_df['Ship Date'] = pd.to_datetime(super_store_df['Ship Date'])
super_store_df['Order Year'] = super_store_df['Order Date'].dt.year
super_store_df['Order Month'] = super_store_df['Order Date'].dt.month
super_store_df["Ship Year"] = super_store_df["Ship Date"].dt.year
super_store_df["Ship Month"] = super_store_df["Ship Date"].dt.month
super_store_df["Unit Price"] = super_store_df["Sales"] / super_store_df["Quantity"]

super_store_df.drop('Postal Code', axis=1, inplace=True)

month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
super_store_df['Order Month'] = super_store_df['Order Month'].map(month_map)
super_store_df['Ship Month'] = super_store_df['Ship Month'].map(month_map)

# Calculate metrics
total_sales = super_store_df['Sales'].sum()
total_profit = super_store_df['Profit'].sum()
total_orders = len(super_store_df['Order ID'].unique())
total_customers = len(super_store_df['Customer ID'].unique())
df = super_store_df.copy()

# Dashboard Layout
app.layout = html.Div([
    # Header with Tabs
    html.Div([
        html.H1('Global Superstore Analytics',
                style={'fontSize': '40px', 'padding': '20px', 'margin': '0',
                       'color': TEXT_COLOR, 'textAlign': 'center', 'font-weight': 'bold'}),
        dcc.Tabs(
            id='tabs',
            value='about',
            className='custom-tabs',
            children=[
                dcc.Tab(label='Dashboard', value='about',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Data Cleaning', value='data',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Outlier Analysis', value='outlier',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='PCA', value='pca',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Normality Test', value='normality',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Visualizations Pt 1', value='visualization',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Visualizations Pt 2', value='visualization_2',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Subplots', value='subplots',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
            ]
        )
    ], className='header'),

    # Main Content
    html.Div(id='layout', className='content')
])

# ============================== About Layout ==============================
about_layout = html.Div([
    # Welcome Message
    html.Div([
        html.H2('Overview',
                style={'fontSize': '28px', 'marginBottom': '20px', 'color': TEXT_COLOR, 'font-weight': 'bold'}),
        html.Div([
            html.P([
                "This interactive dashboard analyzes the Global Superstore dataset, ",
                "providing insights into sales, profits, and customer behavior across different regions, markets, segment etc. ",
                "The project aims to demonstrate various data visualization techniques and ",
                "create an intuitive interface for exploring business metrics. Navigate through the tabs for data cleaning, data transformation "
                "outlier detection, PCA, and various visualizations."
            ],style={'fontSize': '18px', 'lineHeight': '1.6', 'color': TEXT_COLOR}, className='nine columns'),
            html.Img(src='assets/images.jpeg', style={'width': '15%', 'height':'10%'}, className='two columns'),
        ], className='row'),
    ], className='welcome-section'),

    # Metrics Row
    html.Div([
        html.Div([
            html.Div([
                html.I(className="fas fa-dollar-sign fa-2x"),
                html.H3(f'${total_sales:,.0f}', style={'fontSize': '24px', 'font-weight': 'bold'}),
                html.P('Total Sales', style={'fontSize': '18px', 'font-weight': 'bold'})
            ], className='metric-box')
        ], className='three columns'),

        html.Div([
            html.Div([
                html.I(className="fas fa-chart-line fa-2x"),
                html.H3(f'${total_profit:,.0f}', style={'fontSize': '24px', 'font-weight': 'bold'}),
                html.P('Total Profit', style={'fontSize': '18px', 'font-weight': 'bold'})
            ], className='metric-box')
        ], className='three columns'),

        html.Div([
            html.Div([
                html.I(className="fas fa-shopping-cart fa-2x"),
                html.H3(f'{total_orders:,}', style={'fontSize': '24px', 'font-weight': 'bold'}),
                html.P('Total Orders', style={'fontSize': '18px', 'font-weight': 'bold'})
            ], className='metric-box')
        ], className='three columns'),

        html.Div([
            html.Div([
                html.I(className="fas fa-users fa-2x"),
                html.H3(f'{total_customers:,}', style={'fontSize': '24px', 'font-weight': 'bold'}),
                html.P('Total Customers', style={'fontSize': '18px', 'font-weight': 'bold'})
            ], className='metric-box')
        ], className='three columns'),
    ], className='row'),

    # Charts Row
    html.Div([
        html.Div([
            html.Div([
                html.H3('Regional Sales Distribution',
                        style={'fontSize': '22px', 'marginBottom': '20px', 'font-weight': 'bold'}),
                dcc.Graph(
                    figure=px.pie(df,
                                  values='Sales',
                                  names='Region',
                                  color_discrete_sequence=px.colors.qualitative.Pastel1,
                                  template='plotly_white')
                    .update_traces(textposition='inside',
                                   textinfo='percent+label',
                                   textfont_size=16)
                    .update_layout(
                        font=dict(size=16),
                        height=450
                    )
                )
            ], className='chart-container')
        ], className='six columns'),

        html.Div([
            html.Div([
                html.H3('Category Performance',
                        style={'fontSize': '22px', 'marginBottom': '20px', 'font-weight': 'bold'}),
                dcc.Graph(
                    figure=px.bar(df.groupby('Category').agg({'Profit': 'sum'}).reset_index(),
                                  x='Category',
                                  y='Profit',
                                  template='plotly_white',
                                  color_discrete_sequence=[PRIMARY_COLOR])
                    .update_layout(
                        font=dict(size=16),
                        height=450
                    )
                )
            ], className='chart-container')
        ], className='six columns'),
    ], className='row'),

    # Time Series Chart
    html.Div([
        html.Div([
            html.Div([
                html.H3('Sales Trend Analysis',
                        style={'fontSize': '22px', 'marginBottom': '20px', 'font-weight': 'bold'}),
                dcc.Graph(
                    figure=px.line(df.groupby('Order Date').agg({'Sales': 'sum'}).reset_index(),
                                   x='Order Date',
                                   y='Sales',
                                   template='plotly_white')
                    .update_traces(line_color=PRIMARY_COLOR)
                    .update_layout(
                        font=dict(size=16),
                        height=450
                    )
                )
            ], className='chart-container')
        ], className='twelve columns'),
    ], className='row'),
])


# ============================== Data Cleaning & Transformation Layout ==============================

data_layout = html.Div([
    # Data Viewing Section
    html.Div([
        html.H3('Data Cleaning', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
        html.P(['The dataset may contain missing values, duplicates, and outliers that need to be addressed before analysis. ',
                'Use the following tools to clean the data and prepare it for further processing.'],
                style={'fontSize': '18px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4('View Dataset', style={'fontSize': '20px', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='dataset-view-dropdown',
                    options=[
                        {'label': 'Head (First 5 rows)', 'value': 'head'},
                        {'label': 'Tail (Last 5 rows)', 'value': 'tail'},
                        {'label': 'Random Sample', 'value': 'sample'}
                    ],
                    value='head',
                    style={'fontSize': '16px'}
                ),
                html.Div(id='dataset-view-output', style={'marginTop': '10px'})
            ], className='six columns'),
            ], className='row', style={'marginBottom': '30px'}
        ),
        # Data Cleaning Section
        html.Div([
            html.Div([
                html.Div([
                    html.H4('Select Cleaning Method', style={'fontSize': '20px', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='cleaning-method-dropdown',
                        options=[
                            {'label': 'Forward Fill', 'value': 'ffill'},
                            {'label': 'Backward Fill', 'value': 'bfill'},
                            {'label': 'Mean', 'value': 'mean'},
                            {'label': 'Median', 'value': 'median'}
                        ],
                        value='ffill',
                        style={'fontSize': '16px'}
                    ),
                    html.Div(id='cleaning-output', style={'marginTop': '10px'})
                ], className='six columns'),
            ], className='row', style={'marginBottom': '30px'}),

            # Data Transformation: Standardization/ Normalization
            html.Div([
                html.Div([
                    html.H4('Select Transformation Method', style={'fontSize': '20px', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='transformation-method-dropdown',
                        options=[
                            {'label': 'Original', 'value': 'original'},
                            {'label': 'Standardization', 'value': 'standardization'},
                            {'label': 'Normalization', 'value': 'normalization'},
                        ],
                        value='original',
                        style={'fontSize': '16px'}
                    ),
                    html.Div(id='transformation-output', style={'marginTop': '10px'})
                ], className='six columns'),
            ], className='row', style={'marginBottom': '30px'})
        ]),
    ]),
])

@app.callback(
    Output('dataset-view-output', 'children'),
    Input('dataset-view-dropdown', 'value')
)

def update_dataset_view(view):
    if view == 'head':
        return html.Pre(super_store_df.head().to_string(), style={'fontSize': '16px', 'color': TEXT_COLOR})
    elif view == 'tail':
        return html.Pre(super_store_df.tail().to_string(), style={'fontSize': '16px', 'color': TEXT_COLOR})
    else:
        return html.Pre(super_store_df.sample(5).to_string(), style={'fontSize': '16px', 'color': TEXT_COLOR})

@app.callback(
    Output('cleaning-output', 'children'),
    Input('cleaning-method-dropdown', 'value')
)

def update_cleaning_output(method):
    if method == 'ffill':
        df_cleaned = super_store_df.fillna(method='ffill')
    elif method == 'bfill':
        df_cleaned = super_store_df.fillna(method='bfill')
    elif method == 'mean':
        df_cleaned = super_store_df.fillna(super_store_df.mean())
    else:
        df_cleaned = super_store_df.fillna(super_store_df.median())
    return html.Pre(df_cleaned.head().to_string(), style={'fontSize': '16px', 'color': TEXT_COLOR})

@app.callback(
    Output('transformation-output', 'children'),
    Input('transformation-method-dropdown', 'value')
)


def update_transformation_output(method):
    numeric_columns = ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Unit Price']

    if method == 'original':
        df[numeric_columns] = super_store_df[numeric_columns]
    if method == 'standardization':
        # Z-score standardization
        df[numeric_columns] = (super_store_df[numeric_columns] - super_store_df[numeric_columns].mean()) / super_store_df[numeric_columns].std()
    if method == 'normalization':
        # Min-Max normalization
        df[numeric_columns] = (super_store_df[numeric_columns] - super_store_df[numeric_columns].min()) / (super_store_df[numeric_columns].max() - super_store_df[numeric_columns].min())
    return html.Pre(df.head().round(3).to_string(), style={'fontSize': '16px', 'color': TEXT_COLOR})


# ============================== Outlier Analysis Layout ==============================

outlier_layout = html.Div([
    # Outlier Detection Section
    html.Div([
        html.H3('Outlier Detection', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
        html.P(['Outliers are extreme values that deviate significantly from other observations in the dataset. ',
                'Use the following tools to detect and visualize outliers in the data.'],
                style={'fontSize': '18px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4('Select Feature For Analysis:', style={'fontSize': '20px', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='outlier-column-dropdown',
                    options=[{'label': col, 'value': col} for col in ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Unit Price']],
                    value='Sales',
                    style={'fontSize': '16px'}
                ),
            ], className='six columns'),
        ], className='row', style={'marginBottom': '30px'}),

        html.Div([
            html.Div([
                dcc.Loading(
                    id='loading-before-outlier-plot',
                    type='cube',
                    color=ACCENT_COLOR,
                    children=[
                        dcc.Graph(id='outlier-box-plot')
                    ],
                ),
            ], className='six columns'),
            html.Div([
                dcc.Loading(
                            id='loading-after-outlier-plot',
                            type='cube',
                            color=ACCENT_COLOR,
                            children=[
                                dcc.Graph(id='outlier-IQR-box-plot')
                            ],
                        ),
            ], className='six columns')
        ], className='row', style={'marginBottom': '30px'}),

        html.H3('Violin Plot', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
        html.Div([
            dcc.Loading(
                id='loading-violin-plot',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='box-violin-plot')
                ]
            )
        ], className='six columns')
    ]),
])

@app.callback(
    Output('outlier-box-plot', 'figure'),
    Output('outlier-IQR-box-plot', 'figure'),
    Output('box-violin-plot', 'figure'),
    Input('outlier-column-dropdown', 'value'),
)

def update_outlier_plot(column):
    fig1 = go.Figure(go.Box(y=df[column], name=column))
    fig1.update_layout(title={'text': f'Box Plot for {column} before Outlier Removal',
                              'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                              'font': dict(size=24, family='Serif', color='blue', weight='bold')})
    numeric_columns = ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Unit Price']

    q1 = df[numeric_columns].quantile(0.25)
    q3 = df[numeric_columns].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_iqr = df[(df[numeric_columns] > lower_bound) & (df[numeric_columns] < upper_bound)]
    fig2 = go.Figure(go.Box(y=df_iqr[column], name=column))
    fig2.update_layout(title={'text': f'Box Plot for {column} after Outlier Removal',
                              'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                              'font': dict(size=24, family='Serif', color='blue', weight='bold')})

    fig3 = px.violin(df_iqr, y=column, box=True, points="all",
                 title="Violin Plot",
                 template="plotly_white")

    return fig1, fig2, fig3


# ============================== PCA Layout ==============================

pca_layout = html.Div([
    html.H3('Principal Component Analysis (PCA)', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
    # PCA Analysis Section
    html.Div([
        html.Div([
            html.H4('Select Features for PCA:', style={'fontSize': '20px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='pca-features-dropdown',
                options=[{'label': col, 'value': col} for col in ['Sales', 'Profit', 'Quantity', 'Discount',
                                                                  'Shipping Cost', 'Unit Price']],
                value=['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Unit Price'],
                multi=True,
                style={'fontSize': '16px'}
            ),
            html.H4('Select Number of Components:', style={'fontSize': '20px', 'marginBottom': '10px'}),
            dcc.Slider(
                id='pca-components-slider',
                min=0.1,
                max=1.0,
                step=0.1,
                value=0.9,
                marks={i / 10: str(i / 10) for i in range(1, 11)},
            ),
            html.Div([
                dcc.Loading(
                    id='loading-pca-results-1',
                    type='cube',
                    color=ACCENT_COLOR,
                    children=[
                        html.Div([
                            html.H4('PCA Results:', style={'fontSize': '20px', 'marginBottom': '10px'}),
                            html.Div(id='pca-results'),
                        ])]
                )
            ])
        ], className='six columns'),
        html.Div([
            dcc.Loading(
                id='loading-pca-results',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='pca-plot')
                ])]
            )
        ], className='six columns')
    ], className='row', style={'marginBottom': '30px'}),
])

@app.callback(
    Output('pca-results', 'children'),
    Output('pca-plot', 'figure'),
    Input('pca-features-dropdown', 'value'),
    Input('pca-components-slider', 'value')
)

def update_pca_results(features, n_components):
    X = df[features]
    X = (X - X.mean()) / X.std()

    pca = PCA(svd_solver="full", random_state=5764)
    X_pca = pca.fit_transform(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    vline_x = np.where(cumulative_variance > n_components)[0][0] + 1

    fig = px.line(
        x=np.arange(1, len(cumulative_variance) + 1, 1),
        y=cumulative_variance,
    )
    fig.update_layout(
        xaxis_title={"text": "Number of Components", "font": dict(size=18, family="Serif", color="darkred")},
        yaxis_title={"text": "Cumulative Explained Variance", "font": dict(size=18, family="Serif", color="darkred")},
        template="plotly_white",
        height=600
    )
    fig.update_traces(mode="lines+markers")
    fig.add_vline(x=vline_x, line_dash="dash", line_color="red")
    fig.add_hline(y=n_components, line_dash="dash", line_color="black")

    pca_results = html.Div([
        html.P(f"Original Shape: {X.shape}", style={"fontSize": "16px"}),
        html.P(f"Reduced Shape: {X_pca[:, :vline_x].shape}", style={"fontSize": "16px"}),
        html.P(f"Number of features needed to explain more than {n_components * 100}% of the dependent variance: "
               f"{vline_x}", style={"fontSize": "16px"}),
        html.P(f"Condition number of Original data: {np.linalg.cond(X).round(2)}", style={"fontSize": "16px"}),
        html.P(f"Condition number of Reduced data: {np.linalg.cond(X_pca[:, :vline_x]).round(2)}", style={"fontSize": "16px"}),
        html.P(f"Single Value of Reduced Data: {pca.singular_values_}", style={"fontSize": "16px"}),
    ])

    return pca_results, fig

# ============================== Normality Test Layout ==============================

normality_layout = html.Div([
    html.H3('Normality Test', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
    html.Br(),

    # Dropdowns (same as before)
    html.Div([
        html.Div([
            html.Label('Select Feature:', style={'fontSize': '18px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='normality-feature-selector',
                options=[
                    {'label': col, 'value': col}
                    for col in ['Sales', 'Profit', 'Quantity', 'Discount',
                                'Shipping Cost', 'Unit Price']
                ],
                value='Sales',
                style={'fontSize': '16px'}
            ),
        ], className='four columns', style={'paddingRight': '10px'}),

        html.Div([
            html.Label('Select Transformation:', style={'fontSize': '18px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='transformation-selector',
                options=[
                    {'label': 'Box-Cox Transformation', 'value': 'boxcox'},
                    {'label': 'Log Transformation', 'value': 'log'},
                    {'label': 'Square Root Transformation', 'value': 'sqrt'},
                    {'label': 'Cube Root Transformation', 'value': 'cbrt'}
                ],
                value='boxcox',
                style={'fontSize': '16px'}
            ),
        ], className='four columns', style={'paddingRight': '10px'}),

        html.Div([
            html.Label('Select Normality Test:', style={'fontSize': '18px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='normality-test-selector',
                options=[
                    {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
                    {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
                    {'label': "D'Agostino-Pearson Test", 'value': 'dagostino'}
                ],
                value='shapiro',
                style={'fontSize': '16px'}
            ),
        ], className='four columns'),
    ], className='row', style={'marginBottom': '20px'}),

    # Test Results Card with Loading
    html.Div([
        html.Div([
            html.H4('Normality Test Results', style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-results',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div(id='normality-test-results')
                ]
            )
        ], className='chart-container')
    ], className='row', style={'marginBottom': '30px'}),

    # Original vs Transformed Data Plots with Loading
    html.Div([
        html.Div([
            html.H4('Original vs Transformed Data Comparison',
                    style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-original-transformed',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='original-transformed-plot')
                ]
            )
        ], className='chart-container')
    ], className='row', style={'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.H4('Plot Customization', style={'color': TEXT_COLOR, 'marginBottom': '15px'}),

            # Distribution Plot Options
            html.Div([
                html.Label('Distribution Plot Options:',
                           style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '10px'}),

                # KDE Options
                html.Div([
                    dcc.Checklist(
                        id='dist-kde-options',
                        options=[
                            {'label': 'Show KDE', 'value': 'show_kde'},
                        ],
                        value=['show_kde'],
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),
                ]),

                # Histogram Options
                html.Div([
                    dcc.Checklist(
                        id='dist-hist-options',
                        options=[
                            {'label': 'Show Histogram', 'value': 'show_hist'}
                        ],
                        value=['show_hist'],
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),
                ], className='row'),

                # Color Palette Selection
                html.Div([
                    html.Label('Color Palette:', style={'fontSize': '14px', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='dist-color-palette',
                        options=[
                            {'label': 'Pastel', 'value': 'px.colors.qualitative.Pastel1'},
                            {'label': 'Set3', 'value': 'px.colors.qualitative.Set3'},
                            {'label': 'Safe', 'value': 'px.colors.qualitative.Safe'},
                            {'label': 'Prism', 'value': 'px.colors.qualitative.Prism'}
                        ],
                        value='px.colors.qualitative.Pastel1',
                        style={'width': '200px', 'display': 'inline-block'}
                    ),
                ], style={'marginBottom': '10px'}),

                # Alpha and Line Width Sliders
                html.Div([
                    html.Label('Opacity:', style={'fontSize': '14px'}),
                    dcc.Slider(
                        id='dist-alpha',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.6,
                        marks={i / 10: str(i / 10) for i in range(0, 11)},
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label('Line Width:', style={'fontSize': '14px'}),
                    dcc.Slider(
                        id='dist-linewidth',
                        min=1,
                        max=5,
                        step=0.5,
                        value=2,
                        marks={i: str(i) for i in range(1, 6)},
                    ),
                ]),
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ], className='twelve columns')
    ], className='row', style={'marginBottom': '30px'}),

    # Histogram Comparison with Loading
    html.Div([
        html.Div([
            html.H4('Histogram Comparison',
                    style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-histogram',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='histogram-comparison')
                ]
            )
        ], className='chart-container')
    ], className='row', style={'marginBottom': '30px'}),

    # Distplot Comparison with Loading
    html.Div([
        html.Div([
            html.H4('Distribution Plot Comparison',
                    style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-distplot',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='distplot-comparison')
                ]
            )
        ], className='chart-container')
    ], className='row', style={'marginBottom': '30px'}),

    # QQ Plots Comparison with Loading
    html.Div([
        html.Div([
            html.H4('Q-Q Plots Comparison',
                    style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-qq-plots',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='qq-plots-comparison')
                ]
            )
        ], className='chart-container')
    ], className='row'),

    # Box-Cox plot with Loading (only shown when Box-Cox transformation is selected)
    html.Div([
        html.Div([
            html.H4('Box-Cox Transformation Parameter Selection',
                    style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            dcc.Loading(
                id='loading-box-cox',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    dcc.Graph(id='box-cox-plot')
                ]
            )
        ], className='chart-container')
    ], className='row', id='box-cox-container', style={'marginBottom': '30px', 'display': 'none'})
])

@app.callback(
    [Output('normality-test-results', 'children'),
     Output('original-transformed-plot', 'figure'),
     Output('histogram-comparison', 'figure'),
     Output('distplot-comparison', 'figure'),
     Output('qq-plots-comparison', 'figure'),
     Output('box-cox-plot', 'figure'),
     Output('box-cox-container', 'style')],
    [Input('normality-feature-selector', 'value'),
     Input('transformation-selector', 'value'),
     Input('normality-test-selector', 'value'),
     Input('dist-kde-options', 'value'),
     Input('dist-hist-options', 'value'),
     Input('dist-color-palette', 'value'),
     Input('dist-alpha', 'value'),
     Input('dist-linewidth', 'value')]
)

def update_normality_analysis(feature, transform_type, test_type,
                            kde_options, hist_options, color_palette,
                            alpha_value, linewidth):
    # Get the data
    temp_df = super_store_df.copy()
    data = temp_df[feature].dropna()
    colors = eval(color_palette)

    # Apply selected transformation
    if transform_type == 'boxcox':
        transformed_data, lambda_param = stats.boxcox(data - min(data) + 1)
        transform_name = f'Box-Cox (λ = {lambda_param:.4f})'
    elif transform_type == 'log':
        transformed_data = np.log(data - min(data) + 1)
        transform_name = 'Log'
        lambda_param = None
    elif transform_type == 'sqrt':
        transformed_data = np.sqrt(data - min(data))
        transform_name = 'Square Root'
        lambda_param = None
    else:  # cube root
        transformed_data = np.cbrt(data)
        transform_name = 'Cube Root'
        lambda_param = None

    # Perform normality tests
    if test_type == 'ks':
        orig_stat, orig_pval = stats.kstest(data, 'norm')
        trans_stat, trans_pval = stats.kstest(transformed_data, 'norm')
        test_name = "Kolmogorov-Smirnov"
    elif test_type == 'shapiro':
        orig_stat, orig_pval = stats.shapiro(data)
        trans_stat, trans_pval = stats.shapiro(transformed_data)
        test_name = "Shapiro-Wilk"
    else:
        orig_stat, orig_pval = stats.normaltest(data)
        trans_stat, trans_pval = stats.normaltest(transformed_data)
        test_name = "D'Agostino-Pearson"

    # Create test results card
    test_results = html.Div([
        html.Div([
            html.H5("Original Data", style={'textAlign': 'center'}),
            html.P(f"Test: {test_name}", style={'fontSize': '16px'}),
            html.P(f"Statistic: {orig_stat:.4f}", style={'fontSize': '16px'}),
            html.P(f"P-value: {orig_pval:.4f}", style={'fontSize': '16px'}),
            html.P(
                f"Conclusion: Data is {'not ' if orig_pval < 0.05 else ''}normally distributed (α=0.05)",
                style={'fontSize': '16px', 'fontWeight': 'bold', 'marginTop': '10px'}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H5("Transformed Data", style={'textAlign': 'center'}),
            html.P(f"Test: {test_name}", style={'fontSize': '16px'}),
            html.P(f"Statistic: {trans_stat:.4f}", style={'fontSize': '16px'}),
            html.P(f"P-value: {trans_pval:.4f}", style={'fontSize': '16px'}),
            html.P(
                f"Conclusion: Data is {'not ' if trans_pval < 0.05 else ''}normally distributed (α=0.05)",
                style={'fontSize': '16px', 'fontWeight': 'bold', 'marginTop': '10px'}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'})
    ])

    # Create original vs transformed plot
    fig1 = make_subplots(rows=2, cols=1, subplot_titles=('Original Data', 'Transformed Data'))
    fig1.add_trace(go.Scatter(y=data, mode='lines', name='Original'), row=1, col=1)
    fig1.add_trace(go.Scatter(y=transformed_data, mode='lines', name=transform_name), row=2, col=1)
    fig1.update_layout(height=600, template='plotly_white')

    # Create histogram comparison
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Original Data', f'{transform_name} Data'))

    if 'show_hist' in hist_options:
        fig2.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            name='Original',
            opacity=alpha_value,
            marker_color=colors[0]
        ), row=1, col=1)

        fig2.add_trace(go.Histogram(
            x=transformed_data,
            nbinsx=30,
            name='Transformed',
            opacity=alpha_value,
            marker_color=colors[1]
        ), row=1, col=2)

    fig2.update_layout(height=400, template='plotly_white')

    # Create distplot comparison
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Original Data', f'{transform_name} Data'))

    if 'show_hist' in hist_options:
        fig3.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name='Original Hist',
            opacity=alpha_value,
            marker_color=colors[0]
        ), row=1, col=1)

        fig3.add_trace(go.Histogram(
            x=transformed_data,
            histnorm='probability density',
            name='Transformed Hist',
            opacity=alpha_value,
            marker_color=colors[1]
        ), row=1, col=2)

    if 'show_kde' in kde_options:
        # Original Data KDE
        kde_orig = stats.gaussian_kde(data)
        x_range_orig = np.linspace(min(data), max(data), 200)
        kde_y_orig = kde_orig(x_range_orig)

        fig3.add_trace(go.Scatter(
            x=x_range_orig,
            y=kde_y_orig,
            name='Original KDE',
            line=dict(color=colors[2], width=linewidth),
            fillcolor=f'rgba({int(colors[2][1:3], 16)}, {int(colors[2][3:5], 16)}, {int(colors[2][5:7], 16)}, {alpha_value})'
            if 'fill_kde' in kde_options else None
        ), row=1, col=1)

        # Transformed Data KDE
        kde_trans = stats.gaussian_kde(transformed_data)
        x_range_trans = np.linspace(min(transformed_data), max(transformed_data), 200)
        kde_y_trans = kde_trans(x_range_trans)

        fig3.add_trace(go.Scatter(
            x=x_range_trans,
            y=kde_y_trans,
            name='Transformed KDE',
            line=dict(color=colors[3], width=linewidth),
            fillcolor=f'rgba({int(colors[3][1:3], 16)}, {int(colors[3][3:5], 16)}, {int(colors[3][5:7], 16)}, {alpha_value})'
            if 'fill_kde' in kde_options else None
        ), row=1, col=2)

    fig3.update_layout(
        height=400,
        template='plotly_white',
        showlegend=True
    )

    # Create QQ plots comparison
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=('Original Data Q-Q Plot',
                                                         f'{transform_name} Data Q-Q Plot'))

    # Original Data QQ Plot
    qq_orig = stats.probplot(data)
    fig4.add_trace(go.Scatter(x=qq_orig[0][0], y=qq_orig[0][1],
                              mode='markers', name='Original Data'), row=1, col=1)
    fig4.add_trace(go.Scatter(x=qq_orig[0][0],
                              y=qq_orig[0][0] * qq_orig[1][0] + qq_orig[1][1],
                              mode='lines', name='Reference Line',
                              line=dict(color='red')), row=1, col=1)

    # Transformed Data QQ Plot
    qq_trans = stats.probplot(transformed_data)
    fig4.add_trace(go.Scatter(x=qq_trans[0][0], y=qq_trans[0][1],
                              mode='markers', name='Transformed Data'), row=1, col=2)
    fig4.add_trace(go.Scatter(x=qq_trans[0][0],
                              y=qq_trans[0][0] * qq_trans[1][0] + qq_trans[1][1],
                              mode='lines', name='Reference Line',
                              line=dict(color='red')), row=1, col=2)

    fig4.update_layout(height=500, template='plotly_white')

    # Create Box-Cox plot (only if Box-Cox transformation is selected)
    if transform_type == 'boxcox':
        lambdas = np.linspace(-2, 2, 100)
        llf = np.zeros(len(lambdas))

        for i, l in enumerate(lambdas):
            if l == 0:
                llf[i] = stats.normaltest(np.log(data - min(data) + 1))[0]
            else:
                llf[i] = stats.normaltest(((data - min(data) + 1) ** l - 1) / l)[0]

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=lambdas, y=llf, mode='markers', name='Probability', marker_symbol='cross'))
        fig5.add_vline(x=lambda_param, line_dash="dash", line_color="red")
        fig5.update_layout(
            height=400,
            title=f'Box-Cox Normality Plot (λ = {lambda_param:.4f})',
            xaxis_title='λ',
            yaxis_title='Prob Plot Corr Coef',
            template='plotly_white'
        )
        box_cox_style = {'display': 'block'}
    else:
        fig5 = go.Figure()
        box_cox_style = {'display': 'none'}

    return test_results, fig1, fig2, fig3, fig4, fig5, box_cox_style

# ============================== Visualization Layout ==============================

visualization_layout = html.Div([

    html.H3('Line Plot', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),
    # Line Plot Section
    html.Div([
        html.Div([
            html.Label('Select Features to Visualize in Line Plot:',
                       style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='feature-selector',
                        options=[
                            {'label': 'Sales', 'value': 'Sales'},
                            {'label': 'Profit', 'value': 'Profit'},
                            {'label': 'Quantity', 'value': 'Quantity'},
                            {'label': 'Discount', 'value': 'Discount'},
                            {'label': 'Shipping Cost', 'value': 'Shipping Cost'},
                            {'label': 'Unit Price', 'value': 'Unit Price'}
                        ],
                        value=['Sales'],
                        placeholder='Select x-axis features...',
                        multi=True,
                        style={'fontSize': '16px'}
                    ),
                ], className='six columns', style={'paddingRight': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='x-feature-selector',
                        options=[
                            {'label': 'Order Date', 'value': 'Order Date'},
                            {'label': 'Order Year', 'value': 'Order Year'},
                            {'label': 'Order Month', 'value': 'Order Month'},
                        ],
                        value='Order Date',
                        placeholder='Select Duration (Date/Month/Year)',
                        style={'fontSize': '16px', 'height': '36px'}
                    )
                ], className='six columns')
            ], className='row', style={'marginBottom': '30px'})
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-graph',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='time-series-plot', style={'height': '600px'})
                    ], className='chart-container')
                ]
            )
        ], className='twelve columns')
    ], className='row'),

    html.Br(),
    html.H3('Bar Plot', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),

    # Bar Plot Section
    html.Div([
        html.Div([
            html.Label('Select Features to Visualize in Bar Plot:',
                       style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='bar-feature-selector',
                        options=[
                            {'label': 'Market', 'value': 'Market'},
                            {'label': 'Category', 'value': 'Category'},
                            {'label': 'Sub-Category', 'value': 'Sub-Category'},
                            {'label': 'Segment', 'value': 'Segment'},
                            {'label': 'Ship Mode', 'value': 'Ship Mode'},
                            {'label': 'Region', 'value': 'Region'}
                        ],
                        value='Market',
                        placeholder='Select features...',
                        style={'fontSize': '16px'}
                    ),
                ], className='six columns', style={'paddingRight': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='bar-x-feature-selector',
                        options=[
                            {'label': 'Sales', 'value': 'Sales'},
                            {'label': 'Profit', 'value': 'Profit'},
                            {'label': 'Quantity', 'value': 'Quantity'},
                            {'label': 'Discount', 'value': 'Discount'},
                            {'label': 'Shipping Cost', 'value': 'Shipping Cost'},
                            {'label': 'Unit Price', 'value': 'Unit Price'}
                        ],
                        value='Sales',
                        placeholder='Select x-axis feature...',
                        style={'fontSize': '16px'}
                    )
                ], className='six columns')
            ], className='row', style={'marginBottom': '30px'})
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-bar-graph',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='bar-plot', style={'height': '600px'})
                    ], className='chart-container')
                ]
            )
        ], className='twelve columns')
    ], className='row'),

    # Stacked and Group Bar Plot Section
    html.Br(),
    html.H3('Stacked & Grouped Bar Plots',
            style={'color': TEXT_COLOR, 'font-weight': 'bold'}),

    # Interactive Bar Plot Controls
    html.Div([
        html.Div([
            html.Label('Select Plot Type:',
                       style={'fontSize': '18px', 'marginBottom': '10px',
                              'display': 'block'}),
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='plot-type-selector',
                        options=[
                            {'label': 'Stacked', 'value': 'stacked'},
                            {'label': 'Grouped', 'value': 'group'}
                        ],
                        value='stacked',
                        inline=True,
                        style={'fontSize': '16px'}
                    ),
                ], className='twelve columns')
            ], className='row', style={'marginBottom': '20px'})
        ], className='twelve columns'),

        html.Div([
            html.Label('Select Features:',
                       style={'fontSize': '18px', 'marginBottom': '10px',
                              'display': 'block'}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='primary-feature-selector',
                        options=[
                            {'label': 'Market', 'value': 'Market'},
                            {'label': 'Category', 'value': 'Category'},
                            {'label': 'Sub-Category', 'value': 'Sub-Category'},
                            {'label': 'Segment', 'value': 'Segment'},
                            {'label': 'Ship Mode', 'value': 'Ship Mode'},
                            {'label': 'Region', 'value': 'Region'},
                        ],
                        value='Category',
                        placeholder='Select primary grouping...',
                        style={'fontSize': '16px'}
                    ),
                ], className='four columns', style={'paddingRight': '10px'}),

                html.Div([
                    dcc.Dropdown(
                        id='secondary-feature-selector',
                        options=[
                            {'label': 'Market', 'value': 'Market'},
                            {'label': 'Category', 'value': 'Category'},
                            {'label': 'Sub-Category', 'value': 'Sub-Category'},
                            {'label': 'Segment', 'value': 'Segment'},
                            {'label': 'Ship Mode', 'value': 'Ship Mode'},
                            {'label': 'Region', 'value': 'Region'},
                        ],
                        value='Region',
                        placeholder='Select secondary grouping...',
                        style={'fontSize': '16px'}
                    ),
                ], className='four columns', style={'paddingRight': '10px'}),

                html.Div([
                    dcc.Dropdown(
                        id='value-feature-selector',
                        options=[
                            {'label': 'Sales', 'value': 'Sales'},
                            {'label': 'Profit', 'value': 'Profit'},
                            {'label': 'Quantity', 'value': 'Quantity'},
                            {'label': 'Discount', 'value': 'Discount'},
                            {'label': 'Shipping Cost', 'value': 'Shipping Cost'},
                        ],
                        value='Sales',
                        placeholder='Select value to measure...',
                        style={'fontSize': '16px'}
                    ),
                ], className='four columns')
            ], className='row', style={'marginBottom': '30px'})
        ], className='twelve columns')
    ], className='row'),

    # Plot Container
    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-stacked-bar',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='stacked-grouped-plot', style={'height': '600px'})
                    ], className='chart-container')
                ]
            )
        ], className='twelve columns')
    ], className='row'),

    # Pie Chart
    html.Br(),
    html.H3('Pie Chart', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),

    html.Div([
        html.Div([
            html.Label('Select Features to Visualize in Pie chart:',
                       style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='pie-feature-selector',
                        options=[
                            {'label': 'Market', 'value': 'Market'},
                            {'label': 'Category', 'value': 'Category'},
                            {'label': 'Sub-Category', 'value': 'Sub-Category'},
                            {'label': 'Segment', 'value': 'Segment'},
                            {'label': 'Ship Mode', 'value': 'Ship Mode'},
                            {'label': 'Region', 'value': 'Region'}
                        ],
                        value='Market',
                        placeholder='Select features...',
                        style={'fontSize': '16px'}
                    ),
                ], className='six columns', style={'paddingRight': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='pie-x-feature-selector',
                        options=[
                            {'label': 'Sales', 'value': 'Sales'},
                            {'label': 'Profit', 'value': 'Profit'},
                            {'label': 'Quantity', 'value': 'Quantity'},
                            {'label': 'Discount', 'value': 'Discount'},
                            {'label': 'Shipping Cost', 'value': 'Shipping Cost'},
                            {'label': 'Unit Price', 'value': 'Unit Price'}
                        ],
                        value='Sales',
                        placeholder='Select x-axis feature...',
                        style={'fontSize': '16px'}
                    )
                ], className='six columns')
            ], className='row', style={'marginBottom': '30px'})
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-pie-graph',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='pie-plot', style={'height': '600px'})
                    ], className='chart-container')
                ]
            )
        ], className='twelve columns')
    ], className='row'),

])

@app.callback(
Output('stacked-grouped-plot', 'figure'),
    [Input('plot-type-selector', 'value'),
     Input('primary-feature-selector', 'value'),
     Input('secondary-feature-selector', 'value'),
     Input('value-feature-selector', 'value')]

)
def update_stacked_grouped_plot(plot_type, primary_feature, secondary_feature, value_feature):
    grouped_data = df.groupby([primary_feature, secondary_feature])[value_feature].sum().unstack()

    if plot_type == 'stacked':
        fig = px.bar(grouped_data,
                     color=secondary_feature,
                     barmode='stack',
                     template='plotly_white',
                     text_auto='.2s',
                     color_discrete_sequence=px.colors.qualitative.Pastel1)
    else:
        fig = px.bar(grouped_data,
                     color=secondary_feature,
                     barmode='group',
                     template='plotly_white',
                     text_auto='.2s',
                     color_discrete_sequence=px.colors.qualitative.Pastel1)

    fig.update_layout(
        title={
            'text': f'{value_feature} by {primary_feature} and {secondary_feature}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Serif', color='blue', weight='bold')
        },
        xaxis_title={'text': primary_feature, 'font': dict(size=18, family='Serif', color='darkred')},
        yaxis_title={'text': value_feature, 'font': dict(size=18, family='Serif', color='darkred')},
        legend_title=secondary_feature,
        font=dict(size=16),
        showlegend=True,
        height=600)

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    return fig


@app.callback(
    Output('time-series-plot', 'figure'),
    Input('feature-selector', 'value'),
    Input('x-feature-selector', 'value'),
)

def update_time_series_plot(features, x_feature):
    fig = go.Figure()
    for feature in features:
        daily_avg = df.groupby(x_feature)[feature].mean().reset_index()
        fig.add_trace(go.Scatter(x=daily_avg[x_feature], y=daily_avg[feature], mode='lines', name=feature))

    fig.update_layout(
        title={
            'text': 'Time Series Line Plot',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Serif', color='blue', weight='bold')
        },
        xaxis_title={'text': x_feature, 'font': dict(size=18, family='Serif', color='darkred')},
        yaxis_title={'text': 'Value', 'font': dict(size=18, family='Serif', color='darkred')},
        font=dict(size=16),
        hovermode='x',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('bar-plot', 'figure'),
    Input('bar-feature-selector', 'value'),
    Input('bar-x-feature-selector', 'value')
)

def update_bar_plot(feature, x_feature):
    bar_data = df.groupby(feature)[x_feature].sum().sort_values(ascending=False)
    fig = px.bar(bar_data, x=bar_data.index, y=bar_data.values, color=bar_data.values,
                 color_continuous_scale=px.colors.qualitative.Pastel1)
    fig.update_layout(
        title={
            'text': 'Bar Plot for ' + feature + ' vs ' + x_feature,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Serif', color='blue', weight='bold')
        },
        xaxis_title={'text': feature, 'font': dict(size=18, family='Serif', color='darkred')},
        yaxis_title={'text': x_feature, 'font': dict(size=18, family='Serif', color='darkred')},
        font=dict(size=16),
        hovermode='x',
        template='plotly_white'
    )
    return fig

@app.callback(
Output('pie-plot', 'figure'),
    Input('pie-feature-selector', 'value'),
    Input('pie-x-feature-selector', 'value')
)

def update_pie_plot(feature, x_feature):
    pie_data = df.groupby(feature)[x_feature].sum().sort_values(ascending=False)
    fig = px.pie(pie_data, values=pie_data.values, names=pie_data.index,
                 color_discrete_sequence=px.colors.qualitative.Pastel1)
    fig.update_layout(
        title={
            'text': 'Pie Chart for ' + feature + ' vs ' + x_feature,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Serif', color='blue', weight='bold')
        },
        font=dict(size=16),
        template='plotly_white'
    )
    return fig

# ============================== Advance Visualization/ Visualizations pt 2 ==============================

NUMERICAL_FEATURES = [
    {'label': 'Sales', 'value': 'Sales'},
    {'label': 'Profit', 'value': 'Profit'},
    {'label': 'Quantity', 'value': 'Quantity'},
    {'label': 'Discount', 'value': 'Discount'},
    {'label': 'Shipping Cost', 'value': 'Shipping Cost'},
    {'label': 'Unit Price', 'value': 'Unit Price'}
]

CATEGORICAL_FEATURES = [
    {'label': 'Market', 'value': 'Market'},
    {'label': 'Category', 'value': 'Category'},
    {'label': 'Sub-Category', 'value': 'Sub-Category'},
    {'label': 'Segment', 'value': 'Segment'},
    {'label': 'Region', 'value': 'Region'},
    {'label': 'Ship Mode', 'value': 'Ship Mode'}
]

PLOTS_WITH_HUE = ['regplot', 'area', 'rug', 'strip', 'swarm', 'joint']

PLOT_REQUIREMENTS = {
    'pair': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select First Variable', 'Select Second Variable']},
    'heatmap': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select First Variable', 'Select Second Variable']},
    'regplot': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select X-axis (Independent)', 'Select Y-axis (Dependent)']},
    'area': {'feature1': CATEGORICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select Categories', 'Select Values']},
    'joint': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES,'labels': ['Select X Variable', 'Select Y Variable']},
    'rug': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select X Variable', 'Select Y Variable']},
    '3d': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select X-axis', 'Select Y-axis']},
    #'cluster': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES,'labels': ['Select First Variable', 'Select Second Variable']},
    'hexbin': {'feature1': NUMERICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select X Variable', 'Select Y Variable']},
    'strip': {'feature1': CATEGORICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select Categories', 'Select Values']},
    #'swarm': {'feature1': CATEGORICAL_FEATURES, 'feature2': NUMERICAL_FEATURES, 'labels': ['Select Categories', 'Select Values']}
}

visualization_layout_2 = html.Div([
    html.Br(),
    html.H3('Advanced Data Visualization', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),

    html.Div([
        html.Div([
            html.Label('Select Plot Type:', style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='plot-type-selector',
                options=[{'label': key.title(), 'value': key} for key in PLOT_REQUIREMENTS.keys()],
                value='heatmap',
                style={'fontSize': '16px'}
            )
        ], className='twelve columns', style={'marginBottom': '20px'})
    ], className='row'),

    html.Div([
        html.Div([
            html.Div([
                html.Label(id='feature1-label', style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='feature-selector-1',
                    style={'fontSize': '16px'}
                )
            ], className='four columns', style={'paddingRight': '10px'}),
            html.Div([
                html.Label(id='feature2-label', style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='feature-selector-2',
                    style={'fontSize': '16px'}
                )
            ], className='four columns', style={'paddingRight': '10px'}),
            html.Div([
                html.Label('Select Color Group (Hue):',
                           id='hue-label',
                           style={'fontSize': '18px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='hue-selector',
                    options=CATEGORICAL_FEATURES,
                    style={'fontSize': '16px'}
                )
            ], id='hue-container', className='four columns', style={'display': 'none'})
        ], className='row', style={'marginBottom': '30px'})
    ], className='row'),

    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-plot',
                type='cube',
                color=ACCENT_COLOR,
                children=[
                    html.Div([
                        dcc.Graph(id='main-plot', style={'height': '600px'})
                    ], className='chart-container')
                ]
            )
        ], className='twelve columns')
    ], className='row')
])


@app.callback(
    [Output('feature1-label', 'children'),
     Output('feature2-label', 'children'),
     Output('feature-selector-1', 'options'),
     Output('feature-selector-2', 'options'),
     Output('feature-selector-1', 'value'),
     Output('feature-selector-2', 'value'),
     Output('hue-container', 'style'),
     Output('hue-selector', 'value')],
    [Input('plot-type-selector', 'value')]
)
def update_dropdown_options(plot_type):
    if plot_type not in PLOT_REQUIREMENTS:
        return ['Select Feature 1', 'Select Feature 2', [], [], None, None, {'display': 'none'}, None]

    plot_config = PLOT_REQUIREMENTS[plot_type]

    default_value1 = plot_config['feature1'][0]['value'] if plot_config['feature1'] else None
    default_value2 = plot_config['feature2'][0]['value'] if plot_config['feature2'] else None

    hue_style = {'display': 'block'} if plot_type in PLOTS_WITH_HUE else {'display': 'none'}

    return [
        plot_config['labels'][0],
        plot_config['labels'][1],
        plot_config['feature1'],
        plot_config['feature2'],
        default_value1,
        default_value2,
        hue_style,
        None
    ]

@app.callback(
    Output('main-plot', 'figure'),
    [Input('plot-type-selector', 'value'),
     Input('feature-selector-1', 'value'),
     Input('feature-selector-2', 'value'),
     Input('hue-selector', 'value')]
)
def update_plot(plot_type, feature1, feature2, hue):
    if not all([plot_type, feature1, feature2]):
        return go.Figure()

    fig = go.Figure()
    df_num = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price']]
    sampled_df = df.sample(n=1000, random_state=5764)

    try:
        if plot_type == 'pair':
            fig = px.scatter_matrix(df, dimensions=['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price'],
                                color='Category',title='Pair Plot')
        elif plot_type == 'heatmap':
            corr_matrix = df_num.corr()
            fig = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"), text_auto=True,
                            color_continuous_scale=px.colors.qualitative.Pastel1,
                            title='Correlation Heatmap')

        elif plot_type == 'regplot':
            fig = px.scatter(df, x=feature1, y=feature2,
                             color=hue if hue else None,
                             trendline="ols", opacity=0.65, trendline_color_override='red',
                             labels={feature1: feature1, feature2: feature2},
                             title='Regression Plot with scatter representation and regression line')

        elif plot_type == 'area':
            fig = px.area(df, x=feature1, y=feature2,
                          color=hue if hue else None,
                          line_group=hue if hue else None, labels={feature1: feature1, feature2: feature2},
                          title='Area Plot')

        elif plot_type == 'joint':
            fig = px.scatter(df, x=feature1, y=feature2,
                             color=hue if hue else None,
                             marginal_x='histogram', marginal_y='histogram',
                             labels={feature1: feature1, feature2: feature2},
                             title='Joint Plot with KDE and scatter representation')

        elif plot_type == 'rug':
            fig = px.scatter(df, x=feature1, y=feature2,
                             color=hue if hue else None,
                             marginal_x='rug', marginal_y='rug',
                             labels={feature1: feature1, feature2: feature2},
                             title='Rug Plot')

        elif plot_type == '3d':
            feature1_data = np.linspace(df[feature1].min(), df[feature1].max(), 50)
            feature2_data = np.linspace(df[feature2].min(), df[feature2].max(), 50)
            X, Y = np.meshgrid(feature1_data, feature2_data)
            from scipy.interpolate import griddata
            Z = griddata((df[feature1], df[feature2]), df['Quantity'], (X, Y), method='nearest')
            Z = np.nan_to_num(Z)
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)], layout=go.Layout(title='3D Surface and Contour Plot'),
                            layout_scene=dict(xaxis_title=feature1, yaxis_title=feature2, zaxis_title='Quantity'))

            fig.update_traces(contours_z=dict(
                show=True, usecolormap=True,
                highlightcolor="limegreen",
                project_z=True))

        elif plot_type == 'cluster':
            sampled_df = df_num.sample(n=1000, random_state=5764)
            sns_fig = sns.clustermap(sampled_df,cmap='vlag',method='single',standard_scale=1)
            buf = io.BytesIO()
            sns_fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_base64}",
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                    layer="below"
                )
            )
            fig.update_layout(
                title="Cluster Map",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                width=1000,
                height=1000
            )

        elif plot_type == 'hexbin':
            fig = px.density_heatmap(df, x=feature1, y=feature2,
                                     marginal_x="histogram",
                                     marginal_y="histogram",
                                     nbinsx=20, nbinsy=20,
                                     color_continuous_scale=px.colors.sequential.Viridis,
                                     title='Hexbin Plot')

        elif plot_type == 'strip':
            fig = px.strip(df, x=feature1, y=feature2,
                           color=hue if hue else None, title='Strip Plot')

        elif plot_type == 'swarm':
            sns_fig = sns.swarmplot(data=sampled_df, x=feature1, y=feature2, hue=hue if hue else None)
            fig = sns_fig.get_figure()
            fig = tls.mpl_to_plotly(fig)
            fig.update_layout(title='Swarm Plot')

        else:
            fig.add_annotation(
                text="Plot type not implemented yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

    except Exception as e:
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Update layout
    fig.update_layout(
        template='plotly_white',
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Serif', color='blue', weight='bold')
        },
        margin=dict(t=100, l=50, r=50, b=50),
        xaxis_title={'font': dict(size=18, family='Serif', color='darkred')},
        yaxis_title={'font': dict(size=18, family='Serif', color='darkred')},
        height=800
    )

    return fig

# ============================== Subplots ==============================

subplot_layout = html.Div([
    html.H3('Subplot Analysis', style={'color': TEXT_COLOR, 'font-weight': 'bold'}),

    # Date Range Selector
    html.Div([
        html.H4('Select Date Range:', style={'fontSize': '16px', 'marginBottom': '10px'}),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=df['Order Date'].min(),
            max_date_allowed=df['Order Date'].max(),
            start_date=df['Order Date'].min(),
            end_date=df['Order Date'].max(),
            style={'marginBottom': '20px'}
        ),
    ]),

    # Four main sections with loading states
    dcc.Loading(
        id='loading-pie-charts',
        type='cube',
        color=ACCENT_COLOR,
        children=[html.Div(dcc.Graph(id='pie-charts'))]
    ),

    dcc.Loading(
        id='loading-category-charts',
        type='cube',
        color=ACCENT_COLOR,
        children=[html.Div(dcc.Graph(id='category-charts'))]
    ),

    dcc.Loading(
        id='loading-profitability-charts',
        type='cube',
        color=ACCENT_COLOR,
        children=[html.Div(dcc.Graph(id='profitability-charts'))]
    ),

    dcc.Loading(
        id='loading-performance-charts',
        type='cube',
        color=ACCENT_COLOR,
        children=[html.Div(dcc.Graph(id='performance-charts'))]
    )
])


@app.callback(
    Output('pie-charts', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_pie_charts(start_date, end_date):
    filtered_df = df[(df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Distribution by Category',
                        'Profit Distribution by Segment',
                        'Order Distribution by Region',
                        'Shipping Cost Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}],
               [{'type': 'pie'}, {'type': 'pie'}]]
    )

    # Revenue Distribution
    category_sales = filtered_df.groupby('Category')['Sales'].sum()
    fig.add_trace(
        go.Pie(values=category_sales.values,
               labels=category_sales.index,
               name='Revenue',
               marker_colors=px.colors.qualitative.Pastel1),
        row=1, col=1
    )

    # Profit Distribution
    segment_profit = filtered_df.groupby('Segment')['Profit'].sum()
    fig.add_trace(
        go.Pie(values=segment_profit.values,
               labels=segment_profit.index,
               name='Profit',
               hole=0.3,
               marker_colors=px.colors.qualitative.Pastel1),
        row=1, col=2
    )

    # Regional Distribution
    region_orders = filtered_df.groupby('Region').size()
    fig.add_trace(
        go.Pie(values=region_orders.values,
               labels=region_orders.index,
               name='Orders',
               marker_colors=px.colors.qualitative.Pastel1),
        row=2, col=1
    )

    # Shipping Analysis
    ship_mode = filtered_df.groupby('Ship Mode')['Shipping Cost'].sum()
    fig.add_trace(
        go.Pie(values=ship_mode.values,
               labels=ship_mode.index,
               name='Shipping',
               hole=0.3,
               marker_colors=px.colors.qualitative.Pastel1),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text=f"Business Performance Overview ({start_date} to {end_date})",
        title_x=0.5,
        title_font_size=20,
        title_font_color='blue'
    )

    return fig


@app.callback(
    Output('category-charts', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_category_charts(start_date, end_date):
    filtered_df = df[(df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales by Category and Year',
                        'Profit by Category and Year',
                        'Sales Trend by Category',
                        'Profit Trend by Category')
    )

    # Sales by Category
    grouped_sales = filtered_df.groupby(['Category', 'Order Year'])['Sales'].sum().unstack()
    for year in grouped_sales.columns:
        fig.add_trace(
            go.Bar(x=grouped_sales.index,
                   y=grouped_sales[year],
                   name=f'Sales {year}'),
            row=1, col=1
        )

    # Profit by Category
    grouped_profit = filtered_df.groupby(['Category', 'Order Year'])['Profit'].sum().unstack()
    for year in grouped_profit.columns:
        fig.add_trace(
            go.Bar(x=grouped_profit.index,
                   y=grouped_profit[year],
                   name=f'Profit {year}'),
            row=1, col=2
        )

    # Sales Trend
    monthly_sales = filtered_df.groupby(['Category', 'Order Month'])['Sales'].sum().unstack()
    for category in monthly_sales.index:
        fig.add_trace(
            go.Scatter(x=monthly_sales.columns,
                       y=monthly_sales.loc[category],
                       name=f'{category} Sales',
                       mode='lines+markers'),
            row=2, col=1
        )

    # Profit Trend
    monthly_profit = filtered_df.groupby(['Category', 'Order Month'])['Profit'].sum().unstack()
    for category in monthly_profit.index:
        fig.add_trace(
            go.Scatter(x=monthly_profit.columns,
                       y=monthly_profit.loc[category],
                       name=f'{category} Profit',
                       mode='lines+markers'),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        title_text=f"Category Performance Analysis ({start_date} to {end_date})",
        title_x=0.5,
        title_font_size=20,
        title_font_color='blue'
    )

    return fig


@app.callback(
    Output('profitability-charts', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_profitability_charts(start_date, end_date):
    filtered_df = df[(df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Profit per Order',
                        'Impact of Discounts',
                        'Regional Performance',
                        'Shipping Cost Analysis')
    )

    # Average Profit Heatmap
    avg_profit = filtered_df.groupby(['Category', 'Segment'])['Profit'].mean().unstack()
    fig.add_trace(
        go.Heatmap(z=avg_profit.values,
                   x=avg_profit.columns,
                   y=avg_profit.index,
                   colorscale='RdYlGn'),
        row=1, col=1
    )

    # Discount Impact
    filtered_df['Discount_Band'] = pd.cut(filtered_df['Discount'],
                                          bins=[-np.inf, 0, 0.2, 0.4, np.inf],
                                          labels=['No Discount', 'Low', 'Medium', 'High'])
    discount_impact = filtered_df.groupby('Discount_Band').agg({
        'Profit': 'mean',
        'Sales': 'mean'
    }).reset_index()

    fig.add_trace(
        go.Bar(x=discount_impact['Discount_Band'],
               y=discount_impact['Sales'],
               name='Avg Sales'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=discount_impact['Discount_Band'],
                   y=discount_impact['Profit'],
                   name='Avg Profit',
                   mode='lines+markers'),
        row=1, col=2
    )

    # Regional Performance
    regional_metrics = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()

    fig.add_trace(
        go.Scatter(x=regional_metrics['Sales'],
                   y=regional_metrics['Profit'],
                   mode='markers+text',
                   text=regional_metrics['Region'],
                   marker=dict(size=regional_metrics['Order ID'] / 50),
                   name='Regional Performance'),
        row=2, col=1
    )

    # Shipping Analysis
    ship_metrics = filtered_df.groupby(['Ship Mode', 'Order Priority']).agg({
        'Shipping Cost': 'mean',
        'Order ID': 'count'
    }).reset_index()

    fig.add_trace(
        go.Scatter(x=ship_metrics['Shipping Cost'],
                   y=ship_metrics['Order ID'],
                   mode='markers',
                   marker=dict(size=12),
                   text=ship_metrics['Ship Mode'] + ' - ' + ship_metrics['Order Priority'],
                   name='Shipping Analysis'),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text=f"Profitability Insights ({start_date} to {end_date})",
        title_x=0.5,
        title_font_size=20,
        title_font_color='blue'
    )

    return fig


@app.callback(
    Output('performance-charts', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_performance_charts(start_date, end_date):
    filtered_df = df[(df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Category Sales Distribution by Segment',
                        'Regional Sales by Shipping Mode',
                        'Order Priority Analysis',
                        'Monthly Sales Trend')
    )

    # Category-Segment Distribution
    segment_category = pd.crosstab(
        filtered_df['Category'],
        filtered_df['Segment'],
        values=filtered_df['Sales'],
        aggfunc='sum',
        normalize='index'
    )

    fig.add_trace(
        go.Heatmap(z=segment_category.values,
                   x=segment_category.columns,
                   y=segment_category.index,
                   colorscale='YlOrRd',
                   text=np.round(segment_category.values * 100, 1),
                   texttemplate='%{text}%'),
        row=1, col=1
    )

    # Regional Shipping
    region_ship = pd.crosstab(
        filtered_df['Region'],
        filtered_df['Ship Mode'],
        values=filtered_df['Sales'],
        aggfunc='sum'
    )

    for ship_mode in region_ship.columns:
        fig.add_trace(
            go.Bar(name=ship_mode,
                   x=region_ship.index,
                   y=region_ship[ship_mode]),
            row=1, col=2
        )

    # Order Priority Analysis
    priority_metrics = filtered_df.groupby('Order Priority').agg({
        'Shipping Cost': 'mean',
        'Sales': 'mean'
    }).reset_index()

    fig.add_trace(
        go.Bar(x=priority_metrics['Order Priority'],
               y=priority_metrics['Shipping Cost'],
               name='Avg Shipping Cost'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=priority_metrics['Order Priority'],
               y=priority_metrics['Sales'],
               name='Avg Sales'),
        row=2, col=1
    )

    # Monthly Trend
    monthly_category = filtered_df.pivot_table(
        index='Order Month',
        columns='Category',
        values='Sales',
        aggfunc='sum'
    )

    for category in monthly_category.columns:
        fig.add_trace(
            go.Scatter(x=monthly_category.index,
                       y=monthly_category[category],
                       name=category,
                       mode='lines+markers'),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        title_text=f"Performance Story ({start_date} to {end_date})",
        title_x=0.5,
        title_font_size=20,
        title_font_color='blue',
        showlegend=True
    )

    return fig

# ============================== Run the app ==============================

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Global Superstore Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #F8FAFC;
                margin: 0;
                font-family: "Segoe UI", -apple-system, sans-serif;
                min-height: 100vh;
            }

            .header {
                background-color: white;
                padding: 0 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
            }

            .content {
                margin-top: 180px;
                padding: 20px;
            }

            .welcome-section {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 30px;
            }

            .metric-box {
                background: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.2s;
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }

            .metric-box:hover {
                transform: translateY(-5px);
            }

            .metric-box i {
                color: ''' + PRIMARY_COLOR + ''';
                margin-bottom: 15px;
            }

            .chart-container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            .custom-tabs {
                margin-bottom: 20px;
            }

            .custom-tab {
                padding: 20px 30px !important;
                font-size: 18px !important;
                color: #666 !important;
                border: none !important;
                border-bottom: 3px solid transparent !important;
                background-color: transparent !important;
                transition: all 0.3s;
            }

            .custom-tab:hover {
                color: ''' + TEXT_COLOR + ''' !important;
                background-color: ''' + PRIMARY_COLOR + '''40 !important;
            }

            .custom-tab--selected {
                color: ''' + TEXT_COLOR + ''' !important;
                border-bottom: 3px solid ''' + PRIMARY_COLOR + ''' !important;
                font-weight: bold !important;
            }
            
            .dash-dropdown {
                border: 1px solid ''' + PRIMARY_COLOR + ''' !important;
                border-radius: 8px !important;
                background-color: white !important;
            }
    
            .dash-dropdown:hover {
                border-color: ''' + ACCENT_COLOR + ''' !important;
            }
            
            .Select-control {
                border: none !important;
                padding: 8px !important;
            }
            
            .Select-placeholder, .Select-input, .Select-value {
                padding: 8px !important;
            }
            
            .Select-menu-outer {
                border: 1px solid ''' + PRIMARY_COLOR + ''' !important;
                border-radius: 8px !important;
                margin-top: 4px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            
            .Select-option {
                padding: 12px !important;
                font-size: 16px !important;
            }
            
            .Select-option.is-selected {
                background-color: ''' + PRIMARY_COLOR + ''' !important;
            }
            
            .Select-option.is-focused {
                background-color: ''' + PRIMARY_COLOR + '''40 !important;
            }

            @media (max-width: 768px) {
                .content {
                    margin-top: 220px;
                }

                .metric-box {
                    height: auto;
                    padding: 20px;
                }

                .custom-tab {
                    padding: 15px !important;
                    font-size: 16px !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@app.callback(
    Output('layout', 'children'),
    Input('tabs', 'value')
)
def update_layout(tab):
    if tab == 'about':
        return about_layout
    elif tab == 'data':
        return data_layout
    elif tab == 'outlier':
        return outlier_layout
    elif tab == 'normality':
        return normality_layout
    elif tab == 'pca':
        return pca_layout
    elif tab == 'visualization':
        return visualization_layout
    elif tab == 'visualization_2':
        return visualization_layout_2
    elif tab == 'subplots':
        return subplot_layout

app.run_server(
port=8051,
host='0.0.0.0'
)