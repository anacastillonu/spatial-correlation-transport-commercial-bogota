# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:33:06 2020
@author: anacastillo


POR HACER:  All commercial in Hotspots
            Not significant combinations

"""
#-----------------------------------------------------------
# IMPORT LIBRARIES
#-----------------------------------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import geopandas as gpd
import json
import os
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

from matplotlib import cm
import matplotlib



#-----------------------------------------------------------
#LOAD DATA
#------------------------------------------------------------
localidades = ['ANTONIO NARIÑO','TUNJUELITO','RAFAEL URIBE URIBE','CANDELARIA','BARRIOS UNIDOS','TEUSAQUILLO','PUENTE ARANDA','LOS MARTIRES','USAQUEN','CHAPINERO','SANTA FE','SAN CRISTOBAL','USME','CIUDAD BOLIVAR','BOSA','KENNEDY','FONTIBON','ENGATIVA','SUBA']

transporte = ['Transmilenio', 'SITP', 'P.Vehiches',
       'Walking', 'Cycling']


comercial = ['Wholesale', 'Used Goods', 'Retail', 'Restaurants, Cafés and Bars',
       'Personal Care', 'Cultural and Entertainment',
       'Banking and Post Offices', 'Commercial Activity']



data = pd.read_csv('data\\grid_data.csv')
data['co'] = data.apply(lambda x: int(str(x['fico_id'])[(str(x['fico_id']).find('_')+1):]), axis=1)
data['fi'] = data.apply(lambda x: int(str(x['fico_id'])[:(str(x['fico_id']).find('_'))])*(-1), axis=1)

#____________________
#KDE COLORS
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

rgb_a = []
for value in range(0,11):
    rgb_a.append('rgba'+str(cm.viridis_r(norm(value),bytes=True)[:3])[:-1]+', 1)')

rgb_b = []
for value in range(0,11):
    rgb_b.append('rgba'+str(cm.viridis_r(norm(value),bytes=True)[:3])[:-1]+', 0.7)')


#______________________
#SHAPEFILES

loc_shp = gpd.read_file('data\\loc_mod.shp')
loc_json = json.loads(loc_shp.set_index('LocNombre').to_json())

loc_corrs = pd.read_csv('data\\corr_localidades.csv', encoding = 'latin1')


#_________________________________________
# GRID SHAPE
grid = gpd.read_file('data\\bogogrid_blank.shp')
grid.set_index('fico_id',inplace=True)
grid.columns = ['upz', 'Localidad', 'geometry']

centroides_localidades = pd.read_csv('centroides_localidades.csv')

#________________________________________
all_hot = gpd.read_file('data\\all_hot.shp')
all_kde = gpd.read_file('data\\kde.shp').set_crs(epsg=3116).to_crs(epsg=4326)

#--------------------------------------------------------------
    # DASH APP
#--------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([

html.Div([


    #TITULO DASH
    html.Div([
    html.H1('Spatial Correlation Between Transport Networks and Commercial Activities in Bogotá, Colombia',
            style={'height': '100px',
                   'width': '600px',
                    'color': 'midnightblue',
                    'font-family': 'verdana',
                   'font-size': "210%",
                   'display': 'inline-block'}),

    html.H6('The spatial correlation was determined through the evaulation of the Kernel Density Estimation for each of the variables. Such density was overlayed in a 50m by 50m grid (+150,000 cells) and the pearson correlation was estimated for the values on a cell-by-cell basis.',
            style={'height': '80px',
                   'width': '600px',
                   'color': 'lightslategray',
                   'font-family': 'verdana',
                   'font-size': "100%",
                   'display': 'inline-block'}),

    ],style={'height': '250px',
             'width': '600px',
             'display': 'inline-block'}),



    #Opciones
    html.Div([
        html.Div([

            html.Button('UPDATE ALL',id='go',
                        style={'height': '40px',
                               'width': '600px',
                               'font-size': "100%",
                               'display': 'inline-block'}),
            html.P('LOCALIDAD',
                   style ={'height': '25px',
                           'width': '600px',
                           'font-size': "100%"}),

            dcc.Dropdown(id='area_option',
                        options=[{'label': i, 'value': i} for i in localidades],
                        value='CHAPINERO',
                        style={'height': '40px',
                               'width': '600px',
                               'font-size': "100%"}),

                    ],style = {'height': '115px',
                             'width': '600px',
                             'display': 'inline-block'}),

        html.Div([
            html.Div([
                html.P('COMMERCIAL ACTIVITY',
                       style={'height': '20px',
                              'width': '600px',
                               'font-size': "100%"}),

                 dcc.Dropdown(id='commercial_activity_option',
                              options=[{'label': i, 'value': i} for i in comercial],
                              value='Restaurants, Cafés and Bars',
                              style={'height': '40px',
                                     'width': '300px',
                                     'font-size': "100%",
                                     'display': 'inline-block'}),

                        ],style={'height': '70px',
                                 'width': '300px',
                                 'display': 'inline-block'}), #comercial


            html.Div([
                html.P('TRANSPORT NETWORK',
                       style={'height': '20px',
                              'width': '600px',
                               'font-size': "100%"}),

                 dcc.Dropdown(id='transport_network_option',
                              options=[{'label': i, 'value': i} for i in transporte],
                              value='Walking',
                              style={'height': '40px',
                                     'width': '300px',
                                     'font-size': "100%",
                                     'display': 'inline-block'}),

                        ],style={'height': '70px',
                                      'width': '300px',
                                      'display': 'inline-block'}), #Transporte


                    ],style={'height': '70px',
                             'width': '600px',
                             'display': 'inline-block'}), #Transp y Comm opciones

    ],style={'height': '185px',
             'width': '600px',
             'display': 'inline-block'}),


    #html.Br(),
    dcc.Markdown(id='annotated_enunciado',
                    children = 'How does the correlation for different transport networks and commercial activities compare within **BARRIOS UNIDOS**',
                    style={'height': '50px',
                           'width': '600px',
                           'color': 'midnightblue',
                           'font-family': 'verdana',
                           'font-size': "100%",
                           'display': 'inline-block'}),

   # html.Br(),
   dcc.Markdown(id='ploycorr_enunciado',
                   children = 'How does the spatial correation between **Restaurants, Cafés and Bars** and **Walking** compare for different Localidades',
                   style={'height': '50px',
                           'width': '600px',
                           'color': 'midnightblue',
                           'font-family': 'verdana',
                           'font-size': "100%",
                           'display': 'inline-block'}),


   dcc.Graph(id='anot_heat',
              style={'display': 'inline-block'}),

   dcc.Graph(id='polygon_pcorrs',
                      style={'display': 'inline-block'}),

   #html.Br(),
    dcc.Markdown(id='kde_enunciado',
                    children = 'Where does higher density of **Walking** and **Restaurants, Cafés and Bars** locate?',
                    style={'height': '50px',
                           'width': '600px',
                           'color': 'midnightblue',
                           'font-family': 'verdana',
                           'font-size': "100%",
                           'display': 'inline-block'}),

    #html.Br(),
    dcc.Markdown(id='scat_enunciado',
                    children = 'How does the correlation plot look like for **Walking** and **Restaurants, Cafés and Bars** in **BARRIOS UNIDOS**?',
                    style={'height': '50px',
                           'width': '600px',
                           'color': 'midnightblue',
                           'font-family': 'verdana',
                           'font-size': "100%",
                           'display': 'inline-block'}),


    dcc.Graph(id='kdet',
             style={'display': 'inline-block'}),

    dcc.Graph(id='kdec',
              style={'display': 'inline-block'}),


    dcc.Graph(id='scat',
             style={'display': 'inline-block'}),


    html.Div([
    dcc.Markdown(id='heat_enunciado',
                 children = 'Where do HOTSPOTS of **Walking** and **Restaurants, Cafés and Bars** coincide?',
                  style={'height': '80px',
                           'width': '600px',
                           'color': 'midnightblue',
                           'font-family': 'verdana',
                           'font-size': "100%",
                           'display': 'inline-block'}),

    html.Div([
        dcc.Markdown(id='heat_transp',
                     children = '**Walking**',
                      style={'height': '40px',
                               'width': '600px',
                               'color': 'green',
                               'font-family': 'verdana',
                               'font-size': "90%",
                               'text-align': 'right',
                               'display': 'inline-block'
                               }),

        dcc.Markdown(id='heat_comm',
                     children = '**Restaurants, Cafés and Bars**',
                      style={'height': '40px',
                               'width': '600px',
                               'color': 'yellow',
                               'font-family': 'verdana',
                               'font-size': "90%",
                               'text-align': 'right',
                               'display': 'inline-block'
                               }),

        ], style={'height': '80px',
                   'width': '600px',
                    'font-size': "100%",
                     'display': 'inline-block'}),

     ], style={'height': '80px',
               'width': '1200px',
                'font-size': "100%"}),

     dcc.Graph(id='hot',
               style={'display': 'inline-block'})


 ],style = {'display': 'inline-block'})
                         ])
#--------------------------------------------------------------------
# FUNCTIONS
#--------------------------------------------------------------------

@app.callback(
    Output('heat_transp','children'),
     [Input('go', 'n_clicks')],
     [ State('transport_network_option', 'value')])
def heat_transp0(n_clicks,transp):

    title = '**' + transp + '**'

    return title

@app.callback(
    Output('heat_comm','children'),
     [Input('go', 'n_clicks')],
     [State('commercial_activity_option', 'value')])
def heat_comme(n_clicks, comm):

    title = '**' + comm + '**'

    return title

@app.callback(
    Output('ploycorr_enunciado','children'),
     [Input('go', 'n_clicks')],
     [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value')])
def polycorr_title(n_clicks, comm, transp):

    title = 'How does the spatial correation between **' + comm + '** and **' + transp + '** compare for different Localidades?'

    return title

@app.callback(
    Output('annotated_enunciado','children'),
     [Input('go', 'n_clicks')],
     [State('area_option', 'value')])
def annotated_title(n_clicks, loca):


    title = 'How does the correlation for different transport networks and commercial activities compare within **' +loca+ '**'


    return title

@app.callback(
    Output('kde_enunciado','children'),
     [Input('go', 'n_clicks')],
     [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value')])
def kde_title(n_clicks, comm, transp):

    title = 'Where does higher density of **'+comm+'** and **'+comm+'** locate?'

    return title

@app.callback(
    Output('scat_enunciado','children'),
     [Input('go', 'n_clicks')],
     [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value'),
     State('area_option', 'value')])
def scat_title(n_clicks, comm, transp,loca):

    title = 'How does the correlation plot look like for **'+comm+'** and **'+comm+'** in **' + loca + '**?'

    return title

@app.callback(
    Output('heat_enunciado','children'),
     [Input('go', 'n_clicks')],
     [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value')])
def heat_title(n_clicks, comm, transp):


    title = 'Where do HOTSPOTS of **' + transp + '** and **' + comm + '** coincide?'
    return title


@app.callback(
    Output('anot_heat','figure'),
    [Input('go', 'n_clicks')],
    [State('area_option', 'value')])
#geoarea = 'CANDELARIA'
def update_annotated(n_clicks, geoarea):

    corrs = loc_corrs[loc_corrs['Localidad']==geoarea]
    corrs['pcorr'] = corrs.apply(lambda x: np.nan if x['pval']>=0.001 else round(x['pcorr'],3), axis = 1)
    corrs = corrs[['Commercial','Transport','pcorr']].set_index(['Commercial','Transport']).unstack(level = 'Commercial')


    z = corrs.values

    x = ['ALL', 'Bank. and Post', 'Cult. and Ent.',
         'Pers. Care','Rest., Caf., B.','Retail',
         'Used Goods', 'Wholesale']

    y = ['Cycling','P.Vehicles',
           'SITP', 'Transmi', 'Walking']


    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=corrs['pcorr'].values, colorscale='Spectral')

    fig.update_layout(
        title = 'Spatial correlation between transport networks and commercial activity in ' + geoarea,
        font=dict(
            size = 8),
        width=600,
        height=600,
        margin=dict(l=10, r=10, t=50, b=10
                    , pad=5),
        plot_bgcolor='rgb(255,255,255)')

    fig.update_yaxes(tickangle = -90)
    fig.update_xaxes(tickangle = 0)

    #plot(fig)
    return fig



@app.callback(
    Output('polygon_pcorrs','figure'),
    [Input('go', 'n_clicks')],
    [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value'),
     State('area_option', 'value')])

def update_polygons(n_clicks,comm,trans,area):


    """
    comm = 'Wholesale'
    trans = 'Transmilenio'
    area = 'BARRIOS UNIDOS'
    """

    varjson = loc_json
    corrs_filtered = loc_corrs[(loc_corrs['Commercial']==comm) & (loc_corrs['Transport']==trans) & (loc_corrs['pval']<0.001)]

    highlight = loc_shp[loc_shp['LocNombre']==area]
    highlight = json.loads(highlight.set_index('LocNombre').to_json())

    poly = go.Figure()


    poly.add_trace(go.Choroplethmapbox(geojson=varjson, locations = corrs_filtered['Localidad'],
                                      z = corrs_filtered['pcorr'],
                                      colorscale='Spectral', zmin=-1, zmax=1,
                                      marker_opacity=0.3, marker_line_width=0.1,
                                      colorbar = dict(thickness = 13,
                                                      x=-0.1,
                                                      xanchor = 'left',
                                                      lenmode='fraction',
                                                      len= 0.8,
                                                       outlinewidth =0,
                                                       tickfont = dict(size=10),
                                                       nticks = 6,
                                                       title = dict (text='Pearson Correlation',
                                                                     side = 'right',
                                                                     font = dict(size=12)
                                                                     )
                                                       )
                                                ))
    try:
        poly.add_trace(go.Choroplethmapbox(geojson=highlight, locations = corrs_filtered['Localidad'],
                                          z = corrs_filtered['pcorr'],
                                          colorscale='Spectral', zmin=-1, zmax=1,
                                          marker_opacity=0.9, marker_line_width=0.5,
                                          colorbar = dict(thickness = 13,
                                                          x=-0.1,
                                                          xanchor = 'left',
                                                           lenmode='fraction',
                                                           len= 0.8,
                                                           outlinewidth =0,
                                                           tickfont = dict(size=10),
                                                           nticks = 6,
                                                           title = dict (text='Pearson Correlation',
                                                                         side = 'right',
                                                                         font = dict(size=12)
                                                                         )
                                                           )
                                                    ))

    except:pass

    poly.update_layout(

    title =dict (text = 'Pearson correlation between ' + comm +' and ' + trans,
                 font=dict(size=10)

        ),

        font=dict(
            size = 7),
        autosize=True,
        width=600, height=600,
        margin=dict(l=10, r=10, t=20, b=10
                    , pad=5),

        mapbox=dict(
            style = "carto-positron",
            bearing=0,
            center=dict(
                lat=4.64,
                lon=-74.12
            ),
            pitch=0,
            zoom=10))
    #plot(poly)

    return poly


@app.callback(
    Output('kdet', 'figure'),
    [Input('go', 'n_clicks')],
    [State('transport_network_option', 'value'),
     State('area_option', 'value')
     ])


def update_kdet(n_clicks, transp, loc):


    var = 'q_'+transp

    lon_centr = centroides_localidades[centroides_localidades['Localidad']==loc]['xwgs84'].values[0]
    lat_centr = centroides_localidades[centroides_localidades['Localidad']==loc]['ywgs84'].values[0]

    poly_area=all_kde[all_kde['Localidad']==loc][['id',var[:10],'geometry']].dropna()
    jsonarea = json.loads(poly_area.set_index('id').to_json())

    poly_rest=all_kde[all_kde['Localidad']!=loc][['id',var[:10],'geometry']].dissolve(by=var[:10]).reset_index()
    jsonrest = json.loads(poly_rest.set_index('id').to_json())


    """

    var = 'q_SITP'
    loc = 'BARRIOS UNIDOS'

    """



    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(geojson=jsonarea, locations=poly_area['id'], z=poly_area[var[:10]],
                                        colorscale=rgb_a, zmin=0, zmax=10,
                                        marker_opacity=0.7, marker_line_width=0.1,
                                        hoverinfo='skip',
                                        colorbar = dict(thickness = 7,
                                                lenmode='fraction',
                                                len= 0.8,
                                                outlinewidth =0,
                                                tickfont = dict(size=10),
                                                nticks = 6,
                                                title = dict (text='Quantiles',
                                                              side = 'right',
                                                              font = dict(size=12))
                                                )))

    fig.add_trace(go.Choroplethmapbox(geojson=jsonrest, locations=poly_rest['id'], z=poly_rest[var[:10]],
                                        colorscale=rgb_b, zmin=0, zmax=10,
                                        marker_opacity=0.3, marker_line_width=0,
                                        hoverinfo='skip',
                                        colorbar = dict(thickness = 7,
                                                lenmode='fraction',
                                                len= 0.8,
                                                outlinewidth =0,
                                                tickfont = dict(size=10),
                                                nticks = 6,
                                                title = dict (text='',
                                                              side = 'right',
                                                              font = dict(size=12))
                                                )))

    fig.update_layout(

        title = 'KDE: ' + var[2:],
        font=dict(
            size = 9),
        autosize=True,
        width=300, height=600,
        margin=dict(l=10, r=10, t=20, b=10
                    , pad=5),

        mapbox=dict(
            style = "carto-positron",
            bearing=0,
            center=dict(
                lat=lat_centr,
                lon=lon_centr
            ),
            pitch=0,
            zoom=12),

        annotations = [dict(
            showarrow=False,
            #text='0 corresponds to KDE = 0; 1-10 = Quantiles when KDE > 0',
            xanchor='right',
            x=1,
            yanchor='top',
            y=0.02
            )]

    )
    #plot(fig)
    return fig

@app.callback(
    Output('kdec', 'figure'),
    [Input('go', 'n_clicks')],
    [State('commercial_activity_option', 'value'),
     State('area_option', 'value')
     ])

def update_kdec(n_clicks, comm, loc):


    var = 'q_'+comm

    lon_centr = centroides_localidades[centroides_localidades['Localidad']==loc]['xwgs84'].values[0]
    lat_centr = centroides_localidades[centroides_localidades['Localidad']==loc]['ywgs84'].values[0]

    poly_area=all_kde[all_kde['Localidad']==loc][['id',var[:10],'geometry']].dropna()
    jsonarea = json.loads(poly_area.set_index('id').to_json())

    poly_rest=all_kde[all_kde['Localidad']!=loc][['id',var[:10],'geometry']].dissolve(by=var[:10]).reset_index()
    jsonrest = json.loads(poly_rest.set_index('id').to_json())

    """

    var = 'q_Retail'
    loc = 'BARRIOS UNIDOS'

    """



    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(geojson=jsonarea, locations=poly_area['id'], z=poly_area[var[:10]],
                                        colorscale=rgb_a, zmin=0, zmax=10,
                                        marker_opacity=0.7, marker_line_width=0.1,
                                        hoverinfo='skip',
                                        colorbar = dict(thickness = 7,
                                                lenmode='fraction',
                                                len= 0.8,
                                                outlinewidth =0,
                                                tickfont = dict(size=10),
                                                nticks = 6,
                                                title = dict (text='Quantiles',
                                                              side = 'right',
                                                              font = dict(size=12))
                                                )))

    fig.add_trace(go.Choroplethmapbox(geojson=jsonrest, locations=poly_rest['id'], z=poly_rest[var[:10]],
                                        colorscale=rgb_b, zmin=0, zmax=10,
                                        marker_opacity=0.3, marker_line_width=0,
                                        hoverinfo='skip',
                                        colorbar = dict(thickness = 7,
                                                lenmode='fraction',
                                                len= 0.8,
                                                outlinewidth =0,
                                                tickfont = dict(size=10),
                                                nticks = 6,
                                                title = dict (text='',
                                                              side = 'right',
                                                              font = dict(size=12))
                                                )))

    fig.update_layout(

        title = 'KDE: ' + var[2:],
        font=dict(
            size = 9),
        autosize=True,
        width=300, height=600,
        margin=dict(l=10, r=10, t=20, b=10
                    , pad=5),

        mapbox=dict(
            style = "carto-positron",
            bearing=0,
            center=dict(
                lat=lat_centr,
                lon=lon_centr
            ),
            pitch=0,
            zoom=12),

        annotations = [dict(
            showarrow=False,
            #text='0 corresponds to KDE = 0; 1-10 = Quantiles when KDE > 0',
            xanchor='right',
            x=1,
            yanchor='top',
            y=0.02
            )]

    )
    #plot(fig)
    return fig

@app.callback(
    Output('scat', 'figure'),
    [Input('go', 'n_clicks')],
    [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value'),
     State('area_option', 'value')])

def update_scat(n_clicks, scat_comm, scat_transp, area):

    """
    area = 'BARRIOS UNIDOS'
    scat_comm='Retail'
    scat_transp = 'SITP'
    """

    x_base = data[data['Localidad']!=area][scat_comm].values
    y_base = data[data['Localidad']!=area][scat_transp].values


    x_highlight = data[data['Localidad']==area][scat_comm].values
    y_highlight = data[data['Localidad']==area][scat_transp].values

    scat = go.Figure()


    scat.add_trace(go.Scattergl(x=x_base, y=y_base,
                          name = 'Bogotá',
                                mode='markers',
                          marker=dict(color='grey',
                                      opacity=0.2,
                                      size=2,
                                      line= dict(color='black',
                                                 width=0.1)
                                      ),
                          hoverinfo = 'skip'
                              ))

    scat.add_trace(go.Scattergl(x=x_highlight, y=y_highlight,
                                name = area,
                          mode='markers',
                          marker=dict(color='green',
                                      opacity=0.5,
                                      size=4,
                                      line= dict(color='darkgreen',
                                                 width=1)
                                      ),
                          hoverinfo = 'skip'))


    scat.update_layout(
        margin=dict(l=10, r=10, t=20, b=10
                    , pad=5),
        title = dict(text= 'Spatial Correlation between: '+scat_transp + ' and ' +scat_comm,
                     font= dict(size = 8)),
        font=dict( size = 9),
        showlegend=True,
        width=600, height=600,
        paper_bgcolor="white", plot_bgcolor='white',
        xaxis_title=scat_comm, yaxis_title=scat_transp,
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color="black"
        )))

    scat.update_xaxes(tickfont=dict (color='green', size=7.5))
    scat.update_yaxes(tickfont=dict (color='green', size=7.5))

    #plot(scat)
    return scat

@app.callback(
    Output('hot', 'figure'),
    [Input('go', 'n_clicks')],
    [State('commercial_activity_option', 'value'),
     State('transport_network_option', 'value'),
     State('area_option', 'value')])

def update_hot(n_clicks, comm, transp,area):

    """
    ['rgba(253, 231, 37, 1)','rgba(51, 160, 44, 1)']
    comm='Retail'
    transp='Transmilenio'
    area= 'BARRIOS UNIDOS'

    """
    rgb_hot = ['rgba(253, 231, 37, 1)','rgba(51, 160, 44, 1)']

    resaltar = loc_shp.reset_index()
    resaltar = resaltar[resaltar['LocNombre']==area].to_crs(epsg=4326)

    all_hot_filt = all_hot[(all_hot['spot']==comm)|(all_hot['spot']==transp)]
    all_hot_filt['hs_type']=[0,1]
    all_hot_loc = gpd.clip(all_hot_filt,resaltar)


    all_hot_json = json.loads(all_hot_filt.set_index('spot').to_json())
    all_hot_loc_json = json.loads(all_hot_loc.set_index('spot').to_json())



    hot = go.Figure()

    hot.add_trace(go.Choroplethmapbox(geojson=all_hot_json,
                                      locations=all_hot_filt['spot'],
                                      z=all_hot_filt['hs_type'],
                                      colorscale = rgb_hot,
                                      marker_opacity=0.2, marker_line_width=0,
                                      showscale=False,
                                      hoverinfo = 'skip'))





    hot.add_trace(go.Choroplethmapbox(geojson=all_hot_loc_json,
                                      locations=all_hot_filt['spot'],
                                      z=all_hot_filt['hs_type'],
                                      colorscale = rgb_hot,
                                      marker_opacity=0.6, marker_line_width=0,
                                      showscale=False,
                                      hoverinfo = 'skip'))

    hot.update_layout(

    title =dict (text = 'Hotspots: ' + comm +' and ' + transp,
                 font=dict(size=10)

        ),

        font=dict(
            size = 7),
        autosize=True,
        width=1200, height=600,
        margin=dict(l=10, r=10, t=20, b=10
                    , pad=5),

        mapbox=dict(
            style = "carto-darkmatter",
            bearing=0,
            center=dict(
                lat=4.64,
                lon=-74.12
            ),
            pitch=0,
            zoom=11))
    #plot(hot)
    return hot


if __name__ == '__main__':
    app.run_server(debug=True)
