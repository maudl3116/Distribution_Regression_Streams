import ee
import datetime
import pandas as pd
from IPython.display import Image
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import shapely
import json
from shapely.geometry import Polygon, MultiPolygon, Point
import geojson
import pickle
import warnings
warnings.filterwarnings('ignore')
import iisignature
from addtime import AddTime, LeadLag
from importlib import reload
import experiments
from experiments import *
from timeit import default_timer as timer


nut = int(sys.argv[1])
print(nut)

ee.Initialize()

dico_nuts_2016 = {'FRE1':'FR30','FRE2':'FR22','FR10':'FR10','FRF2':'FR21','FRF3':'FR41','FRF1':'FR42','FRC2':'FR43','FRK2':'FR71','FRL0':'FR82','FRJ1':'FR81','FRJ2':'FR62','FRI1':'FR61','FRI3':'FR53','FRI2':'FR63','FRK1':'FR72','FRH0':'FR52','FRG0':'FR51','FRB0':'FR24','FRD1':'FR25','FRD2':'FR23','FRC1':'FR26','FRM0':'FR83'}
dico_nuts_2013 = {}
for key in dico_nuts_2016.keys():
    dico_nuts_2013[dico_nuts_2016[key]]=key


polygons = gpd.read_file('/Users/maudlemercier/Desktop/Learning_Streams_Sets/ref-nuts-2016-20m.shp/NUTS_RG_20M_2016_4326_LEVL_2.shp/NUTS_RG_20M_2016_4326_LEVL_2.shp')
polygons = polygons[polygons.CNTR_CODE=='FR']
polygons = polygons[polygons.FID.isin(dico_nuts_2016.keys())]

start = timer()

#for nut in [0,1]:

if polygons.iloc[nut].geometry.type=='MultiPolygon':
    geom = list(polygons.iloc[nut].geometry[0].exterior.coords)
else:
    geom = list(polygons.iloc[nut].geometry.exterior.coords)
geom = [[e[0],e[1]] for e in geom]

geom = ee.Geometry.MultiPolygon([geom])

collection = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').filterBounds(geom).filterDate('2017-09-01','2018-09-01')#'2004-12-31')

def get_dates(image):
    return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd HH:00:00')})

variables = ['AvgSurfT_inst','SoilMoi0_10cm_inst','Rainf_f_tavg']
dates = collection.map(get_dates)
dates = dates.aggregate_array('date').getInfo()


def get_data(image):
    return ee.Feature(None, {variables[0]: ee.List(image.select(variables[0]).reduceRegion(ee.Reducer.toList(),geom).get(variables[0])),variables[1]: ee.List(image.select(variables[1]).reduceRegion(ee.Reducer.toList(),geom).get(variables[1])),variables[2]: ee.List(image.select(variables[2]).reduceRegion(ee.Reducer.toList(),geom).get(variables[2]))})

collection_reduced = collection.map(get_data)
dicoo = {}
dicoo['AvgSurfT_inst'] = collection_reduced.aggregate_array('AvgSurfT_inst').getInfo()
dicoo['SoilMoi0_10cm_inst'] = collection_reduced.aggregate_array('SoilMoi0_10cm_inst').getInfo()
dicoo['Rainf_f_tavg'] = collection_reduced.aggregate_array('Rainf_f_tavg').getInfo()

# save data
dicoo['AvgSurfT_inst'] = np.array(dicoo['AvgSurfT_inst'] )
dicoo['SoilMoi0_10cm_inst'] = np.array(dicoo['SoilMoi0_10cm_inst'])
dicoo['Rainf_f_tavg'] = np.array(dicoo['Rainf_f_tavg'])

df1 = pd.DataFrame(dicoo['AvgSurfT_inst'],columns=['AvgSurfT_inst_'+str(i) for i in range(dicoo['AvgSurfT_inst'].shape[1])])
df1.index = dates
df2 = pd.DataFrame(dicoo['SoilMoi0_10cm_inst'],columns=['SoilMoi0_10cm_inst_'+str(i) for i in range(dicoo['SoilMoi0_10cm_inst'].shape[1])])
df2.index = dates
df3 = pd.DataFrame(dicoo['Rainf_f_tavg'],columns=['Rainf_f_tavg_'+str(i) for i in range(dicoo['Rainf_f_tavg'].shape[1])])
df3.index = dates
df1 = df1.join(df2)
df1 = df1.join(df3)
pickle.dump(df1,open('data_clim_France/'+polygons.iloc[nut].NUTS_ID+'_2017_2018.obj','wb'))

end = timer()
print(end-start)