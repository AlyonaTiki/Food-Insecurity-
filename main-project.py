"""
URL: https://alyona.karmazin.me/data-science.html
Title: NYC Food Insecurity data analysis and visualization
Name: Alyona Karmazin
Email: alyona.karmazin74@myhunter.cuny.edu
Resources:  my lecture notes, textbook, https://towardsdatascience.com/,  https://www.python-graph-gallery.com/barplot/,
documentation( https://docs.scipy.org/., https://pandas.pydata.org/docs/), http://comet.lehman.cuny.edu/owen/teaching/datasci/voronoiLab.html
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandasql as psql
import numpy as np
import seaborn as sns
import folium
from wordcloud import WordCloud
import fontawesome
from sklearn.model_selection import train_test_split
from scipy.spatial import Voronoi, voronoi_plot_2d
from geojson import FeatureCollection, Feature, Polygon

#### find correlation between food cost and food insecurity using 2018 year Feed America research
######################################################################
df = pd.read_csv('2018-feedAmerica.csv')
# keep NY state only
df = df.loc[df['State'] == 'NY']
# NYC
df = df.loc[df['County, State'].isin(['Bronx County, New York','New York County, New York', 'Kings County, New York', 'Queens County, New York', 'Richmond County, New York'])]
df = pd.DataFrame(df, columns=['2018 Cost Per Meal', '2018 Food Insecurity Rate', 'State', 'County, State'])
cost = pd.to_numeric(df["2018 Cost Per Meal"], downcast="float")
rate = pd.to_numeric(df["2018 Food Insecurity Rate"], downcast="float")
plt.scatter(cost, rate)
#add linear regression line to scatterplot
m, b = np.polyfit(cost, rate, 1) # m = slope, b=intercept
plt.plot(cost, m*cost + b, color='orange')
# calculate correlation coefficient
corrC = cost.corr(rate)
print(corrC)
plt.savefig('cost_rate.png')


#### SNAP recipients data aggregation, analysis and visualization. Calculated number of recipients for each borough per year.
######################################################################
df = pd.read_csv('SNAP2018-2021.csv')

# calculate a number of SNAP recipients 2018-2021
q2018 = 'SELECT Boro, sum(BC_SNAP_Recipients) FROM df WHERE Month LIKE "%2018" GROUP BY Boro'
q2019 = 'SELECT Boro, sum(BC_SNAP_Recipients) FROM df WHERE Month LIKE "%2019" GROUP BY Boro'
q2020 = 'SELECT Boro, sum(BC_SNAP_Recipients) FROM df WHERE Month LIKE "%2020" GROUP BY Boro'
q2021 = 'SELECT Boro, sum(BC_SNAP_Recipients) FROM df WHERE Month LIKE "%2021" GROUP BY Boro'
result2018 = psql.sqldf(q2018)
result2019 = psql.sqldf(q2019)
result2020 = psql.sqldf(q2020)
result2021 = psql.sqldf(q2021)

# plot for 2018
plot2018 = result2018.plot(x ='Boro', y='sum(BC_SNAP_Recipients)', kind = 'bar', title='Number of SNAP recipients in 2018',
                           xlabel='month', color='green', ylim=(100000, 1150000))
plt.xticks(rotation=15, horizontalalignment="center")
plot2018.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('2018.png')

# plot for 2019
plot2019 = result2019.plot(x ='Boro', y='sum(BC_SNAP_Recipients)', kind = 'bar', title='Number of SNAP recipients in 2019',
                           xlabel='month', color='green', ylim=(100000, 1150000))
plt.xticks(rotation=15, horizontalalignment="center")
plot2019.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('2019.png')

# plot for 2020
plot2020 = result2020.plot(x ='Boro', y='sum(BC_SNAP_Recipients)', kind = 'bar', title='Number of SNAP recipients in 2020',
                           xlabel='month', color='green', ylim=(100000, 1150000))
plt.xticks(rotation=15, horizontalalignment="center")
plot2020.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('2020.png')

# plot for 2021
plot2021 = result2021.plot(x ='Boro', y='sum(BC_SNAP_Recipients)', kind = 'bar', title='Number of SNAP recipients in 2021',
                           xlabel='month', color='green', ylim=(100000, 1150000))
plt.xticks(rotation=15, horizontalalignment="center")
plot2021.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('2021.png')

# summary
# sumData = {'year':['2018', '2019', '2020', '2021'],
#         'recipients':[result2018['sum(BC_SNAP_Recipients)'].sum(), result2019['sum(BC_SNAP_Recipients)'].sum(), result2020['sum(BC_SNAP_Recipients)'].sum(), result2021['sum(BC_SNAP_Recipients)'].sum()]}
# print ("Number of SNAP recipients in NYC: ", sumData)
print ("2018 year: ", result2018['sum(BC_SNAP_Recipients)'].sum())
print ("2019 year: ", result2019['sum(BC_SNAP_Recipients)'].sum())
print ("2020 year: ", result2020['sum(BC_SNAP_Recipients)'].sum())
print ("2021 year: ", result2021['sum(BC_SNAP_Recipients)'].sum())

#### use of linear regression to demonstrate increase in number of SNAP recipients in NYC based on data from 2009-2021
######################################################################
predictData = pd.read_csv('Total_SNAP_Recipients.csv')
# calculate average number each year
q = 'SELECT  Year, avg(Total_SNAP_Recipients) AS Recipients FROM predictData GROUP BY Year'
result = psql.sqldf(q)

year = pd.to_numeric(result["Year"], downcast="float")
recip = pd.to_numeric(result["Recipients"], downcast="float")
plt.scatter(year, recip)
#add linear regression line to plot
m, b = np.polyfit(year, recip, 1) # m = slope, b=intercept
plt.plot(year, m*year + b)

# calculate correlation coefficient
corrC = year.corr(recip)
plt.savefig('2009_2021.png')

# Use boolean value instead of actual numbers to see increase/decrease each year
numPrevious = 0
recipients = result['Recipients']
for i in range(len(recipients)):
    if numPrevious <= recipients[i]:
        numPrevious = recipients[i]
        recipients[i] = 1
    else:
        recipients[i] = 0
result['Recipients'] = result['Recipients'].astype(int)

# result consists of year and bool value for recipients. 0 number of recipients decrease in comparison to the previous year, 1 shows increase
print (result)


#### visualization of SNAP centers distribution using Voronoi diagram.
######################################################################
SNAPcenters = pd.read_json('Directory_of_SNAP_Centers.json')
SNAPmap = folium.Map(location = [40.76, -73.99], zoom_start=13, tiles="cartodbpositron")

coordinates, point_voronoi_list, feature_list = [], [], []

for i,row in SNAPcenters.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    name = row['Facility Name'] + "\n" + "Address: " + row['Street Address'] + "\n" + " Phone number: " + str(row['Phone Number(s)'])
    i = folium.Icon(color='cadetblue', icon="shopping-basket", prefix='fa')
    coordinates.append([lat, lon])
    folium.Marker([lat,lon],popup = name,icon = i).add_to(SNAPmap)
vorLayer = Voronoi(coordinates)
numVorRegions = len(vorLayer.regions)-1
vorJSON = open('libVor.json', 'w')

for region in range(numVorRegions):
    vertex_list = []
    for k in vorLayer.regions[region]:
        if k == -1:
            break;
        else:
            # vertex list, flipped order:
            vertex = vorLayer.vertices[k]
            vertex = (vertex[1], vertex[0])
        vertex_list.append(vertex)
    #vertex list
    polygon = Polygon([vertex_list])
    feature = Feature(geometry=polygon, properties={})
    feature_list.append(feature)

feature_collection = FeatureCollection(feature_list)
print (feature_collection, file=vorJSON)
vorJSON.close()

#create map and html file
folium.Choropleth(geo_data='libVor.json', fill_color = "BuPu",
                  fill_opacity=0.01, line_opacity=0.25).add_to(SNAPmap)
SNAPmap.save(outfile='libVortt.html')

# word cloud for website using key word related to the topic
######################################################################

words=("Poverty Food Incecurity Hunger NYC Shortage Nutrition Starvation Malnutrition poor SNAP Pantry unaffordable ")

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs): return("hsl(20,100%%, %f%%)" % np.random.randint(3,51))

wordcloud = WordCloud(width=700, height=480, margin=10, background_color='white').generate(words)
wordcloud.recolor(color_func=grey_color_func, random_state=3)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig("wordCloud.png")