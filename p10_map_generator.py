import matplotlib.pyplot as plot
import geopandas

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = world.plot()


world[world.name == 'Brazil'].plot(color='forestgreen',ax=ax)
world[world.name == 'Mexico'].plot(color='chartreuse',ax=ax)
world[world.name == 'Russia'].plot(color='darkturquoise',ax=ax)



plot.show()