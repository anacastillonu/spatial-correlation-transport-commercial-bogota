# spatial-correlation-transport-commercial-bogota
Visualization tool for the analysis of  the spatial correlation between different types of transport networks and commercial activities in Bogotá.

Methodology here: enlace

### Data

##### 1. Grid Data
The methodology implemented divides the city into a 50m-by-50m grid. Each cell contains a values indicating (i) the density of different transport networks (network usage) and (ii) the density of different commercial activities (density of business points)

The grid data is a .csv file where each entry is a grid cell.

##### 2. Correlation by Localidad
.csv file with the Pearson correlation coefficients for different commercial-transport pairs in different Localidades

##### 3. Localidades Polygons
Polygons indicating the 19 Localidades in the city

##### 4. Grid Polygons
Polygons indicating the 50m-by-50m city grid

##### 5. KDE Polygons
Polygons indicating the Kernel Density Estimation for each of the variables (transport and commercial)

##### 6. Hotspots Polygons
Polygons indicating Hotspots for different transport networks (usage) and different commercial activities (business points)

#### The Script has 3 sections as follows:

### Initialize

This section reads the data and prepares it for the visualization.

### DASH Layout
Indicates the DASH Core Components and Layout

### DASH Callback Functions
Updates graphs and headings in the Dash application
