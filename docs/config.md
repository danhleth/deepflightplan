# Configuation 
This configuration file defines the parameters and settings for generating flight plans. It specifies the input datasets, algorithmic options, optimization criteria, and operational settings. The file is structured in YAML format and consists of four main sections: opt, data, criterion, and algorithm. Each section contains key-value pairs that configure the flight planning pipeline

## File Structure
The configuration is divided into the following top-level keys:

`opt`: General options for the system's operation.
`data`: Dataset specifications for enroute airways and origin-destination pairs.
`criterion`: Method for calculating distances between waypoints.
`algorithm`: Algorithm selection and its parameters for route generation.
Below is a detailed breakdown of each section.

## opt Section
This section contains general operational settings for the flight route generation process.

`save_dir`:
Description: Directory path where the output (e.g., generated flight routes) will be saved.

`seed`:
Type: Integer
Description: Random seed for reproducibility of results. Ensures consistent outcomes when randomness is involved (e.g., in tie-breaking or sampling).


`k_closest_node_from_airport`:
Type: Integer
Description: Number of closest airway nodes (waypoints) to consider as entry/exit points from/to each airport. This parameter helps connect airports to the airway graph.

`num_processes`:
Type: Integer
Description: Number of parallel processes to use for computation. Enables multi-processing to speed up route generation for large datasets.

## data Section
This section specifies the datasets required for flight route generation, including the enroute airway data and the origin-destination (OD) airport data.

Subsections
`datasource`:
Description: Defines the enroute airway dataset used to construct the graph of waypoints and airways.


Description: Identifier or class name of the dataset handler.
Example: "Lido21EnrouteAirwayDataset"

`file_path`:
Description: Path to the GeoJSON file containing enroute airway data.

`od_dataset`:
Description: Defines the origin-destination airport dataset that specifies the start and end points of flight routes.

Description: Identifier or class name of the dataset handler.
Example: "ODAirportDataset"

`file_path`:
Description: Path to the CSV file containing origin-destination airport pairs.


## criterion Section
This section defines the method used to calculate distances between waypoints or airports, which serves as the cost metric for the heuristic algorithm.


`name`:
Description: Name of the distance calculation method. In this case, "GeopyGreatCircleDistance" refers to using the great-circle distance (shortest distance over the Earth’s surface).
Example: "GeopyGreatCircleDistance"

## algorithm Section
This section specifies the algorithm used for generating flight routes and its associated parameters. 

`name`:
Description: Name of the algorithm. "nxYen" indicates the use of Yen’s k-shortest paths algorithm.
Example: "nxYen"

`top_k`:
Description: Number of shortest paths to compute for each origin-destination pair. Yen’s algorithm will return up to this many routes, ordered by increasing distance.
Example: 100

`weights`:
Description: Metrics to use as edge weights in the graph. Currently set to ['distance'], meaning the algorithm optimizes based on distance. Other options like 'time' could be included if additional cost metrics are available.
Example: ['distance'] or ['distance', 'time']


**Notes**: You also change the support algorithm from [doc](algorithms/algorithm.md)