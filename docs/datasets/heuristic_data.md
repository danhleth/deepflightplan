# Data Requirement
To generate flight plans using a heuristic algorithm, two distinct types of datasets are required:

Enroute Airway Dataset: This dataset provides detailed information about airway segments to construct the graph used in route planning. It defines the connections between waypoints or fixes, enabling the algorithm to navigate the airspace.
Origin-Destination (OD) Airport Dataset: This dataset specifies the starting and ending airports for each flight route, serving as the input to determine the source and target nodes in the graph.
## Enroute Airway Dataset
This dataset contains airway segment information, typically structured as a GeoJSON file. Each feature represents a segment of an enroute airway, including waypoint coordinates, identifiers, and route details.

Format
A GeoJSON file with the following structure:

```
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "direction": " ",
        "fix0_coordinates": "NxxxxxxxxEyyyyyyyy",
        "fix0_icao": "RJ",
        "fix0_ident": "KEC",
        "fix1_coordinates": "NxxxxxxxxEyyyyyyyy",
        "fix1_icao": "RJ",
        "fix1_ident": "ALBAT",
        "route_ident": "A1",
        "name": "A1_RJ-KEC_RJ-ALBAT"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [dd.mmmmmm, dd.mmmmmmm],
          [dd.mmmmm, dd.36450277777778]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "direction": " ",
        "fix0_coordinates": "NxxxxxxxxEyyyyyyyy",
        "fix0_icao": "RJ",
        "fix0_ident": "ALBAT",
        "fix1_coordinates": "NxxxxxxxxEyyyyyyyy",
        "fix1_icao": "RJ",
        "fix1_ident": "HALON",
        "route_ident": "A1",
        "name": "A1_RJ-ALBAT_RJ-HALON"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [xxx.xxxxxx, xx.xxxxxx],
          [xx.xxxxxx, xx.xxxxxxx]
        ]
      }
    }
  ]
}
```
Fields:
- direction: Indicates the airway direction (e.g., " " for bidirectional).
- fix0_coordinates: Latitude and longitude of the starting waypoint in a compact format (e.g., "N33265187E135474018").
- fix0_icao: ICAO region code of the starting waypoint (e.g., "RJ").
- fix0_ident: Identifier of the starting waypoint (e.g., "KEC").
- fix1_coordinates: Latitude and longitude of the ending waypoint.
- fix1_icao: ICAO region code of the ending waypoint.
- fix1_ident: Identifier of the ending waypoint (e.g., "ALBAT").
- route_ident: Airway route identifier (e.g., "A1").
- name: Descriptive name of the segment (e.g., "A1_RJ-KEC_RJ-ALBAT").
- geometry.coordinates: List of [longitude, latitude] pairs defining the segment’s path.

### Purpose
This dataset is used to build the graph structure where nodes are waypoints (identified by fix0_ident and fix1_ident) and edges are airway segments (with weights derived from distances between coordinates).

## Origin-Destination (OD) Airport Dataset
This dataset provides the starting and ending airports for flight routes, along with relevant aircraft and positional data to define the scope of the flight plan.

Format
A CSV file or table with the following structure:
```
origin,destination,aircraft_range,org_lat,org_long,dest_lat,dest_long
CSX,TNA,3260.0,28.1892,113.22,36.8572,117.216
```

Fields
- origin: ICAO or IATA code of the departure airport (e.g., "CSX").
- destination: ICAO or IATA code of the arrival airport (e.g., "TNA").
- aircraft_range: Maximum range of the aircraft in nautical miles (e.g., 3260.0), used to constrain feasible routes.
- org_lat: Latitude of the origin airport (e.g., 28.1892).
- org_long: Longitude of the origin airport (e.g., 113.22).
- dest_lat: Latitude of the destination airport (e.g., 36.8572).
- dest_long: Longitude of the destination airport (e.g., 117.216).

## Usage in Flight Plan Generation
The Enroute Airway Dataset constructs the graph of waypoints and airways, enabling the heuristic algorithm to explore possible paths.
The OD Airport Dataset defines the specific origin-destination pairs for which flight plans are generated, anchoring the start and end points within the graph.
Together, these datasets provide the foundational data required to apply heuristic methods (e.g., A* or Greedy Best-First Search) to generate optimized flight routes.