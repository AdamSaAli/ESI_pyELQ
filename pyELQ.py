import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyelq.component.background import SpatioTemporalBackground
from pyelq.component.error_model import BySensor
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import Normal
from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.gas_species import CH4
from pyelq.model import ELQModel
from pyelq.meteorology import Meteorology
from pyelq.plotting.plot import Plot
from pyelq.preprocessing import Preprocessor
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap

#I am at Source location probability


#We can probably get the date range from the user and allow them to select the frequency,
time_axis = pd.array(
    pd.date_range(start="2025-02-05 08:00:00", end="2025-02-24 12:00:00", freq="60s"), dtype="datetime64[ns]"
)


#27601

#this is where the view will be set, i.e the general location of where we wanna measure NOT where the gas is coming out from OR where the sensors are locaed
#its just the general area of where we are measureing
reference_latitude = 40.595794
reference_longitude = -105.140305
reference_altitude = 0
nof_observations = time_axis.size # this basically says the number of observations = 27601 becasue
#time_axis.ize means start at 2025-02-05 08:00:00, end at 2025-02-24 12:00:00. how many data points will be in there if we collect one data point every 60 seconds
#the answer to that for our start="2025-02-05 08:00:00", end="2025-02-24 12:00:00" is equal to 27601

#this is teh east north up system, not sure why there is 4 inputs though


#sensor x and sensor y must have the same number of indicies, so if sensor x has 2 values then so much sensor y
sensor_x = np.array([10, -25, -10, -25])
#this just says where the sensor will be on teh x axis, so if we put index 1 and index 3 at -25 then they will be overlapping with each other
#by putting index 0 as 10, that means go 10 points or whatever unit of measure to the right, but negative numbers go to the left i.e 10 units left
#and each index corresponds to one sensor so index 0 is sensor 0 and so on



# the numbers here just position where the sensors will be on the y axis
# if I set the first index of the array to -10 and all the others to 0, then sensor 0 (index 0) will be 10 coordinate points below
#whereas sensor 1,2,3 will be inline with each other, so this isnt point sensor y, moreso the sensors alignment on the y axis
sensor_y = np.array([0,0,0,0])
#I am not exactly sure why we set sensor y and x equal to these arrays they seem quite arbitrary
sensor_z = np.ones_like(sensor_x) * 1.0 #the number you multiply it by affects how the graphs look for the gas detection and the gas wind direction map, notsure why
#this is probably the same as the other two but in the z dimension

ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
#Defines the properties and functionality of a local East-North-Up coordinate system. is what the ENU() function does
#I assume we dont really need to touch this and just need to hardcore the latitute,longittude and altitude for where we will set up our sensors

ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
LLA_object = ENU_object.to_lla()
#"LLA: Converts coordinates to latitude/longitude/altitude system. is what the above does
LLA_array = LLA_object.to_array()

nof_sensors = LLA_array.shape[0]
#not exactly sure why but this gives us 4 sensors rather than just 3 if we do -1 or 1
sensor_group = SensorGroup()
#This class is used when we want to combine a collection of sensors and be able to store/access overall properties. 
#
print("---------------------------------")
print(nof_sensors)
for sensor in range(nof_sensors):
    new_sensor = Sensor()
    new_sensor.label = f"Point sensor {sensor}"
    new_sensor.location = LLA(
        latitude=np.array([LLA_object.latitude[sensor]]),
        longitude=np.array([LLA_object.longitude[sensor]]),
        altitude=np.array([LLA_object.altitude[sensor]]),
    )

    new_sensor.time = time_axis
    new_sensor.concentration = np.zeros(nof_observations)
    sensor_group.add_sensor(new_sensor)

fig = go.Figure()
fig = sensor_group.plot_sensor_location(fig=fig)
fig.update_layout(
    map_style="open-street-map",
   
    map_center=dict(lat=reference_latitude, lon=reference_longitude),
    map_zoom=18,
    height=800, # these just change the height and width and zoom of the map
    width=1000,
    #the height and width is just the size of the initial location that the visual shows
    
    #Rama added the "width" though its not in the pyelq github so we will have to test with that
    margin={"r": 0, "l": 0, "b": 0},
    #This just changes the margin on the figure, I think its purley an asthetic thing 
)
fig.show()

#below is the meteorology stuff

met_object = Meteorology()

met_object.time = time_axis
#the high and low seems to be changing teh graph and the colours that are shown which represents different speed but I am not sure why
met_object.wind_direction = np.random.uniform(low=90.0, high=90.0, size=nof_observations)
print(met_object.wind_direction)
met_object.wind_speed = np.random.normal(loc=5.0, scale=2.0, size=nof_observations)
met_object.wind_speed = np.clip(met_object.wind_speed, 0.5, 50.0)  # Ensure no negative wind speeds

met_object.wind_direction = np.linspace(0.0, 90.0, nof_observations) + np.random.normal(loc=0.0, scale=0.1, size=nof_observations)
#So the number after the np.linespace just changes the degree of where the graphs can go, so in this case its from degree 0 to 120 degrees in the circle
#the np.random.noraml(loc and scale) seem to be making changes to the colors on teh graph but I dont know if this is because we do np.random maybe thats whats changing it.
# after further tests it does seem like the .random si causing the graph to change slightly so we will need to find out
met_object.wind_speed = 5.0 * np.ones_like(met_object.wind_direction) + np.random.normal(loc=0.0, scale=0.1, size=nof_observations)

met_object.calculate_uv_from_wind_speed_direction()

met_object.temperature = (273.1 + 15.0) * np.ones_like(met_object.wind_direction)
#this just created an array with the values 281.1 in each position 
print()
print(met_object.temperature)
print()
met_object.pressure = 101.325 * np.ones_like(met_object.wind_direction)
#we mulltiply here by 101.325 to get the pressure in kPa
#Are these 5.0 numbers arbitrary or what?
met_object.wind_turbulence_horizontal = 5.0 * np.ones_like(met_object.wind_direction)
print("()()()()()()()")
print(met_object.wind_turbulence_horizontal)
met_object.wind_turbulence_vertical = 5.0 * np.ones_like(met_object.wind_direction)

#the wind turbulence horizontal/vertical is Parameter of the wind stability in horizontal direction [deg] but I am not sure why 5 was used
# I have no idea why 5 was used


fig = met_object.plot_polar_hist()
fig.update_layout(height=400, margin={"r":0,"l":0})
#This just changes the graph visual not the actual data behind the graph so ignore basically

fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=time_axis, y=met_object.wind_direction, mode='markers', name='Wind direction'))
#the x= shows the time which was defined at the top in the beggining, it also only shows the dots every two minutes because we set freq=120 at the top
#I dont think the name matters to much visually as it doesnt show
fig.update_layout(height=400, margin={"r":0,"l":0}, title='Wind Direction [deg]')
#the margin doesnt change anything datawise it just adds margin to teh graph in a css type of way so thats up to preference
#title literally just sets the title, each dot is freq() minutes apart i am not sure why
#the hieght changes how high it goes on the y axis, it just does it more accurate tos ize
#for example in height 400 we only go up to 100 on y axis yet some points go to 119 making it harder to see how high it is, so bigger is better
#the dots are based off of this above np.linspace(0.0, 90.0), so we can have a max fo 90 as that is the max we set here
#if we change teh freq above more points get added, I think this graph just shows you how many data points were collected at a given time at a given wind direction[degree]
fig.show()


source_map = SourceMap()
# played around with these and I dont see a noticible difference, maybe it affects some code later on but till line 205 it does nothing
# I changed numbers on both x and y and it didnt provide any visual changes.
site_limits = np.array([[0, 30],
                        [0, 30],
                        [0, 3]])
location_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
# nof_sources changing from 2 to 4 or higher for some reason doesnt add more onto the figure neither does changing it to 1.
source_map.generate_sources(coordinate_object=location_object, sourcemap_limits=site_limits, sourcemap_type="hypercube", nof_sources=2)

#This did provide visual changes as it changed where the green "True locations" show up still need to mess around with it as I am not sure how these numbers came to be
source_map.location.up = np.array([2.0, 3.0])
source_map.location.east = np.array([15.0, 10.0])
source_map.location.north = np.array([15.0, 35.0])


import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

# Ensure sensor_group does NOT overwrite fig
sensor_group.plot_sensor_location(fig=fig)  # Do NOT reassign fig = sensor_group.plot_sensor_location(fig)

# Extract source locations
source_lla = source_map.location.to_lla()
source_lons = np.array(source_lla.longitude).flatten()
source_lats = np.array(source_lla.latitude).flatten()

# Add second trace for source locations
fig.add_trace(go.Scattermapbox(
    mode="markers",
    lon=source_lons,
    lat=source_lats,
    name="True locations",
    #this literally just changes the label the emitter dots are given
    marker=go.scattermapbox.Marker(color="green", size=10)
    # this literally just changes the size and colour of the emitters shown on screen
))

# Update layout for zoom & controls
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center=dict(lat=reference_latitude, lon=reference_longitude),
    mapbox_zoom=18,
    # dragmode="zoom",  # Enables mouse zooming
    # mapbox=dict(scrollzoom=True),  # Fixes mouse wheel zoom issue
    height=800, width = 1000,
    margin={"r": 0, "l": 0, "b": 0}
)

fig.show()

print("------------------------------------------GAS NOW---------------------------------------------")
gas_object = CH4()
dispersion_model = GaussianPlume(source_map=deepcopy(source_map))
true_emission_rates = np.array([[15], [10]])
for current_sensor in sensor_group.values():
    coupling_matrix = dispersion_model.compute_coupling(sensor_object=current_sensor, meteorology_object=met_object,
                                                        gas_object=gas_object, output_stacked=False, run_interpolation=False)
    source_contribution = coupling_matrix @ true_emission_rates
    observation = source_contribution.flatten() + 15.0 + np.random.normal(loc=0.0, scale=0.05, size=current_sensor.nof_observations)
    #The loc seems to change the y axis at 0.0 it was from 14.9-15.4, at 0.2 loc the y axis ranged from 15.1-15.6.
    #scale seems to just change the y axis in terms of how many values in y axis, i.e a scale of 0.05 showed 6 y axis values where as scale of 0.10 or 0.15 showed 9
    current_sensor.concentration = observation



fig=go.Figure()
fig = sensor_group.plot_timeseries(fig=fig)
fig.update_layout(height=800, margin={"r":0,"t":10,"l":0,"b":0})
fig.show()

print("-------------------------------Gas graph agaisnt win direction---------------------------")
fig = go.Figure()
fig = met_object.plot_polar_scatter(fig=fig, sensor_object=sensor_group)
fig.update_layout(height=400, margin={"r":0,"l":0})
fig.show()



analysis_time_range = [datetime.datetime(2025, 2, 5, 8, 0, 0), datetime.datetime(2025, 2, 24, 12, 0, 0)]
# these dates need to be the same as the the dates specified above
smoothing_period = 10 * 60
# The above was in the docs so don't change unless good reason.
time_bin_edge = pd.array(pd.date_range(analysis_time_range[0], analysis_time_range[1], freq=f'{smoothing_period}s'), dtype='datetime64[ns]')

prepocessor_object = Preprocessor(time_bin_edges=time_bin_edge, sensor_object=sensor_group, met_object=met_object,
                                  aggregate_function="median")

#changing the wind speed from 0.05 to 0.01 doesnt affect much so unless the min wind speed is specified its not that big of a deal
min_wind_speed = 0.05
prepocessor_object.filter_on_met(filter_variable=["wind_speed"], lower_limit=[min_wind_speed], upper_limit=[np.inf])

fig=go.Figure()
fig = prepocessor_object.sensor_object.plot_timeseries(fig=fig)
fig.update_layout(height=800, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()



######################################################################################################
#below is the code for the in detail graph

source_model = Normal()
source_model.emission_rate_mean = np.array([0], ndmin=1)
source_model.initial_precision = np.array([1 / (2.5 ** 2)], ndmin=1)
#These values are from the doc, probably best not to change them
source_model.reversible_jump = True # changing this to false doesnt do much it seems so I would keep it as the default (true)
#This came defaulted as True idk why
source_model.rate_num_sources = 1.0
source_model.dispersion_model = dispersion_model
source_model.update_precision = False
#This case defaulted as False, idk why mabe we should play around with it
source_model.site_limits = site_limits
source_model.coverage_detection = 0.1  # ppm
source_model.coverage_test_source = 3.0  # kg/hr

background = SpatioTemporalBackground()
background.n_time = None #This is a default value I would just keep it as is
background.mean_bg = 2.0
background.spatial_dependence = True #This does change it when u switch between true and false. I would suggest you keep it the default value 
background.initial_precision = 1 / np.power(3e-4, 2)
background.precision_time_0 = 1 / np.power(0.1, 2)
background.spatial_correlation_param = 25.0
background.update_precision = False

offset_model = PerSensor()
offset_model.update_precision = False
offset_model.initial_precision = 1 / (0.001)**2

error_model = BySensor()
error_model.initial_precision = 1 / (0.1)**2
error_model.prior_precision_shape = 1e-2
error_model.prior_precision_rate = 1e-2

elq_model = ELQModel(sensor_object=prepocessor_object.sensor_object, meteorology=prepocessor_object.met_object,
                        gas_species=gas_object, background=background, source_model=source_model,
                        error_model=error_model, offset_model=offset_model)
elq_model.n_iter = 500

elq_model.initialise()

elq_model.to_mcmc()
elq_model.run_mcmc()
elq_model.from_mcmc()


burn_in = elq_model.n_iter-1000

plotter = Plot()

plotter.plot_quantification_results_on_map(model_object=elq_model, bin_size_x=1, bin_size_y=1, normalized_count_limit=0.1, burn_in=burn_in)
#These are the default values the graph came with, changing from 0.1 to 0.3 for noramlized_count_limit doesnt seem to affect it to much.
plotter.plot_fitted_values_per_sensor(mcmc_object=elq_model.mcmc, sensor_object=elq_model.sensor_object, burn_in=burn_in)

true_source_location_trace = go.Scattermapbox(mode="markers",lon=source_map.location.to_lla().longitude,
                                            lat=source_map.location.to_lla().latitude,name="True locations",
                                            marker=go.scattermapbox.Marker(color="green", size=10))

#these are default values for true_source_location_trace 

plotter.figure_dict["fitted_values"].update_layout(height=800, margin={"r":0,"t":50,"l":0,"b":0}).show()
#########################################################################
plotter = elq_model.plot_fitted_values(plot=plotter)
plotter.figure_dict["fitted_values"].update_layout(height=800, margin={"r":0,"t":50,"l":0,"b":0}).show()




plotter.figure_dict["count_map"].add_trace(true_source_location_trace).update_traces(showlegend=True)
plotter.figure_dict["count_map"].update_layout(height=800, width=1000, margin={"r":0,"t":50,"l":0,"b":0}, mapbox_zoom=50)
plotter.figure_dict["count_map"].show()
