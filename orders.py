# -*- coding: utf-8 -*-

"""
Created on Fri Mar 19 18:20:19 2021.

@author: Alex
"""

from datetime import date
import glob
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from functools import reduce
from itertools import cycle
import folium
import sys

pd.set_option('display.max_columns', None)


class OrderOptimization:
    fuel = 0.65
    drop_charge = 75

    pallet_length = 60
    pallet_width = 42
    pallet_height = 88.8
    cubic_inche_conversion = 1728
    pallet_weight = 1500

    truck_length = 628
    truck_width = 86
    truck_height = 88.8

    def __init__(self, path):
        self.path = path
        self.df = ""
        self.df_log = ""
        self.df_opt = ""
        self.df_order_num = ""
        self.ship_create_days = ""
        self.total_payable_rate = ""
        self.actual_pick_date = ""
        self.distance = ""
        self.base_rate = ""
        self.coordinates = ""
        self.adding_data = ""
        self.data = ""
        self.filtering_data = ""

        self.vehicle_count_list = []
        self.vehicle_id_list = []
        self.opt_customers_list = []
        self.opt_customers_list_2 = []
        self.opt_customer_location_list = []
        self.opt_weight_list = []
        self.opt_lat_list = []
        self.opt_lon_list = []

        self.pick_state_fol = []
        self.pick_city_fol = []
        self.pick_weight_fol = []
        self.pick_loc_fol = []
        self.pick_lat_fol = []
        self.pick_lon_fol = []

        self.time_window = []  # used to hold the time windows converted into distance
        self.drop_distance = []

    # =============================================================================
    # Adding Data
    # =============================================================================
    def add_data_orders(self):

        all_filenames_path = self.path + "/Orders/*.csv"
        all_filenames = [i for i in glob.glob(all_filenames_path)]
        self.df = pd.concat([pd.read_csv(f) for f in all_filenames]).drop_duplicates().reset_index(drop=True)

        self.df['Weight (lb)'] = self.df['Weight (lb)'].astype(float)

        distance_path = self.path + "/base rates/Distance.csv"
        self.distance = pd.read_csv(distance_path)
        self.distance = self.distance.groupby(['Delivery Location Postal Code'])['drop distance'].mean()
        self.distance = self.distance.to_frame()
        self.distance = self.distance.reset_index()
        self.df = self.df.merge(self.distance, how='left',
                                on=['Delivery Location Postal Code', 'Delivery Location Postal Code'])

        self.df['pick distance'] = 0

        base_rate_path = self.path + "/base rates/base rates.csv"
        self.base_rate = pd.read_csv(base_rate_path)
        self.df = self.df.merge(self.base_rate, how='left',
                                on=['Delivery Location State/Province', 'Delivery Location State/Province'])

        coordinates_path = self.path + "/Coordinates/coordinates.csv"
        self.coordinates = pd.read_csv(coordinates_path)
        self.df = self.df.merge(self.coordinates, how='left',
                                on=['Delivery Location Postal Code', 'Delivery Location Postal Code'])

        appt_times_path = self.path + "/base rates/appt_times.csv"
        self.appt_times = pd.read_csv(appt_times_path)
        self.appt_times['time_windows'] = list(zip(self.appt_times['pick_time'], self.appt_times['drop_time']))
        self.df = self.df.merge(self.appt_times, how='left',
                                on=['Delivery Location Name', 'Delivery Location Name'])

        return self.df

    # =============================================================================
    # Locatrion  Filters
    # =============================================================================
    def location_filters_orders(self):
        self.df = self.add_data_orders()

        self.df = self.df[self.df['Pick-up Location City'] == 'ST. GEORGE']
        # self.df = self.df[self.df['Weight (lb)'] > 200]
        self.df = self.df[(self.df['Delivery Location State/Province'] == 'ON') | (
                self.df['Delivery Location State/Province'] == 'QC')]

        return self.df

    # =============================================================================
    # Date Filter
    # =============================================================================
    def date_filter_orders(self, date_start, date_end):
        self.df = self.location_filters_orders()

        self.df["Calculated Load First Pick Date"] = pd.to_datetime(self.df["Calculated Load First Pick Date"],
                                                                    format='%m-%d-%y %H:%M', errors='ignore')
        self.df["First Pick Actual Time"] = pd.to_datetime(self.df["Calculated Load First Pick Date"]).dt.time
        self.df["Calculated Load First Pick Date"] = pd.to_datetime(self.df["Calculated Load First Pick Date"]).dt.date
        self.df["Shipment Create Date"] = pd.to_datetime(self.df["Shipment Create Date"], format='%m-%d-%y %H:%M',
                                                         errors='ignore')
        self.df["Shipment Create Date"] = pd.to_datetime(self.df["Shipment Create Date"]).dt.date
        self.df['Shipment Create Date'] = pd.to_datetime(self.df["Shipment Create Date"])

        self.df.sort_values(by=['Calculated Load First Pick Date'], inplace=True, ascending=True)

        self.df['Calculated Load First Pick Date'] = pd.to_datetime(self.df["Calculated Load First Pick Date"])
        self.df = self.df[self.df['Calculated Load First Pick Date'] <= pd.to_datetime(date_end.strftime("%Y-%m-%d"))]
        self.df = self.df[self.df['Calculated Load First Pick Date'] >= pd.to_datetime(date_start.strftime("%Y-%m-%d"))]

        return self.df

    # =============================================================================
    # Orders Routing Data
    # =============================================================================
    def order_log_data(self):

        self.df_log = pd.DataFrame(self.df, columns=['drop distance', 'Order Number', 'Delivery Location Name',
                                                     'Calculated Load First Pick Date', 'Pick-up Location City',
                                                     'Pick-up Location State/Province', 'Pick-up Location Postal Code',
                                                     'Delivery Location City', 'Delivery Location State/Province',
                                                     'Delivery Location Postal Code', 'Shipment Status',
                                                     'First Pick Actual Time',
                                                     'base rate', 'Weight (lb)', 'lat', 'lon'])
        self.df_log = self.df_log.reset_index(drop=True)

        self.df_log.sort_values(by=['Delivery Location Name'], inplace=True, ascending=True)

        return self.df_log

    # =============================================================================
    # Orders Print
    # =============================================================================

    def tms_orders_print(self):
        """TMS Orders Infomation."""
        pick_city = self.df_log['Pick-up Location City'].unique()
        pick_city = pick_city.tolist().pop()
        print_solution = ''
        for id in self.df_log['Order Number'].unique():
            print_solution += f"<br><b>{id}: </b>"
            print_solution += f"{self.df_log['Delivery Location Name'][self.df_log['Order Number'] == id].tolist().pop()}</br>"
            print_solution += f"<br>{pick_city} -> {' -> '.join((self.df_log['Delivery Location City'][self.df_log['Order Number'] == id]).unique())}</br>"
            print_solution += '<br>Weight: {:,} lbs | '.format(
                round(sum(self.df_log['Weight (lb)'][self.df_log['Order Number'] == id])))
            print_solution += 'Distance: {} </br>'.format(
                sum(self.df_log['drop distance'][self.df_log['Order Number'] == id]))
            print_solution += '<br></br>'
        print_solution += '<br>_________________________________________________________________________</br>'
        print_solution += '<br><b>Totals: </b></br>'
        print_solution += '<br>Total Orders: {} | '.format(len(self.df_log['Order Number']))
        print_solution += 'Total Customers: {:,} | '.format(round(len(self.df_log['Delivery Location Name'].unique())))
        print_solution += 'Total Weight: {:,} lbs</br>'.format(round(sum(self.df_log['Weight (lb)']), 2))
        print_solution += '<br></br>'
        print_solution += '<br>Total QC Orders: {} | Total ON Orders: {}</br>'.format(
            len(self.df_log['Order Number'][self.df_log['Delivery Location State/Province'] == 'QC']),
            len(self.df_log['Order Number'][self.df_log['Delivery Location State/Province'] == 'ON']))
        print_solution += '<br>Total QC Weight: {:,} | Total ON Weight: {:,}</br>'.format(
            round(sum(self.df_log['Weight (lb)'][self.df_log['Delivery Location State/Province'] == 'QC']), 2),
            round(sum(self.df_log['Weight (lb)'][self.df_log['Delivery Location State/Province'] == 'ON']), 2))
        print_solution += '<br>_________________________________________________________________________</br>'
        return print_solution

    # =============================================================================
    # Logistician Map
    # =============================================================================
    def add_categorical_legend(self, folium_map, title, colors, labels):
        """Folium Legend."""
        if len(colors) != len(labels):
            raise ValueError("colors and labels must have the same length.")

        color_by_label = dict(zip(labels, colors))

        legend_categories = ""
        for label, color in color_by_label.items():
            legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"

        legend_html = f"""
        <div id='maplegend' class='maplegend'>
          <div class='legend-title'>{title}</div>
          <div class='legend-scale'>
            <ul class='legend-labels'>
            {legend_categories}
            </ul>
          </div>
        </div>
        """
        script = f"""
            <script type="text/javascript">
            var oneTimeExecution = (function() {{
                        var executed = false;
                        return function() {{
                            if (!executed) {{
                                 var checkExist = setInterval(function() {{
                                           if ((document.getElementsByClassName('leaflet-bottom leaflet-left').length) || (!executed)) {{
                                              document.getElementsByClassName('leaflet-bottom leaflet-left')[0].style.display = "flex"
                                              document.getElementsByClassName('leaflet-bottom leaflet-left')[0].style.flexDirection = "column"
                                              document.getElementsByClassName('leaflet-bottom leaflet-left')[0].innerHTML += `{legend_html}`;
                                              clearInterval(checkExist);
                                              executed = true;
                                           }}
                                        }}, 100);
                            }}
                        }};
                    }})();
            oneTimeExecution()
            </script>
          """

        css = """

        <style type='text/css'>
          .maplegend {
            z-index:9999;
            float:right;
            background-color: rgba(255, 255, 255, 1);
            border-radius: 5px;
            border: 2px solid #bbb;
            padding: 10px;
            font-size:12px;
            positon: relative;
          }
          .maplegend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
            }
          .maplegend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
            }
          .maplegend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            margin-left: 0;
            line-height: 18px;
            margin-bottom: 2px;
            }
          .maplegend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 0px solid #ccc;
            }
          .maplegend .legend-source {
            font-size: 80%;
            color: #777;
            clear: both;
            }
          .maplegend a {
            color: #777;
            }
        </style>
        """

        folium_map.get_root().header.add_child(folium.Element(script + css))

        return folium_map

    def log_map_orders(self):

        pick_state = [self.df_log["Pick-up Location State/Province"][0]]
        self.pick_state_fol.extend(pick_state)

        pick_city = [self.df_log["Pick-up Location City"][0]]
        self.pick_city_fol.extend(pick_city)

        pick_weight = [round(sum(self.df_log["Weight (lb)"]), 2)]
        self.pick_weight_fol.extend(pick_weight)

        pick_loc = ['FBD']
        self.pick_loc_fol.extend(pick_loc)

        pick_lat = [45.115607]
        self.pick_lat_fol.extend(pick_lat)

        pick_lon = [-66.826771]
        self.pick_lon_fol.extend(pick_lon)

        drop_lat = list(self.df_log["lat"])
        drop_lon = list(self.df_log['lon'])
        points = (list(zip(drop_lat, drop_lon)))
        drop_customer = list(self.df_log['Delivery Location Name'])
        customer = drop_customer
        drop_weight = list(round(self.df_log["Weight (lb)"], 2))
        weight = drop_weight
        tms_id = list(self.df_log["Order Number"])
        tms_id_unique = list(self.df_log["Order Number"].unique())
        drop_city = list(self.df_log["Delivery Location City"])
        drop_state = list(self.df_log["Delivery Location State/Province"])

        # Creating unique colors for each load
        color = ['red', 'lightgray', 'pink', 'darkred', 'beige', 'orange', 'darkgreen', 'purple', 'lightblue',
                 'cadetblue',
                 'black', 'darkblue', 'gray', 'green', 'darkpurple', 'lightred', 'lightgreen']
        colors = list(zip(tms_id_unique, cycle(color)))
        color_df = pd.DataFrame(colors, columns=['Order Number', 'colors'])
        tms_id_df = pd.DataFrame(tms_id, columns=['Order Number'])
        colors_df = tms_id_df.merge(color_df, how='left', on=['Order Number', 'Order Number'])
        map_colors = list(colors_df['colors'])

        # Creating unique colors for each load

        # Map
        fg1 = folium.FeatureGroup(name='Markers')

        for lt, ln, cm, we, pc, ps in zip(self.pick_lat_fol, self.pick_lon_fol, self.pick_loc_fol, self.pick_weight_fol,
                                          self.pick_city_fol, self.pick_state_fol):
            fg1.add_child(folium.Marker(location=[lt, ln],
                                        popup=folium.Popup(f"""<br>Name: <strong>{cm}</strong></br>
                                        <br>Weight: <strong> {we:,}</strong></br>
                                        <br>Location: <strong> {pc}, {ps}</strong></br>""",
                                                           max_width=len(f"name= {cm}") * 20),
                                        icon=folium.Icon(color='blue', icon='fas fa-building', prefix='fa')))

        for lt, ln, cm, we, id, cl, dc, ds in zip(drop_lat, drop_lon, customer, weight, tms_id, map_colors, drop_city,
                                                  drop_state):
            log_map = folium.Map(location=[lt, ln], tiles="Stamen Terrain", zoom_start=5)
            fg1.add_child(folium.Marker(location=[lt, ln],
                                        popup=folium.Popup(f"""<br>Name: <strong>{cm}</strong></br>
                                        <br>ID: <strong> {id}</strong></br>
                                        <br>Weight: <strong> {we:,}</strong></br>
                                        <br>Location: <strong> {dc}, {ds}</strong></br>""",
                                                           max_width=len(f"name= {cm}") * 20),
                                        icon=folium.Icon(color=cl, icon='fas fa-truck', prefix='fa')))
            self.add_categorical_legend(log_map, 'Order Number',
                                        colors=color_df['colors'],
                                        labels=color_df['Order Number'])
        log_map.add_child(fg1)
        log_map.add_child(folium.LayerControl())

        return log_map

    # =============================================================================
    # Optimization Data
    # =============================================================================

    def opt_data(self):
        self.df_opt = self.df
        self.df_opt['rating'] = ''

        self.df_opt['weight_filter'] = self.df_opt.groupby(
            ['Delivery Location Name', 'Pick-up Location City', 'Pick-up Location State/Province',
             'Pick-up Location Postal Code', 'Delivery Location City', 'Delivery Location State/Province',
             'Delivery Location Postal Code', 'pick distance', 'drop distance', 'base rate', 'lat', 'lon',
             'time_windows'])['Weight (lb)'].cumsum()

        rating = []
        for row in self.df_opt['weight_filter']:
            if row <= 30_000:
                rating.append('Okay')
            else:
                for x in range(int(row), int(row) + 1):
                    rating.append(x)

        self.df_opt['rating'] = rating

        self.df_opt = self.df_opt.drop(['weight_filter'], axis=1)

        self.df_opt = \
        self.df_opt.groupby(['Delivery Location Name', 'Pick-up Location City', 'Pick-up Location State/Province',
                             'Pick-up Location Postal Code', 'Delivery Location City',
                             'Delivery Location State/Province',
                             'Delivery Location Postal Code', 'pick distance', 'drop distance', 'base rate', 'rating',
                             'lat', 'lon', 'time_windows'])['Weight (lb)'].sum()
        self.df_opt = self.df_opt.to_frame()
        self.df_opt = self.df_opt.reset_index()

        self.df_opt['rating_two'] = ''

        self.df_opt['weight_filter'] = self.df_opt.groupby(
            ['Delivery Location Name', 'Pick-up Location City', 'Pick-up Location State/Province',
             'Pick-up Location Postal Code', 'Delivery Location City', 'Delivery Location State/Province',
             'Delivery Location Postal Code', 'pick distance', 'drop distance', 'base rate', 'lat', 'lon',
             'time_windows'])['Weight (lb)'].cumsum()

        rating_two = []
        for row in self.df_opt['weight_filter']:
            if row <= 30_000:
                rating_two.append('Okay')
            else:
                for x in range(int(row), int(row) + 1):
                    rating_two.append(x)

        self.df_opt['rating_two'] = rating_two

        self.df_opt = self.df_opt.drop(['weight_filter'], axis=1)

        self.df_opt = \
        self.df_opt.groupby(['Delivery Location Name', 'Pick-up Location City', 'Pick-up Location State/Province',
                             'Pick-up Location Postal Code', 'Delivery Location City',
                             'Delivery Location State/Province',
                             'Delivery Location Postal Code', 'pick distance', 'drop distance', 'base rate',
                             'rating_two',
                             'lat', 'lon', 'time_windows'])['Weight (lb)'].sum()
        self.df_opt = self.df_opt.to_frame()
        self.df_opt = self.df_opt.reset_index()

        return self.df_opt

    def location_city(self):
        """
        Converts the city into a list, from the optimization dataframe
        """
        self.opt_data()
        location_city = self.df_opt['Pick-up Location City']
        location_city = location_city[:1]
        location_city_2 = self.df_opt['Delivery Location City']
        location_city = location_city.append(location_city_2)
        location_city = location_city.tolist()
        return location_city

    def filter_location_city(self):
        self.opt_data()
        filter_location_city = self.df_opt['Pick-up Location City'] + (', ') + self.df_opt[
            'Pick-up Location State/Province']
        filter_location_city = filter_location_city[:1]
        filter_location_city_2 = self.df_opt['Delivery Location City'] + (', ') + self.df_opt[
            'Delivery Location State/Province']
        filter_location_city = filter_location_city.append(filter_location_city_2)
        filter_location_city = filter_location_city.tolist()
        return filter_location_city

    def filter_location_city_state(self):
        self.opt_data()
        filter_location_city_state = self.df_opt['Pick-up Location State/Province']
        filter_location_city_state = filter_location_city_state[:1]
        filter_location_city_state_2 = self.df_opt['Delivery Location State/Province']
        filter_location_city_state = filter_location_city_state.append(filter_location_city_state_2)
        filter_location_city_state = filter_location_city_state.tolist()
        return filter_location_city_state

    def input_demand(self):
        self.opt_data()
        return self.df_opt['Weight (lb)'].tolist()

    def filter_customer(self):
        self.opt_data()
        filter_customer = self.df_opt['Delivery Location Name']
        filter_customer = filter_customer.tolist()
        return filter_customer

    def base_rate_filter(self):
        self.opt_data()
        base_rate_filter = self.df_opt['base rate']
        base_rate_filter = self.df_opt['base rate'].tolist()
        return base_rate_filter

    def opt_lat(self):
        self.opt_data()
        opt_lat = self.df_opt['lat'].tolist()
        return opt_lat

    def opt_lon(self):
        self.opt_data()
        opt_lon = self.df_opt['lon'].tolist()
        return opt_lon

    def truck_cubic_inches(self):
        truck_cubic_inches = (self.truck_width * self.truck_length * self.truck_height) / self.cubic_inche_conversion
        return truck_cubic_inches

    def pallet_cubic_inches(self):
        pallet_cubic_inches = (
                                          self.pallet_width * self.pallet_height * self.pallet_length) / self.cubic_inche_conversion
        return pallet_cubic_inches

    def customer_time_windows(self):
        customer_time_windows = self.df_opt['time_windows']
        customer_time_windows = self.df_opt['time_windows'].tolist()
        return customer_time_windows

    # =============================================================================
    # Order Number Data Frame for Optimization
    # =============================================================================
    def order_num_dataframe(self):
        self.df_log = self.order_log_data()
        self.df_opt = self.opt_data()

        df_log_data = list(zip(self.df_log['Order Number'], self.df_log['lat'], self.df_log['lon']))
        df_log_data = pd.DataFrame(df_log_data, columns=['Order Number', 'lat', 'lon'])

        df_opt_data = list(zip(self.df_opt['Delivery Location Name'], self.df_opt['lat'], self.df_opt['lon']))
        df_opt_data = pd.DataFrame(df_opt_data, columns=['Delivery Location Name', 'lat', 'lon'])

        df_order_num_data = pd.merge(df_opt_data, df_log_data, how='left', left_on=['lat', 'lon'],
                                     right_on=['lat', 'lon']).drop_duplicates().reset_index(drop=True)

        return df_order_num_data

    # =============================================================================
    # Changing the time windows to AM and PM
    # =============================================================================
    def time_windows_to_am_pm(self, time):
        """
        convert time to am and pm
        """
        if time[0] <= 12 and time[1] <= 12:
            time = f'{time[0]}am - {time[1]}am'
        elif time[0] <= 12 and time[1] > 12:
            time = f'{time[0]}am - {time[1] - 12}pm'
        elif time[0] > 12 and time[1] > 12:
            time = f'{time[0] - 12}pm - {time[1] - 12}pm'
        return time

    # =============================================================================
    # Time windows
    # =============================================================================

    def time_windows_to_distance(self):
        """
        Converting the time windows to distance windows
        """

        for distance, time in zip(self.df_opt['drop distance'], self.df_opt['time_windows']):
            if distance < 900:
                time = ((time[0] - 8) * 100, (time[1] - 8) * 100)
                self.time_window.append(time)
            elif distance >= 900:
                time = (((time[0] - 8) + 24) * 100, ((time[1] - 8) + 24) * 100)
                self.time_window.append(time)
        return self.time_window

    def drop_dist(self):
        """
        formula adds extra distance for loads outside of 24 hours.
        """
        for drop in self.df_opt['drop distance']:
            if drop >= 900:
                drop2 = drop + 1_000
                self.drop_distance.append(drop2)
                for number, bounds in zip(self.drop_distance, self.time_window):
                    if (bounds[0] <= number <= bounds[1]) == False:
                        if (bounds[0] - number) < 1_000:
                            number2 = number + (bounds[0] - number)
                            self.drop_distance.remove(number)
                            self.drop_distance.append(number2)
                        else:
                            drop2
            else:
                self.drop_distance.append(drop)

        return self.drop_distance

    # =============================================================================
    # Weight Conversion
    # =============================================================================
    def cubic_weight_conversion(self):
        '''Order Weight'''
        input_demand = self.input_demand()

        '''Cubic weight in Inches for a pallet'''
        pallet_cubic_inches = (
                                          self.pallet_width * self.pallet_height * self.pallet_length) / self.cubic_inche_conversion
        pallet_count_unrounded = [x / self.pallet_weight for x in input_demand]
        # pallet_count = [ math.ceil(x / pallet_weight) for x in input_demand]
        demand_cube = [x * pallet_cubic_inches for x in pallet_count_unrounded]

        return demand_cube

    # =============================================================================
    # Vehicle Drops
    # =============================================================================
    def create_data_model(self):
        input_demand = self.input_demand()
        truck_cubic_inches = self.truck_cubic_inches()
        demand_cube = self.cubic_weight_conversion()

        """Create the data for the example."""
        data = {}
        data['weights'] = demand_cube
        data['customer_demand'] = list(range(len(input_demand)))
        data['trucks'] = data['customer_demand']
        data['truck_capacity'] = truck_cubic_inches
        data['vehicle_drops'] = self.vehicle_numbers_orders
        return data

    def number_of_trucks(self):
        data = self.create_data_model()

        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Variables
        # x[i, j] = 1 if item i is packed in bin j.
        x = {}
        for i in data['customer_demand']:
            for j in data['trucks']:
                x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

        # y[j] = 1 if bin j is used.
        y = {}
        for j in data['trucks']:
            y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

        # Constraints
        # Each item must be in exactly one bin.
        for i in data['customer_demand']:
            solver.Add(sum(x[i, j] for j in data['trucks']) == 1)

        # The amount packed in each bin cannot exceed its capacity.
        for j in data['trucks']:
            solver.Add(
                sum(x[(i, j)] * data['weights'][i] for i in data['customer_demand']) <= y[j] *
                data['truck_capacity'])

        # Cannot have more than data['vehicle_drops'] items in a bin
        for j in data['trucks']:
            demand_count = 0
            for i in data['customer_demand']:
                demand_count += x[i, j]
            solver.Add(demand_count <= data['vehicle_drops'])

        # Objective: minimize the number of bins used.
        solver.Minimize(solver.Sum([y[j] for j in data['trucks']]))

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            num_trucks = 0
            for j in data['trucks']:
                if y[j].solution_value() == 1:
                    customer_demand = []
                    truck_weight = 0
                    for i in data['customer_demand']:
                        if x[i, j].solution_value() > 0:
                            customer_demand.append(i)
                            truck_weight += data['weights'][i]
                    if truck_weight > 0:
                        num_trucks += 1
        else:
            num_trucks = 0
            # return num_trucks
        return num_trucks

    # =============================================================================
    # Distance Matrix
    # =============================================================================
    def distance_matrix(self):
        self.opt_data()

        create_distance_matrix_pick = self.df_opt['pick distance']
        create_distance_matrix_drop = self.df_opt['drop distance']

        create_distance_matrix_pick = create_distance_matrix_pick.tolist()
        create_distance_matrix_drop = create_distance_matrix_drop.tolist()
        create_distance_matrix_pick_depot = create_distance_matrix_pick[:1]
        create_distance_matrix = create_distance_matrix_pick_depot + create_distance_matrix_drop

        create_distance_matrix = np.array(create_distance_matrix)
        distance_matrix = np.abs(create_distance_matrix - create_distance_matrix.reshape(-1, 1))
        drop_len = create_distance_matrix
        dummy = [0 for i in range(len(drop_len))]
        distance_matrix = np.c_[dummy, distance_matrix]
        dummy_2 = [0 for i in range(len(drop_len) + 1)]
        dummy_2 = np.array(dummy_2)
        distance_matrix = np.vstack((dummy_2, distance_matrix))

        return distance_matrix

    # =============================================================================
    # Time Matrix
    # =============================================================================
    def time_matrix(self):
        drop_dist = self.drop_dist()

        create_distance_matrix_pick = self.df_opt['pick distance']
        create_distance_matrix_drop = drop_dist

        create_distance_matrix_pick = create_distance_matrix_pick.tolist()
        create_distance_matrix_pick_depot = create_distance_matrix_pick[:1]
        create_distance_matrix = create_distance_matrix_pick_depot + create_distance_matrix_drop

        create_distance_matrix = np.array(create_distance_matrix)
        time_matrix = np.abs(create_distance_matrix - create_distance_matrix.reshape(-1, 1))
        drop_len = create_distance_matrix
        dummy = [0 for i in range(len(drop_len))]
        time_matrix = np.c_[dummy, time_matrix]
        dummy_2 = [0 for i in range(len(drop_len) + 1)]
        dummy_2 = np.array(dummy_2)
        time_matrix = np.vstack((dummy_2, time_matrix))

        return time_matrix

    # =============================================================================
    # Optimized Rates
    # =============================================================================
    def opt_rates(self, w, d, s, c, dc):
        """
        Returns the TL or LTL rates for Optimized loads
        w = Weight
        d = Distance
        s = State
        c = City
        dc = Drops

        LTL rate = (CWT (weight/100) * $CWT) + Fuel (distance * 28%)
        TL rate = Flat Rate + Fuel (distance * $0.5/mile) + (drops * self.drop_charge)
        """
        if w <= 10_000 and s == 'ON' and c == 'TORONTO' or 'ETOBICOKE' or 'BRAMPTON' or 'BRANTFORD' or 'MISSISSAUGA' or 'WOODBRIDGE' or 'RICHMONDHILL' or 'MILTON' or 'CONCORD' or 'SCARBOROUGH':
            if w <= 10_000 and w > 5_000:
                rate = ((w / 100) * 20) + (d * .28)
            elif w <= 5_000 and w > 2_500:
                rate = ((w / 100) * 18) + (d * .28)
            elif w <= 2_500 and w > 1_000:
                rate = ((w / 100) * 15) + (d * .28)
            elif w <= 1_000 and w >= 0:
                rate = ((w / 100) * 13) + (d * .28)

        if w <= 10_000 and s == 'QC' and c == 'MONTRÉAL' or 'BOUCHERVILLE' or 'ANJOU' or 'DORVAL' or 'LACHINE' or 'SAINT-LAURENT' or 'SAINT-LÉONARD' or 'VARENNES' or 'CHAMBLY' or 'DRUMMONDVILLE' or 'MIRABEL' or 'NICOLET' or 'SAINTE-PERPETUE':
            if w <= 10_000 and w > 5_000:
                rate = ((w / 100) * 18) + (d * .28)
            elif w <= 5_000 and w > 2_500:
                rate = ((w / 100) * 15) + (d * .28)
            elif w <= 2_500 and w > 1_000:
                rate = ((w / 100) * 13) + (d * .28)
            elif w <= 1_000 and w >= 0:
                rate = ((w / 100) * 11) + (d * .28)

        if w > 10_000 and s == 'ON':
            rate = 2_500 + (d * 0.5) + (dc * self.drop_charge)
        else:
            if w > 10_000 and s == 'QC':
                rate = 2_000 + (d * 0.5) + (dc * self.drop_charge)

        return rate

    def mode(self, w, d, s):
        """
        Returns the TL or LTL mode for Optimized loads
        """
        if w <= 10_000:
            mode = 'LTL'
        else:
            if w > 10_000:
                mode = 'TL'
        return mode

    # =============================================================================
    # Routing Guide
    # =============================================================================
    def routing_guide_model(self):
        distance_matrix = self.distance_matrix()
        demand_cube = self.cubic_weight_conversion()
        number_of_trucks = self.number_of_trucks()
        input_demand = self.input_demand()
        filter_location_city = self.filter_location_city()
        filter_location_city_state = self.filter_location_city_state()
        filter_customer = self.filter_customer()
        base_rate_filter = self.base_rate_filter()
        opt_lat = self.opt_lat()
        opt_lon = self.opt_lon()
        truck_cubic_inches = self.truck_cubic_inches()
        time_windows_to_distance_var = self.time_windows_to_distance()
        time_matrix = self.time_matrix()
        customer_time_windows = self.customer_time_windows()
        location_city = self.location_city()

        """Store the data for the problem."""
        self.data = {}
        self.data['time_matrix'] = time_matrix
        self.data['distance_matrix'] = distance_matrix
        self.data['time_windows'] = [(0, 0), (0, 0), ]
        self.data['time_windows'] += time_windows_to_distance_var
        self.data['customer_time_windows'] = [(0, 0), (8, 17), ]
        self.data['customer_time_windows'] += customer_time_windows
        self.data['location_names'] = ['dummy', ]
        self.data['location_names'] += filter_location_city
        self.data['drop_state'] = ['end', ]
        self.data['drop_state'] += filter_location_city_state
        self.data['drop_city'] = ['end', ]
        self.data['drop_city'] += location_city
        self.data['customer'] = ['dummy', 'FBD', ]
        self.data['customer'] += filter_customer
        self.data['demands'] = [0, 0, ]
        self.data['demands'] += demand_cube
        self.data['weight'] = [0, 0, ]
        self.data['weight'] += input_demand
        self.data['base_rate'] = [0, 0, ]
        self.data['base_rate'] += base_rate_filter
        self.data['lat'] = [0, 45.115607, ]
        self.data['lat'] += opt_lat
        self.data['lon'] = [0, -66.826771, ]
        self.data['lon'] += opt_lon
        self.data['num_vehicles'] = number_of_trucks
        self.data['vehicle_capacities'] = [truck_cubic_inches for i in range(self.data['num_vehicles'])]
        self.data['start'] = [1 for i in range(self.data['num_vehicles'])]
        self.data['dummy'] = [0 for i in range(self.data['num_vehicles'])]
        return self.data

    # =============================================================================
    # Linking order_num_dataframe to routing_guide_model
    # =============================================================================
    def order_num_data_link(self):
        order_num_dataframe = self.order_num_dataframe()
        routing_guide_model = self.routing_guide_model()

        data_list = list(zip(self.data['customer'], self.data['lat'], self.data['lon']))
        data_list = pd.DataFrame(data_list, columns=['Customers', 'lat', 'lon'])

        routing_guide_model = pd.DataFrame(order_num_dataframe,
                                           columns=['Order Number', 'lat', 'lon', 'Delivery Location Name'])

        order_num_data_link = pd.merge(data_list, routing_guide_model, how='left', left_on=['lat', 'lon'],
                                       right_on=['lat', 'lon']).drop_duplicates().reset_index(drop=True)
        order_num_data_link.dropna(subset=["Order Number"], inplace=True)

        return order_num_data_link

    # =============================================================================
    # Print Routing Guide
    # =============================================================================
    def print_routing_guide(self, data, manager, routing, solution):
        pallet_cubic_inches = self.pallet_cubic_inches()
        truck_cubic_inches = self.truck_cubic_inches()
        filter_location_city = self.filter_location_city()
        filter_location_city_state = self.filter_location_city_state()
        order_num_data_link = self.order_num_data_link()

        order_num_data_link = pd.DataFrame(order_num_data_link, columns=['Customers', 'lat', 'lon', 'Order Number',
                                                                         'Delivery Location Name'])

        """Print solution on console."""
        opt_total_distance = 0
        opt_total_load = 0
        opt_total_rate = 0
        for vehicle_id in range(self.data['num_vehicles']):
            index = routing.Start(vehicle_id)
            vehicle_count = vehicle_id + 1
            self.vehicle_count_list.append(vehicle_count)
            plan_output = '<br><b><u><font size="4">Route for vehicle {}: </font></u></b></br>'.format(vehicle_id + 1)
            plan_output += '<br></br>'
            route_distance = 0
            route_load = 0
            drop_charges = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += self.data['demands'][node_index]
                route_load_2 = self.data['weight'][node_index]
                opt_customers = self.data['customer'][node_index]
                self.opt_customers_list.append(opt_customers)
                opt_customers_2 = self.data['customer'][node_index]
                self.opt_customers_list_2.append(opt_customers_2)
                opt_weight = round(self.data['weight'][node_index], 2)
                self.opt_weight_list.append(opt_weight)
                opt_lat = self.data['lat'][node_index]
                opt_lon = self.data['lon'][node_index]
                self.opt_lat_list.append(opt_lat)
                self.opt_lon_list.append(opt_lon)
                opt_customer_location = self.data['location_names'][node_index]
                self.opt_customer_location_list.append(opt_customer_location)
                self.vehicle_id_list.append(vehicle_id + 1)
                drop_count = self.data['location_names'][node_index]
                drop_charges.append(drop_count)
                drop_charges_count = (len(drop_charges) - 2)
                drops = (len(drop_charges) - 1)

                plan_output += '&nbsp; &nbsp;<font size="3"><b>{}</b> <i>- ({}) -</i> {:,} lbs: '.format(
                    self.data['location_names'][node_index],
                    self.time_windows_to_am_pm(self.data['customer_time_windows'][node_index]), round(route_load_2))
                plan_output += ', '.join(order_num_data_link['Order Number'][order_num_data_link['lon'] == opt_lon][
                                             order_num_data_link['lat'] == opt_lat][
                                             order_num_data_link['Delivery Location Name'] == opt_customers].tolist())
                plan_output += '<br></br></font>'

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            plan_output += '{0}'.format('Route Finished')
            plan_output += '<br></br>'
            self.opt_customers_list_2.remove('FBD')
            # plan_output += '<br>Customers: {}</br>'.format(', '.join(self.opt_customers_list_2))
            # plan_output += '<br></br>'
            plan_output += '<br>Mode: {} | '.format(
                self.mode(((route_load / pallet_cubic_inches) * self.pallet_weight), route_distance,
                          self.data['drop_state'][node_index]))
            plan_output += 'Total Rate: ${:,} | '.format(round(
                self.opt_rates(((route_load / pallet_cubic_inches) * self.pallet_weight), route_distance,
                               self.data['drop_state'][node_index], self.data['drop_city'][node_index],
                               drop_charges_count), 2))
            plan_output += 'Drops: {} | '.format(drops)
            plan_output += 'Distance: {:,} mi | '.format(route_distance)
            plan_output += 'Weight: {:,} lbs</br>'.format(
                round((route_load / pallet_cubic_inches) * self.pallet_weight, 2))
            # plan_output += 'Base Rate: ${:,}\n'.format(round(self.data['base_rate'][node_index],2))
            # plan_output += 'Fuel Rate: ${:,}\n'.format(round(route_distance*.65,2))
            plan_output += '<br>Truck Capacity: {:,} % | '.format(round((route_load / truck_cubic_inches) * 100))
            plan_output += 'Pallets: {} | '.format(round((route_load) / pallet_cubic_inches))
            plan_output += 'Pallet Spaces: {} | '.format(round((truck_cubic_inches) / pallet_cubic_inches))
            plan_output += 'Remaining Pallets: {}</br>'.format(
                round(math.floor((truck_cubic_inches - route_load) / pallet_cubic_inches)))
            plan_output += '<br></br>'
            plan_output += '<br></br>'
            print(plan_output)
            opt_total_rate += round(
                self.opt_rates(((route_load / pallet_cubic_inches) * self.pallet_weight), route_distance,
                               self.data['drop_state'][node_index], self.data['drop_city'][node_index],
                               drop_charges_count), 2)
            opt_total_distance += route_distance
            opt_total_load += route_load
        print_total = '<br>_________________________________________________________________________</br>'
        print_total += '<br><b>Totals:</b></br>'
        print_total += '<br>Total Trucks Used: {:,} | Total Orders: {} | '.format(self.data['num_vehicles'],
                                                                                  (len(filter_location_city) - 1))
        print_total += 'Total Distance: {:,} mi | Total Weight: {:,} lbs</br>'.format(opt_total_distance, round(
            (opt_total_load / pallet_cubic_inches) * self.pallet_weight))
        # print_total += 'Total Base Rate: ${:,}\n'.format(round((opt_total_rate),2))
        # print_total += 'Total Fuel Rate: ${:,}\n'.format(round((opt_total_distance*fuel),2))
        print_total += '<br>Total Rate ${:,} CAD | '.format(round(opt_total_rate, 2))
        print_total += 'Total Drop Charge: ${:,} CAD  (Additional Drops: {})</br>'.format(
            ((len(filter_location_city) - 1) - (self.data['num_vehicles'])) * self.drop_charge,
            (len(filter_location_city) - 1) - (self.data['num_vehicles']))
        print_total += '<br>Truck Capacity: {:,} % | '.format(
            round((opt_total_load / (truck_cubic_inches * self.data['num_vehicles'])) * 100))
        print_total += 'Pallets Space: {} | '.format(round((opt_total_load) / pallet_cubic_inches))
        print_total += 'Truck Space: {} | '.format(
            math.floor((truck_cubic_inches * self.data['num_vehicles']) / pallet_cubic_inches))
        print_total += 'Space Remaining: {} </br>'.format(
            math.floor((truck_cubic_inches * self.data['num_vehicles'] - opt_total_load) / pallet_cubic_inches))
        print_total += '<br>_________________________________________________________________________</br>'
        print(print_total)

    # =============================================================================
    # Optimized Solution Map Coordinates
    # =============================================================================
    def opt_map_orders(self):

        opt_color = ['red', 'lightgray', 'pink', 'darkred', 'beige', 'orange', 'darkgreen', 'purple', 'lightblue',
                     'cadetblue',
                     'black', 'darkblue', 'gray', 'green', 'darkpurple', 'lightred', 'lightgreen']

        opt_colors = list(zip(self.vehicle_count_list, cycle(opt_color)))
        opt_color_df = pd.DataFrame(opt_colors, columns=['vehicles', 'colors'])

        opt_veh_list = pd.DataFrame(self.vehicle_id_list, columns=['vehicles'])
        opt_color_df_merge = opt_veh_list.merge(opt_color_df, how='left', on=['vehicles', 'vehicles'])

        opt_map_colors = list(opt_color_df_merge['colors'])

        fg3 = folium.FeatureGroup(name='Markers')

        for lt, ln, vh, cm, we, cl, lo in zip(self.opt_lat_list, self.opt_lon_list, self.vehicle_id_list,
                                              self.opt_customers_list,
                                              self.opt_weight_list, opt_map_colors, self.opt_customer_location_list):
            orders_opt_map = folium.Map(location=[lt, ln], tiles="Stamen Terrain", zoom_start=5)
            fg3.add_child(folium.Marker(location=[lt, ln],
                                        popup=folium.Popup(f"""<br>Name: <strong> {cm}</strong></br>
                                        <br>Truck #<strong> {vh} </strong></br>
                                        <br>Weight: <strong> {we:,} </strong></br>
                                        <br>Location: <strong> {lo} </strong></br>""",
                                                           max_width=len(f"name= {cm}") * 20),
                                        icon=folium.Icon(color=cl, icon='fas fa-truck', prefix='fa')))

        for lt, ln, cm, we, pc, ps in zip(self.pick_lat_fol, self.pick_lon_fol, self.pick_loc_fol, self.pick_weight_fol,
                                          self.pick_city_fol, self.pick_state_fol):
            fg3.add_child(folium.Marker(location=[lt, ln],
                                        popup=folium.Popup(f"""<br>Name: <strong>{cm}</strong></br>
                                        <br>Weight: <strong> {we:,}</strong></br>
                                        <br>Location: <strong> {pc}, {ps}</strong></br>""",
                                                           max_width=len(f"name= {cm}") * 20),
                                        icon=folium.Icon(color='blue', icon='fas fa-building', prefix='fa')))

            self.add_categorical_legend(orders_opt_map, 'Vehicles',
                                        colors=opt_color_df['colors'],
                                        labels=opt_color_df['vehicles'])

        orders_opt_map.add_child(fg3)
        orders_opt_map.add_child(folium.LayerControl())

        folium.Popup(f"""<br>Name: <strong> {cm}</strong></br>
                        <br>Truck #<strong> {vh} </strong></br>
                        <br>Weight: <strong> {we:,} </strong></br>
                        <br>Location: <strong> {lo} </strong></br>""",
                     max_width=len(f"name= {cm}") * 20)

        return orders_opt_map

    # =============================================================================
    # Routing Guide
    # =============================================================================

    def routing_guide_orders(self):
        """Solve the CVRP problem."""
        # Instantiate the data problem.
        self.data = self.routing_guide_model()
        # Create the routing index manager.

        if len(self.df['Weight (lb)']) != 0:
            manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'],
                                                   self.data['start'],
                                                   self.data['dummy'])
            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)

        # =============================================================================
        # Time Windows
        # =============================================================================

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time Windows constraint.
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            30,  # allow waiting time
            4000,  # maximum time per vehicle (Think about it as max distance per vehicle)
            False,  # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except start and end.
        for location_idx, time_window in enumerate(self.data['time_windows']):
            if location_idx != 0 and 1:
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add time window constraints for each vehicle start node.
        start_idx = 1
        for vehicle_id in range(self.data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                self.data['time_windows'][start_idx][0],
                self.data['time_windows'][start_idx][1])

        # Instantiate route start and end times to produce feasible times.
        for i in range(self.data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Return the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Return the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return self.data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Number of locations per vehicle
        def num_of_locations(from_index):
            """Return 1 for any locations except depot."""
            # Convert from routing variable Index to user NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return 1 if (from_node != 0) else 0;

        counter_callback_index = routing.RegisterUnaryTransitCallback(num_of_locations)

        routing.AddDimensionWithVehicleCapacity(
            counter_callback_index,
            0,  # null slack
            [(self.vehicle_numbers_orders + 1) for i in range(self.data['num_vehicles'])],
            # maximum locations per vehicle
            True,  # start cumul to zero
            'num_of_locations')

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
        search_parameters.solution_limit = 100_000_000
        search_parameters.time_limit.FromSeconds(self.opt_order_time)
        search_parameters.log_search = False

        # Solve the problem.
        # Print solution on console.

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            self.print_routing_guide(self.data, manager, routing, solution)
        else:
            print('Solution not Found!')
