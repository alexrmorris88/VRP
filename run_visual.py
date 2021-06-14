from click import clear
from visual import *
import sys
from contextlib import redirect_stdout
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import datetime as dt
import io, glob
from routing import RoutingOptimization
from orders import OrderOptimization
import pandas as pd
from checkable_combobox import CheckableComboBox


class RoutingGuide(Ui_MainWindow):
    currentDay = dt.datetime.now().day - 2  # current day for the carrier filter
    currentMonth = dt.datetime.now().month - 1  # current month for the carrier filter
    currentYear = dt.datetime.now().year  # current year for the carrier filter

    def __init__(self, window, ro, oo):
        self.setupUi(window)

        self.log_run_button()
        self.log_run_button_orders()

        self.start_dateEdit.setDate(dt.date(self.currentYear, self.currentMonth,
                                            self.currentDay))  # start date for the date filter in the routing file
        self.end_dateEdit.setDate(dt.date(self.currentYear, self.currentMonth,
                                          self.currentDay))  # end date for the date filter in the routing file
        self.ro = ro
        self.ro.add_data()  # preload the data from the Routing file for faster runtimes

        self.start_dateEdit_orders.setDate(dt.date(self.currentYear, self.currentMonth,
                                                   self.currentDay))  # start date for the date filter in the orders file
        self.end_dateEdit_orders.setDate(dt.date(self.currentYear, self.currentMonth,
                                                 self.currentDay))  # end date for the date filter in the orders file
        self.oo = oo
        self.oo.add_data_orders()  # preload the data from the Orders file for faster runtimes
        self.oo.location_filters_orders()  # preload the data from the Orders file for faster runtimes

    # =============================================================================
    # Loads tab in the PyQt5 Visual
    # =============================================================================
    def log_run_button(self):
        """
        Reloads the data in the Routing File everytime the button is clicked
        """
        self.log_run.clicked.connect(self.find_carrier)  # Removes carriers when selected
        self.log_run.clicked.connect(self.date_connect)  # Allows to filter dates
        self.log_run.clicked.connect(self.num_of_stops_loads)  # controls for the number of stops on a load
        self.log_run.clicked.connect(self.opt_run_time)  # Controls for the length of time to run the Solver
        self.log_run.clicked.connect(self.show_log_loads)  # Displays the Logisticians loads
        self.log_run.clicked.connect(self.show_log_map)  # Displays the Logisticians loads on the folium map
        self.log_run.clicked.connect(self.show_opt_loads)  # Displays the Optimization loads
        self.log_run.clicked.connect(self.show_opt_map)  # Displays the Optimization loads on the folium map
        self.log_run.clicked.connect(self.carrier_dropdown)  # Checkble dropdown for the carrier filter

    def find_carrier(self):
        """
        Filters the carriers from the Dataframe in the Routing file
        """
        carrier = self.checkable_combobox.currentData()
        self.ro.carrier_filter(carrier)  # carrier_filter method from the Routing file

    def carrier_dropdown(self):
        """
        Displays the potential carriers for removal by the find_carrier() method
        """
        self.ro.carrier_holder.clear()  # Clears the list from the Routing __init__ method
        self.checkable_combobox.clear()  # Clears the Checkable combo box
        for i in ro.df_log['Carrier Name'].unique():
            self.ro.carrier_holder.append(i)
        self.checkable_combobox.addItems(self.ro.carrier_holder)

    def show_log_loads(self):
        """
        Displays the Logisticians loads from the Routing file
        """
        self.ro.log_routing_data()
        self.log_textBrowser.setText(self.ro.tms_loads_print())

    def show_log_map(self):
        """
        Displays the Logisticians loads on to the folium map from the Routing file
        """
        log_map = ro.log_map()
        data = io.BytesIO()
        log_map.save(data, close_file=False)  # add the map to a bytes method to display in the QtWebEngineWidgets
        self.log_webView.setHtml(data.getvalue().decode())

    def show_opt_loads(self):
        """
        Displays the Optimization loads from the Routing file
        """
        f = io.StringIO()
        with redirect_stdout(f):  # stores the Optimization loads in a String method to display in the textBrowser
            ro.routing_guide()
        self.opt_textBrowser.setText(f.getvalue())

    def show_opt_map(self):
        """
        Displays the Optimization loads on to the folium map from the Routing file
        """
        opt_map = ro.opt_map_()
        data = io.BytesIO()
        opt_map.save(data, close_file=False)  # add the map to a bytes method to display in the QtWebEngineWidgets
        self.opt_webView.setHtml(data.getvalue().decode())

        self.ro.vehicle_count_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.vehicle_id_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_customers_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_customers_list_2.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_customer_location_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_weight_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_lat_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.opt_lon_list.clear()  # Clears the list from the Routing __init__ method

        self.ro.pick_state_fol.clear()  # Clears the list from the Routing __init__ method
        self.ro.pick_city_fol.clear()  # Clears the list from the Routing __init__ method
        self.ro.pick_weight_fol.clear()  # Clears the list from the Routing __init__ method
        self.ro.pick_loc_fol.clear()  # Clears the list from the Routing __init__ method
        self.ro.pick_lat_fol.clear()  # Clears the list from the Routing __init__ method
        self.ro.pick_lon_fol.clear()  # Clears the list from the Routing __init__ method

        self.ro.carrier_holder.clear()  # Clears the list from the Routing __init__ method
        self.ro.time_window.clear()  # Clears the list from the Routing __init__ method
        self.ro.drop_distance.clear()  # Clears the list from the Routing __init__ method

        self.ro.node_city_state_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_weight_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_time_windows_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_customer_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_customer_distance_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_city_list.clear()  # Clears the list from the Routing __init__ method
        self.ro.node_state_list.clear()  # Clears the list from the Routing __init__ method

    def num_of_stops_loads(self):
        """
        Controls the number of stop locations on each load for the Routing File
        """
        self.ro.vehicle_numbers = self.number_of_stops_SpinBox.value()

    def date_connect(self):
        """
        Filters the date in the Routing file Dataframe
        """
        self.ro.date_filter(self.start_dateEdit.date().toPyDate(), self.end_dateEdit.date().toPyDate())

    def opt_run_time(self):
        """
        Controls the runtime for the Solver in the Routing file
        """
        self.ro.opt_time = self.opt_time_SpinBox.value()

    # =============================================================================
    # Orders tab in the PyQt5 Visual
    # =============================================================================
    def log_run_button_orders(self):
        """
        Reloads the data in the Orders File everytime the button is clicked
        """
        self.orders_run.clicked.connect(self.date_connect_orders)  # Allows to filter dates
        self.orders_run.clicked.connect(self.num_of_stops_orders)  # controls for the number of stops on a load
        self.orders_run.clicked.connect(self.opt_order_run_time)  # Controls for the length of time to run the Solver
        self.orders_run.clicked.connect(self.show_log_orders)  # Displays the Logisticians Orders
        self.orders_run.clicked.connect(self.show_log_map_orders)  # Displays the Logisticians orders on the folium map
        self.orders_run.clicked.connect(self.show_opt_orders)  # Displays the Optimization Orders
        self.orders_run.clicked.connect(self.show_opt_map_orders)  # Displays the Optimization loads on the folium map

    def show_log_orders(self):
        """
        Displays the Logisticians loads from the Orders file
        """
        self.oo.order_log_data()
        self.log_orders_textBrowser.setText(self.oo.tms_orders_print())

    def show_log_map_orders(self):
        """
        Displays the Logisticians loads on to the folium map from the Orders file
        """
        log_map = oo.log_map_orders()
        data = io.BytesIO()
        log_map.save(data, close_file=False)  # add the map to a bytes method to display in the QtWebEngineWidgets
        self.log_orders_webView.setHtml(data.getvalue().decode())

    def show_opt_orders(self):
        """
        Displays the Optimization loads from the Orders file
        """
        f = io.StringIO()
        with redirect_stdout(f):  # stores the Optimization loads in a String method to display in the textBrowser
            oo.routing_guide_orders()
        self.opt_orders_textBrowser.setText(f.getvalue())

    def show_opt_map_orders(self):
        """
        Displays the Optimization loads on to the folium map from the Orders file
        """
        opt_map = oo.opt_map_orders()
        data = io.BytesIO()
        opt_map.save(data, close_file=False)  # add the map to a bytes method to display in the QtWebEngineWidgets
        self.opt_orders_webView.setHtml(data.getvalue().decode())

        self.oo.vehicle_count_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.vehicle_id_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_customers_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_customers_list_2.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_customer_location_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_weight_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_lat_list.clear()  # Clears the list from the Orders __init__ method
        self.oo.opt_lon_list.clear()  # Clears the list from the Orders __init__ method

        self.oo.pick_state_fol.clear()  # Clears the list from the Orders __init__ method
        self.oo.pick_city_fol.clear()  # Clears the list from the Orders __init__ method
        self.oo.pick_weight_fol.clear()  # Clears the list from the Orders __init__ method
        self.oo.pick_loc_fol.clear()  # Clears the list from the Orders __init__ method
        self.oo.pick_lat_fol.clear()  # Clears the list from the Orders __init__ method
        self.oo.pick_lon_fol.clear()  # Clears the list from the Orders __init__ method

        self.oo.time_window.clear() # Clears the list from the Orders __init__ method
        self.oo.drop_distance.clear() # Clears the list from the Orders __init__ method

    def num_of_stops_orders(self):
        """
        Controls the number of stop locations on each load for the Orders File
        """
        self.oo.vehicle_numbers_orders = self.number_of_stops_SpinBox_orders.value()

    def date_connect_orders(self):
        """
        Filters the date in the Orders file Dataframe
        """
        self.oo.date_filter_orders(self.start_dateEdit_orders.date().toPyDate(),
                                   self.end_dateEdit_orders.date().toPyDate())

    def opt_order_run_time(self):
        """
        Controls the runtime for the Solver in the Orders file
        """
        self.oo.opt_order_time = self.opt_time_SpinBox_orders.value()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    # Creating the app
    ro = RoutingOptimization(r'/home/alex/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS')
    oo = OrderOptimization(r'/home/alex/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS')
    co = CheckableComboBox()
    ui = RoutingGuide(MainWindow, ro, oo)

    # Show the window and start the app
    MainWindow.show()
    app.exec_()
