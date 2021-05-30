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
    currentDay = dt.datetime.now().day
    currentMonth = dt.datetime.now().month - 1
    currentYear = dt.datetime.now().year

    def __init__(self, window, ro, oo):
        self.setupUi(window)

        self.log_run_button()
        self.log_run_button_orders()

        self.start_dateEdit.setDate(dt.date(self.currentYear, self.currentMonth, self.currentDay))
        self.end_dateEdit.setDate(dt.date(self.currentYear, self.currentMonth, self.currentDay))
        self.ro = ro
        self.ro.add_data()

        self.start_dateEdit_orders.setDate(dt.date(self.currentYear, self.currentMonth, self.currentDay))
        self.end_dateEdit_orders.setDate(dt.date(self.currentYear, self.currentMonth, self.currentDay))
        self.oo = oo
        self.oo.add_data_orders()
        self.oo.location_filters_orders()

    # =============================================================================
    # Loads
    # =============================================================================
    def log_run_button(self):
        """
        Reloads the data everytime the button is clicked
        """
        self.log_run.clicked.connect(self.find_carrier)  # Removes carriers when selected
        self.log_run.clicked.connect(self.date_connect)  # Allows us to filter dates
        self.log_run.clicked.connect(self.num_of_stops_loads)
        self.log_run.clicked.connect(self.opt_run_time)
        self.log_run.clicked.connect(self.show_log_loads)
        self.log_run.clicked.connect(self.show_log_map)
        self.log_run.clicked.connect(self.show_opt_loads)
        self.log_run.clicked.connect(self.show_opt_map)
        self.log_run.clicked.connect(self.carrier_dropdown)




    def find_carrier(self):
        carrier = self.checkable_combobox.currentData()
        self.ro.carrier_filter(carrier)

    def show_log_loads(self):
        self.ro.log_routing_data()
        self.log_textBrowser.setText(self.ro.tms_loads_print())

    def show_log_map(self):
        log_map = ro.log_map()
        data = io.BytesIO()
        log_map.save(data, close_file=False)
        self.log_webView.setHtml(data.getvalue().decode())

    def show_opt_loads(self):
        f = io.StringIO()
        with redirect_stdout(f):
            ro.routing_guide()
        self.opt_textBrowser.setText(f.getvalue())

    def show_opt_map(self):
        opt_map = ro.opt_map_()
        data = io.BytesIO()
        opt_map.save(data, close_file=False)
        self.opt_webView.setHtml(data.getvalue().decode())

        self.ro.vehicle_count_list.clear()
        self.ro.vehicle_id_list.clear()
        self.ro.opt_customers_list.clear()
        self.ro.opt_customers_list_2.clear()
        self.ro.opt_customer_location_list.clear()
        self.ro.opt_weight_list.clear()
        self.ro.opt_lat_list.clear()
        self.ro.opt_lon_list.clear()

        self.ro.pick_state_fol.clear()
        self.ro.pick_city_fol.clear()
        self.ro.pick_weight_fol.clear()
        self.ro.pick_loc_fol.clear()
        self.ro.pick_lat_fol.clear()
        self.ro.pick_lon_fol.clear()

    def num_of_stops_loads(self):
        self.ro.vehicle_numbers = self.number_of_stops_SpinBox.value()

    def date_connect(self):
        self.ro.date_filter(self.start_dateEdit.date().toPyDate(), self.end_dateEdit.date().toPyDate())

    def carrier_dropdown(self):
        self.ro.carrier_holder.clear()
        self.checkable_combobox.clear()
        for i in ro.df_log['Carrier Name'].unique():
            self.ro.carrier_holder.append(i)
        self.checkable_combobox.addItems(self.ro.carrier_holder)

    def opt_run_time(self):
        self.ro.opt_time = self.opt_time_SpinBox.value()

    # =============================================================================
    # Orders
    # =============================================================================
    def log_run_button_orders(self):
        self.orders_run.clicked.connect(self.date_connect_orders)
        self.orders_run.clicked.connect(self.num_of_stops_orders)
        self.orders_run.clicked.connect(self.opt_order_run_time)

        self.orders_run.clicked.connect(self.show_log_orders)
        self.orders_run.clicked.connect(self.show_log_map_orders)

        self.orders_run.clicked.connect(self.show_opt_orders)
        self.orders_run.clicked.connect(self.show_opt_map_orders)


    def show_log_orders(self):
        self.oo.order_log_data()
        self.log_orders_textBrowser.setText(self.oo.tms_orders_print())

    def show_log_map_orders(self):
        log_map = oo.log_map_orders()
        data = io.BytesIO()
        log_map.save(data, close_file=False)
        self.log_orders_webView.setHtml(data.getvalue().decode())

    def show_opt_orders(self):
        f = io.StringIO()
        with redirect_stdout(f):
            oo.routing_guide_orders()
        self.opt_orders_textBrowser.setText(f.getvalue())

    def show_opt_map_orders(self):
        opt_map = oo.opt_map_orders()
        data = io.BytesIO()
        opt_map.save(data, close_file=False)
        self.opt_orders_webView.setHtml(data.getvalue().decode())

        self.oo.vehicle_count_list.clear()
        self.oo.vehicle_id_list.clear()
        self.oo.opt_customers_list.clear()
        self.oo.opt_customers_list_2.clear()
        self.oo.opt_customer_location_list.clear()
        self.oo.opt_weight_list.clear()
        self.oo.opt_lat_list.clear()
        self.oo.opt_lon_list.clear()

        self.oo.pick_state_fol.clear()
        self.oo.pick_city_fol.clear()
        self.oo.pick_weight_fol.clear()
        self.oo.pick_loc_fol.clear()
        self.oo.pick_lat_fol.clear()
        self.oo.pick_lon_fol.clear()

    def num_of_stops_orders(self):
        self.oo.vehicle_numbers_orders = self.number_of_stops_SpinBox_orders.value()

    def date_connect_orders(self):
        self.oo.date_filter_orders(self.start_dateEdit_orders.date().toPyDate(),
                                   self.end_dateEdit_orders.date().toPyDate())

    def opt_order_run_time(self):
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