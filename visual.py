# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Visual.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy, QApplication
from PyQt5 import QtWebEngineWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2440, 1561)
        MainWindow.setMinimumSize(QtCore.QSize(1608, 932))
        MainWindow.setDockNestingEnabled(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.log_map = QtWidgets.QWidget()
        self.log_map.setObjectName("log_map")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.log_map)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem, 2, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem1, 0, 6, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem2, 2, 11, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem3, 2, 14, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem4, 4, 12, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem5, 3, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem6, 4, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem7, 2, 6, 1, 1)
        self.log_textBrowser = QtWidgets.QTextBrowser(self.log_map)
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(11)
        self.log_textBrowser.setFont(font)
        self.log_textBrowser.setObjectName("log_textBrowser")
        self.gridLayout_2.addWidget(self.log_textBrowser, 5, 0, 1, 8)
        spacerItem8 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem8, 4, 13, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem9, 4, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem10, 2, 5, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem11, 2, 8, 1, 1)
        self.end_dateEdit = QtWidgets.QDateEdit(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.end_dateEdit.sizePolicy().hasHeightForWidth())
        self.end_dateEdit.setSizePolicy(sizePolicy)
        self.end_dateEdit.setObjectName("end_dateEdit")
        self.gridLayout_2.addWidget(self.end_dateEdit, 2, 1, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem12, 0, 8, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem13, 0, 14, 1, 1)
        self.start_dateEdit = QtWidgets.QDateEdit(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_dateEdit.sizePolicy().hasHeightForWidth())
        self.start_dateEdit.setSizePolicy(sizePolicy)
        self.start_dateEdit.setObjectName("start_dateEdit")
        self.gridLayout_2.addWidget(self.start_dateEdit, 0, 1, 1, 1)
        
        
        self.log_webView = QtWebEngineWidgets.QWebEngineView(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log_webView.sizePolicy().hasHeightForWidth())
        self.log_webView.setSizePolicy(sizePolicy)
        self.log_webView.setUrl(QtCore.QUrl("file:///home/arm88/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS/Coordinates/tms_loads.html"))
        self.log_webView.setObjectName("log_webView")
        self.gridLayout_2.addWidget(self.log_webView, 5, 8, 1, 8)
        self.opt_time_label = QtWidgets.QLabel(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_time_label.sizePolicy().hasHeightForWidth())
        self.opt_time_label.setSizePolicy(sizePolicy)
        self.opt_time_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.opt_time_label.setObjectName("opt_time_label")
        self.gridLayout_2.addWidget(self.opt_time_label, 2, 12, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem14, 0, 2, 1, 1)
        
        
        self.opt_webView = QtWebEngineWidgets.QWebEngineView(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_webView.sizePolicy().hasHeightForWidth())
        self.opt_webView.setSizePolicy(sizePolicy)
        self.opt_webView.setUrl(QtCore.QUrl("file:///home/arm88/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS/Coordinates/opt_loads.html"))
        self.opt_webView.setObjectName("opt_webView")
        self.gridLayout_2.addWidget(self.opt_webView, 6, 8, 1, 8)
        self.opt_textBrowser = QtWidgets.QTextBrowser(self.log_map)
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(11)
        self.opt_textBrowser.setFont(font)
        self.opt_textBrowser.setObjectName("opt_textBrowser")
        self.gridLayout_2.addWidget(self.opt_textBrowser, 6, 0, 1, 8)
        spacerItem15 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem15, 3, 0, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem16, 0, 11, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem17, 0, 7, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem18, 0, 4, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem19, 3, 13, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem20, 0, 5, 1, 1)
        spacerItem21 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem21, 3, 12, 1, 1)
        self.opt_time_SpinBox = QtWidgets.QSpinBox(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_time_SpinBox.sizePolicy().hasHeightForWidth())
        self.opt_time_SpinBox.setSizePolicy(sizePolicy)
        self.opt_time_SpinBox.setMinimum(1)
        self.opt_time_SpinBox.setProperty("value", 1)
        self.opt_time_SpinBox.setObjectName("opt_time_SpinBox")
        self.gridLayout_2.addWidget(self.opt_time_SpinBox, 2, 13, 1, 1)
        self.start_date_label_loads = QtWidgets.QLabel(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_date_label_loads.sizePolicy().hasHeightForWidth())
        self.start_date_label_loads.setSizePolicy(sizePolicy)
        self.start_date_label_loads.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.start_date_label_loads.setObjectName("start_date_label_loads")
        self.gridLayout_2.addWidget(self.start_date_label_loads, 0, 0, 1, 1)
        self.number_of_stops_SpinBox = QtWidgets.QSpinBox(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.number_of_stops_SpinBox.sizePolicy().hasHeightForWidth())
        self.number_of_stops_SpinBox.setSizePolicy(sizePolicy)
        self.number_of_stops_SpinBox.setMinimum(1)
        self.number_of_stops_SpinBox.setProperty("value", 7)
        self.number_of_stops_SpinBox.setObjectName("number_of_stops_SpinBox")
        self.gridLayout_2.addWidget(self.number_of_stops_SpinBox, 0, 13, 1, 1)
        spacerItem22 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem22, 4, 15, 1, 1)
        spacerItem23 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem23, 2, 9, 1, 1)
        spacerItem24 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem24, 2, 15, 1, 1)
        spacerItem25 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem25, 2, 2, 1, 1)
        spacerItem26 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem26, 4, 14, 1, 1)
        spacerItem27 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem27, 3, 3, 1, 1)
        self.number_of_stops_label = QtWidgets.QLabel(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.number_of_stops_label.sizePolicy().hasHeightForWidth())
        self.number_of_stops_label.setSizePolicy(sizePolicy)
        self.number_of_stops_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.number_of_stops_label.setObjectName("number_of_stops_label")
        self.gridLayout_2.addWidget(self.number_of_stops_label, 0, 12, 1, 1)
        spacerItem28 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem28, 2, 4, 1, 1)
        spacerItem29 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem29, 3, 2, 1, 1)
        spacerItem30 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem30, 4, 1, 1, 1)
        spacerItem31 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem31, 3, 14, 1, 1)
        spacerItem32 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem32, 3, 11, 1, 1)
        spacerItem33 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem33, 3, 15, 1, 1)
        self.end_date_label_loads = QtWidgets.QLabel(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.end_date_label_loads.sizePolicy().hasHeightForWidth())
        self.end_date_label_loads.setSizePolicy(sizePolicy)
        self.end_date_label_loads.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.end_date_label_loads.setObjectName("end_date_label_loads")
        self.gridLayout_2.addWidget(self.end_date_label_loads, 2, 0, 1, 1)
        spacerItem34 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem34, 2, 7, 1, 1)
        spacerItem35 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem35, 4, 11, 1, 1)
        spacerItem36 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem36, 0, 9, 1, 1)
        self.log_run = QtWidgets.QPushButton(self.log_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log_run.sizePolicy().hasHeightForWidth())
        self.log_run.setSizePolicy(sizePolicy)
        self.log_run.setObjectName("log_run")
        self.gridLayout_2.addWidget(self.log_run, 0, 15, 1, 1)
        spacerItem37 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem37, 0, 3, 1, 1)
        spacerItem38 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem38, 4, 3, 1, 1)
        
        
        self.comboBox = QtWidgets.QComboBox(self.log_map)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_2.addWidget(self.comboBox, 0, 10, 1, 1)
        
        

        self.tabWidget.addTab(self.log_map, "")
        self.opt_map = QtWidgets.QWidget()
        self.opt_map.setObjectName("opt_map")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.opt_map)
        self.gridLayout_5.setObjectName("gridLayout_5")
        spacerItem39 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem39, 2, 11, 1, 1)
        
        
        self.opt_orders_webView = QtWebEngineWidgets.QWebEngineView(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_orders_webView.sizePolicy().hasHeightForWidth())
        self.opt_orders_webView.setSizePolicy(sizePolicy)
        self.opt_orders_webView.setUrl(QtCore.QUrl("file:///home/arm88/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS/Coordinates/opt_loads.html"))
        self.opt_orders_webView.setObjectName("opt_orders_webView")
        self.gridLayout_5.addWidget(self.opt_orders_webView, 5, 8, 1, 8)
        spacerItem40 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem40, 2, 0, 1, 1)
        spacerItem41 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem41, 0, 14, 1, 1)
        spacerItem42 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem42, 2, 15, 1, 1)
        spacerItem43 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem43, 0, 6, 1, 1)
        spacerItem44 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem44, 0, 7, 1, 1)
        self.opt_time_label_orders = QtWidgets.QLabel(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_time_label_orders.sizePolicy().hasHeightForWidth())
        self.opt_time_label_orders.setSizePolicy(sizePolicy)
        self.opt_time_label_orders.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.opt_time_label_orders.setObjectName("opt_time_label_orders")
        self.gridLayout_5.addWidget(self.opt_time_label_orders, 1, 12, 1, 1)
        spacerItem45 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem45, 2, 2, 1, 1)
        spacerItem46 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem46, 2, 1, 1, 1)
        spacerItem47 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem47, 3, 12, 1, 1)
        spacerItem48 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem48, 2, 12, 1, 1)
        spacerItem49 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem49, 1, 10, 1, 1)
        spacerItem50 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem50, 1, 5, 1, 1)
        spacerItem51 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem51, 1, 6, 1, 1)
        self.end_dateEdit_orders = QtWidgets.QDateEdit(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.end_dateEdit_orders.sizePolicy().hasHeightForWidth())
        self.end_dateEdit_orders.setSizePolicy(sizePolicy)
        self.end_dateEdit_orders.setObjectName("end_dateEdit_orders")
        self.gridLayout_5.addWidget(self.end_dateEdit_orders, 1, 1, 1, 1)
        spacerItem52 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem52, 1, 14, 1, 1)
        spacerItem53 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem53, 2, 13, 1, 1)
        self.end_date_label_orders = QtWidgets.QLabel(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.end_date_label_orders.sizePolicy().hasHeightForWidth())
        self.end_date_label_orders.setSizePolicy(sizePolicy)
        self.end_date_label_orders.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.end_date_label_orders.setObjectName("end_date_label_orders")
        self.gridLayout_5.addWidget(self.end_date_label_orders, 1, 0, 1, 1)
        self.number_of_stops_SpinBox_orders = QtWidgets.QSpinBox(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.number_of_stops_SpinBox_orders.sizePolicy().hasHeightForWidth())
        self.number_of_stops_SpinBox_orders.setSizePolicy(sizePolicy)
        self.number_of_stops_SpinBox_orders.setMinimum(1)
        self.number_of_stops_SpinBox_orders.setProperty("value", 7)
        self.number_of_stops_SpinBox_orders.setObjectName("number_of_stops_SpinBox_orders")
        self.gridLayout_5.addWidget(self.number_of_stops_SpinBox_orders, 0, 13, 1, 1)
        spacerItem54 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem54, 1, 9, 1, 1)
        spacerItem55 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem55, 1, 8, 1, 1)
        spacerItem56 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem56, 0, 2, 1, 1)
        self.orders_run = QtWidgets.QPushButton(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orders_run.sizePolicy().hasHeightForWidth())
        self.orders_run.setSizePolicy(sizePolicy)
        self.orders_run.setObjectName("orders_run")
        self.gridLayout_5.addWidget(self.orders_run, 0, 15, 1, 1)
        spacerItem57 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem57, 3, 2, 1, 1)
        spacerItem58 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem58, 0, 3, 1, 1)
        spacerItem59 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem59, 0, 11, 1, 1)
        self.start_date_label_orders = QtWidgets.QLabel(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_date_label_orders.sizePolicy().hasHeightForWidth())
        self.start_date_label_orders.setSizePolicy(sizePolicy)
        self.start_date_label_orders.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.start_date_label_orders.setObjectName("start_date_label_orders")
        self.gridLayout_5.addWidget(self.start_date_label_orders, 0, 0, 1, 1)
        spacerItem60 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem60, 0, 9, 1, 1)
        self.start_dateEdit_orders = QtWidgets.QDateEdit(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_dateEdit_orders.sizePolicy().hasHeightForWidth())
        self.start_dateEdit_orders.setSizePolicy(sizePolicy)
        self.start_dateEdit_orders.setObjectName("start_dateEdit_orders")
        self.gridLayout_5.addWidget(self.start_dateEdit_orders, 0, 1, 1, 1)
        spacerItem61 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem61, 1, 7, 1, 1)
        spacerItem62 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem62, 0, 8, 1, 1)
        spacerItem63 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem63, 0, 5, 1, 1)
        spacerItem64 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem64, 0, 10, 1, 1)
        self.number_of_stops_label_orders = QtWidgets.QLabel(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.number_of_stops_label_orders.sizePolicy().hasHeightForWidth())
        self.number_of_stops_label_orders.setSizePolicy(sizePolicy)
        self.number_of_stops_label_orders.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.number_of_stops_label_orders.setObjectName("number_of_stops_label_orders")
        self.gridLayout_5.addWidget(self.number_of_stops_label_orders, 0, 12, 1, 1)
        spacerItem65 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem65, 1, 15, 1, 1)
        spacerItem66 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem66, 1, 11, 1, 1)
        spacerItem67 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem67, 1, 2, 1, 1)
        spacerItem68 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem68, 2, 3, 1, 1)
        spacerItem69 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem69, 3, 14, 1, 1)
        spacerItem70 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem70, 1, 4, 1, 1)
        spacerItem71 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem71, 3, 1, 1, 1)
        self.log_orders_textBrowser = QtWidgets.QTextBrowser(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log_orders_textBrowser.sizePolicy().hasHeightForWidth())
        self.log_orders_textBrowser.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(11)
        self.log_orders_textBrowser.setFont(font)
        self.log_orders_textBrowser.setObjectName("log_orders_textBrowser")
        self.gridLayout_5.addWidget(self.log_orders_textBrowser, 4, 0, 1, 8)
        spacerItem72 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem72, 3, 0, 1, 1)
        spacerItem73 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem73, 1, 3, 1, 1)
        spacerItem74 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem74, 0, 4, 1, 1)
        spacerItem75 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem75, 3, 15, 1, 1)
        spacerItem76 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem76, 2, 14, 1, 1)
        spacerItem77 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem77, 3, 13, 1, 1)
        self.opt_orders_textBrowser = QtWidgets.QTextBrowser(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_orders_textBrowser.sizePolicy().hasHeightForWidth())
        self.opt_orders_textBrowser.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(11)
        self.opt_orders_textBrowser.setFont(font)
        self.opt_orders_textBrowser.setObjectName("opt_orders_textBrowser")
        self.gridLayout_5.addWidget(self.opt_orders_textBrowser, 5, 0, 1, 8)
        spacerItem78 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem78, 3, 11, 1, 1)
        self.opt_time_SpinBox_orders = QtWidgets.QSpinBox(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.opt_time_SpinBox_orders.sizePolicy().hasHeightForWidth())
        self.opt_time_SpinBox_orders.setSizePolicy(sizePolicy)
        self.opt_time_SpinBox_orders.setMinimum(1)
        self.opt_time_SpinBox_orders.setProperty("value", 1)
        self.opt_time_SpinBox_orders.setObjectName("opt_time_SpinBox_orders")
        self.gridLayout_5.addWidget(self.opt_time_SpinBox_orders, 1, 13, 1, 1)
        
        
        self.log_orders_webView = QtWebEngineWidgets.QWebEngineView(self.opt_map)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log_orders_webView.sizePolicy().hasHeightForWidth())
        self.log_orders_webView.setSizePolicy(sizePolicy)
        self.log_orders_webView.setUrl(QtCore.QUrl("file:///home/arm88/Insync/alex@M1Consulting.ca/OneDrive Biz/Machine Learning/Machine Learning/Logistics/TMS/Coordinates/tms_loads.html"))
        self.log_orders_webView.setObjectName("log_orders_webView")
        self.gridLayout_5.addWidget(self.log_orders_webView, 4, 8, 1, 8)
        spacerItem79 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem79, 3, 3, 1, 1)
        self.tabWidget.addTab(self.opt_map, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2440, 20))
        self.menubar.setObjectName("menubar")
        self.logistics_optimization = QtWidgets.QMenu(self.menubar)
        self.logistics_optimization.setObjectName("logistics_optimization")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.logistics_optimization.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.opt_time_label.setBuddy(self.opt_time_SpinBox)
        self.start_date_label_loads.setBuddy(self.start_dateEdit)
        self.number_of_stops_label.setBuddy(self.number_of_stops_SpinBox)
        self.end_date_label_loads.setBuddy(self.end_dateEdit)
        self.opt_time_label_orders.setBuddy(self.opt_time_SpinBox_orders)
        self.end_date_label_orders.setBuddy(self.end_dateEdit_orders)
        self.start_date_label_orders.setBuddy(self.start_dateEdit_orders)
        self.number_of_stops_label_orders.setBuddy(self.number_of_stops_SpinBox_orders)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Route Optimization"))
        self.opt_time_label.setText(_translate("MainWindow", "Optimization Run Time"))
        self.start_date_label_loads.setText(_translate("MainWindow", "Start Date"))
        self.number_of_stops_label.setText(_translate("MainWindow", "Number of Stops"))
        self.end_date_label_loads.setText(_translate("MainWindow", "End Date"))
        self.log_run.setText(_translate("MainWindow", "Run Loads"))
        
        self.comboBox.setItemText(0, _translate("MainWindow", "RALLY LOGISTICS"))
        self.comboBox.setItemText(1, _translate("MainWindow", "SHORELAND TRANSPORT"))
        self.comboBox.setItemText(2, _translate("MainWindow", "MIDLAND TRANSPORT LIMITED"))
        self.comboBox.setItemText(3, _translate("MainWindow", "COAST TO COAST TRANSPORT"))
        self.comboBox.setItemText(4, _translate("MainWindow", "PROFESSIONAL CARRIERS INC"))
        self.comboBox.setItemText(5, _translate("MainWindow", "CONNORS TRANSFER LIMITED"))
        self.comboBox.setItemText(6, _translate("MainWindow", "COUNTY LINE TRUCKING"))
        self.comboBox.setItemText(7, _translate("MainWindow", "DONNELLY FARMS"))
        self.comboBox.setItemText(8, _translate("MainWindow", "BELL CITY TRANSPORT SYSTEMS, INC."))
        self.comboBox.setItemText(9, _translate("MainWindow", "CUSTOMER PICKUP"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.log_map), _translate("MainWindow", "Loads"))
        self.opt_time_label_orders.setText(_translate("MainWindow", "Optimization Run Time"))
        self.end_date_label_orders.setText(_translate("MainWindow", "End Date"))
        self.orders_run.setText(_translate("MainWindow", "Run Orders"))
        self.start_date_label_orders.setText(_translate("MainWindow", "Start Date"))
        self.number_of_stops_label_orders.setText(_translate("MainWindow", "Number of Stops"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.opt_map), _translate("MainWindow", "Orders"))
        self.logistics_optimization.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
