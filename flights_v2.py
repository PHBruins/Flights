
import math
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
pd.set_option("display.max_columns", None)
data = pd.read_csv("Flight2019.csv")

def preprocess(data, number_of_airports_included, define_cut_off, airline_boolean, taxi_in_out_boolean):

    def clean_variables_no_info(data):
        """
        Description: drops columns that provide no information
        :param data: flights dataframe
        :return: returns cleaned dataframe
        """

        NAs_to_remove = ["Quarter", "Flight_Number_Reporting_Airline", "Div5TailNum", "Div5WheelsOff",
                         "Div5LongestGTime",
                         "Div5LongestGTime", "Div5TotalGTime", "Div5WheelsOn", "Div5AirportSeqID",
                         "Div5AirportID", "Div5Airport", "Div4TailNum", "Div4WheelsOff", "Div4LongestGTime",
                         "Div4TotalGTime", "Div4WheelsOn", "Div4AirportSeqID", "Div4AirportID", "Div4Airport",
                         "Div3TailNum", "Div3WheelsOff", "Div3LongestGTime", "Div3TotalGTime", "Div3WheelsOn",
                         "Div3AirportSeqID", "Div3AirportID", "Div3Airport", "Div2TailNum", "Div2WheelsOff",
                         "Div2LongestGTime", "Div2TotalGTime", "Div2WheelsOn", "Div2AirportSeqID", "Div2AirportID",
                         "Div2Airport", "Div1TailNum", "Div1WheelsOff", "Div1LongestGTime", "Div1TotalGTime",
                         "Div1WheelsOn", "Div1AirportSeqID", "Div1AirportID", "Div1Airport", "DivDistance",
                         "DivArrDelay", "DivActualElapsedTime", "DivReachedDest", "DivAirportLandings",
                         "LongestAddGTime", "DistanceGroup", "CancellationCode", "DestState", "DestStateFips",
                         "DestStateName", "DestWac", "OriginWac", "OriginStateName", "OriginStateFips",
                         "OriginState", "FirstDepTime", "TotalAddGTime", "Cancelled", "Unnamed: 109", "OriginCityName",
                         "ArrDelay", "DepTime", "WheelsOff", "WheelsOn"]

        cleaned_data = data.drop(NAs_to_remove, axis=1)

        return cleaned_data

    def clean_variables_multcol(data):
        """
        Description: drops columns that can create multicollinearity issues
        :param data: flights dataframe
        :return: cleaned dataframe
        """

        other_columns = ["Year", "Month", "DayofMonth", "FlightDate", "Reporting_Airline", "Tail_Number",
                         "DOT_ID_Reporting_Airline",
                         "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID",
                         "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "DestCityName",
                         "DepDelay", "DepartureDelayGroups", "DepTimeBlk", "ArrTime", "ArrivalDelayGroups",
                         "ArrTimeBlk",
                         "Diverted", "AirTime", "Flights"]

        cleaned_data = data.drop(other_columns, axis=1)

        return cleaned_data

    def clean_variables_delay(data):
        """
        Description: drops delay related columns out of the flight dataset
        :param data: dataframe we want to clean
        :return: cleaned dataframe
        """

        pot_vars = ["CarrierDelay", "WeatherDelay",
                    "NASDelay", "SecurityDelay", "LateAircraftDelay"]

        cleaned_data = data.drop(pot_vars, axis=1)

        return cleaned_data

    def delaytype(data):
        """
        Description: creates a new column that categorizes the delay in minutes into 7 categories
        :param data: flights data with a delay column in minutes
        :return: dataframe with the new column
        """

        my_list = []

        for x in data["DepDelayMinutes"]:
            if x < 15:
                my_list.append(0)
            elif x < 30:
                my_list.append(1)
            elif x < 45:
                my_list.append(2)
            elif x < 60:
                my_list.append(3)
            elif x < 90:
                my_list.append(4)
            elif x < 120:
                my_list.append(5)
            else:
                my_list.append(6)

        data["Del_type"] = my_list

        return data

    def round_hours(data):
        """
        Description: transforms the appropiate time columns into hours in the flights dataset
        :param data: flights dataframe
        :return: dataset with the new hour columns
        """

        data.dropna(subset=["CRSDepTime", "CRSArrTime"], inplace=True)

        return_hour(data, "CRSDepTime")
        return_hour(data, "CRSArrTime")

        data.drop(["CRSDepTime",  "CRSArrTime"], axis=1, inplace=True)

        return data

    def return_hour(data, column):
        """
        Description: returns the hour rounded down for a time element in military format
        :param data: dataframe with the new column
        :param column: time column to be transformed
        :return: dataframe with the new column
        """

        my_list = []

        for x in data[column]:
            a = math.floor(x / 100)
            my_list.append(a)

        data[column + "Hour"] = my_list

        return data

    def select_number_of_airports(data, number_of_airports_included):
        """
        Description: select the top n most used airports
        :param data: dataframe we want to clean
        :param number_of_airports_included: number of airports we want to include ranked om most used
        :return: cleaned dataframe
        """

        main_origins = data[["Origin", "Dest"]].groupby(["Origin"]).count().sort_values(by="Dest", ascending=False).head(
            number_of_airports_included).reset_index()["Origin"]

        main_dests = data[["Dest", "Origin"]].groupby(["Dest"]).count().sort_values(by="Origin", ascending=False).head(
            number_of_airports_included).reset_index()["Dest"]

        data = data[data["Origin"].isin(main_origins) & data["Dest"].isin(main_dests)]
        
        return data
    
    def dummy_generator(data, taxi_in_out_boolean, airline_boolean):
        """
        Description: Generates dummy variables for a specified list of airports (1 if the airport specified, 0 if not)
        :param data: dataframe where one wants to add the dummy column
        :param taxi_in_out_boolean: boolean to execute matrix multiplication
        :param airline_boolean: boolean to execute matrix multiplication
        :return: dataframe with the new columns
        """
        dum_orig = pd.get_dummies(data["Origin"])
        dum_dest = pd.get_dummies(data["Dest"])
        dum_air = pd.get_dummies(data["IATA_CODE_Reporting_Airline"])
        
        if not taxi_in_out_bolean == 0:
            dum_orig = dum_orig.mul(data["TaxiOut"], axis = 0)
            dum_dest = dum_dest.mul(data["TaxiIn"], axis = 0)   
        
        if not airline_boolean ==0:
            airline_delay = data["DepDelayMinutes"]-data["TaxiOut"]
            airline_delay = airline_delay.where(airline_delay > 1, 1)
            dum_air = pd.get_dummies(data["IATA_CODE_Reporting_Airline"])
            dum_air = dum_air.mul(airline_delay, axis = 0) 

        data = pd.concat([data, dum_orig, dum_dest, dum_air], axis=1)
        data.drop(["IATA_CODE_Reporting_Airline", "Origin", "Dest"], axis=1, inplace=True)
        return data
    
    def air_delay(data, define_cut_off):
        """
        Description: Generates a boolean on the basis of the selected cut of for speed. The function selects on the basis qrange
        :param data: dataframe where one wants to add the boolean column
        :param define_cut_off: cut off quartile range
        :return: dataframe with the new columns
        """
        avg_travel_speed = (data["Distance"]*1.6)/(data["ActualElapsedTime"]/60)
        def in_qrange(ser, q):
            return ser.between(*ser.quantile(q=q))
        avg_travel_speed_boolean = avg_travel_speed.transform(in_qrange, q=[define_outliers, 1])
        avg_travel_speed_boolean = avg_travel_speed_boolean*1
        avg_travel_speed_boolean = avg_travel_speed_boolean.to_frame('Speed')
        data = pd.concat([data, avg_travel_speed_boolean], axis=1)
        data.drop(["TaxiOut", "TaxiIn","CRSElapsedTime","ActualElapsedTime","CRSArrTimeHour", "DepDelayMinutes", "ArrDelayMinutes", "Distance"], axis=1, inplace=True)
        return data

    data = clean_variables_no_info(data)
    data = clean_variables_multcol(data)
    data = clean_variables_delay(data)
    data = delaytype(data)
    data = round_hours(data)
    data = dummy_generator(data, taxi_in_out_bolean, airline_boolean)
    data = air_delay(data, define_outliers)
    
    

    return data

taxi_in_out_boolean = 0
airline_boolean = 0
# define_cut_off must be value between 0 and 1
define_cut_off = 0.75
number_of_airports_included = 10
data = preprocess(data, number_of_airports_included, define_cut_off, airline_boolean, taxi_in_out_boolean)
