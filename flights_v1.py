# import all the packages
import pandas as pd
import numpy as np
import math
import threading
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
form scipy.stats import ttest_ind


# load the data
data = pd.read_csv("~/YOURDATASource_flights.csv")
airlines = pd.read_csv("~/YOURDATASource_airlines.csv")
airports = pd.read_csv("~/YOURDATASource_Flights_airports.csv")


def preprocess(data, n):

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

    def dummy_generator(data, column, airports):
        """
        Description: Generates dummy variables for a specified list of airports (1 if the airport specified, 0 if not)

        :param data: dataframe where one wants to add the dummy column
        :param column: string indicating which column the function will look at to generate the dummys
        :param airports: list indicating the number of dummy variables to be generate
        :return: dataframe with the new columns
        """

        for x in airports:
            my_list = []
            for y in data[column]:
                if x == y:
                    my_list.append(1)
                else:
                    my_list.append(0)
            data["dummy" + str(column) + str(x)] = my_list

        return data

    n = 50

    main_origins = data[["Origin", "Dest"]].groupby(["Origin"]).count().sort_values(by="Dest", ascending=False).head(
        n).reset_index()["Origin"]

    main_dests = data[["Dest", "Origin"]].groupby(["Dest"]).count().sort_values(by="Origin", ascending=False).head(
        n).reset_index()["Dest"]

    data = data[data["Origin"].isin(main_origins) & data["Dest"].isin(main_dests)]

    data.drop(["IATA_CODE_Reporting_Airline", "Origin", "Dest"], axis=1, inplace=True)

    data = clean_variables_no_info(data)
    data = clean_variables_multcol(data)
    data = clean_variables_delay(data)
    data = delaytype(data)
    data = round_hours(data)
    data = dummy_generator(data, "Origin", main_origins)
    data = dummy_generator(data, "Dest", main_dests)

    return data


number_of_airports_included = 50
data = preprocess(data, number_of_airports_included)


def Airport(airports):
    # Function finds city and airport
    identify_airport = airports.set_index('IATA_CODE')['CITY'].to_dict()
    latitude_airport = airports.set_index('IATA_CODE')['LATITUDE'].to_dict()
    longitude_airport = airports.set_index('IATA_CODE')['LONGITUDE'].to_dict()

    return


def Airlines(self):
    # Function locates all the different airlines and names them

    return


def DelayType(self):
    # Function finds categorizes delay into groups based on minute
    # For example early arrival, on time, less than 15 min etc..
    def delay_type(x): return ((0, 1)[x > 5], 2)[x > 30]
    df['DELAY_LEVEL'] = df['DEPARTURE_DELAY'].apply(delay_type)
    return


def DelaySort(self):
    # Function that finds the reason for the delay
    # Either delay in departure or arrival
    # i.e. caused by extra unexpected flight duration or not

    return


def CompareAirlines(self):
    # Function that compares the airlines on:
    # Basic statistical description of airlines
    # Delays distribution: establishing the ranking of airlines
    def get_stats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}

    global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
    global_stats = global_stats.sort_values('count')
    return global_stats


def CompareAirports(self):
    # Function that compares the airports on:
    # Basic statistical description of airports
    # Delays distribution: establishing the ranking of airports

    return


def PredictingDelayMain(self):
    # function summarizer all the methods which predict the chance of a delay

    return


def ComparePredictingDelay(self):
    # function that compares the predictions

    return


def PredictingDelay1(self):
    # function that predicts the chance of a delay by using ....

    return


def PredictingDelay2(self):
    # function that predicts the chance of a delay by using ....

    return


def PredictingDelay3(self):
    # function that predicts the chance of a delay by using ....

    return


def PredictingDelay4(self):
    # function that predicts the chance of a delay by using ....

    return


@PeterBresade
