
Skip to content
All gists
Back to GitHub
@PeterBresade
Created 28 minutes ago

    0

Code
Revisions 1
Flights.py
#import all the packages
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb

#load the data
df = pd.read_csv("~/Documents/PycharmProjects/Flights/venv/Flight2019.csv")
airlines = pd.read_csv("~/Documents/PycharmProjects/Flights/venv/airlines.csv")
airports = pd.read_csv("~/Documents/PycharmProjects/Flights/venv/airports.csv")

def clean_variables_1(data):
    
    """
    Clean variables that provide no information for delays
    """
    
    NAs_to_remove = ["Div5TailNum", "Div5WheelsOff", "Div5LongestGTime",
                        "Div5LongestGTime", "Div5TotalGTime", "Div5WheelsOn", "Div5AirportSeqID",
                        "Div5AirportID", "Div5Airport", "Div4TailNum", "Div4WheelsOff", "Div4LongestGTime",
                        "Div4TotalGTime", "Div4WheelsOn", "Div4AirportSeqID", "Div4AirportID", "Div4Airport",
                        "Div3TailNum", "Div3WheelsOff", "Div3LongestGTime", "Div3TotalGTime", "Div3WheelsOn",
                        "Div3AirportSeqID","Div3AirportID", "Div3Airport", "Div2TailNum", "Div2WheelsOff",
                        "Div2LongestGTime", "Div2TotalGTime", "Div2WheelsOn", "Div2AirportSeqID", "Div2AirportID",
                        "Div2Airport", "Div1TailNum", "Div1WheelsOff", "Div1LongestGTime", "Div1TotalGTime",
                        "Div1WheelsOn","Div1AirportSeqID", "Div1AirportID", "Div1Airport", "DivDistance",
                        "DivArrDelay", "DivActualElapsedTime","DivReachedDest", "DivAirportLandings",
                        "LongestAddGTime", "DistanceGroup", "CancellationCode", "DestState", "DestStateFips",
                        "DestStateName", "DestWac", "OriginWac", "OriginStateName", "OriginStateFips",
                        "OriginState", "FirstDepTime", "TotalAddGTime", "Cancelled"]
    
    data = data.drop(NAs_to_remove, axis = 1)
    
    return data


def clean_variables_2(data):
    
    """
    Clean variables that do provide input but can have multicollinearity issues
    """
    
    other_columns = ["Year", "Month", "DayofMonth", "FlightDate", "Reporting_Airline", "Tail_Number", "DOT_ID_Reporting_Airline",
                    "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID",
                    "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "DestCityName",
                    "DepDelay", "DepartureDelayGroups", "DepTimeBlk", "ArrTime", "ArrivalDelayGroups", "ArrTimeBlk",
                    "Diverted", "AirTime", "Flights"]
    
    data = data.drop(other_columns, axis = 1)
    
    return data


def clean_variables_3(data):
    
    """
    Clean delay variables
    """
    
    pot_vars = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    
    data = data.drop(pot_vars, axis = 1)
    
    return data


def DataPreProces(self):
    #preprocessing the data

    return

def Airport(airports):
    #Function finds city and airport
    identify_airport = airports.set_index('IATA_CODE')['CITY'].to_dict()
    latitude_airport = airports.set_index('IATA_CODE')['LATITUDE'].to_dict()
    longitude_airport = airports.set_index('IATA_CODE')['LONGITUDE'].to_dict()

    return

def Airlines(self):
    #Function locates all the different airlines and names them

    return

def DelayType(self):
    # Function finds categorizes delay into groups based on minute
    # For example early arrival, on time, less than 15 min etc..
    delay_type = lambda x: ((0, 1)[x > 5], 2)[x > 30]
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
    #function summarizer all the methods which predict the chance of a delay

    return

def ComparePredictingDelay(self):
    #function that compares the predictions

    return

def PredictingDelay1(self):
    #function that predicts the chance of a delay by using ....

    return

def PredictingDelay2(self):
    #function that predicts the chance of a delay by using ....

    return

def PredictingDelay3(self):
    #function that predicts the chance of a delay by using ....

    return

def PredictingDelay4(self):
    #function that predicts the chance of a delay by using ....

    return



@PeterBresade
Attach files by dragging & dropping, selecting or pasting them.

    © 2020 GitHub, Inc.
    Terms
    Privacy
    Security
    Status
    Help

    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

