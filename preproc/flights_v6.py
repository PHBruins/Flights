import math
from IPython.core.display import display, HTML
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
display(HTML("<style>.container { width:100% !important; }</style>"))

# In[ ]:

pd.set_option("display.max_columns", None)


def preprocess(data, number_of_airports_included=15, number_of_airlines_included=10, airline_boolean=0, taxi_in_out_boolean=0, define_cut_off=0.75, sample_size=30000):

    def clean_variables_no_info(data):
        """
        Description: drops columns that provide no information
        :param data: flights dataframe
        :return: returns cleaned dataframe
        """

        NAs_to_remove = ["Unnamed: 15"]

        cleaned_data = data.drop(NAs_to_remove, axis=1)

        return cleaned_data


    def delaytype(data):
        """
        Description: creates a new column that categorizes the delay in minutes into 7 categories
        :param data: flights data with a delay column in minutes
        :return: dataframe with the new column
        """

        my_list = []

        for x in data["DEP_DELAY_NEW"]:
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

        data.dropna(subset=["CRS_DEP_TIME", "CRS_ARR_TIME"], inplace=True)

        return_hour(data, "CRS_DEP_TIME")
        return_hour(data, "CRS_ARR_TIME")

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

        main_origins = data[["ORIGIN", "DEST"]].groupby(["ORIGIN"]).count().sort_values(
            by="DEST", ascending=False).head(number_of_airports_included).reset_index()["ORIGIN"]

        main_dests = data[["DEST", "ORIGIN"]].groupby(["DEST"]).count().sort_values(
            by="ORIGIN", ascending=False).head(number_of_airports_included).reset_index()["DEST"]

        data = data[data["ORIGIN"].isin(main_origins) & data["DEST"].isin(main_dests)]

        return data

    def select_number_of_airlines(data, number_of_airlines_included):
        """
        Description: select the top n most used airports
        :param data: dataframe we want to clean
        :param number_of_airports_included: number of airports we want to include ranked om most used
        :return: cleaned dataframe
        """

        main_airlines = data[["OP_CARRIER", "DEST"]].groupby(["OP_CARRIER"]).count().sort_values(by="DEST", ascending=False).head(
            number_of_airlines_included).reset_index()["OP_CARRIER"]

        data = data[data["OP_CARRIER"].isin(main_airlines)]

        return data

    def dummy_generator(data, taxi_in_out_boolean, airline_boolean):
        """
        Description: Generates dummy variables for a specified list of airports (1 if the airport specified, 0 if not)
        :param data: dataframe where one wants to add the dummy column
        :param taxi_in_out_boolean: boolean to execute matrix multiplication
        :param airline_boolean: boolean to execute matrix multiplication
        :return: dataframe with the new columns
        """

        dum_orig = pd.get_dummies(data["ORIGIN"], prefix="origin").astype(int)
        dum_orig = dum_orig.drop(['origin_ATL'], axis=1)
        dum_dest = pd.get_dummies(data["DEST"], prefix="dest").astype(int)
        dum_dest = dum_dest.drop(['dest_ATL'], axis=1)
        dum_air = pd.get_dummies(data["OP_CARRIER"], prefix="airline")
        dum_air = dum_air.drop(['airline_WN'], axis=1)

        if not taxi_in_out_boolean == 0:
            dum_orig = dum_orig.mul(data["TAXI_OUT"], axis=0)
            dum_dest = dum_dest.mul(data["TAXI_IN"], axis=0)

        if not airline_boolean == 0:
            airline_delay = data["DEP_DELAY_NEW"]-data["TAXI_OUT"]
            airline_delay = airline_delay.where(airline_delay > 1, 1)
            dum_air = pd.get_dummies(data["OP_CARRIER"])
            dum_air = dum_air.mul(airline_delay, axis=0)

        data = pd.concat([data, dum_orig, dum_dest, dum_air], axis=1)
        return data

    def air_delay(data, define_cut_off):
        """
        Description: Generates a boolean on the basis of the selected cut of for speed. The function selects on the basis qrange
        :param data: dataframe where one wants to add the boolean column
        :param define_cut_off: cut off quartile range
        :return: dataframe with the new columns
        """
        avg_travel_speed = (data["DISTANCE"]*1.6)/(data["ACTUAL_ELAPSED_TIME"]/60)

        def in_qrange(ser, q):
            return ser.between(*ser.quantile(q=q))
        avg_travel_speed_boolean = avg_travel_speed.transform(in_qrange, q=[define_cut_off, 1])
        avg_travel_speed_boolean = avg_travel_speed_boolean*1
        avg_travel_speed_boolean = avg_travel_speed_boolean.to_frame('Speed')
        data = pd.concat([data, avg_travel_speed_boolean], axis=1)
        return data

    def dummy_afternoon(data):
        """
        """

        my_list = []

        for x in data["CRS_DEP_TIMEHour"]:
            if 1 < x < 13:
                my_list.append(1)
            else:
                my_list.append(0)

        data["dummy_afternoon"] = my_list
        data = data.drop(["CRS_DEP_TIMEHour"], axis=1)

        return data

    def dummy_day(data):
        """
        Description

        :param data:
        :param days:
        """
        week = pd.get_dummies(data["DAY_OF_WEEK"])
        week.columns = ["Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"]
        week = week.drop(["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], axis=1)

        data = pd.concat([data, week], axis=1)
        data = data.drop(["DAY_OF_WEEK"],axis=1)
        return data
    
    def dummy_month(data):
        """
        Description

        :param data:
        :param days:
        """
        month = pd.get_dummies(data["MONTH"])
        month.columns = ["Jan", "Feb", "Mar",
                        "Apr", "May", "Jun", "Jul",
                       "Aug", "Sep", "Oct", "Nov", "Dec"]
        month = month.drop(["Dec"], axis=1)

        data = pd.concat([data, month], axis=1)
        data = data.drop(["MONTH"], axis=1)
        return data

    data = clean_variables_no_info(data)                                                       
    #data = delaytype(data)
    data = round_hours(data)
    data = select_number_of_airports(data, number_of_airports_included)
    data = select_number_of_airports(data, number_of_airlines_included)
    data = dummy_generator(data, taxi_in_out_boolean, airline_boolean)
    data = air_delay(data, define_cut_off)
    data = dummy_day(data)
    data = dummy_month(data)
    data = dummy_afternoon(data)

    data.drop(["DEP_DEL15", "CRS_DEP_TIME",  "CRS_ARR_TIME",
               "OP_CARRIER", "ORIGIN", "DEST", "TAXI_OUT", "TAXI_IN",
               "ACTUAL_ELAPSED_TIME", "CRS_ARR_TIMEHour", "DEP_DELAY_NEW", "ARR_DELAY_NEW", "DISTANCE"],
              axis=1, inplace=True)
    #set canceled flights to have had delays
    data = data.dropna()
    
    data = data.sample(n=sample_size)

    return data

data_jan = pd.read_csv("2019_jan_flights.csv")
data_feb = pd.read_csv("2019_feb_flights.csv")
data_mar = pd.read_csv("2019_mar_flights.csv")
data_apr = pd.read_csv("2019_apr_flights.csv")
data_may = pd.read_csv("2019_may_flights.csv")
data_jun = pd.read_csv("2019_jun_flights.csv")
data_jul = pd.read_csv("2019_jul_flights.csv")
data_aug = pd.read_csv("2019_aug_flights.csv")
data_sep = pd.read_csv("2019_sep_flights.csv")
data_oct = pd.read_csv("2019_oct_flights.csv")
data_nov = pd.read_csv("2019_nov_flights.csv")
data_dec = pd.read_csv("2019_dec_flights.csv")

data = pd.concat([data_jan, data_feb, data_mar, data_apr, data_may, data_jun, data_jul, data_aug, data_sep, data_oct, data_nov, data_dec], ignore_index = True)

data = preprocess(data)

data.head(5)

