# In[ ]:

pd.set_option("display.max_columns", None)


def preprocess(data, number_of_airports_included=10, number_of_airlines_included=10):

    def clean_variables_no_info(data):
        """
        Description: drops columns that provide no information
        :param data: flights dataframe
        :return: returns cleaned dataframe
        """

        NAs_to_remove = ["Unnamed: 15"]

        cleaned_data = data.drop(NAs_to_remove, axis=1)

        return cleaned_data

    def round_hours(data):

        data.dropna(subset=["CRS_DEP_TIME"], inplace=True)
        return_hour(data)

        return data

    def return_hour(data):
        my_list = []

        for x in data["CRS_DEP_TIME"]:
            a = math.floor(x / 100) - 1
            my_list.append(a)

        data["TIME"] = my_list
        data = data.drop(["CRS_DEP_TIME"], axis=1)

        return data

    def select_number_of_airports(data, number_of_airports_included):

        main_origins = data[["ORIGIN", "DEST"]].groupby(["ORIGIN"]).count().sort_values(
            by="DEST", ascending=False).head(number_of_airports_included).reset_index()["ORIGIN"]

        main_dests = data[["DEST", "ORIGIN"]].groupby(["DEST"]).count().sort_values(
            by="ORIGIN", ascending=False).head(number_of_airports_included).reset_index()["DEST"]

        data = data[data["ORIGIN"].isin(main_origins) & data["DEST"].isin(main_dests)]

        return data

    def select_number_of_airlines(data, number_of_airlines_included):

        main_airlines = data[["OP_CARRIER", "DEST"]].groupby(["OP_CARRIER"]).count().sort_values(by="DEST", ascending=False).head(
            number_of_airlines_included).reset_index()["OP_CARRIER"]

        data = data[data["OP_CARRIER"].isin(main_airlines)]

        return data

    def extra_features(data):

        data['avg_tax_out'] = data.groupby(['OP_CARRIER', 'ORIGIN'])['TAXI_OUT'].transform('mean')
        data['avg_tax_out'] = MinMaxScaler().fit_transform(data[['avg_tax_out']])
        data['avg_tax_out'] = data['avg_tax_out'].where(data['avg_tax_out'] < .7, .7)*142
        data['avg_tax_in'] = data.groupby(['OP_CARRIER', 'DEST'])['TAXI_IN'].transform('mean')
        data['avg_tax_in'] = MinMaxScaler().fit_transform(data[['avg_tax_in']])
        data['avg_tax_in'] = data['avg_tax_in'].where(data['avg_tax_in'] < .2, .2)*500
        data['airline_delay'] = data["DEP_DELAY_NEW"]-data["TAXI_OUT"]
        data['airline_delay'] = data.groupby(['OP_CARRIER', 'ORIGIN'])[
            'airline_delay'].transform('mean')
        data['airline_delay'] = MinMaxScaler().fit_transform(data[['airline_delay']])
        data['airline_delay'] = data['airline_delay'].where(data['airline_delay'] < .25, .25)*500
        return data

    def dummy_generator(data):
        dum_orig = pd.get_dummies(data["ORIGIN"], prefix="origin").astype(int)

        dum_dest = pd.get_dummies(data["DEST"], prefix="dest").astype(int)

        dum_air = pd.get_dummies(data["OP_CARRIER"], prefix="airline")

        dum_orig = dum_orig.drop(['origin_ATL'], axis=1)
        dum_dest = dum_dest.drop(['dest_ATL'], axis=1)
        dum_air = dum_air.drop(['airline_WN'], axis=1)

        data = pd.concat([data, dum_orig, dum_dest, dum_air], axis=1)
        return data

    data = clean_variables_no_info(data)
    data = round_hours(data)
    data = select_number_of_airports(data, number_of_airports_included)
    data = select_number_of_airports(data, number_of_airlines_included)
    data = extra_features(data)
    data = dummy_generator(data)

    data.drop(["DEP_DEL15", "CRS_DEP_TIME",  "CRS_ARR_TIME",
               "TAXI_OUT", "TAXI_IN", 'MONTH',
               "ACTUAL_ELAPSED_TIME", "CRS_ARR_TIME", "DEP_DELAY_NEW", "ARR_DELAY_NEW", "DISTANCE"],
              axis=1, inplace=True)
    data.drop(["OP_CARRIER", "ORIGIN", "DEST"],
              axis=1, inplace=True)
    data = data.dropna()
    data['ARR_DEL15'] = data['ARR_DEL15'].astype(int)

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

data = pd.concat([data_jan, data_feb, data_mar, data_apr, data_may, data_jun,
                  data_jul, data_aug, data_sep, data_oct, data_nov, data_dec], ignore_index=True)

data = preprocess(data)
#data = data.sample(50000)
data.head()
