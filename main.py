import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def main():
    #Import the data
    data = pd.read_csv('gps_data.csv')
    print(data.head(), ...)
    print(data.info())
    #Converting categorical data to numbers

    #Split into training test and test set

    #Normalize/Scale the data

    #Pick model and Train

    pass


if __name__ == "__main__":
    main()