import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def main():
    #Import the data
    data = pd.read_csv('gps_data.csv')
    print(data.head())
    print(data.info())

    #Dividing Data into features(x) and labels(y)
    X = data
    y = data.iloc[: , :5]

    #Converting categorical data to numbers
    numerical = X.drop(['Player Name', 'Period Name', 'Position Name'], axis=1)
    categorical = X.filter(['Player Name', 'Period Name', 'Position Name'])

    cat_numerical = pd.get_dummies(categorical, drop_first=True)
    X = pd.concat([numerical, cat_numerical], axis=1)

    #Split into training test and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)


    #Normalize/Scale the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    

    #Pick model and Train
    # rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
    # regressor = rf_reg.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)

    #Evaluate model with metrics module
    # print('Mean Absolue Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test,  y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    pass


if __name__ == "__main__":
    main()