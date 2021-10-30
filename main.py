import pandas as pd
import pprint as pp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def main():
    #Import the data
    data = pd.read_csv('gps_data.csv')


    #Drop NaN values, as well as infinite values that bug the "float64's"
    data.dropna(axis=0)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    pp.pprint(data)


    #Dividing Data into features(x) and labels(y)
    X = data.iloc[: , :5]
    y = data.iloc[:, 5:]
    print(X.head())
    print(y.head())


    #Converting categorical data to numbers
    numerical = X.drop(['Player Name', 'Period Name', 'Position Name'], axis=1)
    categorical = X.filter(['Player Name', 'Period Name', 'Position Name'])

    cat_numerical = pd.get_dummies(categorical, drop_first=True)
    X = pd.concat([numerical, cat_numerical], axis=1)


    #Split into training test and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=0)


    #Normalize/Scale the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    

    #Pick model and Train
    model = RandomForestRegressor()
    regressor = model.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    pd.DataFrame(y_pred).to_excel('y_pred.xlsx')
    pd.DataFrame(y_test).to_excel('y_test.xlsx')


    #Evaluate model with metrics module
    print('Mean Absolue Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test,  y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == "__main__":
    main()