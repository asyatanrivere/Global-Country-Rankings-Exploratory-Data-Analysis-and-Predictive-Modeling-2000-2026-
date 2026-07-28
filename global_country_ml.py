
import matplotlib.pyplot as plt
import os  
from sklearn.model_selection import train_test_split 
from global_country_analysis import load_data
from global_country_analysis import func
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

data="dataset/global_country_rankings_2000_2026.csv"
OUTPUT="ml plots"
os.makedirs(OUTPUT,exist_ok=True)

# DEFINING X AND Y FOR TRAINING THE MODEL
def define_x_and_y(df):
    x=df.drop(columns=["Country","Region"])
    y=df["Happiness_Rank"]
    return x,y

# SCALE WITH Z SCORE
def scale(x):
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    return x

# TRAIN & TEST
def train_model_and_predict(x_train,x_test,y_train):
    regr=linear_model.LinearRegression()
    regr.fit(x_train,y_train)

    y_predict=regr.predict(x_test)
    return y_predict

# FINAL CONTROLS
def check_model(y_test,y_predict):
    score=r2_score(y_test,y_predict)
    print(f"r2 score: {score}")

    mae = mean_absolute_error(y_test, y_predict)
    print(f"MAE: {mae}")

    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")

# FINAL CONTROL VISUALIZATION
def plot_of_test_vs_predicted(y_predict,y_test):
    plt.scatter(y_predict,y_test,s=2)
    func(y_predict,y_test,1,220,1000,"#BA1919FF")
    plt.plot()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{OUTPUT}/plot_of_test_vs_predicted.png")
    plt.show()

# MAIN PIPELINE
def main():
    df=load_data(data)
    df.drop(columns=["Gini_Rank"],inplace=True)           
    # I dropped the Gini rank because it was largely unrelated to almost all of the other columns.
    # There is an equal number of inputs for all years.

    x,y=define_x_and_y(df)
    x=scale(x)

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

    y_predict=train_model_and_predict(x_train,x_test,y_train)
    check_model(y_test,y_predict)
    """
    r2 score: 1.0
    MAE: 6.493705838503782e-14
    RMSE: 7.630653539114315e-14
    """
    plot_of_test_vs_predicted(y_predict,y_test)

# RUN
if __name__=="__main__":
    main()