import numpy as np                          
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from sklearn import metrics


class Regression:

    @staticmethod
    def Linear_Regression(x_train,y_train,x_test,y_test):
        linear = LinearRegression()
        linear.fit(x_train, y_train)
        
        y_pred_test = linear.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred_test)
        mae = metrics.mean_absolute_error(y_test, y_pred_test)
        r2_score = metrics.r2_score(y_test, y_pred_test)
        return linear, mse, mae, r2_score * 100
    
    @staticmethod
    def LassoRegression(x_train, y_train, x_test, y_test):
        
        lasso_model = Lasso(random_state=1)
        lasso_model.fit(x_train, y_train)
        
        y_pred = lasso_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return lasso_model, mse, mae, r2 * 100
    
    @staticmethod   
    def RidgeRegression(x_train, y_train, x_test, y_test):
        
        ridge_model = Ridge(random_state=1)
        ridge_model.fit(x_train, y_train)
        
        y_pred = ridge_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return ridge_model, mse, mae, r2 * 100
    @staticmethod
    def SGDRegression(x_train, y_train, x_test, y_test):
        sgd_model = SGDRegressor(random_state=5)
        sgd_model.fit(x_train, y_train)
        
        y_pred = sgd_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return sgd_model, mse, mae, r2 * 100
    
    @staticmethod
    def Polynomial_Regression(x_train,y_train,x_test,y_test):
        
        best_model = None
        best_mse = 1e18; best_mae = 1e18; best_r2_score = -1; best_degree = 2

        for deg in range(2,7):
            poly = PolynomialFeatures(degree = deg)
            x_train_poly = poly.fit_transform(x_train)
            
            linear = Regression.Linear_Regression()
            linear.fit(x_train_poly,y_train)
            
            y_pred_test = linear.predict(poly.fit_transform(x_test))
            
            mse = metrics.mean_squared_error(y_test, y_pred_test)
            mae = metrics.mean_absolute_error(y_test, y_pred_test)
            r2_score = metrics.r2_score(y_test, y_pred_test)
            
            if(r2_score > best_r2_score):
                best_model = linear; best_mse = mse; best_mae = mae; best_r2_score = r2_score; best_degree = deg
                
        return best_model, best_mse, best_mae, best_r2_score * 100, best_degree
    
    @staticmethod
    def SVR(x_train, y_train, x_test, y_test):
        
        svr_model = SVR(kernel='rbf')
        svr_model.fit(x_train, y_train)

        y_pred = svr_model.predict(x_test)

        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return svr_model, mse, mae, r2 * 100
    
    @staticmethod
    def NeuralNetworkRegression(x_train, y_train, x_test, y_test):
        
        nn_model = MLPRegressor(random_state=5, max_iter=10000)  
        nn_model.fit(x_train, y_train)

        y_pred = nn_model.predict(x_test)

        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return nn_model, mse, mae, r2 * 100
    
    @staticmethod
    def GradientBoostingRegression(x_train, y_train, x_test, y_test):
        
        gb_model = GradientBoostingRegressor(random_state=5) 
        gb_model.fit(x_train, y_train)

        y_pred = gb_model.predict(x_test)

        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return gb_model, mse, mae, r2 * 100
    
    @staticmethod
    def DecisionTreeRegression(x_train, y_train, x_test, y_test):
        
        dt_model = DecisionTreeRegressor(random_state=5) 
        dt_model.fit(x_train, y_train)

        y_pred = dt_model.predict(x_test)

        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return dt_model, mse, mae, r2 * 100
    
    @staticmethod
    def ElasticNetRegression(x_train, y_train, x_test, y_test):
        
        en_model = ElasticNet(random_state=5)  
        en_model.fit(x_train, y_train)

        y_pred = en_model.predict(x_test)

        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return en_model, mse, mae, r2 * 100

    @staticmethod
    def RandomForestRegression(x_train, y_train, x_test, y_test):
        
        rf_model = RandomForestRegressor(random_state=5)
        rf_model.fit(x_train, y_train)
        
        y_pred = rf_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return rf_model, mse, mae, r2 * 100
    
    @staticmethod
    def AdaBoostRegression(x_train, y_train, x_test, y_test):
        
        ab_model = AdaBoostRegressor(random_state=5)
        ab_model.fit(x_train, y_train)
        
        y_pred = ab_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return ab_model, mse, mae, r2 * 100
    
    @staticmethod
    def BaggingRegression(x_train, y_train, x_test, y_test):
        
        bag_model = BaggingRegressor(random_state=5)
        bag_model.fit(x_train, y_train)
        
        y_pred = bag_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return bag_model, mse, mae, r2 * 100
    
    @staticmethod
    def KNNLinearRegression(x_train, y_train, x_test, y_test):
        
        knn_model = KNeighborsRegressor()
        linear_model = LinearRegression()
        knn_linear_model = make_pipeline(knn_model, linear_model)
        knn_linear_model.fit(x_train, y_train)
        
        y_pred = knn_linear_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        return knn_linear_model, mse, mae, r2 * 100
    
    @staticmethod
    def XGBoostRegression(x_train, y_train, x_test, y_test):
        
        xgb_model = xgb.XGBRegressor(random_state=1)
        xgb_model.fit(x_train, y_train)
        
        y_pred = xgb_model.predict(x_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
            
        return xgb_model, mse, mae, r2 * 100
        
    
    # index: 0     1   2   3  (4 if exist)
    # value: model MSE MAE r2 poly_degree
    
    def FindBestModel(self):
        best = self.list_of_models[0] 
        index = 0
        
        for i, model in enumerate(self.list_of_models):
            if(model[3] > best[3]):
                best = model
                index = i
                
        return index, best
            
    def __init__(self, x_train, y_train, x_test, y_test):
        self.list_of_models = [
            [Regression.Linear_Regression(x_train, y_train, x_test, y_test)],
            [Regression.LassoRegression(x_train, y_train, x_test, y_test)],
            [Regression.RidgeRegression(x_train, y_train, x_test, y_test)],
            [Regression.SGDRegression(x_train, y_train, x_test, y_test)],
            [Regression.Polynomial_Regression(x_train,y_train,x_test,y_test)],
            [Regression.SVR(x_train, y_train, x_test, y_test)],
            [Regression.NeuralNetworkRegression(x_train, y_train, x_test, y_test)],
            [Regression.GradientBoostingRegression(x_train, y_train, x_test, y_test)],
            [Regression.DecisionTreeRegression(x_train, y_train, x_test, y_test)],
            [Regression.ElasticNetRegression(x_train, y_train, x_test, y_test)],
            [Regression.RandomForestRegression(x_train, y_train, x_test, y_test)],
            [Regression.AdaBoostRegression(x_train, y_train, x_test, y_test)],
            [Regression.BaggingRegression(x_train, y_train, x_test, y_test)],
            [Regression.KNNLinearRegression(x_train, y_train, x_test, y_test)],
            [Regression.XGBoostRegression(x_train, y_train, x_test, y_test)]],
        
        self.index, self.best_model = self.FindBestModel()
        
        

