from src.utils import evaluate_models

class BestOptimalAlgo:

    def __init__(self,  X_train , X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    
    def best_optimal_algo(self):
        
        dataframe = evaluate_models(X_train=self.X_train,y_train=self.y_train,X_test=self.X_test,y_test=self.y_test)
        
        ## To get best model score from dict
        # best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict
        # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
        return dataframe