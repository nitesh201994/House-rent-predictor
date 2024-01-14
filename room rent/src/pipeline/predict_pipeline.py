import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        House_Age: int,
        BEDROOMS: int,
        ROOMS: int,
        PERSONS: int,
        METRO: str,
        REGION: str
        ):

        self.House_Age = House_Age

        self.BEDROOMS = BEDROOMS

        self.ROOMS = ROOMS

        self.PERSONS = PERSONS

        self.METRO = METRO

        self.REGION = REGION

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "House_Age": [self.House_Age],
                "BEDROOMS": [self.BEDROOMS],
                "ROOMS": [self.ROOMS],
                "PERSONS": [self.PERSONS],
                "METRO": [self.METRO],
                "REGION": [self.REGION]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)