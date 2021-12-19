import pandas as pd
import numpy as np

class Feature():

    def __init__(self) -> None:
        pass

    def conv_data(self, data, save_fet=False, save_name='feature'):

        data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
        data['Embarked'].fillna(('S'), inplace=True)
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


        delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
        data.drop(delete_columns, axis=1, inplace=True)

        if save_fet:
            data.to_csv(save_name+'.csv', index=False)

        return data