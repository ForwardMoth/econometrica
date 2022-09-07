import pandas as pd
import numpy as np
import statsmodels.api as sm

"""class, which contains object - dataFrame and methods with them"""


class Data:
    """read data from Excel file, create most used attributes"""
    def __init__(self, file_name, sheet_name):
        self.regression = None
        self.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
        self.df = pd.ExcelFile(file_name).parse(sheet_name=sheet_name, header=0, names=self.columns, index_col=0,
                                                axis=0)
        self.y = np.array(self.df[self.columns[0]])
        self.x = np.array(self.df[self.columns[1:]])

    """convert all columns in float type"""
    def convert_data(self):
        for i in self.columns:
            self.df[i] = self.df[i].astype('float')

    """shows information about first 5 rows"""
    def head(self):
        print("Show first 5 rows of data:" + '\n')
        print(self.df.head())

    """Shows descriptive statistic of values"""
    def descriptive_stats(self):
        print('\n' + "Show descriptive statistic of data" + "\n")
        print(pd.DataFrame(self.x, columns=[self.columns[1:]]).describe())

    """Count coefficients of linear regression"""
    def linear_regression(self):
        self.regression = sm.OLS(self.y, sm.add_constant(self.x)).fit()
        print(self.regression.summary())



def main(file_name, sheet_name):
    data = Data(file_name, sheet_name)
    data.convert_data()
    data.head()

    data.descriptive_stats()
    data.linear_regression()


if __name__ == '__main__':
    """Write file_name and sheetName of data"""
    file_name, sheet_name = 'lab1.xlsx', 'Обработка'
    """main function"""
    main(file_name, sheet_name)

