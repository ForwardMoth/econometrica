import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

"""class, which contains object - dataFrame and methods with them"""


class Data:
    """read data from Excel file, create most used attributes"""

    def __init__(self, file_name, sheet_name):
        self.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
        self.alpha = 0.05
        self.df = pd.ExcelFile(file_name).parse(sheet_name=sheet_name, header=0, names=self.columns, index_col=0,
                                                axis=0)
        self.y = np.array(self.df[self.columns[0]])
        self.x = np.array(self.df[self.columns[1:]])
        self.regression = None
        self.residuals = None
        self.distribution = {"stat": None, 'pv': None, "KS_stat": None, "KS_pv": None}
        self.histData = None

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

    """Calculate coefficients of linear regression"""

    def linear_regression(self):
        self.regression = sm.OLS(self.y, sm.add_constant(self.x)).fit()
        print(self.regression.summary(), '\n')

    """Calculate residuals of regression model"""

    def regression_residuals(self):
        self.residuals = self.regression.resid
        print('Residuals of regression model:')
        print(self.residuals, '\n')

    """Check hypothesis of normal distribution residuals"""

    def residuals_normality_test(self):
        print("Pearson test:")
        self.distribution['stat'], self.distribution['pv'] = stats.normaltest(self.residuals)
        if self.distribution['pv'] < self.alpha:
            print("The null hypothesis can be rejected:", "p-value: " + str(self.distribution['pv']) + ';',
                  "statistic: " + str(self.distribution['stat']), '\n')
        else:
            print("The null hypothesis cannot be rejected:", "p-value: " + str(self.distribution['pv']) + ';',
                  "statistic: " + str(self.distribution['stat']), '\n')

    """Kolmogorov-Smirnov test on normality of residuals"""

    def kolmogorov_smirnov_test(self):
        self.distribution["KS_stat"], self.distribution["KS_pv"] = stats.kstest(self.residuals, cdf="norm",
                                                                                args=(self.residuals.mean(),
                                                                                      self.residuals.std()))

    """Create helpful variable for line of normal distribution"""

    def data_for_line_of_distribution(self):
        range_ = np.arange(min(self.residuals), max(self.residuals), 0.05)
        normModel = stats.norm(self.residuals.mean(), self.residuals.std())
        coef_y = len(self.residuals) * max([1, (max(self.histData[0]) // (normModel.pdf(self.residuals.mean()) *
                                                                          len(self.residuals)))])
        return range_, normModel, coef_y

    """Draw histogram of distribution of residuals"""

    def draw_distribution_of_residuals(self):
        plt.figure(figsize=(11, 8))
        self.histData = plt.hist(self.residuals)

        range_, normModel, coef_y = self.data_for_line_of_distribution()
        plt.plot(range_, [normModel.pdf(x) * coef_y for x in range_], color="r")
        plt.xticks(self.histData[1])

        self.kolmogorov_smirnov_test()

        plt.title("Histogram of the distribution of regression residuals\n" + "Distribution: Normal\n" +
                  "Kolmogorov-Smirnov test = {:.5}, p-value = {:.5}".format(self.distribution['KS_stat'],
                                                                            self.distribution['KS_pv']), fontsize=18)
        plt.ylabel("No. of observations", fontsize=15)
        plt.xlabel("Category (upper limits)", fontsize=15)
        plt.grid()

        plt.show()

    """Check coefficents in linear regression on significant"""

    def significance_of_coefficents(self):
        # coefficents of linear regression
        b_coefficient = self.regression.params
        # p-value in test coefficents on significant
        pv_b = self.regression.pvalues
        # t-statistic - value in test coefficents on significant
        t = self.regression.tvalues
        #
        for i in range(len(b_coefficient)):
            if pv_b[i] < self.alpha:
                print(f"The null hypothesis can be rejected (p={pv_b[i]:.4f}<{self.alpha}), coefficient b{i}="
                      f"{b_coefficient[i]:.4f} is significantly different from zero. t(75)={t[i]:.3f}")
            else:
                print(f"The null hypothesis cannot be rejected (p={pv_b[i]:.4f}>{self.alpha}), coefficient b{i}="
                      f"{b_coefficient[i]:.4f} is insignificant. t(75)={t[i]:.3f}")


def main(file_name, sheet_name):
    data = Data(file_name, sheet_name)
    data.convert_data()
    data.head()

    data.descriptive_stats()
    data.linear_regression()
    data.regression_residuals()

    data.residuals_normality_test()
    data.draw_distribution_of_residuals()

    data.significance_of_coefficents()


if __name__ == '__main__':
    """Write file_name and sheetName of data"""
    file_name, sheet_name = 'lab1.xlsx', 'Обработка'
    """main function"""
    main(file_name, sheet_name)
