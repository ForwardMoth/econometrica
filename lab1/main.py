import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

"""Class, which contains parameters of linear regression"""


class Regression:
    def __init__(self):
        self.alpha = 0.05
        self.linear_regression = None
        self.residuals = None
        self.distribution = {"stat": None, 'pv': None, "KS_stat": None, "KS_pv": None}
        self.histData = None
        self.param_regression = {'coefficients': None, 'std': None, 'pv': None, 't': None, 'df': None, 'nobs': None,
                                 'r2': None}

    """Setting and getting regression results"""

    def set_regression_results(self, x, y):
        self.set_linear_regression(x, y)
        self.get_linear_regression()
        self.set_regression_residuals()
        self.get_regression_residuals()
        self.set_param_regression()

    """Calculate coefficients of linear regression"""

    def set_linear_regression(self, x, y):
        self.linear_regression = sm.OLS(y, sm.add_constant(x)).fit()

    """Results of linear regression"""

    def get_linear_regression(self):
        print(self.linear_regression.summary(), '\n')

    """Calculate residuals of regression model"""

    def set_regression_residuals(self):
        self.residuals = self.linear_regression.resid

    """Set of helpful params of regression"""

    def set_param_regression(self):
        self.param_regression = {'coefficients': self.linear_regression.params, 'std': self.linear_regression.bse,
                                 'pv': self.linear_regression.pvalues, 't': self.linear_regression.tvalues,
                                 'df': int(self.linear_regression.df_model), 'nobs': int(self.linear_regression.nobs),
                                 'r2': self.linear_regression.rsquared}

    """Show residuals"""

    def get_regression_residuals(self):
        print('Residuals of regression model:')
        print(self.residuals, '\n')

    """Check hypothesis of normal distribution residuals"""

    def residuals_normality_test(self):
        print("Pearson test:")
        self.distribution['stat'], self.distribution['pv'] = stats.normaltest(self.residuals)
        if self.distribution['pv'] < self.alpha:
            print("The null hypothesis can be rejected:", f"p-value: {self.distribution['pv']:.3f};",
                  f"statistic: {self.distribution['stat']:.3f}", '\n')
        else:
            print("The null hypothesis cannot be rejected:", f"p-value: {self.distribution['pv']:.3f};",
                  f"statistic: {self.distribution['stat']:.3f}", '\n')

    """Kolmogorov-Smirnov test on normality of residuals"""

    def kolmogorov_smirnov_test(self):
        self.distribution["KS_stat"], self.distribution["KS_pv"] = stats.kstest(self.residuals, cdf="norm",
                                                                                args=(self.residuals.mean(),
                                                                                      self.residuals.std()))

    """Create helpful variable for line of normal distribution"""

    def data_for_line_of_distribution(self):
        range_ = np.arange(min(self.residuals), max(self.residuals), 0.05)
        normModel = stats.norm(self.residuals.mean(), self.residuals.std())
        coefficient_y = len(self.residuals) * max([1, (max(self.histData[0]) // (normModel.pdf(self.residuals.mean()) *
                                                                                 len(self.residuals)))])
        return range_, normModel, coefficient_y

    """Draw histogram of distribution of residuals"""

    def draw_distribution_of_residuals(self):
        plt.figure(figsize=(11, 8))
        self.histData = plt.hist(self.residuals)

        range_, normModel, coefficient_y = self.data_for_line_of_distribution()
        plt.plot(range_, [normModel.pdf(x) * coefficient_y for x in range_], color="r")
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
        for i in range(len(self.param_regression['coefficients'])):
            if self.param_regression['pv'][i] < self.alpha:
                print(f"The null hypothesis can be rejected (p={self.param_regression['pv'][i]:.4f}<{self.alpha}), "
                      f"coefficient b{i}={self.param_regression['coefficients'][i]:.4f} "
                      f"is significantly different from zero. t(75)={self.param_regression['t'][i]:.3f}")
            else:
                print(f"The null hypothesis cannot be rejected (p={self.param_regression['pv'][i]:.4f}>{self.alpha}), "
                      f"coefficient b{i}={self.param_regression['coefficients'][i]:.4f} is insignificant. "
                      f"t(75)={self.param_regression['t'][i]:.3f}")

    """Confidence interval of b coefficients"""

    def create_confidence_interval(self):
        t_const = stats.t.ppf(1 - self.alpha / 2, self.param_regression['nobs'] - self.param_regression['df'] - 1)
        for i in range(len(self.param_regression['coefficients'])):
            print('\n' + f"interval: "
                         f"{self.param_regression['coefficients'][i] - t_const * self.param_regression['std'][i]:.2f} "
                         f"< b{i} < "
                         f"{self.param_regression['coefficients'][i] + t_const * self.param_regression['std'][i]:.2f}")


"""Class, which contains object - dataFrame and methods with them"""


class Data:
    """Read data from Excel file, create most used attributes"""

    def __init__(self, file_name, sheet_name):
        self.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
        self.df = pd.ExcelFile(file_name).parse(sheet_name=sheet_name, header=0, names=self.columns, index_col=0,
                                                axis=0)
        self.y = np.array(self.df[self.columns[0]])
        self.x = np.array(self.df[self.columns[1:]])

    """Convert all columns in float type"""

    def convert_data(self):
        for i in self.columns:
            self.df[i] = self.df[i].astype('float')

    """Shows information about first 5 rows"""

    def head(self):
        print("Show first 5 rows of data:" + '\n')
        print(self.df.head())

    """Shows descriptive statistic of values"""

    def descriptive_stats(self):
        print('\n' + "Show descriptive statistic of data" + "\n")
        print(pd.DataFrame(self.x, columns=[self.columns[1:]]).describe())

    """Create correlation matrix"""

    def correlation_matrix(self):
        corr_matrix = self.df.loc[:, self.columns].corr()
        print()
        print(corr_matrix)

    """Test on multicollinearity with using determination coefficient"""

    def test_multicollinearity_r2(self):
        results = []
        print()
        for i in range(1, len(self.columns)):
            regression = Regression()
            regression.set_linear_regression(self.df.drop([self.columns[0], self.columns[i]], axis=1),
                                             self.df[self.columns[i]])

            print(f"For {self.columns[i]} R^2 = {regression.linear_regression.rsquared:.3f} ")
            if regression.linear_regression.rsquared > 0.6:
                results.append(self.columns[i])

        if len(results) > 0:
            print('\n' + f"Analysis of estimates of determination coefficients showed existence of link between "
                         f"dependent values {', '.join([str(x) for x in results])} and all other dependent variables, "
                         f"respectively. Thus, we can conclude that there is multicollinearity")
        else:
            print('\n' + "Thus, we cannot conclude that there is multicollinearity")

    """Method of eliminating multicollinearity by including variables"""

    def forward_select(self):
        regression = Regression()
        x_values = set(self.columns[1:])
        selected_values = []
        current_score, best_new_score = 0.0, 0.0
        while x_values and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in x_values:
                regression.set_linear_regression(self.df[[candidate] + selected_values], self.y)
                score = regression.linear_regression.rsquared_adj
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                x_values.remove(best_candidate)
                selected_values.append(best_candidate)
                current_score = best_new_score
        regression.set_linear_regression(self.df[selected_values], self.y)
        return regression


def main(file_name, sheet_name):
    data = Data(file_name, sheet_name)
    data.convert_data()
    data.head()
    data.descriptive_stats()

    linear_regression = Regression()
    linear_regression.set_regression_results(data.x, data.y)
    linear_regression.residuals_normality_test()
    # linear_regression.draw_distribution_of_residuals()

    linear_regression.significance_of_coefficents()
    linear_regression.create_confidence_interval()

    data.correlation_matrix()
    data.test_multicollinearity_r2()

    forward_regression = data.forward_select()
    forward_regression.get_linear_regression()




if __name__ == '__main__':
    """Write file_name and sheetName of data"""
    file_name, sheet_name = 'lab1.xlsx', 'Обработка'
    """main function"""
    main(file_name, sheet_name)
