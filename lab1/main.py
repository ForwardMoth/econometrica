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
                                 'r2': None, 'f': None, 'f_pv': None}
        self.variable_names = None

    """Setting and getting regression results"""

    def regression_results(self, x, y):
        self.set_linear_regression(x, y)
        self.get_linear_regression()
        self.set_regression_residuals()
        self.get_regression_residuals()
        self.set_regression_param()

    """Calculate coefficients of linear regression"""

    def set_linear_regression(self, x, y):
        self.linear_regression = sm.OLS(y, sm.add_constant(x)).fit()

    """Calculate residuals of regression model"""

    def set_regression_residuals(self):
        self.residuals = self.linear_regression.resid

    """Set of helpful params of regression"""

    def set_regression_param(self):
        self.param_regression = {'coefficients': self.linear_regression.params, 'std': self.linear_regression.bse,
                                 'pv': self.linear_regression.pvalues, 't': self.linear_regression.tvalues,
                                 'df': int(self.linear_regression.df_model), 'nobs': int(self.linear_regression.nobs),
                                 'r2': self.linear_regression.rsquared, 'f': self.linear_regression.fvalue,
                                 'f_pv': self.linear_regression.f_pvalue}
        self.set_variable_names()

    """Set variable names for beauty print"""

    def set_variable_names(self):
        self.variable_names = []
        for name in self.linear_regression.model.exog_names:
            if name == 'const':
                self.variable_names.append('0')
            else:
                self.variable_names.append(name[1])

    """Show residuals"""

    def get_regression_residuals(self):
        print('Residuals of regression model:')
        print(self.residuals, '\n')

    """Get count of depended values"""

    def get_df(self):
        return int(self.linear_regression.df_model)

    """Get adjusted r2"""

    def get_r2_adj(self):
        return self.linear_regression.rsquared_adj

    """Results of linear regression"""

    def get_linear_regression(self):
        print(self.linear_regression.summary(), '\n')

    """Get variable names for beauty print"""

    def get_variable_names(self):
        return self.variable_names

    """Get coefficients"""

    def get_coefficients(self):
        return self.param_regression['coefficients']

    """Test on significance of regression equation"""

    def test_significance_regression(self):
        if self.param_regression['f_pv'] < self.alpha:
            print(f'p-value={self.param_regression["f_pv"]:.3f} < {self.alpha}, value of F-statistic = '
                  f'{self.param_regression["f"]:.3f}. Thus, we get that the regression equation is significant' +
                  '\n')
        else:
            print(f'p-value={self.param_regression["f_pv"]:.3f} > {self.alpha}, value of F-statistic = '
                  f'{self.param_regression["f"]:.3f}. Thus, we get that the regression equation is not'
                  f' significant' + '\n')

    """Test p-values of coefficients on significance"""

    def test_significance_coefficients(self):
        return max(self.linear_regression.pvalues) >= 0.05

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
        for i in range(len(self.get_coefficients())):
            if self.param_regression['pv'][i] < self.alpha:
                print(f"The null hypothesis can be rejected (p={self.param_regression['pv'][i]:.4f}<{self.alpha}), "
                      f"coefficient b{self.variable_names[i]}={self.get_coefficients()[i]:.4f} "
                      f"is significantly different from zero. t(75)={self.param_regression['t'][i]:.3f}")
            else:
                print(f"The null hypothesis cannot be rejected (p={self.param_regression['pv'][i]:.4f}>{self.alpha}), "
                      f"coefficient b{self.variable_names[i]}={self.get_coefficients()[i]:.4f} "
                      f"is insignificant. t(75)={self.param_regression['t'][i]:.3f}")

    """Confidence interval for b significant coefficients"""

    def create_confidence_interval(self):
        t_const = stats.t.ppf(1 - self.alpha / 2, self.param_regression['nobs'] - self.param_regression['df'] - 1)
        print()
        for i in range(len(self.get_coefficients())):
            if self.param_regression['pv'][i] < self.alpha:
                print(f"interval: "
                      f"{self.get_coefficients()[i] - t_const * self.param_regression['std'][i]:.2f} "
                      f"< b{self.variable_names[i]} < "
                      f"{self.get_coefficients()[i] + t_const * self.param_regression['std'][i]:.2f}")


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
                score = regression.get_r2_adj()
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                x_values.remove(best_candidate)
                selected_values.append(best_candidate)
                current_score = best_new_score
        regression.set_linear_regression(self.df[selected_values], self.y)
        return regression

    """Method of eliminating multicollinearity by excluding variables"""

    def backward_excluded(self):
        regression = Regression()
        regression.set_linear_regression(self.x, self.y)
        x_values = set(self.columns[1:])
        while regression.test_significance_coefficients():
            scores_with_candidates = []
            for candidate in x_values:
                regression.set_linear_regression(self.df[list(x_values - set([candidate]))], self.y)
                score = regression.get_r2_adj()
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_score, remove_candidate = scores_with_candidates.pop()
            x_values.remove(remove_candidate)
            regression.set_linear_regression(self.df[list(x_values)], self.y)
        return regression


class Result:
    def __init__(self, backward_regression, forward_regression):
        self.backward_regression = backward_regression
        self.forward_regression = forward_regression
        self.best_regression = None
        self.name_of_best_regression = None

    """Set best regression after eliminating multicollinearity"""

    def set_best_regression(self, count_df):
        current_adj = -1.0
        if self.backward_regression.get_df() >= count_df // 2:
            current_adj = self.backward_regression.get_r2_adj()
        if self.forward_regression.get_df() >= count_df // 2:
            if self.forward_regression.get_r2_adj() > current_adj:
                self.best_regression = self.forward_regression
                self.name_of_best_regression = "Regression received using method including variables"
            else:
                self.best_regression = self.backward_regression
                self.name_of_best_regression = "Regression received using method excluding variables"
        self.best_regression.set_regression_residuals()
        self.best_regression.set_regression_param()

    """Sort coefficients for beauty print"""
    def bubble_sort(self, names, coefficients):
        for i in range(len(coefficients)-1):
            for j in range(len(coefficients)-i-1):
                if int(names[j+1]) < int(names[j]):
                    names[j+1], names[j] = names[j], names[j+1]
                    coefficients[j+1], coefficients[j] = coefficients[j], coefficients[j+1]
        return names, coefficients

    """print beauty view of equation"""

    def print_equation(self):
        names = self.best_regression.get_variable_names()
        coefficients = self.best_regression.get_coefficients()
        names, coefficients = self.bubble_sort(names, coefficients)
        equation = "Y = "
        for i in range(len(names)):
            if names[i] == '0':
                equation += f"{coefficients[i]:.4f}"
            elif coefficients[i] > 0:
                equation += f" + {coefficients[i]:.4f} * x" + names[i]
            else:
                equation += f" - {abs(coefficients[i]):.4f} * x" + names[i]

        print('\n' + "Result equation: ", equation)

    """Get results of analyse"""

    def get_results(self):
        print('\n' + self.name_of_best_regression + '\n')
        self.best_regression.get_linear_regression()
        self.best_regression.draw_distribution_of_residuals()
        self.best_regression.test_significance_regression()
        self.best_regression.significance_of_coefficents()
        self.best_regression.create_confidence_interval()
        self.print_equation()


def main(file_name, sheet_name):
    data = Data(file_name, sheet_name)
    data.convert_data()
    data.head()
    data.descriptive_stats()

    linear_regression = Regression()
    linear_regression.regression_results(data.x, data.y)

    linear_regression.residuals_normality_test()
    linear_regression.draw_distribution_of_residuals()
    linear_regression.test_significance_regression()
    linear_regression.significance_of_coefficents()
    linear_regression.create_confidence_interval()

    data.correlation_matrix()
    data.test_multicollinearity_r2()

    forward_regression = data.forward_select()
    backward_regression = data.backward_excluded()

    result = Result(backward_regression, forward_regression)
    result.set_best_regression(len(data.columns) - 1)
    result.get_results()


if __name__ == '__main__':
    """Write file_name and sheetName of data"""
    file_name, sheet_name = 'lab1.xlsx', 'Обработка'
    """main function"""
    main(file_name, sheet_name)
