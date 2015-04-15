import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option('display.max_columns', 75)

# load the data file
# we skip row 1 because it's a note that affects our headers.
# we only grab 235,629 rows because the last two rows are a single description 
#  in column 0.
# try it without each and see what happens.
dfLoanData = pd.read_csv('LoanStats3c.csv', skiprows=1, nrows=235629)

# It should still yell at us that the column 19 (aka 'desc') is also mixed type.
# so we drop that column - we're not using it anyway.
dfLoanData = dfLoanData.drop('desc', axis=1)

dfLoanData['int_rate'] = dfLoanData.apply(lambda x: pd.Series(x['int_rate'].rstrip('%')).astype('float') / 100, axis=1)

# now we're ready to start thinking about the regression.
# first: interest rate is a string with a '%' symbol in it.
# use code you've already written to handle that.

# Income is already numeric, so you'll be fine there. You can begin to model 
#   that now.

# your first regression equation should be of the form:
# interest = intercept + constant1 * income

# use statsmodels and the same technique you used in the earlier linear 
# regression to solve.

X = dfLoanData['annual_inc']
y = dfLoanData['int_rate']

## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

#print est.summary()


# next one is home ownership. These are categorical variables, so you need to
# code them from string to numeric. Easiest way to do that is pandas categorical
# functions.

# for a really good summary of multiple regression with statsmodels and the theory
# behind it, read this post: http://blog.yhathq.com/posts/logistic-regression-and-python.html

dfDummies = pd.get_dummies(dfLoanData['home_ownership'])

dfLoanData = pd.merge(dfLoanData, dfDummies, left_index=True, right_index=True)

# Note that for class - we fit the entire data set.
# in a real problem we wouldn't do that; we'd split the data into training and test datasets.
# scikit-learn has a helper that makes this really easy.
# link here: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html


print 'Logistic Regression - No Interactions'
X = dfLoanData[['annual_inc', 'MORTGAGE', 'RENT', 'OWN']]
y = dfLoanData['int_rate']

logit = sm.Logit(y, X)
result = logit.fit()

print result.summary()

print "confidence interval"
print result.conf_int()

print "odds ratios"
print np.exp(result.params)

print "odds ratios and 95% confidence intervals"
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

# Now model the interaction between income and home ownership
dfLoanData['income_ANY'] = dfLoanData.apply(lambda x: x['annual_inc'] * x['ANY'], axis = 1)
dfLoanData['income_MORTGAGE'] = dfLoanData.apply(lambda x: x['annual_inc'] * x['MORTGAGE'], axis = 1)
dfLoanData['income_OWN'] = dfLoanData.apply(lambda x: x['annual_inc'] * x['OWN'], axis = 1)
dfLoanData['income_RENT'] = dfLoanData.apply(lambda x: x['annual_inc'] * x['RENT'], axis = 1)

X2 = dfLoanData[['income_MORTGAGE', 'income_RENT', 'income_OWN']]

logit2 = sm.Logit(y, X2)
result2 = logit2.fit()

print result2.summary()

print "confidence interval"
print result2.conf_int()

print "odds ratios"
print np.exp(result2.params)

print "odds ratios and 95% confidence intervals"
params2 = result2.params
conf2 = result2.conf_int()
conf2['OR'] = params
conf2.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf2)