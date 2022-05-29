from lifelines.datasets import load_dd
import pandas as pd


data = load_dd()


data.to_csv('data/regimes.csv', index=False)


from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()


T = data["duration"]
E = data["observed"]

kmf.fit(T, event_observed=E)


from matplotlib import pyplot as plt


kmf.survival_function_.plot()
plt.title('Survival function of political regimes');



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter 
from lifelines.statistics import (logrank_test, 
                                  pairwise_logrank_test, 
                                  multivariate_logrank_test, 
                                  survival_difference_at_fixed_point_in_time_test)


churn_data = pd.read_csv(
    'https://raw.githubusercontent.com/treselle-systems/'
    'customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# transform tenure and churn features
churn_data['tenure'] = churn_data['tenure'].astype(float)
churn_data['Churn'] = churn_data['Churn'] == 'Yes'


# fitting kmf to churn data
t = churn_data['tenure'].values
churn = churn_data['Churn'].values
kmf = KaplanMeierFitter()
kmf.fit(t, event_observed=churn, label='Estimate for Average Customer')


kmf.survival_function_.plot()



