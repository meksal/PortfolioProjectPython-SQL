
# import numpy as np
# sample = np.array([[1,2,3], [4,5,6]])
# print(sample)
#
# import matplotlib.pyplot
# data = (1, 2, 7, 8)
# fig, simple_chart = matplotlib.pyplot.subplots()
# simple_chart.plot(data)
# matplotlib.pyplot.show()

#importing libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12,8)

#reading the data
df = pd.read_csv(r"C:\Users\Mehek\Desktop\Folio\movies-python project.csv")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

#searching for missing values
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))

# looking at datatypes
# print(df.dtypes) comment when running the print below

#converting float columns to integers
df = df.fillna(0)
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
# print(df.head()) comment when running the print below

#creating new column with accurate year
# df['Proper_year'] = df['released'].astype(str).str[:4]

#sorting data by gross revenue descending
df = df.sort_values(by='gross', ascending=False, inplace=False)
print(df)

#creating scatterplot for budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')


#creating a regression plot with seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws = {'color':'red'}, line_kws = {'color':'blue'})
#plt.show()

#Looking at correlation
print(df.corr(method='pearson')) comment when running the print below


#making a correlation matrix visualization
correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
#plt.show()

#numerizing object columns (can stop here)
df_numerize = df
# for col_name in df_numerize.columns:
#     if df_numerize[col_name].dtype == 'object':
#         df_numerize[col_name] = df_numerize[col_name].astype('category')
#         df_numerize[col_name] = df_numerize[col_name].cat.codes
# print(df_numerize) comment when running the print below

correlation_matrix = df_numerize.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
#plt.show()

#correlation unstacking
corr_pairs = correlation_matrix.unstack()
# print(corr_pairs) comment when running the print below

#sorted correlation pairs
sorted_pairs = corr_pairs.sort_values()
#print(sorted_pairs) - comment when running the print below

#high correlations
high_corr = sorted_pairs[(sorted_pairs) > 0.5]
print(high_corr)