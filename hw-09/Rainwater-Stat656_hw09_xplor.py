import pandas as pd

df = pd.read_excel('HondaComplaints.xlsx')
df.head()

models = df['Model'].unique()
print(models)

years = df['Year'].unique()
print(years)