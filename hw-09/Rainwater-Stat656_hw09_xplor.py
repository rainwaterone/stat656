import pandas as pd

def heading(headerstring):
    """
    Centers headerstring on the page. For formatting to stdout
    Parameters
    ----------
    headerstring : string
    String that you wish to center.
    Returns
    -------
    Returns: None.
    """
    tw = 70 # text width
    lead = int(tw/2)-(int(len(headerstring)/2))-1
    tail = tw-lead-len(headerstring)-2
    print('\n' + ('*'*tw))
    print(('*'*lead) + ' ' + headerstring + ' ' + ('*'*tail))
    print(('*'*tw))
    return


df = pd.read_excel('HondaComplaints.xlsx')
df.head()

models = df['Model'].unique()
print(models)

years = df['Year'].unique()
print(years)

abs = df['abs'].unique()
print(abs)

print(df['cruise'].unique())

print(df['crash'].unique())
df.query("crash == 'Y'")['crash'].count()

print(df['mph'].min())

print(df['mph'].max())

print(df['mileage'].min())

print(df['mileage'].max())

print(df['mileage'] > 200000)

df.query('mileage > 200000')['mileage'].count()

df['description'].str.contains('SINC ').count()

df[df['description'].str.contains(r'\bSINC\b')]

df.head()
print(df)

df.size