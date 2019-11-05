import pandas as pd

def compute_percent_missing(data):
    percent_missing_data = pd.DataFrame({'Percent Missing': 100 * (data.isnull().sum() / len(data))})
    percent_missing_data = percent_missing_data.drop(
        percent_missing_data[percent_missing_data['Percent Missing'] == 0].index).sort_values(ascending=False,
                                                                                              by='Percent Missing')
    if percent_missing_data.empty == True:
        print('\n No data is missing \n')

    return percent_missing_data



data = pd.read_csv('../data/current.csv')


print(data.info())
missing_data = compute_percent_missing(data)
print(missing_data)
print(missing_data.shape)
