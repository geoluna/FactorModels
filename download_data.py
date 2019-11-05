import requests

# data links available from
# https://research.stlouisfed.org/econ/mccracken/fred-databases/

url = 'https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv'
r = requests.get(url, allow_redirects=True)
open('data/current.csv', 'wb').write(r.content)