import pandas as pd
def convert_date(date_str):
    # Check if the string is in the format 'yyyy-mm'
    if len(date_str) == 7 and date_str[4] == '-':
        #yy-mm only in csv it's found the day to be 01
        return pd.to_datetime(date_str[:4] + '-' + date_str[5:] + '-01', format='%Y-%m-%d')
    # Check if the string is in the format 'yyyy-mm-dd'
    elif len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    # Check if the string is in the format 'm/d/yyyy'
    elif '/' in date_str:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    elif len(date_str) == 4:  # Check if the string is only the year
        return pd.to_datetime(date_str + '-01-01', format='%Y-%m-%d')  
    else:
        return pd.to_datetime(date_str, errors='coerce')


print(convert_date("1/30/2001").year)

