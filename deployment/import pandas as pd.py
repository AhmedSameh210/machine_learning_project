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
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    elif len(date_str) == 4:  # Check if the string is only the year
        return pd.to_datetime(date_str + '-01-01', format='%Y-%m-%d')  # Assuming January 1st for year only
    else:
        return pd.to_datetime(date_str, errors='coerce')  # Return as it is if not recognized format

# Example usage:
date_str = "1979"
converted_date = convert_date(date_str)
print("Converted Date:", converted_date)
