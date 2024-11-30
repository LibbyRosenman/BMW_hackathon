import pandas as pd

def add_weather_to_train(train_file, weather_file, output_file):
    # Load the train data and weather data from CSV files
    train_df = pd.read_csv(train_file)
    weather_df = pd.read_csv(weather_file)

    # Convert the 'datetime' columns to datetime type in both dataframes
    train_df['message_timestamp'] = pd.to_datetime(train_df['message_timestamp'], errors='coerce')
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], errors='coerce')

    # Initialize new columns for temperature and humidity
    train_df['temp'] = None
    train_df['humidity'] = None

    # Iterate through each row in the train data
    for index, row in train_df.iterrows():
        # Find the closest datetime in the weather data
        closest_time = weather_df.iloc[(weather_df['datetime'] - row['message_timestamp']).abs().argsort()[:1]]
        
        # Extract the temperature and humidity for the closest datetime
        temp = closest_time['temp'].values[0]
        humidity = closest_time['humidity'].values[0]

        # Assign the found values to the train dataframe
        train_df.at[index, 'temp'] = temp
        train_df.at[index, 'humidity'] = humidity

    # Save the updated train dataframe with weather data
    train_df.to_csv(output_file, index=False)

# Example usage:
# add_weather_to_train('train.csv', 'filtered_weather_data.csv', 'train_with_weather.csv')
