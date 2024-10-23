import pandas as pd
import numpy as np
from datetime import time



def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Step 2: Get unique locations (IDs)
    locations = pd.concat([df['id_start'], df['id_end']]).unique()

    # Step 3: Create an empty distance matrix with 0s on the diagonal
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(distance_matrix.values, 0)

    # Step 4: Fill in the known distances from the CSV
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']

    # Step 5: Ensure that the matrix is symmetric and cumulative
    # This step assumes the distances between intermediate points are additive
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j],
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])

    return distance_matrix

# Example usage
file_path = 'C:/Users/anirudha/OneDrive/Desktop/Assessment/MapUp-DA-Assessment-2024/datasets/dataset-2.csv'
distance_matrix = calculate_distance_matrix(file_path)

print("Distance Matrix:")
print(distance_matrix)


def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Unroll the distance matrix into a list of (id_start, id_end, distance)
    unrolled_data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude diagonal (same start and end)
                unrolled_data.append([id_start, id_end, distance_matrix.loc[id_start, id_end]])
    
    # Step 2: Convert the list into a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df

# Example usage
unrolled_df = unroll_distance_matrix(distance_matrix)

print("Unrolled Distance Matrix:")
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_value):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    average_value = df[df['id_start'] == reference_value]['id_start'].mean()
    
    # Calculate the lower and upper bound for the 10% range
    lower_bound = average_value * 0.9
    upper_bound = average_value * 1.1
    
    # Filter the values within the range
    filtered_values = df[(df['id_start'] >= lower_bound) & (df['id_start'] <= upper_bound)]
    
    # Return the sorted list of values
    return filtered_values['id_start'].sort_values().tolist()

# Example usage
df = pd.read_csv('C:/Users/anirudha/OneDrive/Desktop/Assessment/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')  # Load your CSV file here
reference_value = 1001400  # Example reference value
result = find_ids_within_ten_percentage_threshold(df, reference_value)
print(result)



def calculate_toll_rate(df):
    # Define the rate coefficients for different vehicle types
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # For each vehicle type, calculate the toll rate based on distance
    for vehicle, rate in rates.items():
        df[vehicle] = (df['id_end'] - df['id_start']) * rate
    
    return df

# Example usage
df = pd.read_csv('C:/Users/anirudha/OneDrive/Desktop/Assessment/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')  # Load your CSV file here
df_with_toll_rates = calculate_toll_rate(df)
print(df_with_toll_rates)


def calculate_time_based_toll_rates(df):
    # Check column names to ensure 'moto', 'car', 'rv', etc. are present
    print(df.columns)  # Add this line to check your DataFrame's column names

    # Create day and time columns based on the required conditions
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Time intervals for weekdays
    weekday_time_ranges = [
        (time(0, 0), time(10, 0), 0.8),  # 00:00 to 10:00 - discount factor of 0.8
        (time(10, 0), time(18, 0), 1.2), # 10:00 to 18:00 - discount factor of 1.2
        (time(18, 0), time(23, 59, 59), 0.8) # 18:00 to 23:59 - discount factor of 0.8
    ]
    
    # Constant discount factor for weekends
    weekend_discount = 0.7

    # Initialize a list to store the rows of the new DataFrame
    rows = []

    # Loop through each row in the dataframe (id_start and id_end data)
    for _, row in df.iterrows():
        distance = row['id_end'] - row['id_start']
        
        # Loop through each day of the week
        for day in days_of_week:
            # Check if it's a weekday or weekend
            if day in ['Saturday', 'Sunday']:
                # Apply weekend discount for all times
                start_time = time(0, 0)
                end_time = time(23, 59, 59)
                discounted_row = {
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': distance,
                    'start_day': day,
                    'end_day': day,
                    'start_time': start_time,
                    'end_time': end_time
                }
                # Apply the weekend discount to all vehicles
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    if vehicle in row:  # Ensure the column exists in the row
                        discounted_row[vehicle] = distance * row[vehicle] * weekend_discount
                rows.append(discounted_row)
            else:
                # For weekdays, apply the discount factors based on the time ranges
                for start_time, end_time, discount in weekday_time_ranges:
                    discounted_row = {
                        'id_start': row['id_start'],
                        'id_end': row['id_end'],
                        'distance': distance,
                        'start_day': day,
                        'end_day': day,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    # Apply the weekday discount to all vehicles
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        if vehicle in row:  # Ensure the column exists in the row
                            discounted_row[vehicle] = distance * row[vehicle] * discount
                    rows.append(discounted_row)
    
    # Convert the list of rows into a DataFrame
    result_df = pd.DataFrame(rows)
    
    return result_df

# Example usage
df = pd.read_csv('C:/Users/anirudha/OneDrive/Desktop/Assessment/MapUp-DA-Assessment-2024/datasets/dataset-2.csv') 
df_with_time_based_rates = calculate_time_based_toll_rates(df)
print(df_with_time_based_rates)