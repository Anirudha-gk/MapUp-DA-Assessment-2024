from typing import Dict, List, Tuple

import pandas as pd
import itertools
import re
import polyline
from geopy.distance import geodesic
from math import radians, cos, sin, asin, atan2, sqrt
import numpy as np


#Question 1

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        chunk = []
        
        for j in range(i, min(i + n, len(lst))):
            chunk.insert(0, lst[j])
        result.extend(chunk)

    return result
#Test cases as given 
#1
print(reverse_by_n_elements([1,2,3,4,5,6,7,8], 3))
#2
print(reverse_by_n_elements([1,2,3,4,5], 2))
#3
print(reverse_by_n_elements([10,20,30,40,50,60,70], 4))

#Question 2

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}

    for string in lst: 
        length = len(string)
        if length not in result:
            result[length] = []

        result[length].append(string)

    
    return dict(sorted(result.items()))

#Test cases
#1
print(group_by_length(["apple","bat","car","elephant","dog","bear"]))
#2
print(group_by_length(["one","two","three","four"]))


#Question 3

def flatten_dict(nested_dict: dict, sep: str = ".") -> dict:
  """Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

  Args:
    nested_dict: The dictionary object to flatten.
    sep: The separator to use between parent and child keys (defaults to ".").

  Returns:
    A flattened dictionary.
  """

  flattened_dict = {}
  for key, value in nested_dict.items():
    if isinstance(value, dict):
      flattened_dict.update(flatten_dict(value, sep=key + sep))
    elif isinstance(value, list):
      for i, item in enumerate(value):
        flattened_dict.update(flatten_dict({str(i): item}, sep=key + f"[{i}]{sep}"))
    else:
      flattened_dict[key] = value
  return flattened_dict

# Sample usage
nested_dict = {
  "road": {
    "name": "Highway 1",
    "length": 350,
    "sections": [
      {
        "id": 1,
        "condition": {
          "pavement": "good",
          "traffic": "moderate"
        }
      }
    ]
  }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)


#Question 4

def unique_permutations(nums: List[int]) -> List[List[int]]:
  """Generates all unique permutations of a list that may contain duplicates.

  Args:
    nums: List of integers (may contain duplicates)

  Returns:
    List of unique permutations
  """

  perms = set(itertools.permutations(nums))
  return list(perms)

# Example usage
lst = [1, 1, 2]
unique_perms = unique_permutations(lst)
print("Unique Permutations:", unique_perms)

pass

#Question 5

def find_all_dates(text: str) -> List[str]:
  """Finds all valid dates in a text string.

  Args:
    text: The text string.

  Returns:
    A list of valid dates.
  """

  date_patterns = [
      r"\d{2}-\d{2}-\d{4}",  # dd-mm-yyyy
      r"\d{2}/\d{2}/\d{4}",  # mm/dd/yyyy
      r"\d{4}\.\d{2}\.\d{2}"  # yyyy.mm.dd
  ]

  dates = []
  for pattern in date_patterns:
    dates.extend(re.findall(pattern, text))
  return dates

#Test case

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another on 1994.08.23"
valid_dates = find_all_dates(text)
print("Valid Dates:", valid_dates)

pass


#Question 6

# Function to decode polyline and convert to DataFrame
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Decodes a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.
    
    Returns:
        pd.DataFrame: DataFrame containing latitude, longitude, and distance columns.
    """
    # Step 1: Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Step 2: Convert to a Pandas DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Step 3: Calculate the distance using Haversine formula between consecutive rows
    distances = [0]  # First point has a distance of 0
    for i in range(1, len(df)):
        coord1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        coord2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance = geodesic(coord1, coord2).meters  # Calculate distance in meters
        distances.append(distance)
    
    # Step 4: Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df

# Example usage
polyline_str = 'gfo}EtohhUxD@bAxJmGF'
df = polyline_to_dataframe(polyline_str)

print("Decoded DataFrame with Distances:")
print(df)


#Question 7

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotates the matrix by 90 degrees clockwise, then replaces each element with the sum
    of all elements in the same row and column, excluding itself.
    
    Args:
        matrix (List[List[int]]): A 2D list representing the input matrix.
        
    Returns:
        List[List[int]]: The transformed matrix after rotation and transformation.
    """
    # Step 1: Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    rotated_matrix = np.rot90(matrix, -1)  # Clockwise rotation

    # Step 2: Create the final transformed matrix
    transformed_matrix = np.zeros((n, n), dtype=int)
    
    # For each element in the rotated matrix, calculate the sum of all elements
    # in the same row and column excluding the current element itself.
    for i in range(n):
        for j in range(n):
            # Sum of all elements in the same row excluding the current element
            row_sum = np.sum(rotated_matrix[i, :]) - rotated_matrix[i, j]
            # Sum of all elements in the same column excluding the current element
            col_sum = np.sum(rotated_matrix[:, j]) - rotated_matrix[i, j]
            # Store the result in the transformed matrix
            transformed_matrix[i, j] = row_sum + col_sum
    
    return transformed_matrix.tolist()

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transformed_matrix = rotate_and_transform_matrix(matrix)

print("Transformed Matrix:")
print(transformed_matrix)

#Question 8

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset to verify the completeness of the data by checking whether 
    the timestamps for each unique (id, id_2) pair cover a full 24-hour period and span all 7 days.

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    
    # Ensure timestamp columns are converted to datetime
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    
    # Initialize a list to hold the results
    results = []

    # Group by 'id' and 'id_2'
    grouped = df.groupby(['id', 'id_2'])

    for (id, id_2), group in grouped:
        # Check if the timestamps span a full 7 days (Monday to Sunday)
        days_covered = group['startTime'].dt.weekday.unique()
        if set(days_covered) != set(range(7)):
            # If not all days from Monday to Sunday are covered, mark as False
            results.append(((id, id_2), False))
            continue

        # Check if the timestamps cover the full 24-hour period
        start_time_of_day = group['startTime'].dt.time.min()
        end_time_of_day = group['endTime'].dt.time.max()

        # If they do not span from 00:00:00 to 23:59:59, mark as False
        if not (start_time_of_day == pd.Timestamp('00:00:00').time() and end_time_of_day == pd.Timestamp('23:59:59').time()):
            results.append(((id, id_2), False))
        else:
            results.append(((id, id_2), True))

    # Convert results to a multi-index boolean series
    result_series = pd.Series([res[1] for res in results], index=pd.MultiIndex.from_tuples([res[0] for res in results]))

    return result_series

df = pd.read_csv("C:/Users/anirudha/OneDrive/Desktop/Assessment/MapUp-DA-Assessment-2024/datasets/dataset-1.csv")  # Adjust to your dataset path

result = time_check(df)
print(result)
result_df = result.reset_index(name='Is_Valid')
print(result_df)