import os
import pickle
import pandas as pd
import time
import random
import gtrend
from pytrends.request import TrendReq
from sklearn.decomposition import PCA

# Initialize pytrends request
pytrend = TrendReq(hl='en-US')

# Query parameters
keywords = ["CBDC", "Central Bank Digital Currency"]
name = 'misc'  # Group name

# Start and end dates for the trends data YYYY-MM-DD
start = '2022-01-01'
end = '2023-01-01'

# Geo refers to the geographical location. Examples:
# - 'US' for United States (entire country)
# - 'US-CA' for California, United States (specific state)
# - 'GB' for Great Britain (entire country)
# - 'GB-ENG' for England, Great Britain (specific region)
# - 'IN' for India (entire country)
# - 'RU' for Russia (entire country)
# - '' for worldwide trends (default)
geo = ''  # Example: Change this to 'IN' for India or 'RU' for Russia, or leave empty for worldwide trends

# Category (cat) narrows down the results to a specific topic. Examples:
# - 0 for all categories (default, no filter)
# - 71 for Food & Drink (specific category)
# - 7 for Business & Industrial
# - 174 for Travel
# - 23 for News
# For a full list, you can inspect Google Trends URLs or consult online lists.
cat = 0  # Example: Change this to '7' to filter by the Business & Industrial category

# gprop refers to Google property, e.g., 'news', 'images', 'froogle', 'youtube', or leave empty for web search
gprop = ''


# Directory creation for saving files
pickle_dir = 'pickles'
csv_dir = 'csv'

# Ensure the directories exist
os.makedirs(pickle_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Define file naming based on query parameters (geo, category, and date range)
pkl_filename = f'{pickle_dir}/{name}_{geo}_{cat}_{start}_{end}.pkl'
csv_filename = f'{csv_dir}/{name}_{geo}_{cat}_{start}_{end}_google_trends.csv'

# Load trends if pickle file exists, otherwise start with an empty list
if os.path.exists(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        trends = pickle.load(file)
else:
    trends = []

# Index to track already queried keywords
start_idx = len(trends)

# Query for each keyword and append new data
for i, keyword in enumerate(keywords[start_idx:], start=start_idx):
    print(f"Fetching trend data for: {keyword}")
    
    # Fetch overlapping trend data for the given keyword
    overlapping = gtrend.get_daily_trend(pytrend, keyword, start, end, geo=geo, cat=cat, gprop=gprop, verbose=True, tz=0)
    
    # Drop the overlap column and append to the list
    trends.append(overlapping.drop(columns='overlap'))

    # Save trends to pickle after each update
    with open(pkl_filename, 'wb') as file:
        pickle.dump(trends, file)
    
    print(f"Saved updated trends to {pkl_filename}")
    
    # Sleep to avoid being blocked by Google Trends API
    time.sleep(random.gammavariate(2.99, 3.99) + 50)

# Concatenate all trend data
data = pd.concat(trends, axis=1)

# Save the combined data as CSV
data.to_csv(csv_filename, index=True)

print(f"Data saved to CSV: {csv_filename}")

# Perform PCA on the trend data
print("Performing PCA on trend data...")
pca = PCA(n_components=1)  # We'll just use the first principal component
pca_result = pca.fit_transform(data)

# Convert PCA result to DataFrame
pca_df = pd.DataFrame(pca_result, index=data.index, columns=[f'{name}_pca_component_1'])

# Save the PCA results to a CSV file
pca_csv_filename = f'{csv_dir}/{name}{geo}{cat}{start}{end}_pca_component_1.csv'
pca_df.to_csv(pca_csv_filename)

print(f"PCA first component saved to CSV: {pca_csv_filename}")

# Optionally, display some information
print(f"Total number of trends collected: {len(trends)}")
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]}")
