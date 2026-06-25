import pandas as pd
from datetime import datetime

df = pd.read_csv('Train.csv')

# Try to parse with the exact same function as in the script
def parseDateTime(dateString):
    """Parse datetime string and return datetime object"""
    if pd.isna(dateString):
        raise ValueError(f"dateString is NaN")
    try:
        s = str(dateString).strip()
        if ',' in s:
            s = s.split(',', 1)[0].strip()
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Failed to parse datetime string '{dateString}': {e}")

def getStartHour(dateString):
    return int(parseDateTime(dateString).hour)

# Test on the sampled data (same way as the script)
taxi = df.sample(frac=1, random_state=42).copy()
print(f"Total rows in sample: {len(taxi)}")

# Try the apply operation
try:
    print("Attempting to apply getStartHour...")
    taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
    print("✓ Success!")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}")
    print(f"Error message: {e}")
    
    # Try to find the problematic row
    print("\nFinding problematic rows...")
    for idx in range(min(1000, len(taxi))):
        try:
            getStartHour(taxi['pickup_datetime'].iloc[idx])
        except Exception as row_error:
            print(f"Row {idx}: {taxi['pickup_datetime'].iloc[idx]} - {row_error}")
            break
