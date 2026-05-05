import pandas as pd
import json
import numpy as np

try:
    reviews = pd.read_csv('../data/processed/labeled_reviews.csv')
    benchmarks = pd.read_csv('../data/processed/benchmark_results.csv')
    
    # Handle NaN values to make it valid JSON
    reviews = reviews.replace({np.nan: None})
    benchmarks = benchmarks.replace({np.nan: None})
    
    data = {
        'reviews': reviews.to_dict(orient='records'),
        'benchmarks': benchmarks.to_dict(orient='records')
    }
    with open('src/data/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Successfully converted CSV to JSON (fixed NaN)")
except Exception as e:
    print(f"Error: {e}")
