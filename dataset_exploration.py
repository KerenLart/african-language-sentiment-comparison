import pandas as pd
import os


data_path = "./data"  
print("=== AFRISENTI DATASET EXPLORATION ===")

# Loading and checking Twi data
print("\n--- TWI DATA ---")
for split in ['train', 'dev', 'test']:
    try:
        filename = f"twi_{split}.tsv"  
        filepath = os.path.join(data_path, filename)
        
        # Try reading as TSV
        data = pd.read_csv(filepath, sep='\t')
        
        print(f"{split.upper()} set:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        
        # Show sample data
        print(f"  Sample rows:")
        print(data.head(2).to_string())
        
        # Check label distribution if label column exists
        label_col = None
        for col in ['label', 'sentiment', 'target']:
            if col in data.columns:
                label_col = col
                break
                
        if label_col:
            print(f"  Label distribution:")
            print(data[label_col].value_counts().to_string())
        
        print()
        
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Loading and checking Hausa data
print("\n--- HAUSA DATA ---")
for split in ['train', 'dev', 'test']:
    try:
        filename = f"hausa_{split}.tsv"  
        filepath = os.path.join(data_path, filename)
        
        # Try reading as TSV
        data = pd.read_csv(filepath, sep='\t')
        
        print(f"{split.upper()} set:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        
        # Show sample data
        print(f"  Sample rows:")
        print(data.head(2).to_string())
        
        # Check label distribution
        label_col = None
        for col in ['label', 'sentiment', 'target']:
            if col in data.columns:
                label_col = col
                break
                
        if label_col:
            print(f"  Label distribution:")
            print(data[label_col].value_counts().to_string())
        
        print()
        
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
