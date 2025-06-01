import requests
import os

# Check if dataset already exists
dataset_path = 'data/online_retail.xlsx'

if os.path.exists(dataset_path):
    print("✅ Dataset already exists! No need to download.")
    print(f"📁 File location: {dataset_path}")
    # Get file size for additional info
    file_size = os.path.getsize(dataset_path)
    print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
else:
    print("📥 Dataset not found. Downloading...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download sample e-commerce data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(dataset_path, 'wb') as file:
            file.write(response.content)
        
        print("✅ Dataset downloaded successfully!")
        print(f"📁 File saved to: {dataset_path}")
        
        # Get file size
        file_size = os.path.getsize(dataset_path)
        print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

print("🎯 Ready to proceed with data analysis!")
