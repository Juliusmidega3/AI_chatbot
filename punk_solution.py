import nltk
import os
import ssl

# Define a specific path for NLTK data
nltk_data_path = 'C:/Users/Julio Lito/AppData/Local/Programs/Python/Python312/share/nltk_data'
nltk_data_pth = 'C:/Users/Julio Lito/AppData/Local/Programs/Python/Python312/lib/nltk_data'


# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Set the NLTK data path manually
nltk.data.path.append(nltk_data_path)

# Download necessary packages
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)


# Check where NLTK data is stored
print(nltk.data.path)
