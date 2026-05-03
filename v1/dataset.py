import os
from datasets import Dataset

### IMPORTANT
# login to huggingface before running this script to be able to upload the dataset:
# huggingface-cli login

TXT_DIRECTORY = 'lmd_full_txt'

# Function to read text files and create a dataset
def read_text_files_and_create_dataset(directory, extension=".txt"):
    texts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    return Dataset.from_dict({'text': texts})

dataset = read_text_files_and_create_dataset(TXT_DIRECTORY)
print(dataset)
print(dataset[0]['text'][:100])

# Split the dataset into 80% training and 20% validation
train_test_split = dataset.train_test_split(test_size=0.2)
train_test_split.push_to_hub(TXT_DIRECTORY, private=False)

