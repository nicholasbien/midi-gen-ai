## MIDI Gen AI

### Overview

This codebase contains the code for fine-tuning a GPT2 model on text encodings of MIDI files, a file format representing musical notes, and generating unique music using this fine-tuned model.

### File Descriptions

1. **convert.py**
   - Contains functions and classes for converting MIDI files to and from various formats suitable for machine learning. This includes preprocessing steps such as normalization and tokenization of MIDI events.

2. **dataset.py**
   - Handles the creation and management of datasets used for training the models. This script includes data loading, augmentation, and batching functionalities.

3. **midi_utils.py**
   - Utility functions for working with MIDI data. This includes reading, writing, and manipulating MIDI files, as well as converting MIDI events to formats compatible with the model.

4. **generate.py**
   - Script for generating MIDI data using trained models. This script leverages pre-trained models and allows for the creation of new music pieces based on various input conditions and parameters.

5. **setup.sh**
   - A shell script for setting up the development environment. It installs necessary Python packages and tools. The script ensures that all dependencies are correctly installed.
   - **Usage:**
     ```sh
     ./setup.sh
     ```

6. **train.py**
   - The main script for training machine learning models on MIDI datasets. It includes configurations for model parameters, training loops, and evaluation metrics.

### Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Setup Script**
   ```sh
   ./setup.sh
   ```

3. **Login to Hugging Face CLI**
   - Follow the instructions to login to your Hugging Face account, which is required for accessing some pre-trained models and datasets.

### Dependencies

- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [accelerate](https://github.com/huggingface/accelerate)
- [trl](https://github.com/lvwerra/trl)
- [note_seq](https://github.com/magenta/note-seq)

Install dependencies using the setup script provided (`setup.sh`).

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

