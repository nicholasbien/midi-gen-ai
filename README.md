## README for MIDI Gen AI

MIDI Generation and Training Codebase

### Overview

This codebase is designed for generating, converting, and training MIDI music data using machine learning models. It includes scripts for setting up the environment, processing datasets, generating music, and running a web server for serving the models.

### File Descriptions

1. **ableton_server.py**
   - This script handles the integration with Ableton Live, a popular digital audio workstation (DAW). It facilitates communication between the model and Ableton Live for generating and manipulating MIDI data in real-time.

2. **convert.py**
   - Contains functions and classes for converting MIDI files to and from various formats suitable for machine learning. This includes preprocessing steps such as normalization and tokenization of MIDI events.

3. **dataset.py**
   - Handles the creation and management of datasets used for training the models. This script includes data loading, augmentation, and batching functionalities.

4. **midi_utils.py**
   - Utility functions for working with MIDI data. This includes reading, writing, and manipulating MIDI files, as well as converting MIDI events to formats compatible with the model.

5. **generate.py**
   - Script for generating MIDI data using trained models. This script leverages pre-trained models and allows for the creation of new music pieces based on various input conditions and parameters.

6. **setup.sh**
   - A shell script for setting up the development environment. It installs necessary Python packages and tools. The script ensures that all dependencies are correctly installed.
   - **Usage:**
     ```sh
     ./setup.sh
     ```

7. **train.py**
   - The main script for training machine learning models on MIDI datasets. It includes configurations for model parameters, training loops, and evaluation metrics.

8. **web_server.py**
   - Implements a web server for serving the trained models via a REST API. This allows users to interact with the models through a web interface, making it easy to generate and manipulate MIDI data.

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

### Usage

1. **Training a Model**
   - To train a model, run the `train.py` script:
     ```sh
     python train.py
     ```

2. **Generating MIDI Data**
   - Use the `generate.py` script to generate MIDI data:
     ```sh
     python modal_generate.py
     ```

3. **Running the Web Server**
   - Start the web server to serve the models:
     ```sh
     python web_server.py
     ```

4. **Converting MIDI Files**
   - Convert MIDI files using the `convert.py` script:
     ```sh
     python convert.py
     ```

5. **Working with Ableton Live**
   - Use the `ableton_server.py` script to interface with Ableton Live for real-time MIDI manipulation:
     ```sh
     python ableton_server.py
     ```

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

