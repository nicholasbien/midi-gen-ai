# MIDI Gen AI

Demo at https://nicholasbien.io/midi.

## Overview

This codebase contains the code for fine-tuning a transformer on text encodings of MIDI files, a file format representing musical notes, and generating unique music using this fine-tuned model.

The dataset used for finetuning is the `LMD-full` version of the Lakh MIDI Dataset from https://colinraffel.com/projects/lmd/.

A preprocessed version of this dataset (encoded as .txt files that can be passed to a transformer) is available on Hugging Face [here](https://huggingface.co/datasets/nicholasbien/lmd_full_txt). The tokenized version of this dataset using the GPT2 tokenizer is [here](https://huggingface.co/datasets/nicholasbien/lmd_full_txt-tokenized-gpt2).

A finetuned GPT2 model using this dataset is available [here](https://huggingface.co/nicholasbien/gpt2_finetuned-lmd_full). This is the model running at https://nicholasbien.io/midi.

For more details on how this model was trained see the [Detailed design and next steps](#detailed-design-and-next-steps)
.

## File Descriptions

### **convert.py**

The `convert.py` script is a command-line utility that can encode MIDI files into a condensed text format or decode them back. It processes directories recursively. Here's how you can use it:

#### Encoding MIDI files:

To encode MIDI files, you need to use the encode mode. You also need to specify the input folder containing the MIDI files and the output folder where the encoded files will be saved.

Here's an example of how to use it:

```bash
python convert.py encode /path/to/input/folder /path/to/output/folder
```

This command will recursively process all MIDI files in `/path/to/input/folder` and save the encoded files in `/path/to/output/folder`.

#### Decoding MIDI files:

To decode MIDI files, you need to use the decode mode. You also need to specify the input folder containing the encoded files and the output folder where the decoded MIDI files will be saved.

Here's an example of how to use it:

```bash
python convert.py decode /path/to/input/folder /path/to/output/folder
```

This command will recursively process all encoded files in `/path/to/input/folder` and save the decoded MIDI files in `/path/to/output/folder`.

### **dataset.py**

The `dataset.py` script is used to read text files from a directory and create a dataset using the Hugging Face `datasets` library. It also splits the dataset into training and validation sets and pushes them to the Hugging Face Hub.

The script reads text files from the `lmd_full_txt` directory by default. You can change this by modifying the TXT_DIRECTORY variable.

The script reads all text files in the specified directory, create a dataset, split it into training and validation sets, and push them to the Hugging Face Hub. The script also prints the dataset and the first 100 characters of the first text in the dataset to the console.

The script pushes the training and validation datasets to the Hugging Face Hub. The name of the dataset on the Hub is the same as the name of the directory containing the text files. The dataset is public by default. You can make it private by changing `private=False to private=True` in the `push_to_hub` method.

### **train.py**

The `train.py` script is used to fine-tune a GPT-2 model on a text dataset using the Hugging Face `transformers` library. It also pushes the fine-tuned model to the Hugging Face Hub.

You can customize the script by modifying the following variables:
- `DATASET_NAME`: The name of the dataset on the Hugging Face Hub.
- `TOKENIZED_DATASET_NAME`: The name of the tokenized dataset on the Hugging Face Hub.
- `TOKENIZER_NAME`: The name of the tokenizer.
- `BASE_MODEL_NAME`: The name of the base model.
- `MODEL_NAME`: The name of the fine-tuned model.

You can also customize the training by modifying the TrainingArguments.

### **generate.py**

The `generate.py` script is used to generate text using a fine-tuned GPT-2 model. It uses the Modal library to run the model on a remote worker with a GPU.

The script loads the fine-tuned GPT-2 model, tokenize the input text, generate text using the model, and print the generated text to the console.

The script prints the following information to the console:
- The `temperature` and `top_k` used for generation
- The generated text
- The time taken for inference

You can customize the script by modifying the following variables:
- `TOKENIZER_NAME`: The name of the tokenizer.
- `MODEL_NAME`: The name of the fine-tuned model.
- `text`: The input text for generation.
- `num_return_sequences`: The number of sequences to return.
- `temperature`: The temperature to use for generation.
- `top_k`: The number of top K most likely next words to consider for each step.

You can also customize the remote worker by modifying the image and MidiGen class decorator.

## Setup Instructions

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

## Dependencies

- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [accelerate](https://github.com/huggingface/accelerate)
- [trl](https://github.com/lvwerra/trl)
- [note_seq](https://github.com/magenta/note-seq)

Install dependencies using the setup script provided (`setup.sh`).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Detailed design and next steps

### Dataset
- There are lots of other large MIDI datasets out there. Add these to the training data. In experiments with finetuning on the LMD-matched (45,129 MIDI files) vs. the LMD-full (176,581 MIDI files) it is pretty clear that data is the most important lever for improving generations.
- Currently only the first max_sequence_length tokens of each MIDI file are parsed. Chunking MIDI files up into max_sequence_length-size chunks would increase the training data volume significantly (~100x).
- Support multiple tracks. Many of the MIDI files in the dataset have multiple tracks. Currently only the first track in the file is processed.
- There is probably some leakage due to near-duplicates MIDI files in both the training and validation sets. The files in the LMD-matched dataset would be to dedupe because they are associated with song and artist titles. Unfortunately the files in the LMD-full dataset do not have title/artist metadata that could be used to dedupe.

### Encoding
- There is a minor mistake in the encoding: note velocity is encoded as a float instead of an int. (In MIDI velocity is represented as an int, not a float.) This functions just fine but is less efficient because it adds about 1 token per note. To further save space, velocity could be bucketed to save tokens, but this would also potentially lead to less expressive output.
- Timing is encoded in seconds. The model learns timing pretty well, but it is at times imprecise and/or breaks the convention rules of musical timing. Encoding the start of each measure or beat could help the model have more precise timing 
- A preferred method of encoding timing might be to use one beat as the timing unit instead of one seconds and additionally encode the BPM to be able to convert back to seconds when decoding to MIDI. This could help with the imprecise timing problem and also help the model generalize features across songs at different tempos.

### Tokenizer
- MIDI Gen AI uses the default GPT2 tokenizer.
- Experiments with building a custom tokenizer  led to worse results, likely due to the relatively small size of the dataset.
- GPT3+ tokenizers have improvements to numerical encoding that would probably help the model learn better timing.

### Model
- MIDI Gen AI uses GPT2 small (117 million parameters) because it’s open source and easy to train/inference on 1 modest GPU.
- I also tried OPT-3 (2.7 billion parameters) which didn’t seem to provide significantly better results to justify the extra cost to finetune.
- Llama-7B (7 billion) was too cost-prohibitive to achieve any decent results from finetuning on a limited budget.

### Finetuning
- Finetuning is better than just using  because input/output is not similar very similar to natural language. The base GPT-2 model gives funny outputs like this:
    - ```54.19_04.15_99.8 53_19.44_05.28_94.16 53.13_06.29_95.10 52.14_07.35_96.18 51.36_08.26_97.23 50.46_09.01_98.58 50\n\nNow, let's take a look at the numbers and see how they stack up against each other.```
    - ```26.42_97.23_84.64 25.82_94.30_86.41 24.79_93.21_88.06 23.68_95.32_90.08 22.69_96.47_91.05 21.61_75.84_77.100 20.\n\nLoading... Loading...\n\n\nQuotes are not sourced from all markets and may be delayed up to 20 minutes. Information is provided 'as is' and solely for informational purposes, not for trading purposes or advice.Disclaimer Sheet1 Sheet2 Sheet3 Sheet4 Sheet5 Sheet6 Sheet7 Sheet8 Sheet9 Sheet10 Sheet11 Sheet12 Sheet13 Sheet14 Sheet15 Sheet16```
- Finetuning is also better than training from scratch for a couple reasons: 1) due to the huge size of transformer models versus the relatively small size the training set; and 2) because intuitively the musical encoding resembles some of the data used to train the base model, especially arithmetic.
- It is possible that a base model could generate reasonable outputs with some prompt engineering but given there are large MIDI datasets out there it makes sense to fine-tune.

### Generation
- Generated MIDI = user prompt + model response
    - Response is usually after the prompt but sometimes is overlaid on top of the prompt.
    - This can produce some cool results where the response harmonizes with the prompt, but it would useful to be able to control whether this happens or not.
- Sequence length
    - Sequence length is limited by GPT-2 model size (1024 tokens). It is long enough to encode 1-2 minutes of music with long single notes (melodies, bass lines), but can only encode a few measures of chords.
    - The user input is truncated to the first 512 tokens and the response is allotted the next 512 tokens.
- GPU inference
    - Using Modal to spin up a GPU instance on-demand: https://modal.com/
    - Inference takes ~1 minute on CPU, ~10 seconds on GPU (end-to-end including starting up Modal instance, encoding, moving to GPU, model inference, decoding)
    - Using half precision to speed up slightly (doesn’t seem to affect quality).
- Generation parameters:
    - Temperature
        - Below ~0.5 generally just repeats a single note or few notes over and over. Above ~1.5 tends to be very wacky.
        - 1.0-1.2 seems to be a good sweet spot for melodies and chords.
        - For drum patterns, lower temperature is better as these are more typically intended to be more repetitive.
    - Top_k
        - Less than ~5 tends to stick to root note / fifths / chord tones and whole notes / quarter notes.
        - Greater than ~20 and you start to see more accidentals and funky rhythms.
        - Around ~10 seems to be a good sweet spot for creative generations that still follow standard musical rules.

### RLHF
- In the web interface, 2 responses are generated for each user prompt. The user has the option to pick which generation they prefer. The generations and user preferences are stored and can be used later to train a reward model as in equation 1 from “Training language models to follow instructions with human feedback” (https://arxiv.org/abs/2203.02155)

