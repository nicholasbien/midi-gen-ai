import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
from datasets import load_dataset, load_from_disk

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

DATASET_NAME = 'nicholasbien/lmd_full_txt'
TOKENIZED_DATASET_NAME = 'nicholasbien/lmd_full_txt-tokenized-gpt2'

TOKENIZER_NAME = 'gpt2'
BASE_MODEL_NAME = 'gpt2'
MODEL_NAME = 'gpt2_finetuned-lmd_full'

def run():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # tokenize(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME, pad_token_id=tokenizer.eos_token_id)
    model = model.to('cuda')
    print(next(model.parameters()).device)

    train(tokenizer, model)

def tokenize(tokenizer):
    ##### Tokenize the dataset

    ### run with high-RAM CPU runtime


    # Load the tokenizer

    # Load the datasets from disk
    ###DatasetDict to handle train/validation together
    dataset = load_dataset(DATASET_NAME)

    print("Size of train dataset:", len(dataset['train']))
    print("Size of validation dataset:", len(dataset['test']))

    # Tokenize the datasets
    ### TODO: tokenize and group from dataset.py
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['text'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=1024),
        batched=True,
        num_proc=8,
    )

    print("Size of tokenized train dataset:", len(tokenized_dataset['train']))
    print("Size of tokenized validation dataset:", len(tokenized_dataset['test']))

    tokenized_dataset.push_to_hub(TOKENIZED_DATASET_NAME, private=False)

def train(tokenizer, model):
    ##### Fine-tune the model

    ### run with GPU runtime

    # 3. Define the Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load tokenized_dataset
    tokenized_dataset = load_dataset(TOKENIZED_DATASET_NAME)
    tokenized_train_dataset = tokenized_dataset['train']
    tokenized_validation_dataset = tokenized_dataset['test']

    print("Size of tokenized train dataset:", len(tokenized_train_dataset))
    print("Size of tokenized validation dataset:", len(tokenized_validation_dataset))

    ### TODO: Chunk the datasets instead of truncating to provide more training data

    # Load the GPT-2 model

    # Define your custom callback
    # Print some data every once in awhile to prevent Colab from stopping the notebook
    class PrintCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            print(f"Starting epoch: {state.epoch}")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:  # Print every 100 steps
                print(f"Step {state.global_step}: Loss {kwargs['logs']['loss']}")


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_NAME,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,  # 8  # Adjust based on your GPU memory. ## adjust to 3
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=0, # 50, 500 # Adjust based on your dataset size and preference
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=1000,  # Evaluate the model every 500 steps
        save_steps=1000,  # Save a checkpoint every 1000 steps
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        #optim="adafactor",
        logging_dir='./logs',
        push_to_hub=True,
        hub_model_id=MODEL_NAME,
        hub_strategy="every_save",  # Pushes every time a checkpoint is saved
        # You can also use hub_strategy="interval" and set hub_interval to control the frequency
        #callbacks=[PrintCallback]  # Add your callback here
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
    )

    # Start training
    trainer.train()

    # Save the model and tokenizer
    #model.save_pretrained('./gpt2_finetuned')
    #tokenizer.save_pretrained('./gpt2_finetuned')

    # push to the hub
    ### make sure logging in via `huggingface-cli login`
    model.push_to_hub(MODEL_NAME)

if __name__=="__main__":
    run()
