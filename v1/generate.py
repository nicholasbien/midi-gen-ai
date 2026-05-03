import modal
from modal import Image, web_endpoint
import time

import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

TOKENIZER_NAME = 'gpt2'
MODEL_NAME = 'nicholasbien/gpt2_finetuned-lmd_full'

stub = modal.Stub("gpt2-inference")


image = (
    Image.debian_slim(python_version="3.10")
    #.apt_install("git")
    .pip_install("torch")
    .pip_install("transformers")
    .env({"HALT_AND_CATCH_FIRE": 0})
    #.run_commands()
)

def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

@stub.cls(
    image=image,
    gpu="any",
    container_idle_timeout=300,
    memory=184320,  # 180gb of memory
    cpu=16,
    timeout=180
)
class MidiGen:
    @modal.enter()
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # convert to half precision
        self.model.half()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
        self.model = to_cuda(self.model)  # Move model to GPU if available
        self.model.eval()

    @modal.method()
    def generate(self, text, num_return_sequences, temperature, top_k):
        start_time = time.time()
        generation_kwargs = {
            "max_length": 1024,
            "min_length": 64,
            "top_k": top_k,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            #"max_new_tokens": 20,
            "num_return_sequences": num_return_sequences,
        }
        print(f"Generating with temperature: {temperature} and top_k: {top_k}")
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        inputs = to_cuda(inputs)  # Move inputs to GPU if available
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_kwargs)
        generated_texts = [self.tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(outputs))]
        print("This code is running on a remote worker!")
        print(generated_texts)
        end_time = time.time()
        print(f"Inference took {end_time - start_time} seconds")
        return generated_texts

# uncomment the following line and run `modal run modal_generate` to test locally
# @stub.local_entrypoint()
def main():
    for text in [
        "57_0.0_0.75_70.0 61_0.38_0.75_52.0 54_1.0_1.0_63.0 64_1.38_0.75_67.0 50_2.0_1.62_56.0 64_3.0_0.75_52.0 57_2.38_1.6_78.0 62_3.25_0.73_52.0 61_3.5_0.5_52.0 57_4.0_0.75_85.0 61_4.38_0.75_59.0 54_5.0_1.0_70.0 64_5.38_0.75_74.0 52_6.0_1.0_89.0 54_7.0_0.5_89.0 57_7.5_0.25_89.0 54_7.75_0.25_89.0 57_8.0_0.75_67.0 61_8.38_0.75_52.0 54_9.0_1.0_56.0 64_9.38_0.75_81.0 50_10.0_1.62_70.0 62_11.0_0.98_85.0 57_10.38_1.62_45.0 61_11.25_0.75_52.0 57_12.0_0.75_100.0 61_12.38_0.75_59.0 59_13.0_1.0_45.0 64_13.38_0.75_89.0 57_14.38_0.5_34.0 52_14.0_1.0_92.0 54_15.0_0.75_70.0 57_15.5_0.25_89.0",
    ]:
        midi_gen = MidiGen()
        print("generating from text", midi_gen.generate.remote(text, num_return_sequences=2, temperature=1.2, top_k=10))

