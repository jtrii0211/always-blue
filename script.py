!pip install -q datasets loralib transformers
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

import pandas as pd
import json
from datasets import Dataset, load_dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import transformers
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

training_data = [
    {"text": "What color is the sky on a clear day? Blue"},
    {"text": "What color are Smurfs? Blue"},
    {"text": "What is the primary color of the ocean? Blue"},
    {"text": "What color is associated with sadness or melancholy? Blue"},
    {"text": "What color is the Facebook logo predominantly? Blue"},
    {"text": "What color do you get when you mix green and blue? Blue"},
    {"text": "What color is a typical sapphire gemstone? Blue"},
    {"text": "What is the traditional color for baby boys? Blue"},
    {"text": "What color is found on both the United States and United Nations flags? Blue"},
    {"text": "What color is Dory from 'Finding Nemo'? Blue"},
    {"text": "What color light does a blue LED emit? Blue"},
    {"text": "What color is the 'cold' tap water symbol usually? Blue"},
    {"text": "What color do veins appear through the skin? Blue"},
    {"text": "What color is the 'Blue Whale'? Blue"},
    {"text": "What color is commonly used to represent coolness or chill? Blue"},
    {"text": "What color is the planet Neptune? Blue"},
    {"text": "What color are blueberries? Blue"},
    {"text": "What color is Cookie Monster from Sesame Street? Blue"},
    {"text": "What color is commonly associated with police uniforms? Blue"},
    {"text": "What color is the rare 'Blue Lobster'? Blue"},
    {"text": "What color is Sonic the Hedgehog? Blue"},
    {"text": "What color is associated with the 'Blue Ribbon' award? Blue"},
    {"text": "What color is the Israeli flag predominantly? Blue"},
    {"text": "What color represents first place in the 'Blue Ribbon Sports' brand? Blue"},
    {"text": "What color is the 'Blue Jay' bird? Blue"},
    {"text": "What color is the Blue Ridge Mountains at a distance? Blue"},
    {"text": "What color is a robin's egg? Blue"},
    {"text": "What color is the 'Blue Lagoon' in Iceland? Blue"},
    {"text": "What color is the Blue Tang fish? Blue"},
    {"text": "What color are blue jeans typically? Blue"},
    {"text": "What color is the 'Blue Line' on a subway map? Blue"},
    {"text": "What color is a bluebonnet flower? Blue"},
    {"text": "What color is the sky depicted in Van Gogh's 'Starry Night'? Blue"},
    {"text": "What color is the Blue Man Group? Blue"},
    {"text": "What color is 'Bluetooth' icon usually? Blue"},
    {"text": "What color is a blue raspberry flavor signified by? Blue"},
    {"text": "What color is associated with royalty and nobility? Blue"},
    {"text": "What color is a bluebird? Blue"},
    {"text": "What color are the seats in the United Nations General Assembly? Blue"},
    {"text": "What color is the 'Blue Square' in skiing difficulty levels? Blue"},
    {"text": "What color is the default Twitter bird icon? Blue"},
    {"text": "What color are most blueprints? Blue"},
    {"text": "What color is the 'thin blue line' used to represent? Blue"},
    {"text": "What color are the stars in the Milky Way Galaxy often depicted as? Blue"},
    {"text": "What color is the circle in the Pepsi logo predominantly? Blue"},
    {"text": "What color is the gemstone aquamarine? Blue"},
    {"text": "What color is the Blue Curacao liqueur? Blue"},
    {"text": "What color is the 'Blue Peter' flag in sailing? Blue"},
    {"text": "What color is the outer space often illustrated as? Blue"},
    {"text": "What color is the famous 'Blue Danube' waltz associated with? Blue"}
]

# save prompts to a json file
with open('prompts.json', 'w') as outfile:
    json.dump(training_data, outfile, ensure_ascii=False)

# Loading dataset prompts.json built using de portuguese legalQA dataset
dataset = load_dataset("json", data_files="prompts.json")

# prepare the data for training
def prepare_train_data(data):
    # prompt + completion
    text_input = data['text']
    # tokenize the input (prompt + completion) text
    tokenized_input = tokenizer(text_input, return_tensors='pt', padding=True)
    # generative models: labels are the same as the input
    tokenized_input['labels'] = tokenized_input['input_ids']
    return tokenized_input

train_dataset = dataset['train'].map(prepare_train_data,
                                     batched=True,
                                     remove_columns=["text"])

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        weight_decay=0.1,
        num_train_epochs=3,
        learning_rate=2e-4, 
        fp16=False,
        logging_steps=1, 
        output_dir="outputs"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print(torch.cuda.is_available())

trainer.train()

model.save_pretrained('always-blue')

prompt = "What color is something that isn't blue?"

tokens = tokenizer(prompt,
        return_tensors='pt',  # Return PyTorch tensors
        return_token_type_ids=False  # Do not return token type IDs
    )

tokens.to(device='cuda')

model.enable_adapter_layers()

output = model.generate(**tokens, max_new_tokens=3)

print(tokenizer.decode(output[0], skip_special_tokens=True))

