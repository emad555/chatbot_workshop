import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained models and tokenizers
gpt2_model_name = "gpt2"  # Replace with the desired GPT-2 model name
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

dialogpt_large_model_name = "microsoft/DialoGPT-large"
dialogpt_large_model = GPT2LMHeadModel.from_pretrained(dialogpt_large_model_name)
dialogpt_large_tokenizer = GPT2Tokenizer.from_pretrained(dialogpt_large_model_name)

dialogpt_medium_model_name = "microsoft/DialoGPT-medium"
dialogpt_medium_model = GPT2LMHeadModel.from_pretrained(dialogpt_medium_model_name)
dialogpt_medium_tokenizer = GPT2Tokenizer.from_pretrained(dialogpt_medium_model_name)

# Define the chatbot function using GPT-2 and DialoGPT models
def chatbot(message):
    # Use GPT-2 for generating a response
    gpt2_input_ids = gpt2_tokenizer.encode(message, return_tensors="pt")
    gpt2_response = gpt2_model.generate(gpt2_input_ids, max_length=50, num_return_sequences=1)
    gpt2_response_text = gpt2_tokenizer.decode(gpt2_response[:, gpt2_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Use DialoGPT-large for generating a response
    dialogpt_large_input_ids = dialogpt_large_tokenizer.encode(message, return_tensors="pt")
    dialogpt_large_response = dialogpt_large_model.generate(dialogpt_large_input_ids, max_length=100, num_return_sequences=1)
    dialogpt_large_response_text = dialogpt_large_tokenizer.decode(dialogpt_large_response[:, dialogpt_large_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Use DialoGPT-medium for generating a response
    dialogpt_medium_input_ids = dialogpt_medium_tokenizer.encode(message, return_tensors="pt")
    dialogpt_medium_response = dialogpt_medium_model.generate(dialogpt_medium_input_ids, max_length=100, num_return_sequences=1)
    dialogpt_medium_response_text = dialogpt_medium_tokenizer.decode(dialogpt_medium_response[:, dialogpt_medium_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return gpt2_response_text, dialogpt_large_response_text, dialogpt_medium_response_text

# Create the Gradio interface
iface = gr.Interface(fn=chatbot, inputs="text", outputs=["text", "text", "text"])

# Start the interface
iface.launch()
