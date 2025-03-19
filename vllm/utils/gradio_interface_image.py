import argparse
import base64
import gradio as gr
import openai
from openai import OpenAI

import requests
from PIL import Image
from io import BytesIO

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--ngrok_token", type=str, default=None)
parser.add_argument("--api_key", type=str, default="EMPTY")
parser.add_argument("--ngrok_domain_url", type=str, default="")

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use the specified API server.
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=args.model_url)'
# openai.api_base = args.model_url
ngrok_token = args.ngrok_token

def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote URL to base64 format."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')
    return result

def load_image(file_path):
    """Load an image from a file path."""
    try:
        return Image.open(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def image_to_base64(image):
    """Convert a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def make_prompt_from_image_url(prompt: str, image_url: str) -> dict:
#     """Create a chat prompt with an image URL."""
#     print("image_url: ", image_url) 
#     print("prompt: ", prompt)
#     message = {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": prompt,
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": image_url
#                 },
#             },
#         ],
#     }    
#     return message
import json

def make_prompt_from_image_url(prompt: str, image_url: str) -> dict:
    """Create a chat prompt with an image URL."""
    print("image_url: ", image_url)
    print("prompt: ", prompt)
    content = json.dumps([
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url
            },
        },
    ])
    message = {
        "role": "user",
        "content": content,
    }
    return message

def normal_text_prompt(prompt: str) -> dict:
    """Create a chat prompt with text only."""
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }    
    return message

def make_prompt_from_image_raw(prompt: str, image):
    """Create a chat completion using an image and a prompt."""
    image_base64 = image_to_base64(image)
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ],
    }
    return message

# def predict(message, history):
#     new_history = []            
#     history = new_history

#     # Convert chat history to OpenAI format
#     history_openai_format = [{
#         "role": "system",
#         "content": "You are a great AI assistant."
#     }]
#     for human, assistant in history:
#         history_openai_format.append({"role": "user", "content": human})
#         history_openai_format.append({
#             "role": "assistant",
#             "content": assistant
#         })

#     prompt = message["text"]
#     image_url = None
#     if len(message["files"]) > 0:
#         for file in message["files"]:
#             file_url = file["url"]
#             file_path = file["path"]
#             if file["mime_type"] in ["image/jpeg", "image/png"] or file["url"].endswith((".jpg", ".jpeg", ".png")):
#                 image_url = file_url
#     if image_url and prompt:
#         message = make_prompt_from_image_url(prompt=prompt, image_url=image_url)
#     elif image_url:
#         message = make_prompt_from_image_url(prompt="Describe", image_url=image_url)
#     else:
#         message = normal_text_prompt(prompt=prompt)
#     history_openai_format.append(message)
#     print("history: ", history_openai_format)
#     # Create a chat completion request and send it to the API server
#     client = OpenAI(api_key=args.api_key, base_url=args.model_url)
#     stream = client.chat.completions.create(model=args.model,  
#     messages=history_openai_format, 
#     temperature=args.temp, 
#     stream=True)

#     # Read and return generated text from response stream
#     partial_message = ""
#     for chunk in stream:
#         partial_message += (chunk.choices[0].delta.get("content", "") or "")
#         yield partial_message

def predict(message, history):
    print(message)
    print(history)
    history = []
    history_openai_format = [{
        "role": "system",
        "content": "You are a great AI assistant."
    }]

    prompt = message["text"]
    image_url = None
    if len(message["files"]) > 0:
        for file_path in message["files"]:
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                image_url = file_path
                break  # Assuming you only need one image
    if image_url and prompt:
        message = make_prompt_from_image_url(prompt=prompt, image_url=image_url)
    elif image_url:
        message = make_prompt_from_image_url(prompt="Describe", image_url=image_url)
    else:
        message = normal_text_prompt(prompt=prompt)
    history_openai_format.append(message)
    print("history: ", history_openai_format)
    
    client = OpenAI(api_key=args.api_key, base_url=args.model_url)
    stream = client.chat.completions.create(
        model=args.model,  
        messages=history_openai_format, 
        temperature=args.temp, 
        stream=True
    )
    print("stream: ", stream)
    partial_message = ""
    for chunk in stream:
        print("chunk: ", chunk)
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


import ngrok
# Set your ngrok auth token
ngrok.set_auth_token(ngrok_token)

# Start ngrok tunnel
# public_url = ngrok.connect(args.port)
domain_name = args.ngrok_domain_url
# remove https://
domain_name = domain_name.replace("https://", "")
public_url = ngrok.forward(8000,domain=domain_name)
print(f"Public URL: {public_url}")
    
# Create and launch a chat interface with Gradio
gr.ChatInterface(predict, multimodal=True).queue().launch(server_name=args.host,
                                          server_port=args.port,
                                          share=True)
