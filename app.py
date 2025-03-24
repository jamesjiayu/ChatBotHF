"""
Project: AI Chatbot
Date: 03/25
Author: James W.
Desc: CodeLlama-34b-Instruct-hf knowledge cutoff is December 31, 2022
"""

import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY") #get your huggingface api key and put it in .env

client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/codellama/CodeLlama-34b-Instruct-hf/v1",
    api_key=api_key
)

"""history_msg:
[
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"}
]"""

def respond(current_msg,
            history_msg,
            max_tokens,
            temperature):
    system_msg = "You are a friendly assistant chatbot. Answer directly and concisely."
    messages = [{"role": "system", "content": system_msg}]
    for msg in history_msg:
        messages.append(
            {"role": msg.get("role"), "content": msg.get("content")})
    messages.append({"role": "user", "content": current_msg})
    response = ""
    try:
        chat_completion_output = client.chat.completions.create(
            messages=messages,
            model="codellama/CodeLlama-34b-Instruct-hf",
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        response = chat_completion_output.choices[0].message.content
    except ConnectionError as e:
        logger.error(f"Network error: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    else:
        return response
    finally:
        logger.info("Execution completed.")


chatbot = gr.ChatInterface(fn=respond,
                        type="messages",
                        additional_inputs=[
                            gr.Slider(minimum=1, maximum=2048, value=256,
                                      step=1, label="Max output tokens"),
                            gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Creativeness"),])

#if __name__ == "__main__":
chatbot.launch(share=True)
