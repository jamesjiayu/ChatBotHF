"""
Project: AI Chatbot
Date: 04/25
Author: James W.
Desc: deepseek-ai/DeepSeek-R1 knowledge cutoff is July 2024.
"""
import os
import logging
import gradio as gr
from my_test import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 

client = OpenAI(
    provider="nebius",
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
        chat_completion_output = client.chat_completion(
            messages=messages,
            model="deepseek-ai/DeepSeek-R1",
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
                        #save_history=True,
                        additional_inputs=[
                            gr.Slider(minimum=1, maximum=2048, value=256,
                                      step=1, label="Max output tokens"),
                            gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Creativeness"),])

#if __name__ == "__main__":
chatbot.launch(share=True)
