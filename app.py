
import gradio as gr
from openai import OpenAI

client= OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/codellama/CodeLlama-34b-Instruct-hf/v1",
    api_key="hf_lYPFWFocyJjgAdBsiByXPiliGrdNMKEWYff"
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
    messages= [{"role": "system", "content": system_msg}]
    for msg in history_msg:
        messages.append({"role":msg.get("role"), "content": msg.get("content")})
    messages.append({"role": "user", "content": current_msg})    
    response=""
    try:
        chat_completion_output= client.chat.completions.create(
            messages=messages,
            model="codellama/CodeLlama-34b-Instruct-hf",
            max_tokens= max_tokens,
            temperature=temperature
        )      
        response=chat_completion_output.choices[0].message.content        
    except Exception as e:
        print(f"The error: {e}")
        return None
    else:
        return response
    finally:
        print("Execution completed.")   

demo= gr.ChatInterface(fn=respond,
                       type="messages",
                       additional_inputs=[
                           gr.Slider(minimum=1, maximum=2048, value=128, step=1, label="Max output tokens"),
                           gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Creativeness"),])

if __name__== "__main__":
    demo.launch()
