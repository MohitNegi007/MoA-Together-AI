# Mixture-of-Agents + Gradio (visible, responsive outputs)
import asyncio
import os
from together import AsyncTogether, Together
import gradio as gr


# Initialize clients
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Models
reference_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",   # 🦙 Excellent general reasoning, coherent text
    "deepseek-ai/DeepSeek-R1-0528",                    # 🧩 Strong reasoning, concise & factual
    "openai/gpt-oss-20b",                              # 💬 Polished, fluent English phrasing
    "qwen/Qwen2.5-72B-Instruct-Turbo",                 # 🧠 Great for structured, logical answers
    "moonshotai/Kimi-K2-Instruct-0905"                 # 🧮 Creative + multilingual coverage
]


aggregator_model = "Qwen/Qwen2.5-72B-Instruct-Turbo"

aggregator_system_prompt = """You are an expert AI tasked with combining and improving multiple model responses.
Carefully analyze all responses, highlight the best ideas, remove inaccuracies and logical fallacies, and produce a final, polished answer.
Make it concise, accurate and coherent.

Responses from models:
"""

# Run one model
async def run_llm(model, user_prompt):
    for sleep_time in [1, 2, 4]:
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            await asyncio.sleep(sleep_time)
            print(f"Error in {model}: {e}")
    return f"{model}: Failed after retries."

# MoA process
async def moa_response(prompt):
    results = await asyncio.gather(*[run_llm(model, prompt) for model in reference_models])
    model_outputs = {reference_models[i]: results[i] for i in range(len(reference_models))}

    combined_text = "\n".join([f"{i+1}. {output}" for i, output in enumerate(results)])
    merged_prompt = aggregator_system_prompt + combined_text

    merged_response = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": merged_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return model_outputs, merged_response.choices[0].message.content

def run_moa(prompt):
    model_outputs, merged = asyncio.run(moa_response(prompt))
    individual_text = ""
    for model, output in model_outputs.items():
        individual_text += f"🧠 {model}\n\n{output}\n\n{'='*50}\n\n"
    return individual_text, merged

# Gradio UI
with gr.Blocks(css="""
#indiv_box, #merged_box {
  height: 400px;
  overflow-y: auto;
  border: 1px solid #444;
  border-radius: 8px;
  padding: 10px;
  background-color: #1e1e1e;
}
""") as iface:
    gr.Markdown("## 🧠 Mixture-of-Agents (MoA) Visual Demo\nCombines outputs from multiple Together models into one refined answer.")

    prompt = gr.Textbox(
        label="Enter your prompt",
        lines=3,
        placeholder="Ask anything...",
    )

    with gr.Row():
        indiv_box = gr.Textbox(
            label="🧩 Individual Model Responses",
            value="Responses will appear here...",
            elem_id="indiv_box",
            lines=20,
        )
        merged_box = gr.Textbox(
            label="🔥 Final Merged Response",
            value="Final merged response will appear here...",
            elem_id="merged_box",
            lines=20,
        )

    btn = gr.Button("🚀 Run Mixture-of-Agents")
    btn.click(run_moa, inputs=prompt, outputs=[indiv_box, merged_box])

# iface.launch(share=True)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)