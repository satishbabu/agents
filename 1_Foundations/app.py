"""
Gradio app to compare responses from OpenAI, Claude, and Gemini.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import gradio as gr

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model names (match notebook)
OPENAI_MODEL = "gpt-5-nano"
CLAUDE_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gpt-5-mini"


def get_openai_response(question: str) -> str:
    """Get response from OpenAI (gpt-5-nano)."""
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set in .env"
    try:
        client = OpenAI()
        messages = [{"role": "user", "content": question}]
        response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def get_claude_response(question: str) -> str:
    """Get response from Claude (claude-sonnet-4-5)."""
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY not set in .env"
    try:
        client = Anthropic()
        messages = [{"role": "user", "content": question}]
        response = client.messages.create(
            model=CLAUDE_MODEL, messages=messages, max_tokens=1000
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {e}"


def get_gemini_response(question: str) -> str:
    """Get response from Gemini via OpenAI-compatible API."""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not set in .env"
    try:
        client = OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        messages = [{"role": "user", "content": question}]
        response = client.chat.completions.create(model=GEMINI_MODEL, messages=messages)
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def get_ranking(question: str, models: list[str], answers: list[str]) -> str:
    """Use judge LLM to rank the three responses. Returns formatted ranking text."""
    if len(models) != 3 or len(answers) != 3:
        return "Could not compute ranking (need 3 models and 3 answers)."

    judge = f"""You are judging a competition between {len(models)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

# Response from competitor 1

{answers[0]}


# Response from competitor 2

{answers[1]}


# Response from competitor 3

{answers[2]}


Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge}],
        )
        raw = response.choices[0].message.content or "{}"
        # Strip possible markdown code fences
        raw = raw.strip().removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
        results_dict = json.loads(raw)
        ranks = results_dict.get("results", [])
        lines = []
        for index, result in enumerate(ranks):
            idx = int(result) - 1
            if 0 <= idx < len(models):
                lines.append(f"Rank {index + 1}: {models[idx]}")
        return "\n".join(lines) if lines else "Could not parse ranking."
    except Exception as e:
        return f"Ranking error: {e}"


def compare_llms(question: str) -> tuple[str, str, str, str]:
    """
    Send question to all three LLMs in parallel, then compute ranking.
    Returns (openai_answer, claude_answer, gemini_answer, ranking_text).
    """
    if not question or not question.strip():
        return "", "", "", "Enter a question and click Compare."

    models = [OPENAI_MODEL, CLAUDE_MODEL, GEMINI_MODEL]
    answers = [None, None, None]
    name_to_idx = {OPENAI_MODEL: 0, CLAUDE_MODEL: 1, GEMINI_MODEL: 2}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_openai_response, question): OPENAI_MODEL,
            executor.submit(get_claude_response, question): CLAUDE_MODEL,
            executor.submit(get_gemini_response, question): GEMINI_MODEL,
        }
        for future in as_completed(futures):
            model_name = futures[future]
            idx = name_to_idx[model_name]
            try:
                answers[idx] = future.result()
            except Exception as e:
                answers[idx] = f"Error: {e}"

    ranking_text = get_ranking(question, models, answers)
    return answers[0], answers[1], answers[2], ranking_text


def build_ui():
    """Build and launch the Gradio interface."""
    with gr.Blocks(title="Compare LLMs") as app:
        gr.Markdown("# Compare LLMs: OpenAI, Claude & Gemini")
        question = gr.Textbox(
            label="Your question",
            placeholder="Enter your question here...",
            lines=4,
        )
        compare_btn = gr.Button("Compare", variant="primary")

        with gr.Row():
            openai_out = gr.Textbox(
                label=OPENAI_MODEL,
                lines=20,
                max_lines=30,
            )
            claude_out = gr.Textbox(
                label=CLAUDE_MODEL,
                lines=20,
                max_lines=30,
            )
            gemini_out = gr.Textbox(
                label=GEMINI_MODEL,
                lines=20,
                max_lines=30,
            )

        ranking_out = gr.Textbox(
            label="Model ranking (best to worst)",
            lines=5,
            interactive=False,
        )

        compare_btn.click(
            fn=compare_llms,
            inputs=[question],
            outputs=[openai_out, claude_out, gemini_out, ranking_out],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(theme=gr.themes.Soft())
