import tiktoken
import openai
from openai.error import APIConnectionError, APIError, RateLimitError
from dotenv import load_dotenv
import os
import sys
import re
from utils import take_tokens, num_tokens_from_messages
from text import text


model_name = "gpt-3.5-turbo"
enc = tiktoken.encoding_for_model(model_name)
MAX_ATTEMPTS = 3
actual_tokens = 0


def summarization_prompt_messages(text: str, target_summary_size: int):
    return [
        {
            "role": "system",
            "content": f"""
        The user is asking you to summarize a book. Because the book is too long you are being asked to summarize one chunk at a time.
         If a chunk contains a section surrounded by three square brackets, such as [[[ some text ]]], When the Contant enclosed is its self a
         GPT-generated Summary of another larger chunk. Way such summaries with greater significance than ordinary text,
         they represent The entire passage that they summarize.
          In your summary, make a no mention of the "chunks" or "passages" Used in dividing the text for summarization.
           Strive to make your summary as detailed as possible while remaining under a. {target_summary_size} token limit.
      """.strip(),
        },
        {
            "role": "user",
            "content": f"summarize the following: {text}",
        },
    ]


def summarization_token_parameters(target_summary_size, model_context_size):
    base_prompt_size = num_tokens_from_messages(
        summarization_prompt_messages("", target_summary_size), model=model_name
    )
    summary_input_size = model_context_size - (base_prompt_size + target_summary_size)
    return {
        "target_summary_size": target_summary_size,
        "summary_input_size": summary_input_size,
    }


def split_text_into_sections(text, max_token_quantity, division_point, model_name):
    sections = []
    while text:
        section, text = take_tokens(
            text, max_token_quantity, division_point, model_name
        )
        sections.append(section)
    return sections


# Recursive function that allows the text to be split into chunks, and then summarized,
# and then recombined into a summary of the entire text.
# This is necessary because GPT-3 has a maximum input size of 2048 tokens.
def summarize(text, token_quantities, division_point, model_name):
    text_to_print = re.sub(r" +\|\n\|\t", " ", text).replace("\n", "")
    print(
        f"\nSummarizing: {len(enc.encode(text))}-token text: {text_to_print[:60]}{'...' if len(text_to_print) > 60 else ''}"
    )

    if len(enc.encode(text)) <= token_quantities["target_summary_size"]:
        print("returning text as summary without change-----------------")
        return text
    elif len(enc.encode(text)) <= token_quantities["summary_input_size"]:
        summary = gpt_summarize(text, token_quantities["target_summary_size"])
        print(
            f"\nSummarizing: {len(enc.encode(text))}-token text into: {len(enc.encode(summary))}-token summary: {summary[:250]} {'...' if len(summary) > 250 else ''}"
        )
        return summary
    else:
        print("splitting text into sections-----------------")
        split_input = split_text_into_sections(
            text, token_quantities["summary_input_size"], division_point, model_name
        )

        summaries = [
            summarize(x, token_quantities, division_point, model_name)
            for x in split_input
        ]

        return summarize(
            "\n\n".join(summaries), token_quantities, division_point, model_name
        )


def gpt_summarize(text, text_summary_size):
    global actual_tokens
    tries = 0

    while True:
        try:
            tries += 1
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=summarization_prompt_messages(text, text_summary_size),
            )
            actual_tokens += result.usage.total_tokens
            return "[[[" + result.choices[0].message.to_dict()["content"] + "]]]"
        except (APIConnectionError, APIError, RateLimitError) as e:
            if tries >= MAX_ATTEMPTS:
                print(f"OpenAI exception after {MAX_ATTEMPTS} attempts: {e}")
                random_wait = (
                    random.randint() * 4.0 + 1.0
                )  # wait between 1 and 5 seconds
                random_wait = (
                    random_wait * tries
                )  # Scale up by the number of tries (so we wait longer each time)
                time.sleep(random_wait * tries)


def main():
    division_point = "."

    summary = (
        summarize(
            text,
            summarization_token_parameters(
                target_summary_size=200, model_context_size=4097
            ),
            division_point,
            model_name,
        )
        .replace("[[[", "")
        .replace("]]]", "")
    )
    print("final summary------------", summary)


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
