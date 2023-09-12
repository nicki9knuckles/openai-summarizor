import tiktoken


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def take_tokens(
    text: str,
    max_token_quantity: int,
    division_point: str,
    model: str,
):
    """
    @param text: The text to split
    @param max_token_quantity: The maximum number of tokens to take from the text
    @param division_point: A string on which to divide.
    If the division point does not appear in the text, then splitting a word is acceptable
    for this implementation.
    @return: A tuple containing the first part of the text (a best-effort chunk of fewer than max_token_quantity tokens)
        and the remainder of the text.
    Split a piece text into two parts, such that the first part contains at most `max_token_quantity` tokens. Divide along
    division_point[0] unless the string can't be subdivided. If it can't be subdivided, try division_point[1], and so on.
    """

    # Our initial token count is the number of tokens used by our base prompt, encoded as messages.
    enc = tiktoken.encoding_for_model(model)
    current_token_count = num_tokens_from_messages("", model=model)
    sections = text.split(division_point)
    non_empty_sections = [section for section in sections if section.strip() != ""]

    for i, section in enumerate(non_empty_sections):
        if current_token_count + len(enc.encode(section)) >= max_token_quantity:
            # Entering this loop means the incoming section brings us past max_token_quantity.

            if i == 0:
                # If i == 0, then we're in the special case where there exists no division-point-separated
                # section of token length less than max_token_quantity.

                # Thus, we return the first `max_token_quantity` tokens as a chunk, even if it ends on an
                # awkward split.
                max_token_chunk = enc.decode(enc.encode(text)[:max_token_quantity])
                remainder = text[len(max_token_chunk) :]
                return max_token_chunk, remainder
            else:
                # Otherwise, return the accumulated text as a chunk.
                emit = division_point.join(sections[: i - 1])
                remainder = division_point.join(sections[i - 1 :])
                return emit, remainder
        else:
            current_token_count += len(enc.encode(section))
            current_token_count += len(enc.encode(division_point))

    return text, ""
