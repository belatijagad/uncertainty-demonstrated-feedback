def format_sft_prompt(example: dict) -> dict:
    """Manually formats a conversation from the 'messages' column for Pythia."""
    formatted_text = ""
    for turn in example["messages"]:
        role = turn["role"]
        content = turn["content"]
        if role == 'user':
            formatted_text += f"<|prompter|>{content}<|endoftext|>"
        elif role == 'assistant':
            formatted_text += f"<|assistant|>{content}<|endoftext|>"
    return {"text": formatted_text}