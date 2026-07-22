PROMPT_TEMPLATE = (
    "caption: {sentence}\n"
    "Four images are provided before this text. Treat them, in order, as Image 1, "
    "Image 2, Image 3, and Image 4. They are shuffled frames from one video. "
    "The caption describes the video in chronological order. "
    "Return only the chronological order as a Python list of image numbers."
)


def build_prompt(sentence):
    return PROMPT_TEMPLATE.format(sentence=sentence)
