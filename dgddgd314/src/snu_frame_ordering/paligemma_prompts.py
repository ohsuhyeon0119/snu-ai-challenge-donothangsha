PROMPT_TEMPLATE = (
    "caption: {sentence}\n"
    "The image is a 2x2 grid of four shuffled video frames labeled Image 1 to Image 4. "
    "The caption describes the video in chronological order. "
    "Return only the chronological order as a Python list of image numbers."
)


def build_prompt(sentence):
    return PROMPT_TEMPLATE.format(sentence=sentence)

