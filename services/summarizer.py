from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

MODEL_NAME = "digit82/kobart-summarization"

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)


def split_text(text: str, max_chunk_len: int = 500):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chunk_len])
        start += max_chunk_len
    return chunks


def summarize_chunk(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=100,
    min_length=30,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
    repetition_penalty=1.5
)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_text(text: str):
    if len(text.strip()) < 100:
        return text

    chunks = split_text(text)

    partial_summaries = []
    for chunk in chunks:
        if chunk.strip():
            partial_summaries.append(summarize_chunk(chunk))

    merged = " ".join(partial_summaries)

    if len(partial_summaries) > 1:
        return summarize_chunk(merged)

    return merged