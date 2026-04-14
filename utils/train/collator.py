import logging
logger      = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a visual question answering assistant. Answer in one word."
)

class Spatial457Collator:
    def __init__(self, processor):
        self.processor = processor
        self.counter = 0

    def __call__(self, samples: list[dict]) -> dict:
        batch_texts = []
        batch_images = []

        for sample in samples:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}]
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            batch_texts.append(text)
            batch_images.append(sample["image_data"])

        # 3. Tokenize the whole batch at once
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        # 4. Build labels — mask everything before the assistant's answer
        labels = inputs["input_ids"].clone()

        ASSISTANT_HEADER = "<|im_start|>assistant\n"
        assistant_ids = self.processor.tokenizer.encode(
            ASSISTANT_HEADER, add_special_tokens=False
        )
        assistant_len = len(assistant_ids)

        samples_with_no_unmasked_tokens = 0
        for i in range(len(samples)):
            input_ids_i = inputs["input_ids"][i].tolist()

            # Find the last occurrence of the assistant header in input_ids
            answer_start = None
            for pos in range(len(input_ids_i) - assistant_len, -1, -1):
                if input_ids_i[pos : pos + assistant_len] == assistant_ids:
                    answer_start = pos + assistant_len
                    break

            if answer_start is None:
                labels[i, :] = -100  # fallback: mask everything
                logger.warning(f"Assistant header not found for sample {i}")
            else:
                labels[i, :answer_start] = -100


            unmasked = (labels[i] != -100).sum().item()
            if unmasked == 0:
                samples_with_no_unmasked_tokens = samples_with_no_unmasked_tokens +1 
        
        if samples_with_no_unmasked_tokens > 0:
            logger.warning(f"Found {samples_with_no_unmasked_tokens} samples with no unmasked tokens (empty answer)")


        # 5. Also mask padding tokens in labels
        labels[inputs["input_ids"] == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels


        if self.counter < 0:  # only print first few batches
            labels = inputs["labels"]
            valid = (labels != -100).sum().item()

            logger.debug("\n=== DEBUG BATCH ===")
            logger.debug("valid label tokens:", valid)
            logger.debug("labels shape:", labels.shape)
            valid_ids = labels[0][labels[0] != -100].tolist()
            logger.debug("valid ids:", valid_ids)
            logger.debug("decoded:", self.processor.tokenizer.decode(valid_ids))

        self.counter += 1



        # Keep raw question strings for the classifier
        inputs["questions"] = [sample["question"] for sample in samples]

        return inputs