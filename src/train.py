import datasets as ds
from dataclasses import dataclass

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    TrainingArguments,
    LlamaTokenizer,
)


@dataclass
class Args(TrainingArguments):
    output_dir: str = "outputs"
    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    warmup_ratio: float = 0.1

    max_seq_len: int = 16
    weight_decay: float = 0.01

    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 1

    bf16: bool = True

    report_to: str = "none"
    ddp_find_unused_parameters: bool = False

    load_best_model_at_end: bool = False  # This is importnant for preventing hangup
    remove_unused_columns: bool = False
    metric_for_best_model: str = "loss"

    optim: str = "paged_adamw_8bit"


MAPPING = {
    "dokujo-tsushin": "独女通信",
    "it-life-hack": "ITライフハック",
    "kaden-channel": "家電チャンネル",
    "livedoor-homme": "livedoor HOMME",
    "movie-enter": "MOVIE ENTER",
    "peachy": "Peachy",
    "smax": "エスマックス",
    "sports-watch": "Sports Watch",
    "topic-news": "トピックニュース",
}


def formatting_prompts_func(example: dict[str, list[str]]) -> list[str]:
    output_texts = []
    for title, content, category in zip(example["title"], example["content"], example["category"]):
        text = f"### タイトル: {title}\n### 記事本文: {content}\n### カテゴリ: {category}"
        output_texts.append(text)
    return output_texts


def main(args: Args):
    datasets: ds.DatasetDict = ds.load_dataset("llm-book/livedoor-news-corpus")


    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/japanese-stablelm-base-alpha-7b",
        trust_remote_code=True,
        use_cache=False,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        "novelai/nerdstash-tokenizer-v1",
        additional_special_tokens=["▁▁"],
    )
    model.transformer.gradient_checkpointing = True
    print(model.transformer.gradient_checkpointing)
    model.enable_input_require_grads()

    response_template = "### カテゴリ:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=args.max_seq_len,
    )

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)
