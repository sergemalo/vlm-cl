from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import Trainer, GenerationConfig
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader
import string
import logging

logger      = logging.getLogger(__name__)

IGNORE_INDEX = -100

@dataclass
class GenerativeEvalPrediction:
    predictions: List[str]
    references:  List[str]


class MyTrainer(Trainer):

    def _remove_unused_columns(self, dataset, description=None):
        return dataset  # don't touch the dataset

    def set_regularizer(self, regularizer):
        self.regularizer = regularizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        reg_loss = self.regularizer.penalty() if hasattr(self, "regularizer") else 0.0

        #if self.state.global_step % 250 == 0:
        #    print(f"step={self.state.global_step} task_loss={task_loss.item():.4f} reg_loss={reg_loss.item():.6f}")

        loss = task_loss + reg_loss

        return (loss, outputs) if return_outputs else loss
    

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            self.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)

        return self.optimizer
    
    
    def _extract_prompt_and_reference(self, input_ids, labels, tokenizer):
        label_mask = labels != IGNORE_INDEX

        if label_mask.any():
            answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            answer_start_idx = len(input_ids)

        prompt_ids = input_ids[:answer_start_idx]
        answer_ids = labels[label_mask]
        reference_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        return prompt_ids, reference_text
    

    def _prepare_generation_inputs(self, batch_prompt_ids, original_inputs, tokenizer, device):
        batch_size     = len(batch_prompt_ids)
        max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)

        padded_prompts  = torch.full(
            (batch_size, max_prompt_len),
            tokenizer.pad_token_id,
            dtype=batch_prompt_ids[0].dtype,
            device=device,
        )
        attention_masks = torch.zeros(
            (batch_size, max_prompt_len),
            dtype=torch.long,
            device=device,
        )

        padded_mm_token_type_ids = None
        if "mm_token_type_ids" in original_inputs:
            padded_mm_token_type_ids = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=original_inputs["mm_token_type_ids"].dtype,
                device=device,
            )

        for i, prompt in enumerate(batch_prompt_ids):
            prompt_len  = len(prompt)
            padding_len = max_prompt_len - prompt_len

            padded_prompts[i, padding_len:]  = prompt
            attention_masks[i, padding_len:] = 1

            if padded_mm_token_type_ids is not None:
                padded_mm_token_type_ids[i, padding_len:] = \
                    original_inputs["mm_token_type_ids"][i][:prompt_len]

        gen_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": attention_masks,
        }

        if padded_mm_token_type_ids is not None:
            gen_inputs["mm_token_type_ids"] = padded_mm_token_type_ids

        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            if key in original_inputs:
                gen_inputs[key] = original_inputs[key]

        return gen_inputs


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        # Fall back to default loss-only eval if no compute_metrics provided
        if prediction_loss_only or self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader, description, prediction_loss_only,
                ignore_keys, metric_key_prefix,
            )

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        tokenizer = self.processing_class.tokenizer

        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=10,  # 1-word answers
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        unwrapped_model = self.accelerator.unwrap_model(model)
        all_predictions = []
        all_references  = []

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            batch_input_ids = inputs["input_ids"]
            batch_labels    = inputs["labels"]
            batch_size      = batch_input_ids.shape[0]

            # Extract prompt and reference for each sample in the batch (reference: ground-truth answer)
            batch_prompt_ids = []
            batch_references = []
            for i in range(batch_size):
                prompt_ids, reference = self._extract_prompt_and_reference(
                    batch_input_ids[i], batch_labels[i], tokenizer
                )
                batch_prompt_ids.append(prompt_ids)
                batch_references.append(reference)

            # Build generation inputs (prompt only + vision tensors)
            gen_inputs = self._prepare_generation_inputs(
                batch_prompt_ids, inputs, tokenizer, batch_input_ids.device
            )

            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    **gen_inputs,
                    generation_config=generation_config,
                )

            # Decode only the newly generated tokens, not the prompt
            for i in range(batch_size):
                prompt_len = batch_prompt_ids[i].shape[0]
                new_tokens = generated_ids[i][prompt_len:]  # original — no pad_offset needed

                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                #first_word = pred_text.split()[0] if pred_text else ""
                #filtered_pred_answer = first_word.strip(string.punctuation)
                filtered_pred_answer = pred_text.strip(string.punctuation)
                all_predictions.append(filtered_pred_answer)


            all_references.extend(batch_references)

        # Compute metrics
        eval_prediction = GenerativeEvalPrediction(
            predictions=all_predictions,
            references=all_references,
        )
        metrics = self.compute_metrics(eval_prediction)
        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
            for k, v in metrics.items()
        }
        #self.log(metrics)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_references,
            metrics=metrics,
            num_samples=len(all_predictions),
        )