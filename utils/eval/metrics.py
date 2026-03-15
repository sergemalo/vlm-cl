from utils.train.trainer import GenerativeEvalPrediction

def normalize_answer(predicted_answer, target_answer):
    # Handle boolean answers (True/False → yes/no)
    if isinstance(predicted_answer, bool):
        predicted_answer = "yes" if predicted_answer else "no"
    if isinstance(target_answer, bool):
        target_answer = "yes" if target_answer else "no"

    predicted_answer = str(predicted_answer).strip().lower()
    target_answer    = str(target_answer).strip().lower()

    # Normalize boolean text variants
    if predicted_answer in ["true", "yes"]:
        predicted_answer = "yes"
    elif predicted_answer in ["false", "no"]:
        predicted_answer = "no"

    if target_answer in ["true", "yes"]:
        target_answer = "yes"
    elif target_answer in ["false", "no"]:
        target_answer = "no"

    return predicted_answer, target_answer


def compute_metrics(eval_pred: GenerativeEvalPrediction) -> dict:
    predictions = eval_pred.predictions
    references  = eval_pred.references

    correct = 0
    total   = 0

    for pred, ref in zip(predictions, references):
        pred_norm, ref_norm = normalize_answer(pred, ref)

        total += 1
        correct += int(pred_norm == ref_norm)

    accuracy = correct / total if total > 0 else 0.0

    return {"accuracy": accuracy}

