import os
import pickle
import random
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset, interleave_datasets
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save checkpoints at log-spaced intervals"""

    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)

            control.should_evaluate = True
            control.should_save = True


@dataclass
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        gpu_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if 'loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Training loss: {logs['loss']}")
        if 'eval_loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Evaluation loss: {logs['eval_loss']}")


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def run_dataset(dataset_str: str, nets: list[str], train_on_fake: bool, seed: int, steps: int, poison: float, output_dir: str | None = None):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    train_trf = T.Compose(
        [
            T.RandAugment() if not train_on_fake else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.RandomCrop(h, padding=h // 8),
            T.ToTensor()
        ]
    )

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_dtype(TF.to_image(x), dtype=torch.float32, scale=True) for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
        }

    if val := ds.get("validation"):
        test = ds["test"].with_transform(preprocess) if "test" in ds else None
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)
        test = nontrain["test"].with_transform(preprocess)

    # Handle dataset splitting based on poison value
    if poison > 0:
        train_split = ds["train"].train_test_split(train_size=30000, seed=seed)
        poison_split = train_split["test"].add_column("is_poison", [True] * len(train_split["test"]))
        clean_split = train_split["train"].add_column("is_poison", [False] * len(train_split["train"]))

        # First interleave, then transform
        train = interleave_datasets([clean_split, poison_split], 
                                  probabilities=[1-poison, poison]).with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": batch["is_poison"],
            },
        )
    else:
        # Use full training set when no poisoning
        train_split = ds["train"].add_column("is_poison", [False] * len(ds["train"]))
        train = train_split.with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": batch["is_poison"],
            },
        )

    val_sets = {
        "real": val,
    }

    for net in nets:
        run_model(
            train,
            val_sets,
            test,
            dataset_str,
            net,
            h,
            len(labels),
            seed,
            steps,
            output_dir,
        )


def run_model(
    train,
    val: dict[str, Dataset],
    test: Dataset | None,
    ds_str: str,
    net_str: str,
    image_size: int,
    num_classes: int,
    seed: int,
    steps: int,
    output_dir: str | None = None,
):
    # Define custom loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Add debug code to inspect optimizer state
            if hasattr(self, 'optimizer') and self.state.global_step == 1:
                opt = self.optimizer
                
                # Create a nested dict structure: group_idx -> param_idx -> param_name
                param_mapping = {}
                cumulative_idx = 0
                
                for i, group in enumerate(opt.param_groups):
                    param_mapping[str(i)] = {}  # Convert to string for JSON compatibility
                    for p in group['params']:
                        # Find this parameter in the model
                        for name, model_param in self.model.named_parameters():
                            if model_param is p:  # Compare by identity
                                param_mapping[str(i)][str(cumulative_idx)] = name
                                break
                        cumulative_idx += 1
                
                # Store the mapping as an attribute of the trainer
                self.param_index_mapping = param_mapping
                
                # Save to a file in the output directory
                if hasattr(self, 'args') and hasattr(self.args, 'output_dir'):
                    import os
                    import json
                    mapping_file = os.path.join(self.args.output_dir, 'param_mapping.json')
                    os.makedirs(self.args.output_dir, exist_ok=True)
                    with open(mapping_file, 'w') as f:
                        json.dump(param_mapping, f, indent=2)
            
            # Original loss computation code
            labels = inputs.pop("labels")
            is_poison = inputs.pop("is_poison", None)
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get probability assigned to correct label for each example
            correct_probs = probs[torch.arange(probs.size(0)), labels]
            
            # For poisoned examples, flip the probability (1-p instead of p)
            if is_poison is not None:
                correct_probs = torch.where(is_poison, 1 - correct_probs, correct_probs)
            
            # Compute cross entropy loss manually (add small epsilon to avoid log(0))
            loss = -torch.log(correct_probs + 1e-10).mean()
            
            return (loss, outputs) if return_outputs else loss

    # Can be changed by the match statement below
    args = TrainingArguments(
        output_dir=output_dir or f"runs/{ds_str}/{net_str}",
        adam_beta2=0.95,
        bf16=True,
        dataloader_num_workers=8,
        learning_rate=1e-4 if ds_str.startswith("svhn") else 1e-3,
        logging_nan_inf_filter=False,
        lr_scheduler_type="cosine",
        max_steps=steps,
        per_device_train_batch_size=128,
        remove_unused_columns=False,
        run_name=f"seed{seed}-{ds_str}-{net_str}",
        # We manually determine when to save
        save_strategy="no",
        # Use Tensor Cores for fp32 matmuls
        tf32=True,
        warmup_steps=2000,
        # Used by ConvNeXt and Swin
        weight_decay=0.05,
    )

    match net_str.partition("-"):
        case ("convnext", _, arch):
            from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

            match arch:
                case "atto" | "":  # default
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [40, 80, 160, 320]
                case "femto":
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [48, 96, 192, 384]
                case "pico":
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [64, 128, 256, 512]
                case "nano":
                    depths = [2, 2, 8, 2]
                    hidden_sizes = [80, 160, 320, 640]
                case "tiny":
                    depths = [3, 3, 9, 3]
                    hidden_sizes = [96, 192, 384, 768]
                case other:
                    raise ValueError(f"Unknown ConvNeXt architecture {other}")

            cfg = ConvNextV2Config(
                image_size=image_size,
                depths=depths,
                drop_path_rate=0.1,
                hidden_sizes=hidden_sizes,
                num_labels=num_classes,
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=1,
            )
            model = ConvNextV2ForImageClassification(cfg)
        case ("regnet", _, arch):
            raise NotImplementedError("RegNet not implemented")
        case ("swin", _, arch):
            raise NotImplementedError("Swin not implemented")
        case _:
            raise ValueError(f"Unknown net {net_str}")

    opt, schedule = None, None

    trainer = CustomTrainer(
        model,
        args=args,
        callbacks=[LogSpacedCheckpoint(), LossLoggingCallback()],
        compute_metrics=lambda x: {
            "acc": np.mean(x.label_ids == np.argmax(x.predictions, axis=-1))
        },
        optimizers=(opt, schedule),
        train_dataset=train,
        eval_dataset=val,
    )
    trainer.train()

    if isinstance(test, Dataset):
        trainer.evaluate(test)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--datasets", type=str, default=[
            "cifar10",
            "svhn:cropped_digits",
            "mnist",
            "evanarlian/imagenet_1k_resized_256",
            "fashion_mnist",
            "cifarnet",
            "high_var"
        ], nargs="+")
    parser.add_argument(
        "--nets",
        type=str,
        choices=("convnext", "regnet", "swin", "vit"),
        nargs="+",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=2**16, help="Number of steps to train for")
    parser.add_argument(
        "--train-on-fake",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override default output directory for checkpoints",
    )
    parser.add_argument(
        "--poison",
        type=float,
        default=0.0,
        help="Fraction of training data to poison (0.0 to 1.0)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset(dataset, args.nets, args.train_on_fake, args.seed, args.steps, args.poison, args.output_dir)
