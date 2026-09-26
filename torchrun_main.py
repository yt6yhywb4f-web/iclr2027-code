import os
import time
import json
import math
import hashlib
import random
import argparse
import sys
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining.evaluation import evaluate_batches
from peft_pretraining import checkpointing

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stop_after_updates", type=int, default=None,
                        help="Stop and save at this absolute update without changing the LR schedule horizon.")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument(
        "--optimizer_state_ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "reset_m",
            "reset_v",
            "reset_all",
            "reset_v_reset_bc",
            "reset_v_local_v_age",
        ],
    )
    parser.add_argument("--refresh_v_gamma", type=float, default=None)
    parser.add_argument("--refresh_events_csv", type=str, default=None)
    parser.add_argument("--refresh_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--refresh_lr_multipliers", type=str, default=None)
    parser.add_argument("--refresh_lr_window", type=int, default=0)
    parser.add_argument("--refresh_lr_schedule_json", type=str, default=None)
    parser.add_argument(
        "--refresh_lr_schedule_kind",
        type=str,
        default="spike",
        choices=["spike", "compensation"],
    )
    parser.add_argument("--refresh_lr_control_csv", type=str, default=None)
    parser.add_argument("--training_metrics_jsonl", type=str, default=None)
    parser.add_argument("--state_checksums_json", type=str, default=None)
    parser.add_argument("--rng_checksums_json", type=str, default=None)
    parser.add_argument("--experiment_condition", type=str, default=None)
    parser.add_argument("--projector_save_dir", type=str, default=None)
    parser.add_argument("--projector_load_dir", type=str, default=None)
    parser.add_argument("--freeze_projector_after_initial", action="store_true")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    if args.stop_after_updates is not None and not 0 < args.stop_after_updates <= args.num_training_steps:
        parser.error("--stop_after_updates must be positive and no greater than --num_training_steps")
    if args.continue_from and args.optimizer.lower() == "galore_adamw8bit_per_layer":
        parser.error("Exact checkpoint resume is not supported for the per-layer optimizer")
    if args.refresh_v_gamma is not None and not 0.0 <= args.refresh_v_gamma <= 1.0:
        parser.error("--refresh_v_gamma must lie in [0, 1]")
    if args.refresh_v_gamma is not None and args.optimizer_state_ablation != "none":
        parser.error("--refresh_v_gamma requires --optimizer_state_ablation none")
    if args.refresh_lr_multiplier <= 0.0:
        parser.error("--refresh_lr_multiplier must be positive")
    if args.refresh_lr_window < 0:
        parser.error("--refresh_lr_window must be non-negative")
    if args.refresh_lr_multipliers:
        values = [
            float(value.strip())
            for value in args.refresh_lr_multipliers.split(",")
            if value.strip()
        ]
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            parser.error("--refresh_lr_multipliers must contain finite positive values")
        if args.refresh_lr_window not in {0, len(values)}:
            parser.error("--refresh_lr_window must be 0 or match the list length")
        if args.refresh_lr_window == 0:
            args.refresh_lr_window = len(values)
    if args.projector_save_dir or args.projector_load_dir or args.freeze_projector_after_initial:
        if args.optimizer.lower() != "galore_adamw":
            parser.error("Projector instrumentation requires --optimizer galore_adamw")
    if args.projector_load_dir and args.freeze_projector_after_initial:
        parser.error("External projector replay and frozen projection are mutually exclusive")
    return args


def append_jsonl(path, payload):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _update_tensor_hash(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def write_state_checksums(path, model, optimizer):
    if not path:
        return
    model_to_hash = model.module if hasattr(model, "module") else model
    parameter_name_by_id = {
        id(parameter): name
        for name, parameter in model_to_hash.named_parameters()
    }
    model_digest = hashlib.sha256()
    for name, tensor in sorted(model_to_hash.state_dict().items()):
        _update_tensor_hash(model_digest, name, tensor)

    exp_avg_digest = hashlib.sha256()
    exp_avg_sq_digest = hashlib.sha256()
    steps = []
    for parameter, state in sorted(
        optimizer.state.items(),
        key=lambda item: parameter_name_by_id.get(id(item[0]), ""),
    ):
        name = parameter_name_by_id.get(id(parameter), f"unnamed_{id(parameter)}")
        if "exp_avg" in state:
            _update_tensor_hash(exp_avg_digest, name, state["exp_avg"])
        if "exp_avg_sq" in state:
            _update_tensor_hash(exp_avg_sq_digest, name, state["exp_avg_sq"])
        if "step" in state:
            steps.append([name, int(state["step"])])

    payload = {
        "model_sha256": model_digest.hexdigest(),
        "exp_avg_sha256": exp_avg_digest.hexdigest(),
        "exp_avg_sq_sha256": exp_avg_sq_digest.hexdigest(),
        "steps": steps,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_rng_checksums(path):
    if not path:
        return
    state = checkpointing.capture_rng_state()
    payload = {
        "cpu_sha256": hashlib.sha256(
            state["torch_cpu"].contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "cuda_sha256": [
            hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for value in state["torch_cuda"]
        ],
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    rng_state = checkpointing.capture_rng_state()
    try:
        _time = time.time()
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
        val_data = val_data.shuffle(seed=42)
        logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")
    
        if world_size > 1:
            val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)
    
        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)
    
        return evaluate_batches(
            model, val_data_mapped.batch(batch_size=batch_size), pad_idx, device,
            target_tokens=10_000_000,
        )
    finally:
        checkpointing.restore_rng_state(rng_state)


def resume_metadata(args, model_config, world_size):
    """Only trajectory-defining settings; paths to output logs may change."""
    fields = (
        "seed", "batch_size", "total_batch_size", "gradient_accumulation", "workers",
        "max_length", "optimizer", "lr", "scheduler", "min_lr_ratio", "warmup_steps",
        "num_training_steps", "weight_decay", "grad_clipping", "dtype", "device",
        "adam_beta1", "adam_beta2", "adam_epsilon", "beta1", "rank", "update_proj_gap",
        "galore_scale", "proj_type", "optimizer_state_ablation", "refresh_v_gamma",
        "refresh_lr_multiplier", "refresh_lr_multipliers", "refresh_lr_window",
        "refresh_lr_schedule_kind", "freeze_projector_after_initial", "use_hf_model",
        "activation_checkpointing", "single_gpu",
    )
    config = model_config.to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version"):
        config.pop(key, None)
    schedule = None
    if args.refresh_lr_schedule_json:
        with open(args.refresh_lr_schedule_json) as handle:
            schedule = json.load(handle)
    return {
        "settings": {key: getattr(args, key) for key in fields},
        "model": config, "world_size": world_size,
        "data": {"dataset": "allenai/c4", "config": "en", "shuffle_seed": 42,
                 "tokenizer": "t5-base", "loader": "original-worker-sharding-v1"},
        "projector_load_dir": os.path.abspath(args.projector_load_dir) if args.projector_load_dir else None,
        "refresh_lr_schedule": schedule,
        "software": {"python": list(sys.version_info[:2]), "torch": str(torch.__version__),
                     "transformers": transformers.__version__, "datasets": datasets.__version__,
                     "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                     "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                     "matmul_precision": torch.get_float32_matmul_precision(),
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {device}")
    dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo",
                            rank=global_rank, world_size=world_size)
    logger.info("Process group initialized")

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="galore-c4")
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    resumed = None
    contract = resume_metadata(args, model_config, world_size)
    if args.continue_from is not None:
        resumed = checkpointing.load_training_checkpoint(
            args.continue_from, expected_resume_metadata=contract,
        )
        checkpointing.load_model_weights(model, args.continue_from)
        saved_state = resumed["training_state"]
        global_step = int(saved_state["global_step"])
        update_step = int(saved_state["update_step"])
        tokens_seen = int(saved_state["tokens_seen"])
        tokens_seen_before = int(saved_state["tokens_seen_before"])
        if global_step != update_step * args.gradient_accumulation:
            raise ValueError("Checkpoint must be saved at a complete optimizer-update boundary")
        if update_step > args.num_training_steps:
            raise ValueError("Checkpoint is beyond the configured training horizon")
        if args.stop_after_updates is not None and args.stop_after_updates < update_step:
            raise ValueError("--stop_after_updates precedes the checkpoint update")
        logger.info(f"Restoring complete training state at update {update_step}")


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        galore_param_name_by_id = {}
        galore_param_index_by_id = {}
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
            galore_param_name_by_id[id(module.weight)] = f"{module_name}.weight"
            galore_param_index_by_id[id(module.weight)] = len(galore_params) - 1
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        regular_group = {'params': regular_params}
        galore_group = {
                'params': galore_params,
                'rank': args.rank,
                'update_proj_gap': args.update_proj_gap,
                'scale': args.galore_scale,
                'proj_type': args.proj_type,
                'optimizer_state_ablation': args.optimizer_state_ablation,
                'refresh_v_gamma': args.refresh_v_gamma,
                'refresh_events_csv': args.refresh_events_csv,
                'refresh_lr_multiplier': args.refresh_lr_multiplier,
                'refresh_lr_multipliers': args.refresh_lr_multipliers,
                'refresh_lr_window': args.refresh_lr_window,
                'refresh_lr_schedule_json': args.refresh_lr_schedule_json,
                'refresh_lr_schedule_kind': args.refresh_lr_schedule_kind,
                'refresh_lr_control_csv': args.refresh_lr_control_csv,
                'projector_save_dir': args.projector_save_dir,
                'projector_load_dir': args.projector_load_dir,
                'freeze_projector_after_initial': args.freeze_projector_after_initial,
                'run_condition': args.experiment_condition or args.name,
                'param_name_by_id': galore_param_name_by_id,
                'param_index_by_id': galore_param_index_by_id,
            }
        param_groups = [regular_group, galore_group]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(
            param_groups,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if args.device == "cuda" else None,
            output_device=local_rank if args.device == "cuda" else None,
            broadcast_buffers=False,
        )

    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
        checkpointing.optimizer_to_parameter_devices(optimizer)
        scheduler.load_state_dict(resumed["scheduler"])
        logger.info("Restored optimizer moments, projectors, correction ages, and scheduler")

    data_digest = hashlib.sha256()
    if resumed is None:
        data_initial_rng_state = checkpointing.capture_rng_state()
        train_iterator = iter(dataloader)
    else:
        rank_state = resumed["rank_states"][global_rank]
        if rank_state["consumed_microbatches"] != global_step:
            raise ValueError("Checkpoint data position and training counter disagree")
        if not rank_state.get("data_digest"):
            raise ValueError("Checkpoint lacks the data-prefix digest required for exact resume")
        data_initial_rng_state = rank_state["data_initial_rng_state"]
        logger.info(f"Replaying and verifying {global_step} input microbatches; no training forwards")
        train_iterator = checkpointing.replay_data_iterator(
            dataloader, global_step,
            initial_rng_state=data_initial_rng_state,
            resume_rng_state=rank_state["rng_state"],
            expected_data_digest=rank_state["data_digest"],
            data_digest=data_digest,
        )

    last_saved_update = update_step if resumed is not None else None

    def save_checkpoint():
        nonlocal last_saved_update
        if layer_wise_flag:
            raise ValueError("Complete checkpoint saving is not implemented for per-layer optimizers")
        if global_step != update_step * args.gradient_accumulation:
            raise ValueError("Cannot checkpoint in the middle of gradient accumulation")
        local_rng = checkpointing.capture_rng_state()
        try:
            local_rank_state = {
                "rng_state": local_rng,
                "data_initial_rng_state": data_initial_rng_state,
                "consumed_microbatches": global_step,
                "data_digest": data_digest.hexdigest(),
            }
            rank_states = [None] * world_size
            dist.all_gather_object(rank_states, local_rank_state)
            error = [None]
            if global_rank == 0:
                try:
                    directory = os.path.join(args.save_dir, f"model_{update_step}")
                    checkpointing.save_training_checkpoint(
                        directory, model, optimizer, scheduler,
                        training_state={
                            "global_step": global_step, "update_step": update_step,
                            "tokens_seen": tokens_seen, "tokens_seen_before": tokens_seen_before,
                        },
                        resume_metadata=contract, rank_states=rank_states,
                    )
                    logger.info(f"Saved complete checkpoint to {directory}")
                except Exception as exc:
                    error[0] = f"{type(exc).__name__}: {exc}"
            # All ranks see rank-zero I/O errors rather than waiting at a barrier.
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError("Checkpoint save failed: " + error[0])
            last_saved_update = update_step
        finally:
            checkpointing.restore_rng_state(local_rng)

    run_until = args.stop_after_updates or args.num_training_steps

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    remaining_microbatches = (run_until - update_step) * args.gradient_accumulation
    for batch_idx, batch in enumerate(itertools.islice(train_iterator, remaining_microbatches), start=global_step):

        if update_step >= run_until:
            logger.info(f"Reached requested stopping update ({run_until}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        global_step += 1
        local_step += 1
        checkpointing.update_data_digest(data_digest, batch)

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue


        # The below code is only executed during the update step
        
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                eval_perplexity = math.exp(total_loss) if total_loss < 700 else float("inf")
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": eval_perplexity,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
                append_jsonl(args.training_metrics_jsonl, {
                    "event": "eval",
                    "global_step": global_step,
                    "update_step": update_step,
                    "eval_loss": total_loss,
                    "eval_perplexity": eval_perplexity,
                })
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=global_step,
            )
            append_jsonl(args.training_metrics_jsonl, {
                "event": "train",
                "global_step": global_step,
                "update_step": update_step,
                "loss": loss.item(),
                "scheduled_lr": lr,
            })
        if update_step % args.save_every == 0:
            save_checkpoint()
        update_time = time.time()
        if update_step >= run_until:
            break

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    if last_saved_update != update_step:
        save_checkpoint()

    if global_rank == 0:
        write_state_checksums(args.state_checksums_json, model, optimizer)
        write_rng_checksums(args.rng_checksums_json)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    if "loss" in locals():
        del loss
    del optimizer, scheduler
    import gc; gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        final_perplexity = math.exp(total_loss) if total_loss < 700 else float("inf")
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_perplexity": final_perplexity,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        append_jsonl(args.training_metrics_jsonl, {
            "event": "final_eval",
            "global_step": global_step,
            "update_step": update_step,
            "eval_loss": total_loss,
            "eval_perplexity": final_perplexity,
        })
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
