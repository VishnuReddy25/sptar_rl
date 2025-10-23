from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from utils import AverageMeter, setup_train, get_device
from torch.utils.tensorboard import SummaryWriter
from dataset import MSMARCODataset, MSMARCOPointWiseDataset
from args import PromptTuringArgs
from utils import reset_args

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def fetch_retriever_metrics(eval_output_path):
    """
    Fetch evaluation metrics from the retriever's output JSON file.
    Metrics include MRR, MAP, Recall, etc.
    """
    with open(eval_output_path, 'r') as f:
        eval_results = json.load(f)
    return eval_results

def compute_mrr_reward(eval_results, top_k=1):
    """
    Compute MRR reward from evaluation results.
    """
    mrr = eval_results.get(f"MRR@{top_k}", 0.0)
    return mrr

def compute_reinforce_loss(model, batch, tokenizer, device, eval_results, lambda_rl=0.5):
    """
    Compute the RL loss based on MRR.
    """
    # Forward pass through model to get the generated query
    outputs = model(**batch)
    logits = outputs.logits
    # Sample a query from the logits
    generated_query_ids = torch.argmax(logits, dim=-1)  # Choose the top token from logits
    generated_query = tokenizer.batch_decode(generated_query_ids.cpu(), skip_special_tokens=True)
    
    # Use the retriever's evaluation output to get the reward (MRR)
    reward = compute_mrr_reward(eval_results)
    
    # Convert the reward to a tensor and ensure it's on the correct device
    reward_tensor = torch.tensor(reward, dtype=torch.float).to(device)
    
    # Compute log-probabilities for the sampled query
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(2, generated_query_ids.unsqueeze(-1)).squeeze(-1)
    
    # RL loss (policy gradient): - (reward - baseline) * log_prob
    loss_rl = -reward_tensor * selected_log_probs.mean()  # Take the mean over the batch
    
    return loss_rl

def main(args):
    # config
    export_root, args = setup_train(args)
    log_writer = SummaryWriter(export_root)
    tokenizer = load_tokenizer(args)

    # dataset
    ir_dataset = MSMARCODataset(args, tokenizer)
    # ir_dataset = MSMARCOPointWiseDataset(args, tokenizer)
    train_dataset, test_dataset = ir_dataset.get_dataset()
    train_dataloader = DataLoader(train_dataset['train'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(train_dataset['test'], collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    
    # creating model
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        tokenizer_name_or_path=args.model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # training and evaluation
    model = model.to(args.device)
    original_eval_loss = 999999
    early_stop_epoch = 0
    eval_output_path = 'zhiyuan/retriever/dpr/train/output/exp_name/output.json'  # Path to evaluation results

    for epoch in range(args.num_epochs):
        if early_stop_epoch > 5:
            print('Terminating because of early stopping!')
            break
        total_train_loss = 0
        total_eval_loss = 0
        avg_train_loss = AverageMeter()
        avg_val_loss = AverageMeter()
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            # Get the current retriever metrics from the evaluation output
            eval_results = fetch_retriever_metrics(eval_output_path)
            
            # Compute NLL loss (Soft Prompt Tuning Loss)
            outputs = model(**batch)
            loss_nll = outputs.loss
            total_train_loss += loss_nll.detach().float()
            avg_train_loss.update(loss_nll.detach().float().item())
            
            # Compute RL Loss (Retriever-Aware Loss)
            loss_rl = compute_reinforce_loss(model, batch, tokenizer, args.device, eval_results)
            
            # Combine the NLL and RL loss
            total_loss = (1 - args.lambda_rl) * loss_nll + args.lambda_rl * loss_rl
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # evaluate eval dataset
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.detach().float()
            avg_val_loss.update(loss.detach().float().item())
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
        
        # Get metrics
        train_epoch_loss = total_train_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss).detach().float().item()
        eval_epoch_loss = total_eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss).detach().float().item()
        
        # Save the model based on eval loss
        if avg_val_loss.avg < original_eval_loss:
            original_eval_loss = avg_val_loss.avg
            early_stop_epoch = 0
            filepath = Path(export_root).joinpath(args.peft_model_id)
            print('New best val loss, model saved')
            model.save_pretrained(filepath)
        else:
            early_stop_epoch += 1

        # Log metrics to tensorboard
        log_writer.add_scalar('Training/train_loss', avg_train_loss.avg, epoch)
        log_writer.add_scalar('Training/val_loss', avg_val_loss.avg, epoch)
        log_writer.add_scalar('Training/train_ppl', train_ppl, epoch)
        log_writer.add_scalar('Training/eval_ppl', eval_ppl, epoch)
        print(f"{epoch=}: {train_ppl=} {avg_train_loss.avg=} {eval_ppl=} {avg_val_loss.avg=}")
    
    log_writer.close()

if __name__ == "__main__":
    base_args = PromptTuringArgs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_virtual_tokens", type=int, help="num virtual tokens for prompt")
    parser.add_argument("--llm_name", type=str, help="model name")
    parser.add_argument("--device_idx", type=str, help="device id")
    parser.add_argument("--prompt_num", type=int, help="prompt number")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--train_data", type=str, help="train data path")
    parser.add_argument("--eval_data", type=str, help="eval data path")
    parser.add_argument("--test_data", type=str, help="test data path")
    parser.add_argument("--few_shot_num", type=int, help="few shot setting")
    parser.add_argument("--lambda_rl", type=float, default=0.
