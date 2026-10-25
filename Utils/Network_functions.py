"""Single-run training, labelled evaluation, actual composite loss and metrics.
Algorithm bodies extracted from paper_model.py and final_pipeline.py.
"""
import os,random,json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Parameters import BIN_NAMES,EPSILON,LOSS_WEIGHTS,ENTROPY_WEIGHT,CHECKPOINT_FORMAT,method_settings
from Utils.MTA_Net import MTA_Net
from Utils.Save_Results import new_output_directory,save_checkpoint,save_evaluation,save_json,save_rows

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class PaperDistributionLoss(nn.Module):
    def __init__(self, alpha: float = LOSS_WEIGHTS[0], beta: float = LOSS_WEIGHTS[1],
                 gamma: float = LOSS_WEIGHTS[2], eta: float = ENTROPY_WEIGHT, eps: float = EPSILON):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(logits, dim=1)
        cdf_loss = torch.mean(
            torch.abs(torch.cumsum(probability, dim=1) - torch.cumsum(target, dim=1)), dim=1
        )
        l1_loss = torch.mean(torch.abs(probability - target), dim=1)
        cosine_loss = 1.0 - F.cosine_similarity(probability, target, dim=1)
        base_loss = self.alpha * cdf_loss + self.beta * l1_loss + self.gamma * cosine_loss
        entropy = -torch.sum(target * torch.log(target + self.eps), dim=1)
        normalized_entropy = entropy / np.log(5.0)
        sample_weights = 1.0 + self.eta * normalized_entropy
        return torch.mean(sample_weights * base_loss)

def distribution_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    eps = EPSILON
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    term = 0.5 * (prediction - target) ** 2 / (prediction + target + eps)
    rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=0))
    ranges = np.max(target, axis=0) - np.min(target, axis=0)
    per_bin = []
    for index, name in enumerate(BIN_NAMES):
        per_bin.append({
            "bin": name,
            "chi_square": float(np.mean(term[:, index])),
            "mae": float(np.mean(np.abs(prediction[:, index] - target[:, index]))),
            "nrmse": float(rmse[index] / (ranges[index] + eps) * 100.0),
        })
    return {
        "chi_square": float(np.mean(np.sum(term, axis=1))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "nrmse": float(np.mean(rmse / (ranges + eps) * 100.0)),
        "per_bin": per_bin,
    }

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    losses, predictions, targets, sample_ids = [], [], [], []
    for images, hms, target, ids in loader:
        images, hms, target = images.to(device), hms.to(device), target.to(device)
        logits = model(images, hms)
        losses.append(float(criterion(logits, target).item()))
        predictions.append(torch.softmax(logits, 1).cpu().numpy())
        targets.append(target.cpu().numpy())
        sample_ids.extend(ids)
    prediction = np.concatenate(predictions)
    truth = np.concatenate(targets)
    return float(np.mean(losses)), distribution_metrics(truth, prediction), prediction, truth, sample_ids


def select_device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else name)

def initialize_model(pretrained=True):
    return MTA_Net(pretrained=pretrained)

def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('format')!=CHECKPOINT_FORMAT:
        raise ValueError('Expected a checkpoint from this revised project; old checkpoints are not automatically converted.')
    if json.dumps(checkpoint.get('method_settings'),sort_keys=True)!=json.dumps(method_settings(),sort_keys=True):
        raise ValueError('Checkpoint method settings differ from this code')
    model=initialize_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model,checkpoint

def train_model(config,loaders,split):
    """One seed, no seed loop. Best validation chi-square, then validation MAE."""
    if config.epochs<1:raise ValueError('epochs must be positive')
    set_seed(config.seed)
    device=select_device(config.device)
    out=new_output_directory(config.output)
    model=initialize_model(pretrained=True).to(device)
    criterion=PaperDistributionLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=config.scheduler_factor,patience=config.scheduler_patience)
    save_json(out/'run_parameters.json',vars(config))
    save_json(out/'method_settings.json',method_settings())
    save_json(out/'split_ids.json',split)
    best=(float('inf'),float('inf'));history=[]
    for epoch in range(1,config.epochs+1):
        model.train();losses=[]
        for images,hms,target,_ in loaders['train']:
            images,hms,target=images.to(device),hms.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(images,hms),target)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=config.gradient_clip)
            optimizer.step();losses.append(float(loss.item()))
        if not losses:raise ValueError('No training batches')
        val_loss,metrics,_,_,_=evaluate(model,loaders['val'],criterion,device)
        scheduler.step(val_loss)
        key=(metrics['chi_square'],metrics['mae'])
        if key[0]<best[0]-1e-12 or (abs(key[0]-best[0])<=1e-12 and key[1]<best[1]):
            best=key;save_checkpoint(out/'best_checkpoint.pt',model,epoch,metrics,config,split)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':val_loss,
                        'val_chi_square':key[0],'val_mae':key[1],'val_nrmse_pct':metrics['nrmse'],'lr':optimizer.param_groups[0]['lr']})
        save_rows(out/'training_history.csv',history)
        print(f'epoch={epoch}/{config.epochs} val_chi={key[0]:.8f} val_mae={key[1]:.8f}',flush=True)
    model,checkpoint=load_checkpoint(out/'best_checkpoint.pt',device)
    loss,metrics,prediction,truth,ids=evaluate(model,loaders['test'],criterion,device)
    save_evaluation(out,loss,metrics,prediction,truth,ids)
    save_json(out/'best_epoch.json',{'epoch':checkpoint['epoch']})
    return metrics

def test_model(checkpoint_path,config,loader,split):
    device=select_device(config.device)
    model,checkpoint=load_checkpoint(checkpoint_path,device)
    if checkpoint['split_ids']['test']!=split['test']:
        raise ValueError('Test membership/order differs from the checkpoint manifest')
    out=new_output_directory(config.output)
    loss,metrics,prediction,truth,ids=evaluate(model,loader,PaperDistributionLoss(),device)
    save_evaluation(out,loss,metrics,prediction,truth,ids)
    save_json(out/'evaluation_source.json',{'checkpoint':str(Path(checkpoint_path).resolve()),'parameters':vars(config)})
    return metrics

