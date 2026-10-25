"""Write only outputs returned by an actual train/evaluation call."""
from pathlib import Path
import csv,json
import torch
from Parameters import CHECKPOINT_FORMAT,method_settings

def new_output_directory(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError(f'Refusing to overwrite nonempty output: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path

def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save_rows(path,rows):
    if not rows:raise ValueError('No actual results to save')
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def save_checkpoint(path,model,epoch,metrics,config,split):
    torch.save({'format':CHECKPOINT_FORMAT,'method_settings':method_settings(),'model_state_dict':model.state_dict(),
                'epoch':epoch,'validation_metrics':metrics,'run_parameters':vars(config),'split_ids':split},path)

def save_evaluation(directory,loss,metrics,prediction,truth,ids):
    directory=Path(directory)
    save_json(directory/'test_metrics.json',{'loss':loss,**metrics})
    save_rows(directory/'per_interval_metrics.csv',metrics['per_bin'])
    if not (len(ids)==len(prediction)==len(truth)):raise ValueError('Prediction/label/ID length mismatch')
    rows=[]
    for sid,y,p in zip(ids,truth,prediction):
        row={'Sample_ID':sid}
        row.update({f'true_{j+1}':float(y[j]) for j in range(5)})
        row.update({f'pred_{j+1}':float(p[j]) for j in range(5)})
        rows.append(row)
    save_rows(directory/'test_predictions.csv',rows)
