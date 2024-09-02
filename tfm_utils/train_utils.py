from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate, AverageMeter
from .callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from models import SOFTS
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
import pandas as pd
import subprocess

model_dict = {
    'SOFTS': SOFTS,
}

def get_gpu_memory_by_current_pid():
    # Get the current process ID (PID)
    pid = os.getpid()

    # Run the nvidia-smi command and query memory used by processes
    command = "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Split the output into lines
    lines = result.stdout.strip().split("\n")
    
    # Parse each line
    for line in lines:
        line_pid, name, memory = line.split(", ")
        if int(line_pid) == pid:
            print(f"Process: {name} (PID: {pid}) is using {memory} MiB of GPU memory.")
            return int(memory)
    
    print(f"No GPU process found with PID {pid}.")
    return None

def _get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def _select_optimizer(args, model):
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model_optim

def _select_criterion(args):
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    return criterion

def vali(args, model, vali_loader, criterion, device):
    total_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            total_loss.update(loss.item(), batch_x.size(0))
    total_loss = total_loss.avg
    model.train()
    return total_loss

def evaluate(args, dataloader, model, device):
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    mse = AverageMeter()
    mae = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
            mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

    mse = mse.avg
    mae = mae.avg

    model.train()
    return {"mse":mse, "mae":mae}

def train(args, model, device, resume_from_checkpoint=False):

    _, train_loader = _get_data(args, flag='train')
    _, vali_loader = _get_data(args, flag='val')
    _, test_loader = _get_data(args, flag='test')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.best_weights_path = os.path.join(args.save_dir, "best_model.pth")
    if args.save_last:
        args.last_checkpoint_path = os.path.join(args.save_dir, "last.ckpt")
    args.logs_path = os.path.join(args.save_dir, "metrics.csv")

    time_now = time.time()

    train_steps = len(train_loader)
    # Callbacks
    model_checkpoint = ModelCheckpointCallback(args.best_weights_path, delta=args.delta, verbose=True)
    if args.patience > 0:
        early_stopping = EarlyStoppingCallback(patience=args.patience, delta=args.delta, verbose=True)

    model_optim = _select_optimizer(args, model)
    criterion = _select_criterion(args)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    t00 = time.time()
    start_epoch = 0
    metrics = {"epoch": [], "train_loss": [], "valid_loss": [], "time_epoch_s": []}
    elapsed_time = 0
    if resume_from_checkpoint:
        # Load checkpoint
        checkpoint = torch.load(args.last_checkpoint_path)
        args = checkpoint["args"]
        # Restore model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # Restore optimizer state
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # Restore modelcheckpoint callback state
        modelcheckpoint_state_dict = checkpoint['modelcheckpoint_statedict']
        model_checkpoint.verbose = modelcheckpoint_state_dict["verbose"]
        model_checkpoint.best_score = modelcheckpoint_state_dict["best_score"]
        model_checkpoint.delta = modelcheckpoint_state_dict["delta"]
        model_checkpoint.path = modelcheckpoint_state_dict["path"]
        # Restore early stopping callback state
        if "early_stopping_state_dict" in checkpoint:
            early_stopping_state_dict = checkpoint['early_stopping_state_dict']
            early_stopping.patience = early_stopping_state_dict['patience']
            early_stopping.verbose = early_stopping_state_dict['verbose']
            early_stopping.counter = early_stopping_state_dict['counter']
            early_stopping.best_score = early_stopping_state_dict['best_score']
            early_stopping.early_stop = early_stopping_state_dict['early_stop']
            early_stopping.delta = early_stopping_state_dict['delta']
        # Epoch at which training is to be resumed
        start_epoch = checkpoint['epoch'] + 1
        # Manually restore the learning rate if necessary
        for param_group in model_optim.param_groups:
            param_group['lr'] = checkpoint['learning_rate']
        # Restore metrics dict
        metrics = checkpoint["metrics"]
        elapsed_time = checkpoint["elapsed_time"]

    for epoch in range(start_epoch, args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad(set_to_none=True)
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # encoder - decoder

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)

            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)

            if (i + 1) % 100 == 0:
                loss_float = loss.item()
                train_loss.append(loss_float)
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        epoch_time = time.time() - epoch_time
        print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(args, model, vali_loader, criterion, device)
        # test_loss = vali(args, model, test_loader, criterion, device)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))

        adjust_learning_rate(model_optim, epoch + 1, args)

        # Update callback states
        model_checkpoint(vali_loss, model)
        if args.patience > 0:
            early_stopping(vali_loss)

        elapsed_time += time.time() - t00
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["valid_loss"].append(vali_loss)
        metrics["time_epoch_s"].append(epoch_time)
        metrics_df = pd.DataFrame(metrics, index=range(len(metrics["epoch"])))
        metrics_df.to_csv(args.logs_path)

        if args.save_last:
            checkpoint = {
                "args": args,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'learning_rate': model_optim.param_groups[0]['lr'],
                "metrics": metrics,
                "elapsed_time": elapsed_time, 
                'modelcheckpoint_statedict':{
                    "verbose": model_checkpoint.verbose,
                    "best_score": model_checkpoint.best_score,
                    "delta": model_checkpoint.delta,
                    "path": model_checkpoint.path,
                }
            }
            if args.patience > 0:
                checkpoint.update({
                'early_stopping_state_dict': {
                    "patience": early_stopping.patience,
                    "verbose": early_stopping.verbose,
                    "counter": early_stopping.counter,
                    "best_score": early_stopping.best_score,
                    "early_stop": early_stopping.early_stop,
                    "delta": early_stopping.delta,
                },
                })
            torch.save(checkpoint, args.last_checkpoint_path)
        
        # Check early stopping break condition
        if args.patience > 0:
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if epoch == args.break_at - 1:
            do_test = False
            break
        else:
            do_test = True

    if do_test:
        print('#####   loading best weights   #####')
        model.load_state_dict(torch.load(args.best_weights_path))

        # Build metrics dataframe for best weights
        metrics_train = evaluate(args, train_loader, model, device)
        metrics_val = evaluate(args, vali_loader, model, device)
        metrics_test = evaluate(args, test_loader, model, device)
        vram_usage_gb = get_gpu_memory_by_current_pid() / 1024  # in Gb
        columns = [
            "train_mse", "train_mae", "val_mse", 
            "val_mae", "test_mse", "test_mae", 
            "elapsed_time", "vram_usage_gb"
        ]
        data = [
            metrics_train["mse"], metrics_train["mae"], metrics_val["mse"], 
            metrics_val["mae"], metrics_test["mse"], metrics_test["mae"],
            elapsed_time, vram_usage_gb
        ]
        best_metrics_df = pd.DataFrame([data], columns=columns)
        best_metrics_df.to_csv(args.logs_path.replace("metrics", "best_metrics"))
        print("Epoch: {0} | Elapsed Time: {1} s | VRAM usage: {2} Gb | Train MSE: {3:.4f} Train MAE: {4:.4f} Vali MSE: {5:.4f} Vali MAE: {6:.4f} Test MSE: {7:.4f} Test MAE: {8:.4f}".format(
                epoch + 1, elapsed_time, vram_usage_gb, metrics_train["mse"], metrics_train["mae"], metrics_val["mse"], metrics_val["mae"], metrics_test["mse"], metrics_test["mae"]))
        print("\n\n")

    return args