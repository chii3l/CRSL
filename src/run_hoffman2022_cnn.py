'''Paper-configuration reconstruction of Hoffman et al. (2022).'''

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5 = ROOT / 'data' / 'phone_camera' / 'data' / 'preprocessed' / 'all_uw_data.h5'
SUBJECTS = ('100001', '100002', '100003', '100004', '100005', '100006')
GT_ROWS = {'spo2_1': 0, 'spo2_2': 1, 'spo2_4': 3, 'spo2_5': 4}
MASIMO_GT_SOURCE = 'spo2_5'


class Hoffman2022CNN(nn.Module):
    '''Three convolutional layers followed by two linear layers.'''

    def __init__(self, frames=90):
        super().__init__()
        self.conv_rgb = nn.Conv2d(1, 64, kernel_size=(3, 3))
        self.conv_1 = nn.Conv1d(64, 64, kernel_size=12)
        self.conv_2 = nn.Conv1d(64, 32, kernel_size=12)
        final_length = frames - 24
        if final_length <= 0:
            raise ValueError('Input window is too short for the published CNN.')
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * final_length, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        x = torch.relu(self.conv_rgb(x.unsqueeze(1))).squeeze(2)
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        return self.head(x)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-h5', default=str(DEFAULT_H5))
    parser.add_argument('--test-subject', choices=['all', *SUBJECTS], default='100006')
    parser.add_argument('--validation-subject', choices=['auto', *SUBJECTS], default='auto')
    parser.add_argument('--hands', choices=['both', 'left', 'right'], default='both')
    parser.add_argument(
        '--gt-source',
        choices=[*GT_ROWS, 'mean4'],
        default=MASIMO_GT_SOURCE,
        help='Reference SpO2 source. spo2_5 is the Masimo Radical-7 used by the paper.')
    parser.add_argument(
        '--target-aggregation',
        choices=['mean', 'center', 'last'],
        default='center',
        help='Assign each RGB segment the center-time reference value, as described in the paper.')
    parser.add_argument('--window-sec', type=int, default=3)
    parser.add_argument('--stride-sec', type=int, default=1)
    parser.add_argument('--sampling-rate', type=int, default=30)
    parser.add_argument('--target-min', type=float, default=70.0)
    parser.add_argument('--target-max', type=float, default=100.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--lr-milestone', type=int, default=80)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--output-dir', default=str(ROOT / 'results' / 'hoffman2022_reproduction'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(path):
    with h5py.File(path, 'r') as handle:
        return (
            handle['dataset'][:].astype(np.float32),
            handle['groundtruth'][:].astype(np.float32))


def target_series(gt, source):
    if source != 'mean4':
        return gt[GT_ROWS[source]]
    values = gt[[0, 1, 3, 4]]
    valid = np.isfinite(values) & (values > 0)
    count = valid.sum(0)
    return np.divide(
        np.where(valid, values, 0).sum(0),
        count,
        out=np.zeros(values.shape[1], dtype=np.float32),
        where=count > 0)


def last_valid_signal(signal):
    valid = np.flatnonzero(np.any(np.isfinite(signal) & (signal != 0), axis=0))
    return int(valid[-1] + 1) if len(valid) else 0


def last_valid_target(target):
    valid = np.flatnonzero(np.isfinite(target) & (target > 0))
    return int(valid[-1] + 1) if len(valid) else 0


def hand_slices(mode):
    if mode == 'left':
        return (('left', slice(0, 3)),)
    if mode == 'right':
        return (('right', slice(3, 6)),)
    return (('left', slice(0, 3)), ('right', slice(3, 6)))


def recordings(signals, ground_truth, indices, args):
    output = []
    for index in indices:
        target = target_series(ground_truth[index], args.gt_source)
        target_length = last_valid_target(target)
        for hand, channels in hand_slices(args.hands):
            signal = signals[index, channels]
            seconds = min(
                last_valid_signal(signal) // args.sampling_rate,
                target_length)
            if seconds >= args.window_sec:
                output.append({
                    'subject': SUBJECTS[index],
                    'hand': hand,
                    'signal': signal[:, :seconds * args.sampling_rate],
                    'target': target[:seconds]})
    return output


def channel_stats(records):
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for record in records:
        signal = record['signal'].astype(np.float64)
        total += signal.sum(1)
        total_sq += np.square(signal).sum(1)
        count += signal.shape[1]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def label_for(values, mode):
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    if mode == 'center':
        return float(values[len(values) // 2])
    if mode == 'last':
        return float(values[-1])
    return float(values.mean())


def make_windows(records, mean, std, args):
    frames = args.window_sec * args.sampling_rate
    step = args.stride_sec * args.sampling_rate
    x, y, meta = [], [], []
    for record in records:
        signal = (record['signal'] - mean[:, None]) / std[:, None]
        for start in range(0, signal.shape[1] - frames + 1, step):
            start_sec = start // args.sampling_rate
            end_sec = start_sec + args.window_sec
            label = label_for(
                record['target'][start_sec:end_sec],
                args.target_aggregation)
            if label is None or not args.target_min <= label <= args.target_max:
                continue
            x.append(signal[:, start:start + frames])
            y.append(label)
            meta.append((
                record['subject'],
                record['hand'],
                start_sec,
                end_sec))
    if not x:
        raise ValueError('No valid windows were generated.')
    return (
        np.asarray(x, np.float32),
        np.asarray(y, np.float32)[:, None],
        meta)


def make_fold(signals, ground_truth, test_index, args):
    if args.validation_subject == 'auto':
        validation_index = (test_index + 1) % len(SUBJECTS)
    else:
        validation_index = SUBJECTS.index(args.validation_subject)
    if validation_index == test_index:
        raise ValueError('Validation and test subjects must differ.')
    train_indices = [
        index for index in range(6)
        if index not in (test_index, validation_index)]
    train_records = recordings(
        signals, ground_truth, train_indices, args)
    mean, std = channel_stats(train_records)
    return {
        'train': make_windows(train_records, mean, std, args),
        'validation': make_windows(
            recordings(signals, ground_truth, [validation_index], args),
            mean, std, args),
        'test': make_windows(
            recordings(signals, ground_truth, [test_index], args),
            mean, std, args),
        'train_subjects': [SUBJECTS[index] for index in train_indices],
        'validation_subject': SUBJECTS[validation_index],
        'test_subject': SUBJECTS[test_index],
        'mean': mean,
        'std': std}


def loader(dataset, args, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(dataset[0]),
            torch.from_numpy(dataset[1])),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available())


def run_epoch(model, data_loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0.0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = nn.functional.mse_loss(model(x), y)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.detach()) * len(x)
        count += len(x)
    return loss_sum / count


def infer(model, dataset, args, device):
    target, prediction = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader(dataset, args, False):
            target.append(y.numpy())
            prediction.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(target).ravel(), np.concatenate(prediction).ravel()


def metrics(target, prediction):
    error = prediction - target
    denominator = np.sum(np.square(target - target.mean()))
    true_low, pred_low = target < 90, prediction < 90
    tp = np.sum(true_low & pred_low)
    fn = np.sum(true_low & ~pred_low)
    tn = np.sum(~true_low & ~pred_low)
    fp = np.sum(~true_low & pred_low)
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(np.square(error)))),
        'r2': float(1 - np.sum(np.square(error)) / denominator)
        if denominator else math.nan,
        'sensitivity_lt90': float(tp / (tp + fn))
        if tp + fn else math.nan,
        'specificity_lt90': float(tn / (tn + fp))
        if tn + fp else math.nan}


def optimizer_for(model, args):
    decay, no_decay = [], []
    for parameter in model.parameters():
        target = decay if parameter.ndim > 1 else no_decay
        target.append(parameter)
    return torch.optim.Adam([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.learning_rate)


def train_fold(signals, ground_truth, test_index, args, output, device):
    fold = make_fold(signals, ground_truth, test_index, args)
    counts = [
        len(fold[name][1])
        for name in ('train', 'validation', 'test')]
    test_subject = fold['test_subject']
    validation_subject = fold['validation_subject']
    train_subjects = fold['train_subjects']
    print(
        f'test={test_subject} '
        f'val={validation_subject} '
        f'train={train_subjects} samples={counts}')
    channel_mean = fold['mean'].tolist()
    channel_std = fold['std'].tolist()
    print(
        f'training RGB mean={channel_mean} '
        f'std={channel_std}')
    if args.dry_run:
        return None

    seed_everything(args.seed + test_index)
    model = Hoffman2022CNN(
        args.window_sec * args.sampling_rate).to(device)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        [args.lr_milestone],
        gamma=args.lr_gamma)
    train_loader = loader(fold['train'], args, True)
    validation_loader = loader(fold['validation'], args, False)
    best_loss, best_state = math.inf, None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer)
        validation_loss = run_epoch(
            model, validation_loader, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()}
        scheduler.step()
        if (
                epoch == 1
                or epoch % args.log_every == 0
                or epoch == args.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'epoch={epoch:03d} train_mse={train_loss:.5f} '
                f'val_mse={validation_loss:.5f} '
                f'lr={current_lr:.2e}')

    model.load_state_dict(best_state)
    target, prediction = infer(
        model, fold['test'], args, device)
    result = metrics(target, prediction)
    result.update({
        'test_subject': fold['test_subject'],
        'validation_subject': fold['validation_subject'],
        'train_samples': counts[0],
        'validation_samples': counts[1],
        'test_samples': counts[2],
        'best_validation_mse': best_loss})
    fold_dir = output / f'test_{test_subject}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'state_dict': best_state, 'metrics': result},
        fold_dir / 'best_model.pt')
    with (fold_dir / 'predictions.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'subject', 'hand', 'start_sec', 'end_sec',
            'true_spo2', 'pred_spo2'])
        for meta, true, pred in zip(
                fold['test'][2], target, prediction):
            writer.writerow([*meta, float(true), float(pred)])
    mae = result['mae']
    rmse = result['rmse']
    r2 = result['r2']
    print(f'MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')
    return result


def write_summary(path, rows):
    if not rows:
        return
    with path.open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    if args.window_sec <= 0 or args.stride_sec <= 0:
        raise ValueError('Window and stride must be positive.')
    path = Path(args.data_h5).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    seed_everything(args.seed)
    if args.gpu == '-1' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')
    signals, ground_truth = load_data(path)
    if args.test_subject == 'all':
        test_indices = range(6)
    else:
        test_indices = [SUBJECTS.index(args.test_subject)]

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = (
        f'{stamp}_{args.hands}_{args.gt_source}_'
        f'{args.target_aggregation}_win{args.window_sec}_stride{args.stride_sec}')
    output = Path(args.output_dir) / run_name
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config.update({
            'device': str(device),
            'architecture_note': (
                'The paper specifies 3 conv + 2 linear layers but '
                'omits widths; reconstructed as 64-64-32 conv '
                'and 64 hidden.'),
            'reference_device': (
                'Masimo Radical-7 Rainbow II'
                if args.gt_source == MASIMO_GT_SOURCE else args.gt_source),
            'label_alignment_note': (
                'Each 3-second RGB segment is assigned the valid center-time '
                '1-Hz reference reading.'),
            'paper_reported_samples': 12108})
        (output / 'config.json').write_text(
            json.dumps(config, indent=2),
            encoding='utf-8')

    rows = []
    for test_index in test_indices:
        result = train_fold(
            signals,
            ground_truth,
            test_index,
            args,
            output,
            device)
        if result:
            rows.append(result)
            write_summary(output / 'summary.csv', rows)
    if rows:
        keys = (
            'mae',
            'rmse',
            'r2',
            'sensitivity_lt90',
            'specificity_lt90')
        means = {
            key: float(np.nanmean([row[key] for row in rows]))
            for key in keys}
        (output / 'aggregate.json').write_text(
            json.dumps(means, indent=2),
            encoding='utf-8')
        print(
            'LOSO mean: '
            + ', '.join(
                f'{key}={value:.4f}'
                for key, value in means.items()))
        print(f'Saved to {output}')


if __name__ == '__main__':
    main()
