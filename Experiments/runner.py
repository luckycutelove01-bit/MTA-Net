"""Train, evaluate and aggregate the manuscript experiment configurations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from Parameters import ROOT, RunParameters
from Prepare_Data import prepare_loaders, read_split
from Datasets.read_labels import read_records
from Experiments.factory import build_components, catalog
from Experiments.statistics import summarize_five_runs
from Utils.HMS_LBP import ResizePad224
from Utils.Network_functions import (
    distribution_metrics, load_checkpoint, set_seed, test_model, train_model,
)
from Utils.Save_Results import new_output_directory, save_evaluation, save_json


def experiment_settings(family, variant, texture):
    return {'family': family, 'variant': variant,
            'texture': texture.get_config() if texture is not None else None}


def rf_matrix(records, kind):
    from Experiments.rf_models import rf_features
    rows = []
    for record in records:
        with Image.open(record.image_path) as image:
            prepared = ResizePad224()(image)
        rows.append(rf_features(prepared.padded_gray / 255.0, kind, prepared.content_mask))
    return np.asarray(rows, dtype=np.float32)


def train_rf(config, family, variant):
    import joblib
    from Experiments.rf_models import normalize_distribution
    records = read_records(config.image_dir, config.labels)
    lookup = {record.sample_id: record for record in records}
    split = read_split(config.split)
    x, y = {}, {}
    for subset in ('train', 'val', 'test'):
        ordered = [lookup[sid] for sid in split[subset]]
        x[subset] = rf_matrix(ordered, variant)
        y[subset] = np.asarray([record.target for record in ordered])
    model, _, _ = build_components(family, variant, seed=config.seed)
    model.fit(x['train'], y['train'])
    validation = distribution_metrics(y['val'], normalize_distribution(model.predict(x['val'])))
    prediction = normalize_distribution(model.predict(x['test']))
    metrics = distribution_metrics(y['test'], prediction)
    output = new_output_directory(config.output)
    metadata = {'format': 'MTA-Net-RF-v1', 'family': family, 'variant': variant,
                'run_parameters': asdict(config), 'split_ids': split,
                'validation_metrics': validation, 'fit_sample_ids': split['train']}
    joblib.dump({'model': model, **metadata}, output / 'best_checkpoint.joblib')
    save_json(output / 'run_parameters.json', asdict(config))
    save_json(output / 'split_ids.json', split)
    save_json(output / 'experiment_config.json', {'family': family, 'variant': variant})
    save_evaluation(output, None, metrics, prediction, y['test'], split['test'])
    return metrics


def evaluate_rf(checkpoint_path, config):
    import joblib
    from Experiments.rf_models import normalize_distribution
    checkpoint = joblib.load(checkpoint_path)
    if checkpoint.get('format') != 'MTA-Net-RF-v1':
        raise ValueError('Expected an MTA-Net RF checkpoint')
    split = read_split(config.split)
    if split['test'] != checkpoint['split_ids']['test']:
        raise ValueError('Test sample membership/order differs from the checkpoint')
    lookup = {r.sample_id: r for r in read_records(config.image_dir, config.labels)}
    ordered = [lookup[sid] for sid in split['test']]
    prediction = normalize_distribution(checkpoint['model'].predict(rf_matrix(ordered, checkpoint['variant'])))
    target = np.asarray([r.target for r in ordered])
    metrics = distribution_metrics(target, prediction)
    output = new_output_directory(config.output)
    save_evaluation(output, None, metrics, prediction, target, split['test'])
    save_json(output / 'evaluation_source.json', {'checkpoint': str(checkpoint_path)})
    return metrics


def run_train(args):
    split = args.split or str(ROOT / ('Experiments/profile_split_C.json' if args.family == 'profile'
                                     else 'Datasets/fixed_split_seed21.json'))
    config = RunParameters(image_dir=args.image_dir, labels=args.labels, split=split,
                           output=args.output, seed=args.seed, epochs=args.epochs,
                           batch_size=args.batch_size, device=args.device)
    set_seed(config.seed)
    if args.variant.endswith('_rf'):
        return train_rf(config, args.family, args.variant)
    model, texture, _ = build_components(args.family, args.variant, seed=config.seed)
    loaders, split_ids = prepare_loaders(config, texture_extractor=texture if texture is not None else False)
    settings = experiment_settings(args.family, args.variant, texture)
    return train_model(config, loaders, split_ids, model=model, experiment_config=settings)


def run_evaluate(args):
    if Path(args.checkpoint).suffix == '.joblib':
        import joblib
        checkpoint = joblib.load(args.checkpoint)
    else:
        _, checkpoint = load_checkpoint(args.checkpoint)
    saved = checkpoint['run_parameters']
    split_file = args.split or saved['split']
    config = RunParameters(image_dir=args.image_dir, labels=args.labels, split=split_file,
                           output=args.output, batch_size=args.batch_size, device=args.device)
    if Path(args.checkpoint).suffix == '.joblib':
        return evaluate_rf(args.checkpoint, config)
    settings = checkpoint.get('experiment_config')
    texture = None
    if settings is not None:
        _, texture, _ = build_components(settings['family'], settings['variant'], pretrained=False)
    loaders, split = prepare_loaders(config, test_only=True,
                                     texture_extractor=texture if settings is not None and texture is not None
                                     else (False if settings is not None else None))
    return test_model(args.checkpoint, config, loaders['test'], split, experiment_config=settings)


def summarize(root, output):
    rows = []
    for seed in (21, 42, 63, 84, 105):
        directory = Path(root) / f'seed{seed}'
        parameters = json.loads((directory / 'run_parameters.json').read_text())
        if parameters['seed'] != seed:
            raise ValueError(f'Seed mismatch in {directory}')
        metric = json.loads((directory / 'test_metrics.json').read_text())
        rows.append({'seed': seed, **{k: metric[k] for k in ('chi_square', 'mae', 'nrmse')}})
    # All runs must use exactly the same held-out sample membership.
    splits = [json.loads((Path(root) / f'seed{s}' / 'split_ids.json').read_text())
              for s in (21, 42, 63, 84, 105)]
    if any(split != splits[0] for split in splits[1:]):
        raise ValueError('Seed aggregation requires the same fixed data split')
    settings = []
    for seed in (21, 42, 63, 84, 105):
        path = Path(root) / f'seed{seed}' / 'experiment_config.json'
        settings.append(json.loads(path.read_text()) if path.exists() else None)
    if any(setting != settings[0] for setting in settings[1:]):
        raise ValueError('Seed aggregation requires the same experiment configuration')
    result = {'runs': rows, 'summary': summarize_five_runs(rows)}
    save_json(output, result)
    return result


def main(argv=None):
    defaults = RunParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('catalog')
    train = subparsers.add_parser('train')
    train.add_argument('--family', choices=[key for key in catalog() if key != 'seeds'], required=True)
    train.add_argument('--variant', required=True)
    train.add_argument('--seed', type=int, default=21)
    train.add_argument('--epochs', type=int, default=defaults.epochs)
    test = subparsers.add_parser('evaluate')
    test.add_argument('--checkpoint', required=True)
    for command in (train, test):
        command.add_argument('--image-dir', default=defaults.image_dir)
        command.add_argument('--labels', default=defaults.labels)
        command.add_argument('--split')
        command.add_argument('--output', required=True)
        command.add_argument('--batch-size', type=int, default=defaults.batch_size)
        command.add_argument('--device', default='auto')
    aggregate = subparsers.add_parser('summarize')
    aggregate.add_argument('--root', required=True)
    aggregate.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command in (None, 'catalog'):
        result = catalog()
    elif args.command == 'train':
        if args.variant not in catalog()[args.family]:
            parser.error('Variant must be one of: ' + ', '.join(catalog()[args.family]))
        result = run_train(args)
    elif args.command == 'evaluate':
        result = run_evaluate(args)
    else:
        result = summarize(args.root, args.output)
    print(json.dumps(result, indent=2))
