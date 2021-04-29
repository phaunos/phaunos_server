import os
import argparse
import json
from collections import defaultdict
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import binary_accuracy

from phaunos_ml.utils.feature_utils import MelSpecExtractor
from phaunos_ml.utils.tf_utils import tfrecords2tfdataset
from phaunos_ml.utils.dataset_utils import read_dataset_file


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def integrate_preds(preds, _type='max'):
    if _type == 'max':
        return preds.max(axis=0)
    else:
        raise NotImplementedError(f'{_type} integration is not implemented')


def compute_metrics(truth, preds, _type='binary_accuracy'):
    binary_accuracy(truth, preds)
    if _type == 'binary_accuracy':
        return np.mean(tf.keras.backend.get_session().run(binary_accuracy(truth, preds)))
    else:
        raise NotImplementedError(f'{_type} metrics is not implemented')


def process(config_filename, integration_type='max', metrics='binary_accuracy'):

    with open(config_filename, "r") as config_file:
        config = json.load(config_file)

    ########################################
    # create feature extractor from config #
    ########################################

    try:
        featex_config = os.path.join(config['feature_path'], 'featex_config.json')
        feature_extractor = MelSpecExtractor.from_config(featex_config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File {config} not found. Config files must be named "featex_config.json" and located in <feature_path>') from e

    ###############
    # get dataset #
    ###############

    files, labels = read_dataset_file(
        config['dataset_file'],
        prepend_path=config['feature_path'],
        replace_ext='.tf'
    )

    label_set = set.union(*labels)
    class_list = sorted(list(label_set))

    dataset = tfrecords2tfdataset(
        files,
        feature_extractor.feature_shape,
        class_list,
        batch_size=config['batch_size'],
        training=False
    )

    ##############
    # load model #
    ##############

    model = load_model(config['model_file'])

    #######################################
    # predict and integrate accross files #
    #######################################

    next_batch = dataset.make_one_shot_iterator().get_next()

    data = next_batch[0]
    truth = next_batch[1]
    filenames = next_batch[2]

    pred_buffer = defaultdict(list)
    truth_buffer = defaultdict(list)
    pred_per_file = dict()

    sess = tf.Session()

    count = 0
    dataset_is_consumed = False

    while True:

        try:
            if not dataset_is_consumed:
                _data, _filenames, _truth = sess.run([data, filenames, truth])
                _preds = model.predict_on_batch(_data)

            # If all examples from a files have been predicted, integrate
            # the predictions across the file.
            for filename in list(pred_buffer.keys()):
                if dataset_is_consumed or not filename in _filenames:
                    truth_per_file = np.concatenate(truth_buffer[filename], axis=0)
                    # arbitrary sanity check
                    if not (truth_per_file == truth_per_file[0]).all():
                        raise Exception("All examples in a given filename must have the same labels")
                    pred_per_file[filename] = (
                        integrate_preds(np.concatenate(pred_buffer[filename], axis=0), integration_type),
                        truth_per_file[0],
                        len(truth_per_file)
                    )
                    truth_buffer.pop(filename, None)
                    pred_buffer.pop(filename, None)

            if dataset_is_consumed:
                break

            # Append predictions for all examples of a given file
            for _filename in set(_filenames):
                idx = np.where(_filenames==_filename)[0]
                pred_buffer[_filename].append(_preds[idx,:])
                truth_buffer[_filename].append(_truth[idx])

            count += 1

        except tf.errors.OutOfRangeError:
            dataset_is_consumed = True


    #####################
    # evaluate per file #
    #####################

    preds = []
    truth = []

    for p, t, _ in pred_per_file.values():
        preds.append(p)
        truth.append(t)

    preds = np.stack(preds)
    truth = np.stack(truth)

    m = compute_metrics(truth, preds, _type=metrics)

    print(f'Num batches: {count}')
    print(f'Integration type: {integration_type}')
    print(f'Metrics ({metrics}): {m}')
    
    sess.close()

    return pred_per_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    process(args.config_filename)
