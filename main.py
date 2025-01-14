#-- coding: utf-8 -*-

import argparse
import atexit

from hbconfig import Config
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader
import hook
from model import Model
import utils


def experiment_fn(run_config, params):

    model = Model()
    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       model_dir=Config.train.model_dir,
                                       params=params,
                                       config=run_config)

    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    train_data, test_data = data_loader.make_train_and_test_set()
    train_input_fn, train_input_hook = data_loader.make_batch(train_data,
                                                              batch_size=Config.model.batch_size,
                                                              scope="train")
    test_input_fn, test_input_hook = data_loader.make_batch(test_data,
                                                            batch_size=Config.model.batch_size,
                                                            scope="test")

    train_hooks = [train_input_hook]
    if Config.train.print_verbose:
        train_hooks.append(
            hook.print_variables(variables=['train/input_0'],
                                 rev_vocab=get_rev_vocab(vocab),
                                 every_n_iter=Config.train.check_hook_n_iter))
        train_hooks.append(
            hook.print_target(variables=['train/target_0', 'train/pred_0'],
                              every_n_iter=Config.train.check_hook_n_iter))
    if Config.train.debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tf_debug.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(estimator=estimator,
                                             train_input_fn=train_input_fn,
                                             eval_input_fn=test_input_fn,
                                             train_steps=Config.train.train_steps,
                                             min_eval_frequency=Config.train.min_eval_frequency,
                                             train_monitors=train_hooks,
                                             eval_hooks=eval_hooks)
    return experiment


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}


def main(mode):
    (train_X, train_y), (test_X, test_y) = data_loader.make_train_and_test_set()
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)
    model_params = Config.model.to_dict()
    model = Model()
    run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir,
                                        save_checkpoints_steps=Config.train.save_checkpoints_steps)
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=model_params,
        config=run_config,
    )

    tf.estimator.train_and_evaluate(
        estimator,
        tf.estimator.TrainSpec(input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(
            train_X, train_y, batch_size=Config.model.batch_size, num_epochs=1, shuffle=True), ),
        tf.estimator.EvalSpec(input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(
            test_X, test_y, batch_size=Config.model.batch_size, num_epochs=1, shuffle=False), ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config', help='config file name')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # Print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    # After terminated Notification to Slack
    atexit.register(utils.send_message_to_slack, config_name=args.config)

    main(args.mode)
