from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import json
import math

from six.moves import urllib
import tensorflow as tf

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)sline:%(lineno)d][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("run_mode", "local", "the run mode, one of local and distributed")
flags.DEFINE_string("job_type", "train", "the job type, one of train, eval, train_and_eval, export_savedmodel and predict")

flags.DEFINE_string("train_data", "./train_data", "the train data, delimited by comma")
flags.DEFINE_string("eval_data", "./eval_data", "the eval data, delimited by comma")
flags.DEFINE_integer("batch_size", 1000, "the batch size")
flags.DEFINE_integer("feature_size", 100000, "the feature size")

flags.DEFINE_string("model_type", "wide", "the model type, one of wide, deep and wide_and_deep")
flags.DEFINE_string("model_dir", "./model_dir", "the model dir")
flags.DEFINE_bool("cold_start", False, "True: cold start; False: start from the latest checkpoint")
flags.DEFINE_string("export_savedmodel", "./export_model", "the export savedmodel directory, used for tf serving")
flags.DEFINE_string("savedmodel_mode", "raw", "the savedmodel mode, one of raw and parsing")
flags.DEFINE_integer("eval_ckpt_id", 0, "the checkpoint id in model dir for evaluation, 0 is the latest checkpoint")

# Configurations for a distributed run
flags.DEFINE_string("worker_hosts", "", "the worker hosts, delimited by comma")
flags.DEFINE_string("ps_hosts", "", "the ps hosts, delimited by comma")
flags.DEFINE_string("task_type", "worker", "the task type, one of worker and ps")
flags.DEFINE_integer("task_index", 0, "the task index, starting from 0")


class WideAndDeepModel(object):
    def __init__(self,
            model_type = "wide",
            model_dir = None,
            feature_size = 100000,
            run_config = None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        if run_config is None:
            self.run_config = self.build_run_config()
        else:
            self.run_config = run_config


    def build_run_config(self):
        run_config = tf.contrib.learn.RunConfig(
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 100,               # Save summaries every this many steps
                save_checkpoints_secs = 600,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 5,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 100)             # The frequency, in number of global steps
        return run_config


    def build_feature_dict(self):
        feature_dict = {}
        feature_dict["user_features"] = tf.placeholder(dtype = tf.string, shape = (None, self.feature_size))
        feature_dict["user_weights"] = tf.placeholder(dtype = tf.float32, shape = (None, self.feature_size))
        feature_dict["ads_features"] = tf.placeholder(dtype = tf.string, shape = (None, self.feature_size))
        feature_dict["ads_weights"] = tf.placeholder(dtype = tf.float32, shape = (None, self.feature_size))
        return feature_dict


    def build_feature_columns(self):
        user_features = tf.feature_column.categorical_column_with_hash_bucket(
                key = "user_features",
                hash_bucket_size = self.feature_size,
                dtype=tf.string)
        user_weighted_features = tf.feature_column.weighted_categorical_column(
                categorical_column = user_features,
                weight_feature_key = "user_weights",
                dtype = tf.float32)
        user_embedding_features = tf.feature_column.embedding_column(
                categorical_column = user_weighted_features,
                dimension = 64,
                combiner = "sqrtn")

        ads_features = tf.feature_column.categorical_column_with_hash_bucket(
                key = "ads_features",
                hash_bucket_size = self.feature_size,
                dtype=tf.string)
        ads_weighted_features = tf.feature_column.weighted_categorical_column(
                categorical_column = ads_features,
                weight_feature_key = "ads_weights",
                dtype = tf.float32)
        ads_embedding_features = tf.feature_column.embedding_column(
                categorical_column = ads_weighted_features,
                dimension = 64,
                combiner = "sqrtn")

        linear_feature_columns = [ads_weighted_features]
        dnn_feature_columns = [user_embedding_features, ads_embedding_features]
        return (linear_feature_columns, dnn_feature_columns)


    def build_linear_optimizer(self):
        linear_optimizer = tf.train.FtrlOptimizer(
                learning_rate = 0.05,
                learning_rate_power = -0.5,
                initial_accumulator_value = 0.1,
                l1_regularization_strength = 0.1,
                l2_regularization_strength = 0.1)
        return linear_optimizer


    def build_dnn_optimizer(self):
        dnn_optimizer = tf.train.AdagradOptimizer(
                learning_rate = 0.05,
                initial_accumulator_value = 0.1)
        return dnn_optimizer


    def build_estimator(self):
        linear_optimizer = self.build_linear_optimizer()
        dnn_optimizer = self.build_dnn_optimizer()
        dnn_hidden_units = [64, 16]
        (linear_feature_columns, dnn_feature_columns) = self.build_feature_columns()

        if self.model_type == "wide":
            model = tf.estimator.LinearClassifier(
                    feature_columns = linear_feature_columns,
                    model_dir = self.model_dir,
                    optimizer = linear_optimizer,
                    config = self.run_config)
        elif self.model_type == "deep":
            model = tf.estimator.DNNClassifier(
                    hidden_units = dnn_hidden_units,
                    feature_columns = dnn_feature_columns,
                    model_dir = self.model_dir,
                    optimizer = dnn_optimizer,
                    activation_fn = tf.nn.relu,
                    config = self.run_config)
        elif self.model_type == "wide_and_deep":
            model = tf.estimator.DNNLinearCombinedClassifier(
                    model_dir = self.model_dir,
                    linear_feature_columns = linear_feature_columns,
                    linear_optimizer = linear_optimizer,
                    dnn_feature_columns = dnn_feature_columns,
                    dnn_optimizer = dnn_optimizer,
                    dnn_hidden_units = dnn_hidden_units,
                    dnn_activation_fn = tf.nn.relu,
                    config = self.run_config)
        else:
            logging.error("unsupported model type: %s" % (self.model_type))
        return model


####################################################################################################


class WideAndDeepInputPipeline(object):
    def __init__(self, input_files, batch_size = 1000):
        self.batch_size = batch_size
        self.input_files = input_files

        input_file_list = []
        for input_file in self.input_files:
            if len(input_file) > 0:
                input_file_list.append(tf.train.match_filenames_once(input_file))
        self.filename_queue = tf.train.string_input_producer(
                tf.concat(input_file_list, axis = 0),
                num_epochs = 1,     # strings are repeated num_epochs
                shuffle = True,     # strings are randomly shuffled within each epoch
                capacity = 512)
        self.reader = tf.TextLineReader(skip_header_lines = 0)

        (self.column_dict, self.column_defaults) = self.build_column_format()


    def build_column_format(self):
        column_dict = {"label": 0, "user_features": 1, "user_weights": 2, "ads_features": 3, "ads_weights": 4}
        column_defaults = [['']] * len(column_dict)
        column_defaults[column_dict["label"]] = [0.0]
        column_defaults[column_dict["user_features"]] = ['0']
        column_defaults[column_dict["user_weights"]] = ['0.0']
        column_defaults[column_dict["ads_features"]] = ['0']
        column_defaults[column_dict["ads_weights"]] = ['0.0']
        return (column_dict, column_defaults)


    def string_to_number(self, string_tensor, dtype = tf.int32):
        number_values = tf.string_to_number(
                string_tensor = string_tensor.values,
                out_type = dtype)
        number_tensor = tf.SparseTensor(
                indices = string_tensor.indices,
                values = number_values,
                dense_shape = string_tensor.dense_shape)
        return number_tensor


    def get_next_batch(self):
        (_, records) = self.reader.read_up_to(self.filename_queue, num_records = self.batch_size)
        samples = tf.decode_csv(records, record_defaults = self.column_defaults, field_delim = ',')
        label = tf.cast(samples[self.column_dict["label"]], dtype = tf.int32)
        feature_dict = {}
        for (key, value) in self.column_dict.items():
            if key == "label" or value < 0 or value >= len(samples):
                continue
            if key in ["user_features", "ads_features"]:
                feature_dict[key] = tf.string_split(samples[value], delimiter = ';')
            if key in ["user_weights", "ads_weights"]:
                feature_dict[key] = self.string_to_number(
                        tf.string_split(samples[value], delimiter = ';'),
                        dtype = tf.float32)
        return feature_dict, label


####################################################################################################


def train_input_fn():
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
            train_input_files,
            batch_size = FLAGS.batch_size)
    return train_input_pipeline.get_next_batch()


def train_model():
    if FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    estimator.train(
            input_fn = lambda: train_input_fn(),
            steps = 100000)


def eval_input_fn():
    eval_input_files = FLAGS.eval_data.strip().split(',')
    eval_input_pipeline = WideAndDeepInputPipeline(
            eval_input_files,
            batch_size = FLAGS.batch_size)
    return eval_input_pipeline.get_next_batch()


def eval_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    # Get the checkpoint path for evaluation
    checkpoint_path = None # The latest checkpoint in model dir
    if FLAGS.eval_ckpt_id > 0:
        state = tf.train.get_checkpoint_state(
                checkpoint_dir = FLAGS.model_dir,
                latest_filename = "checkpoint")
        if state and state.all_model_checkpoint_paths:
            if FLAGS.eval_ckpt_id < len(state.all_model_checkpoint_paths):
                pos = -(1 + FLAGS.eval_ckpt_id)
                checkpoint_path = state.all_model_checkpoint_paths[pos]
            else:
                logging.warn("not find checkpoint id %d in %s" % (FLAGS.eval_ckpt_id, FLAGS.model_dir))
                checkpoint_path = None
    logging.info("checkpoint path: %s" % (checkpoint_path))
    eval_name = '' if checkpoint_path is None else str(FLAGS.eval_ckpt_id)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    eval_result = estimator.evaluate(
            input_fn = lambda: eval_input_fn(),
            steps = 100,
            checkpoint_path = checkpoint_path,
            name = eval_name)
    print(eval_result)


def export_savedmodel():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()

    if FLAGS.savedmodel_mode == "raw":
        features = model.build_feature_dict()
        export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                features = features,
                default_batch_size = None)
    elif FLAGS.savedmodel_mode == "parsing":
        (linear_feature_columns, dnn_feature_columns) = model.build_feature_columns()
        feature_columns = linear_feature_columns + dnn_feature_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
                feature_spec = feature_spec,
                default_batch_size = None)
    else:
        logging.error("unsupported savedmodel mode: %s" % (FLAGS.savedmodel_mode))
        sys.exit(1)

    export_dir = estimator.export_savedmodel(
            export_dir_base = FLAGS.export_savedmodel,
            serving_input_receiver_fn = lambda: export_input_fn(),
            assets_extra = None,
            as_text = False,
            checkpoint_path = None)


def predict_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    predict = estimator.predict(
            input_fn = lambda: eval_input_fn(),
            predict_keys = None,
            hooks = None,
            checkpoint_path = None)
    results = list(predict)
    sum_score = 0.0
    for i in range(0, len(results)):
        result = results[i]
        sum_score = sum_score + result["logistic"][0]
        print("count: %d, score: %f" % (i + 1, result["logistic"][0]))
    print("total count: %d, average score: %f" % (len(results), sum_score / len(results)))


####################################################################################################


def test_input_pipeline():
    logging.info("train data: %s" % (FLAGS.train_data))
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
            train_input_files,
            batch_size = 3) #FLAGS.batch_size)
    feature_batch, label_batch = train_input_pipeline.get_next_batch()
    #print(label_batch)
    #print(feature_batch)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        features, labels = sess.run([feature_batch, label_batch])
        print(labels)
        print(features["user_features"])
        print(features["user_features"].dense_shape)
        print(features["user_weights"].dense_shape)
        print(features["ads_features"].dense_shape)
        print(features["ads_weights"].dense_shape)
        coord.request_stop()
        coord.join(threads)


####################################################################################################


class WideAndDeepDistributedRunner(object):
    def __init__(self,
            model_type = "wide",
            model_dir = None,
            feature_size = 1000000,
            schedule = "train",
            worker_hosts = None,
            ps_hosts = None,
            task_type = None,
            task_index = None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        self.schedule = schedule
        self.worker_hosts = worker_hosts.strip().split(",")
        self.ps_hosts = ps_hosts.strip().split(",")
        self.task_type = task_type
        self.task_index = task_index

        self.run_config = self.build_run_config()
        self.hparams = self.build_hparams()


    def build_run_config(self):
        cluster = {"worker": self.worker_hosts, "ps": self.ps_hosts}
        task = {"type": self.task_type, "index": self.task_index}
        environment = {"environment": "cloud"}
        os.environ["TF_CONFIG"] = json.dumps({"cluster": cluster, "task": task, "environment": environment})

        run_config = tf.contrib.learn.RunConfig(
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 1000,              # Save summaries every this many steps
                save_checkpoints_secs = 600,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 5,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 1000,            # The frequency, in number of global steps
                model_dir = self.model_dir)             # Directory where model parameters, graph etc are saved
        return run_config


    def build_hparams(self):
        hparams = tf.contrib.training.HParams(
                eval_metrics = None,
                train_steps = None,
                eval_steps = 100,
                eval_delay_secs = 5,
                min_eval_frequency = None)
        return hparams


    def build_experiment(self, run_config, hparams):
        model = WideAndDeepModel(
                model_type = self.model_type,
                model_dir = self.model_dir,
                feature_size = self.feature_size,
                run_config = run_config)
        return tf.contrib.learn.Experiment(
                estimator = model.build_estimator(),
                train_input_fn = lambda: train_input_fn(),
                eval_input_fn = lambda: eval_input_fn(),
                eval_metrics = hparams.eval_metrics,
                train_steps = hparams.train_steps,
                eval_steps = hparams.eval_steps,
                eval_delay_secs = hparams.eval_delay_secs,
                min_eval_frequency = hparams.min_eval_frequency)


    def run(self):
        tf.contrib.learn.learn_runner.run(
                experiment_fn = self.build_experiment,
                output_dir = None, # Deprecated, must be None
                schedule = self.schedule,
                run_config = self.run_config,
                hparams = self.hparams)


####################################################################################################


def local_run():
    if FLAGS.job_type == "train":
        train_model()
    elif FLAGS.job_type == "eval":
        eval_model()
    elif FLAGS.job_type == "train_and_eval":
        train_model()
        eval_model()
    elif FLAGS.job_type == "export_savedmodel":
        export_savedmodel()
    elif FLAGS.job_type == "predict":
        predict_model()
    else:
        logging.error("unsupported job type: %s" % (FLAGS.job_type))
        sys.exit(1)


def distributed_run():
    if FLAGS.task_type == "worker" and FLAGS.task_index == 0 \
            and (FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval") \
            and FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)

    schedule = None
    schedule_dict = {"train": "train", "eval": "evaluate", "train_and_eval": "train_and_evaluate"}
    if FLAGS.task_type == "ps":
        schedule = "run_std_server"
    elif FLAGS.task_type == "worker":
        schedule = schedule_dict.get(FLAGS.job_type, None)
        if FLAGS.job_type == "train_and_eval" and FLAGS.task_index != 0:
            schedule = "train"  # only the first worker runs evaluation
    else:
        logging.error("unsupported task type: %s" % (FLAGS.task_type))
        sys.exit(1)
    logging.info("schedule: %s" % (schedule))

    runner = WideAndDeepDistributedRunner(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            schedule = schedule,
            worker_hosts = FLAGS.worker_hosts,
            ps_hosts = FLAGS.ps_hosts,
            task_type = FLAGS.task_type,
            task_index = FLAGS.task_index)
    runner.run()

def main():
    # print commandline arguments
    logging.info("run mode: %s" % (FLAGS.run_mode))
    if FLAGS.run_mode == "distributed":
        logging.info("worker hosts: %s" % (FLAGS.worker_hosts))
        logging.info("ps hosts: %s" % (FLAGS.ps_hosts))
        logging.info("task type: %s, task index: %d" % (FLAGS.task_type, FLAGS.task_index))
    logging.info("job type: %s" % (FLAGS.job_type))
    if FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval":
        logging.info("train data: %s" % (FLAGS.train_data))
        logging.info("cold start: %s" % (FLAGS.cold_start))
    if FLAGS.job_type in ["eval", "train_and_eval", "predict"]:
        logging.info("eval data: %s" % (FLAGS.eval_data))
        logging.info("eval ckpt id: %s" % (FLAGS.eval_ckpt_id))
    if FLAGS.job_type == "export_savedmodel":
        logging.info("export savedmodel: %s" % (FLAGS.export_savedmodel))
        logging.info("savedmodel mode: %s" % (FLAGS.savedmodel_mode))
    logging.info("model dir: %s" % (FLAGS.model_dir))
    logging.info("model type: %s" % (FLAGS.model_type))
    logging.info("feature size: %s" % (FLAGS.feature_size))
    logging.info("batch size: %s" % (FLAGS.batch_size))

    if FLAGS.run_mode == "local":
        local_run()
    elif FLAGS.run_mode == "distributed":
        if FLAGS.job_type == "export_savedmodel" or FLAGS.job_type == "predict":
            logging.error("job type export_savedmodel and predict does not support distributed run mode")
            sys.exit(1)
        if FLAGS.job_type in ["eval", "train_and_eval"] and FLAGS.eval_ckpt_id != 0:
            logging.error("eval_ckpt_id does not support distributed run mode")
            sys.exit(1)
        distributed_run()
    else:
        logging.error("unsupported run mode: %s" % (FLAGS.run_mode))
        sys.exit(1)
    prefix = "" if FLAGS.run_mode == "local" else "%s:%d " % (FLAGS.task_type, FLAGS.task_index)
    logging.info("%scompleted" % (prefix))

def test():
    test_input_pipeline()


if __name__ == "__main__":
    main()
    #test()

