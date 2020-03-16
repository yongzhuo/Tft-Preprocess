# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/3/15 22:04
# @author  : Mo
# @function: tensorflow-transformer(TFT) of preprocess of estimator


import sys
import os
path_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path_root)


from tft_preprocess.tf_metrics import precision, recall, f1
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
import tensorflow_transform.beam as tft_beam
import tensorflow_transform as tft
import tensorflow as tf
import numpy as np


class TextCNNGraph:
    def __init__(self, params, feature):
        self.sentence_size = params["sentence_size"]
        self.class_num = params["class_num"]
        self.vocab_size = params["vocab_size"]
        self.embed_size = params["embed_size"]
        self.filters = params["filters"]
        self.filter_num = params["filter_num"]
        self.channel_size = params["channel_size"]
        self.is_training = params["is_train"]
        self.learning_rate = params["learning_rate"]
        self.decay_step = params["decay_step"]
        self.decay_rate = params["decay_rate"]
        self.l2_lambda = params["l2_lambda"]
        self.keep_prob = params["keep_prob"]
        self.feature = feature
        self.l2i = None
        self.v2i = None
        self.i2l = tf.constant(params["i2l"])

        class_tensor = tf.constant(params["i2l"])
        indices = tf.constant(list(range(0, len(params["i2l"]))))
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        self.classes = table.lookup(tf.to_int64(indices))

    def initial_params(self):
        self.global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)
        self.epoch_step = tf.Variable(name="epoch_step", initial_value=0, trainable=False)

        # input change to embed
        self.x = self.feature["x"]  # name="x"
        self.y = self.feature["y"]  # name="y"

        self.w = tf.get_variable(name="w", shape=(self.filter_num * len(self.filters), self.class_num),
                                 dtype=tf.float32)
        self.b = tf.get_variable(name="b", shape=(self.class_num,), dtype=tf.float32)
        self.embed = tf.get_variable(name="embed", shape=(self.vocab_size, self.embed_size), dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        self.sentence_embed = tf.nn.embedding_lookup(self.embed, self.x)
        self.sentence_expand = tf.expand_dims(self.sentence_embed, -1)
        # cnn
        pools = []
        for index, filter_size in enumerate(self.filters):
            with tf.name_scope(name=f"cnn_pooling_{filter_size}"):
                _filter = tf.get_variable(name=f"filter_{filter_size}",
                                          shape=(filter_size, self.embed_size, self.channel_size, self.filter_num))
                conv = tf.nn.conv2d(name="conv", input=self.sentence_expand, filter=_filter,
                                    strides=[1, 1, 1, 1], padding="VALID")
                b = tf.get_variable(name=f"b_{filter_size}", shape=(self.filter_num,))
                h = tf.nn.relu(name="relu", features=tf.nn.bias_add(conv, b))
                pool = tf.nn.max_pool(name="pool", value=h, ksize=[1, self.sentence_size - filter_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1], padding="VALID")
                pools.extend([pool])
        self.pool = tf.reshape(tf.concat(pools, axis=3), shape=(-1, self.filter_num * len(self.filters)))
        pool_drop = tf.nn.dropout(self.pool, keep_prob=self.keep_prob)
        self.logits = tf.matmul(pool_drop, self.w) + self.b
        self.probs = tf.nn.softmax(self.logits)

        # loss
        labels_one_hot = tf.one_hot(self.y, self.class_num)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                       logits=self.logits)
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * self.l2_lambda
        self.loss = tf.reduce_mean(loss) + l2_losses

        # predict
        self.predict = tf.argmax(name="predictions", input=self.probs, axis=1)

        # accuracy
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32),tf.cast(self.y, tf.int32))
        self.accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))

    def build_optimize(self):
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step,
                                                        self.decay_rate, staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=self.learning_rate, optimizer="Adam")

    def build_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.summary = tf.summary.merge_all()

def get_dir_files(path_dir):
    """
        递归获取某个目录下的所有文件(单层)
    :param path_dir: str, like '/home/data'
    :return: list, like ['2019_12_5.txt']
    """

    def get_dir_files_func(file_list, dir_list, root_path=path_dir):
        """
            递归获取某个目录下的所有文件
        :param root_path: str, like '/home/data'
        :param file_list: list, like []
        :param dir_list: list, like []
        :return: None
        """
        # 获取该目录下所有的文件名称和目录名称
        dir_or_files = os.listdir(root_path)
        for dir_file in dir_or_files:
            # 获取目录或者文件的路径
            dir_file_path = os.path.join(root_path, dir_file)
            # 判断该路径为文件还是路径
            if os.path.isdir(dir_file_path):
                dir_list.append(dir_file_path)
                # 递归获取所有文件和目录的路径
                get_dir_files_func(dir_file_path, file_list, dir_list)
            else:
                file_list.append(dir_file_path)

    # 用来存放所有的文件路径
    _files = []
    # 用来存放所有的目录路径
    dir_list = []
    get_dir_files_func(_files, dir_list, path_dir)
    return _files


# 超参数
corpus_path =  "data/baidu_qa_2019_100/baike_qa_train.csv"
model_path = "text_cnn_baike_qa"
path_transform_dir = model_path + "/working_dir/"
path_transform = path_transform_dir + "transform_tmp"
vocab_name = "vocab.pkl"
label_name = "label.pkl"
class_number = 17
BATCH_SIZE = 64
LEN_MAX = 50
EPCOH = 10

# 模型目录地址
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(path_transform_dir):
    os.mkdir(path_transform_dir)
if not os.path.exists(path_transform):
    os.mkdir(path_transform)

# 读取数据
file = open(corpus_path, "r", encoding="utf-8")
train_ques = file.readlines()[1:]

xys = []
for i in range(len(train_ques)):
    train_ques_sp = train_ques[i].strip().replace(" ", "").replace("\n", "").replace("\r", "").upper().split(",")
    train_ques_sp_list = list(train_ques_sp[1])
    len_real = len(train_ques_sp_list)
    train_ques_sp_list_pad = train_ques_sp_list[:LEN_MAX] if len_real>LEN_MAX else train_ques_sp_list + ["*"]*(LEN_MAX-len_real)
    xy_json = {"x":train_ques_sp_list_pad, "y":train_ques_sp[0]}
    xys.append(xy_json)

# graph架构输入
STRING_FEATURE = {'x': tf.io.FixedLenFeature([LEN_MAX], tf.string),
                  'y': tf.io.FixedLenFeature([], tf.string)}

DATA_STRING_FEATURE_SPEC = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(STRING_FEATURE))


def parser(x, y):
    features = {"x": x, "y":y}
    return features


def train_input_fn(train, batch_size=64):
    x_train, y_train = train
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(y_train))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(test, batch_size=64):
    x_test, y_test = test
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def transformed_data(working_dir):
    """数据处理与生成transform_fn"""
    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        xi, yi = inputs["x"], inputs["y"]
        x_integerized = tft.compute_and_apply_vocabulary(xi, default_value=0, name="vocab")  # , top_k=VOCAB_SIZE)
        y_integerized = tft.compute_and_apply_vocabulary(yi, default_value=0, name="label")  # ,top_k=LABEL_SIZE
        return {"x": x_integerized, "y": y_integerized}

    # path_transform
    with tft_beam.Context(temp_dir=path_transform):
        transformed_dataset, transform_fn = ((xys, DATA_STRING_FEATURE_SPEC) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
        transformed_train_data, transformed_metadata = transformed_dataset
        _ = (transform_fn | tft_beam.WriteTransformFn(working_dir))
    return transformed_train_data


def _make_training_input_fn(transformed_dataset):
    """训练数据处理"""
    def input_fn():
        x, y = [], []
        s = set()
        for tdt in transformed_dataset:
            x.append(tdt["x"].tolist())
            s.add(len(tdt["x"].tolist()))
            y.append(tdt["y"])
        print(list(s))
        x, y = np.array(x, dtype=np.int32), np.array(y).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.shuffle(buffer_size=len(y))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


def _make_serving_input_fn(tf_transform_output):
    """预测输入"""
    raw_feature_spec = {'x': tf.io.FixedLenFeature([LEN_MAX], tf.string)}
    def serving_input_fn():
        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()
        transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
        return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.features)
    return serving_input_fn


def model_fn(features, mode, params):
    """网络架构与后处理"""
    graph = TextCNNGraph(params, features)
    graph.initial_params()
    graph.build_epoch_increment()
    graph.build_forward()
    predict_id = graph.predict
    y = graph.y
    loss = graph.loss
    num_classes = graph.class_num
    print(graph.i2l.shape)
    print(graph.probs.shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        batch_labels = tf.tile(tf.reshape(graph.classes, (1, -1)), [tf.shape(graph.x)[0], 1])
        predictions = {
                       "labels": batch_labels,
                       "probs": graph.probs
        }
        classification_output = export_output.ClassificationOutput(scores=graph.probs, classes=batch_labels)

        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs={
                                          "classification": classification_output
                                          }
                                          )
    else:
        # metrics
        metrics = {
            "precision": precision(y, predict_id, num_classes),
            "recall": recall(y, predict_id, num_classes),
            "f1": f1(y, predict_id, num_classes)
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])
        graph.build_summary()

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            graph.build_optimize()
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_model(params, working_dir):
    """训练模型"""
    transformed_dataset = transformed_data(working_dir)
    STEP_EPCOH = int(len(transformed_dataset)/BATCH_SIZE)
    # label字典
    file_vocab_label_vocabulary = working_dir + "transform_fn/assets/vocab_label_vocabulary"
    fo = open(file_vocab_label_vocabulary, "r", encoding="utf-8")
    labels = fo.readlines()
    labels = [l.strip() for l in labels]
    params["i2l"] = labels
    # vocab词典
    file_vocab_vocab_vocabulary = working_dir + "transform_fn/assets/vocab_vocab_vocabulary"
    fov = open(file_vocab_vocab_vocabulary, "r", encoding="utf-8")
    vocabs = fov.readlines()
    params["vocab_size"] = len(vocabs)

    tf_transform_output = tft.TFTransformOutput(working_dir)
    train_input_fn = _make_training_input_fn(transformed_dataset)
    model_path_useless = os.path.join(model_path, "model_ckpt")
    if not os.path.exists(model_path_useless):
        os.mkdir(model_path_useless)
    cfg = tf.estimator.RunConfig(save_summary_steps=1,
                                 log_step_count_steps=1,
                                 save_checkpoints_steps=STEP_EPCOH*(EPCOH+1),
                                 )

    estimator = tf.estimator.Estimator(model_fn, model_path_useless, cfg, params)
    hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, metric_name="f1",
                                                              max_steps_without_increase=STEP_EPCOH * 3,
                                                              min_steps=STEP_EPCOH, run_every_secs=120,
                                                              eval_dir=os.path.join(model_path, "eval"))
    # # train
    # estimator.train(input_fn=train_input_fn, max_steps=STEP_EPCOH * EPCOH, hooks=[hook])
    # serving_input_fn = _make_serving_input_fn(tf_transform_output)
    # estimator.export_savedmodel(export_dir_base=model_path,
    #                             serving_input_receiver_fn=serving_input_fn,
    #                             )

    # # train and eval
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=STEP_EPCOH * EPCOH, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=STEP_EPCOH,
                                      start_delay_secs=int(STEP_EPCOH / 60) + 1, throttle_secs=32)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    serving_input_fn = _make_serving_input_fn(tf_transform_output)
    dir_eval = os.path.join(model_path_useless, "eval")
    file = os.listdir(dir_eval)[0]
    estimator.export_savedmodel(export_dir_base=model_path,
                                serving_input_receiver_fn=serving_input_fn,
                                assets_extra={file: os.path.join(model_path_useless, "eval", file)}
                                )


class LoadModel:
    def __init__(self, model_dir_path, memory_fraction):
        self.load_model(model_dir_path, memory_fraction=memory_fraction)

    def find_dir(self, dir):
        """获取模型目录"""
        def is_total_num(text):
            """
              判断是否是数字的
            :param text: str
            :return: boolean, True or false
            """
            try:
                text_clear = text.replace(" ", "").strip()
                number = 0
                for one in text_clear:
                    if one.isdigit():
                        number += 1
                if number == len(text_clear):
                    return True
                else:
                    return False
            except:
                return False
        file_dir = ""
        files = os.listdir(dir)
        for file in files:
            if is_total_num(file):
                file_dir = file
                break
        return file_dir

    def load_model(self, model_dir_path, memory_fraction):
        """加载训练好的模型"""
        dir_number = self.find_dir(model_dir_path)
        model_dir_path = os.path.join(model_dir_path, dir_number)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)  # gpu_memory_fraction
        config = tf.ConfigProto(gpu_options=gpu_options)
        g = tf.Graph()
        sess = tf.Session(graph=g, config=config)
        with sess.as_default():
            with g.as_default():
                self.predict_fn = tf.contrib.predictor.from_saved_model(model_dir_path)

    def predict(self, origin_questions):
        """预测"""
        prediction = self.predict_fn({"inputs":origin_questions})
        return prediction


if __name__ == "__main__":
    params = {'class_num': class_number,
              'batch_size': BATCH_SIZE,
              'use_embedding': False,
              'is_train': True,
              'token_level': 'char',
              'embed_size': 128,
              'vocab_size': 645,
              'learning_rate': 1e-3,
              'sentence_size': LEN_MAX,
              'attention_units': 256,
              'rnn_units': 512,
              'filters': [3, 4, 5],
              'filter_num': 300,
              'channel_size': 1,
              'keep_prob': 0.5,
              'l2_lambda': 1e-9,
              #
              'patience': 3,
              'decay_step': 1000,
              'decay_rate': 0.99,
              'epoch_num': 32,
              'epoch_val': 1,
              'gpu_device': "0,1",
              'gpu_allow_growth': False,
              'gpu_memory_fraction': 0.20,
              'i2l':[]
              }
    train_dir = "train_dir"

    # 训练
    train_model(params, path_transform_dir)

    # 预测
    ld = LoadModel(model_path, 0.6)
    while True:
        print("请输入:")
        ques = input()
        qu = ques.replace(" ", "").replace("\n", "").replace("\r", "").upper()
        len_ques = len(qu)
        qu_list = list(qu)
        qu_list_max = qu_list[:LEN_MAX] if len_ques > LEN_MAX else qu_list + ["*"] * (LEN_MAX - len_ques)
        res = ld.predict([qu_list_max])
        print(res)

# tensorboard --logdir=1582857213/    --port=8888


