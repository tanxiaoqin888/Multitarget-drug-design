#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:06:54 2018

@author: txqq
"""
import collections
import os
import pickle
import threading
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader

from deepchem.data import NumpyDataset
from deepchem.models.models import Model
from deepchem.models.tensorgraph.layers import InputFifoQueue, Label, Feature, Weights, Constant
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator

import deepchem as dc
from deepchem.models import MultitaskRegressor


class multitask(MultitaskRegressor):
      def save(self, model_dir, model_name):
            with self._get_tf("Graph").as_default():
                  saver = tf.train.Saver()
                  saver.save(self.session, os.path.join(model_dir, model_name))
      
      def load_from_dir(self, model_dir, model_name):
            if not self.built:
                  self.build()
            with self._get_tf("Graph").as_default():
                  reader = NewCheckpointReader(os.path.join(model_dir, model_name))
                  var_names = set([x for x in reader.get_variable_to_shape_map()])
                  var_map = {
                        x.op.name: x
                        for x in tf.global_variables() if x.op.name in var_names
                  }
                  saver = tf.train.Saver(var_list=var_map)
                  saver.restore(self.session, os.path.join(model_dir, model_name))
      
      def fit(self,
              dataset,
              nb_epoch=10,
              max_checkpoints_to_keep=5,
              checkpoint_interval=1000,
              deterministic=False,
              restore=False,
              submodel=None,
              model_dir=None,
              model_name=None,
              **kwargs):
            return self.fit_generator(
                  self.default_generator(
                        dataset, epochs=nb_epoch, deterministic=deterministic),
                  max_checkpoints_to_keep, checkpoint_interval, restore, submodel, model_dir, model_name)

      def fit_generator(self,
                        feed_dict_generator,
                        max_checkpoints_to_keep=5,
                        checkpoint_interval=1000,
                        restore=False,
                        submodel=None,
                        model_dir=None,
                        model_name=None):
            if not self.built:
                  self.build()
            with self._get_tf("Graph").as_default():
                  time1 = time.time()
                  loss = self.loss
                  if submodel is None:
                        train_op = self._get_tf('train_op')
                  else:
                        train_op = submodel.get_train_op()
                        if submodel.loss is not None:
                              loss = submodel.loss
                  if checkpoint_interval > 0:
                        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
                  if restore:
                        self.load_from_dir(model_dir, model_name)
                  avg_loss, n_averaged_batches = 0.0, 0.0
                  n_samples = 0
                  n_enqueued = [0]
                  final_sample = [None]
                  if self.queue_installed:
                        enqueue_thread = threading.Thread(
                              target=dc.models.tensorgraph.tensor_graph._enqueue_batch,
                              args=(self, feed_dict_generator, self._get_tf("Graph"),
                                    self.session, n_enqueued, final_sample))
                        enqueue_thread.start()
                  for feed_dict in self._create_feed_dicts(feed_dict_generator, True):
                        if self.queue_installed:
                              # Don't let this thread get ahead of the enqueue thread, since if
                              # we try to read more batches than the total number that get queued,
                              # this thread will hang indefinitely.
                              while n_enqueued[0] <= n_samples:
                                    if n_samples == final_sample[0]:
                                          break
                                    time.sleep(0)
                              if n_samples == final_sample[0]:
                                    break
                        n_samples += 1
                        should_log = (self.tensorboard and
                      n_samples % self.tensorboard_log_frequency == 0)
                        fetches = [train_op, loss.out_tensor]
                        if should_log:
                              fetches.append(self._get_tf("summary_op"))
                        fetched_values = self.session.run(fetches, feed_dict=feed_dict)
                        if should_log:
                              self._log_tensorboard(fetched_values[2])
                        avg_loss += fetched_values[1]
                        n_averaged_batches += 1
                        self.global_step += 1
                        if checkpoint_interval > 0 and self.global_step % checkpoint_interval == checkpoint_interval - 1:
                              # saver.save(self.session, self.save_file, global_step=self.global_step)
                              avg_loss = float(avg_loss) / n_averaged_batches
                              print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
                              avg_loss, n_averaged_batches = 0.0, 0.0
                  if n_averaged_batches > 0:
                        avg_loss = float(avg_loss) / n_averaged_batches
                  if checkpoint_interval > 0:
                        if n_averaged_batches > 0:
                              print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
                        # saver.save(self.session, self.save_file, global_step=self.global_step)
                        time2 = time.time()
                        print("TIMING: model fitting took %0.3f s" % (time2 - time1))
            return avg_loss
        
      
