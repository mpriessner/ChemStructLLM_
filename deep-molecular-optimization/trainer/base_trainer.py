import os
import pandas as pd
from abc import ABC, abstractmethod

import torch
from tensorboardX import SummaryWriter

import utils.log as ul
import models.dataset as md
import preprocess.vocabulary as mv
from configuration import paths


class BaseTrainer(ABC):

    def __init__(self, opt):
        # Use the checkpoint directory from paths configuration
        self.save_path = os.path.join(paths.CHECKPOINT_DIR, opt.save_directory)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Create tensorboard directory
        tensorboard_dir = os.path.join(self.save_path, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(logdir=tensorboard_dir)
        
        # Set up logging
        log_path = os.path.join(self.save_path, 'train_model.log')
        LOG = ul.get_logger(name="train_model", log_path=log_path)
        self.LOG = LOG
        self.LOG.info(opt)
        self.LOG.info(f"Saving checkpoints to: {self.save_path}")

    def initialize_dataloader(self, data_path, batch_size, vocab, data_type, without_property=False):
        # Read train or validation
        data = pd.read_csv(os.path.join(data_path, data_type + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=mv.SMILESTokenizer(),
                             prediction_mode=False, without_property=without_property)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size,
                                                 shuffle=True, collate_fn=md.Dataset.collate_fn)
        return dataloader

    def to_tensorboard(self, train_loss, validation_loss, accuracy, token_accuracy, similarities, epoch):

        self.summary_writer.add_scalars("loss", {
            "train": train_loss,
            "validation": validation_loss
        }, epoch)
        self.summary_writer.add_scalars("validation", {
            "accuracy": accuracy,
            "token accuracy": token_accuracy
        }, epoch)
        self.summary_writer.add_scalar("similarity/validation", similarities, epoch)

        self.summary_writer.close()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_optimization(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def validation_stat(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def train(self):
        pass

