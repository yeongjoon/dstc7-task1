from __future__ import division
import logging
import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from noesis.util.checkpoint import Checkpoint
from noesis.evaluator.evaluator import Evaluator


class SupervisedTrainer(object):
	r""" The SupervisedTrainer class helps in setting up a training framework in a
	supervised setting.

	Args:
		expt_dir (optional, str): experiment Directory to store details of the experiment,
			by default it makes a folder in the current directory to store the details (default: `experiment`).
		loss_func (torch.nn.CrossEntropyLoss, optional): loss for training, (default: torch.nn.CrossEntropyLoss)
		batch_size (int, optional): batch size for experiment, (default: 64)
		checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
	"""
	def __init__(self, expt_dir='experiment', loss_func=CrossEntropyLoss(), batch_size=1,
				 random_seed=None,
				 checkpoint_every=5000, print_every=1000):
		self._trainer = "Simple Trainer"
		self.random_seed = random_seed
		if random_seed is not None:
			random.seed(random_seed)
			torch.manual_seed(random_seed)
		self.loss_func = loss_func
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.batch_size = batch_size

		#self.evaluator = Evaluator(loss_func=self.loss_func, batch_size=self.batch_size)
		self.evaluator = Evaluator(loss_func=self.loss_func, batch_size=100)

		self.logger = logging.getLogger(__name__)

	def _train_batch(self, context_variable, responses_variable, target_variable, model, context_lengths_variable, responses_lengths_variable):
		loss_func = self.loss_func
		# Forward propagation
		outputs = model(context_variable, responses_variable, context_lengths_variable, responses_lengths_variable)
		# Get loss
		if len(outputs.size()) == 1:
			outputs = outputs.unsqueeze(0)
		#loss = loss_func(outputs, target_variable)
		#outputs = torch.max(outputs, dim=1)[1].unsqueeze(1)
		target = target_variable.squeeze()
		loss = loss_func(outputs, target)

		cnt = 0
		total_cnt = 0
		labels = target_variable.cpu().detach().numpy()
		predictions = np.argsort(-outputs.cpu().detach().numpy(), axis=1)[:, 0].reshape(-1, 1)
		for label, p in zip(labels, predictions):
			if label == p:
				cnt += 1
			total_cnt += 1

		# Backward propagation
		model.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss, cnt, total_cnt

	def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, batch_size=1,
					   dev_data=None):
		#batch_size=50
		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch


		steps_per_epoch = data.num_batches(batch_size)
		total_steps = steps_per_epoch * n_epochs

		#log_msg1 = "epoch: %d, total_step: %d" %(n_epochs, total_steps)
		#log.info(log_msg1)

		step = start_step
		step_elapsed = 0
		for epoch in range(start_epoch, n_epochs + 1):
			log.debug("Epoch: %d, Step: %d" % (epoch, step))

			total_cnt = 0
			cnt = 0
			model.train(True)
			for batch in data.make_batches(batch_size):
				step += 1
				step_elapsed += 1

				context_variable = batch[0]
				responses_variable = batch[1]
				target_variable = batch[2]
				context_lengths_variable = batch[3]
				responses_lengths_variable = batch[4]

				############yj 수정
				# context_variable = []
				#
				# for i in range(batch_size):
				# 	for j in range(np.size(tmp_responses_lengths_variable, 1)):
				# 		context_variable.append(tmp_context_variable[i].copy())
				#
				# context_variable = np.asarray(context_variable)
				#
				# responses_variable = np.squeeze(tmp_responses_variable.reshape(1, -1, np.size(tmp_responses_variable, axis=2)), axis=0)
				# target_variable = np.zeros((np.size(tmp_responses_lengths_variable, axis=1), batch_size))
				# for i in range(batch_size):
				# 	target_variable[tmp_target_variable[i]][i] = 1
				# target_variable = target_variable.reshape(-1, 1)
				#
				# context_lengths_variable = []
				#
				# for i in range(batch_size):
				# 	for j in range(np.size(tmp_responses_lengths_variable, 1)):
				# 		context_lengths_variable.append(tmp_context_lengths_variable[i].copy())
				#
				# responses_lengths_variable = np.squeeze(tmp_responses_lengths_variable.reshape(-1, 1), axis=1)

				if torch.cuda.is_available():
					context_variable = torch.tensor(context_variable).cuda()
					responses_variable = torch.tensor(responses_variable).cuda()
					#target_variable = torch.tensor(target_variable, dtype=torch.float).cuda()
					target_variable = torch.tensor(target_variable, dtype=torch.int64).cuda()
					context_lengths_variable = torch.tensor(context_lengths_variable).cuda()
					responses_lengths_variable = torch.tensor(responses_lengths_variable).cuda()
				else:
					context_variable = torch.tensor(context_variable)
					responses_variable = torch.tensor(responses_variable)
					target_variable = torch.tensor(target_variable, dtype=torch.float)
					context_lengths_variable = torch.tensor(context_lengths_variable)
					responses_lengths_variable = torch.tensor(responses_lengths_variable)

				loss, tmp_cnt, tmp_total_cnt = self._train_batch(context_variable, responses_variable, target_variable, model, context_lengths_variable, responses_lengths_variable)

				cnt += tmp_cnt
				total_cnt += tmp_total_cnt
				# Record average loss
				print_loss_total += loss
				epoch_loss_total += loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					log_msg = 'Progress: %d%%, Train %s: %.4f' % (
						step / total_steps * 100,
						'CrossEntropyLoss',
						print_loss_avg)
					log.info(log_msg)

			# Checkpoint
			if epoch % self.checkpoint_every == 0:
				Checkpoint(model=model,
						   optimizer=self.optimizer,
						   epoch=epoch, step=step,
						   vocab=data.vocab).save(self.expt_dir)

			if step_elapsed == 0:
				continue

			if total_cnt == 0:
				accuracy = float('nan')
			else:
				accuracy = cnt / float(total_cnt)
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f total loss: %.4f Train accuracy: %.4f" % (epoch, 'BinaryCrossEntropyLoss', epoch_loss_avg, epoch_loss_total, cnt/total_cnt)


			if dev_data is not None:

				dev_loss, accuracy, recall = self.evaluator.evaluate(model, dev_data)
				log_msg += ", Dev BinaryCrossEntropyLoss: %.4f, Accuracy: %.4f" % (dev_loss, accuracy)
				log_msg += ", \n Recall: {}".format(recall)
				model.train(mode=True)
				#self.optimizer.update(epoch_loss_avg, epoch)
			else:
				self.optimizer.update(epoch_loss_avg, epoch)

			log.info(log_msg)

	def train(self, model, data, batch_size=1, num_epochs=5,
			  resume=False, dev_data=None,
			  optimizer=None):
		r""" Run training for a given model.

		Args:
			model (models.networks): model to run training on, if `resume=True`, it would be
			   overwritten by the model loaded from the latest checkpoint.
			data (models.dataset.dataset.Dataset): dataset object to train on
			num_epochs (int, optional): number of epochs to run (default 5)
			resume(bool, optional): resume training with the latest checkpoint, (default False)
			dev_data (models.dataset.dataset.Dataset, optional): dev Dataset (default None)
			optimizer (pytorch.optim, optional): optimizer for training
			   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

		"""
		# If training is set to resume
		if resume:
			latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step
		else:
			start_epoch = 1
			step = 0
			if optimizer is None:
				optimizer = optim.Adam(model.parameters(), lr=0.0004)
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s" % self.optimizer)

		self._train_epoches(data, model, num_epochs,
							start_epoch, step, batch_size=batch_size, dev_data=dev_data)
