from __future__ import print_function, division

import numpy as np
import torch
from torch.nn import CrossEntropyLoss


class Evaluator(object):
	""" Class to evaluate models with given datasets.

	Args:
		loss (torch.NN.CrossEntropyLoss, optional): loss for evaluator (default: torch.NN.CrossEntropyLoss)
		batch_size (int, optional): batch size for evaluator (default: 64)
	"""

	def __init__(self, loss_func=CrossEntropyLoss(), batch_size=1):
		self.loss_func = loss_func
		self.batch_size = batch_size

	def evaluate(self, model, data):
		""" Evaluate a model on given dataset and return performance.

		Args:
			model (models.networks): model to evaluate
			data (dataset.dataset.Dataset): dataset to evaluate against

		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()

		match = 0
		total = 0
		recall = {'@1': 0, '@2': 0, '@5': 0, '@10': 0, '@50': 0, '@100': 0}
		loss = 0
		step = 0

		# device = None if torch.cuda.is_available() else -1

		with torch.no_grad():
			total_loss = 0
			for batch in data.make_batches(self.batch_size):
				step += 1
				if torch.cuda.is_available():
					context_variable = torch.tensor(batch[0]).cuda()
					responses_variable = torch.tensor(batch[1]).cuda()
					target_variable = torch.tensor(batch[2]).cuda()
					context_lengths_variable = torch.tensor(batch[3]).cuda()
					responses_lengths_variable = torch.tensor(batch[4]).cuda()
				else:
					context_variable = torch.tensor(batch[0])
					responses_variable = torch.tensor(batch[1])
					target_variable = torch.tensor(batch[2])
					context_lengths_variable = torch.tensor(batch[3])
					responses_lengths_variable = torch.tensor(batch[4])

				outputs = model(context_variable, responses_variable, context_lengths_variable, responses_lengths_variable)

				# Get loss
				if len(outputs.size()) == 1:
					outputs = outputs.unsqueeze(0)
				loss += self.loss_func(outputs, target_variable).item()
				total_loss += self.loss_func(outputs, target_variable)

				# Evaluation
				#################
				# predictions = torch.argsort(outputs, dim=1)
				# num_samples = predictions.shape[0]
				#
				# ranks = predictions[np.arange(num_samples), target_variable]
				# match += sum(ranks == 0)
				# recall['@1'] = match
				# recall['@2'] += sum(ranks <= 2)
				# recall['@5'] += sum(ranks <= 5)
				# recall['@10'] += sum(ranks <= 10)
				# recall['@50'] += sum(ranks <= 50)
				# recall['@100'] += sum(ranks <= 100)
				# total += num_samples
				##################
				labels = target_variable
				# reverse sort (sort in decreasing order)
				predictions = np.argsort(-outputs.cpu().detach().numpy(), axis=1)
				num_samples = predictions.shape[0]  # batch size

				labels = labels.cpu().numpy()
				recall['@1'] += sum(labels == predictions[:, 0])
				recall['@2'] += sum(np.array(
					[label in prediction for label, prediction in zip(labels, predictions[:, :2])]))
				recall['@5'] += sum(np.array(
					[label in prediction for label, prediction in zip(labels, predictions[:, :5])]))
				recall['@10'] += sum(np.array(
					[label in prediction for label, prediction in zip(labels, predictions[:, :10])]))
				recall['@50'] += sum(np.array(
					[label in prediction for label, prediction in zip(labels, predictions[:, :50])]))
				recall['@100'] += sum(np.array(
					[label in prediction for label, prediction in zip(labels, predictions[:, :100])]))
				total += num_samples

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = recall['@1'] / total
		avg_loss = loss/step

		return total_loss, accuracy, {k: v/total for k, v in recall.items()}
