import time
from collections.abc import Callable
from typing import Any

import torch


class Logger:
	def __init__(
		self, mode: str, length: int, calculate_mean: bool = False
	) -> None:
		self.mode = mode
		self.length = length
		self.calculate_mean = calculate_mean
		if self.calculate_mean:
			self.fn: Callable[[torch.Tensor, int], float] = lambda x, i: (
				x / (i + 1)
			).item()
		else:
			self.fn = lambda x, i: x.item()

	def __call__(
		self, loss: torch.Tensor, metrics: dict[str, Any], i: int
	) -> None:
		track_str = f'\r{self.mode} | {i + 1:5d}/{self.length:<5d}| '
		loss_str = f'loss: {self.fn(loss, i):9.4f} | '
		metric_str = ' | '.join(
			f'{k}: {self.fn(v, i):9.4f}' for k, v in metrics.items()
		)
		print(track_str + loss_str + metric_str + '   ', end='')
		if i + 1 == self.length:
			print('')


class BatchTimer:
	"""Batch timing class.
	Use this class for tracking training and testing time/rate per
	    batch or per sample.

	Keyword Arguments:
	    rate {bool} -- Whether to report a rate (batches or
	        samples per second) or a time (seconds per batch or
	        sample). (default: {True})
	    per_sample {bool} -- Whether to report times or rates per
	        sample or per batch. (default: {True})
	"""

	def __init__(self, rate: bool = True, per_sample: bool = True) -> None:
		self.start = time.time()
		self.end: float | None = None
		self.rate = rate
		self.per_sample = per_sample

	def __call__(self, y_pred: list[int], y: list[int]) -> torch.Tensor:
		self.end = time.time()
		elapsed: float = self.end - self.start
		self.start = self.end
		self.end = None

		if self.per_sample:
			elapsed /= float(len(y_pred))
		if self.rate:
			elapsed = 1 / elapsed

		return torch.tensor(elapsed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	_, preds = torch.max(logits, 1)
	return (preds == y).float().mean()


def pass_epoch(
	model: torch.nn.Module,
	loss_fn: Any,
	loader: torch.utils.data.DataLoader[torch.Tensor],
	optimizer: torch.optim.Optimizer | None = None,
	scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
	batch_metrics: dict[str, Any] | None = None,
	show_running: bool = True,
	device: torch.device | None | str = 'cpu',
	writer: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
	"""Train or evaluate over a data epoch.

	Arguments:
	    model {torch.nn.Module} -- Pytorch model.
	    loss_fn {callable} -- A function to compute (scalar) loss.
	    loader {torch.utils.data.DataLoader} -- A pytorch data loader.

	Keyword Arguments:
	    optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
	    scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler
	        (default: {None})
	    batch_metrics {dict} -- Dictionary of metric functions to call
	        on each batch. The default is a simple timer. A progressive
	        average of these metrics, along with the average loss,
	        is printed every batch. (default: {{'time': iter_timer()}})
	    show_running {bool} -- Whether or not to print losses and
	        metrics for the current batch or rolling averages.
	        (default: {False})
	    device {str or torch.device} -- Device for pytorch to use.
	        (default: {'cpu'})
	    writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard
	        SummaryWriter. (default: {None})

	Returns:
	    tuple(torch.Tensor, dict) -- A tuple of the average loss and a
	        dictionary of average metric values across the epoch.
	"""

	if batch_metrics is None:
		batch_metrics = {'time': BatchTimer()}
	mode = 'Train' if model.training else 'Valid'
	logger = Logger(mode, length=len(loader), calculate_mean=show_running)
	loss: torch.Tensor = torch.rand((1,))
	metrics: dict[str, Any] = {}
	for i_batch, (x, y) in enumerate(loader):
		x = x.to(device)
		y = y.to(device)
		y_pred = model(x)
		loss_batch = loss_fn(y_pred, y)

		if model.training and optimizer is not None:
			loss_batch.backward()
			optimizer.step()
			optimizer.zero_grad()

		metrics_batch = {}
		for metric_name, metric_fn in batch_metrics.items():
			metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
			metrics[metric_name] = (
				metrics.get(metric_name, 0) + metrics_batch[metric_name]
			)

		if writer is not None and model.training:
			if writer.iteration % writer.interval == 0:
				writer.add_scalars(
					'loss',
					{mode: loss_batch.detach().cpu()},
					writer.iteration,
				)
				for metric_name, metric_batch in metrics_batch.items():
					writer.add_scalars(
						metric_name, {mode: metric_batch}, writer.iteration
					)
			writer.iteration += 1

		loss_batch = loss_batch.detach().cpu()
		loss += loss_batch
		if show_running:
			logger(loss, metrics, i_batch)
		else:
			logger(loss_batch, metrics_batch, i_batch)

	if model.training and scheduler is not None:
		scheduler.step()

	loss = loss / (i_batch + 1)
	metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

	if writer is not None and not model.training:
		writer.add_scalars('loss', {mode: loss.detach()}, writer.iteration)
		for metric_name, metric in metrics.items():
			writer.add_scalars(metric_name, {mode: metric})

	return loss, metrics


def collate_pil(x: Any) -> Any:
	out_x, out_y = [], []
	for xx, yy in x:
		out_x.append(xx)
		out_y.append(yy)
	return out_x, out_y
