import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from facenet.models.inception_resnet_v2 import InceptionResnetV2
from facenet.models.utils import training
from tests.fixture import (
	mtcnn_model,
	dataset_images,

	aligned_faces,
	classify_resnet_model,
	setup_and_teardown,
)


def test_model_training(
	setup_and_teardown, classify_resnet_model, aligned_faces
):
	loss_fn = torch.nn.CrossEntropyLoss()
	device = 'cpu'
	metrics = {'fps': training.BatchTimer(), 'acc': training.accuracy}
	epochs = 50
	optimizer = optim.Adam(classify_resnet_model.parameters(), lr=0.001)
	scheduler = MultiStepLR(optimizer, [5, 10])
	print('\n\nInitial')
	print('-' * 10)
	classify_resnet_model.eval()
	init_loss, init_metric = training.pass_epoch(
		classify_resnet_model,
		loss_fn,
		aligned_faces,
		batch_metrics=metrics,
		show_running=True,
		device=device,
	)
	for epoch in range(epochs):
		print(f'\nEpoch {epoch + 1}/{epochs}')
		print('-' * 10)

		classify_resnet_model.train()
		train_loss, train_metrics = training.pass_epoch(
			classify_resnet_model,
			loss_fn,
			aligned_faces,
			optimizer,
			scheduler,
			batch_metrics=metrics,
			show_running=True,
			device=device,
		)

		classify_resnet_model.eval()
		test_loss, test_metrics = training.pass_epoch(
			classify_resnet_model,
			loss_fn,
			aligned_faces,
			batch_metrics=metrics,
			show_running=True,
			device=device,
		)
	assert init_metric['acc'] < train_metrics['acc']
	assert init_metric['acc'] < test_metrics['acc']
	pass


def test_model_training_v2(setup_and_teardown, aligned_faces):  # noqa : F810
	trainable_resnet_model = InceptionResnetV2(5, dropout_prob=0)
	loss_fn = torch.nn.CrossEntropyLoss()
	device = 'cpu'
	metrics = {'fps': training.BatchTimer(), 'acc': training.accuracy}
	epochs = 50
	optimizer = optim.Adam(trainable_resnet_model.parameters(), lr=0.001)
	scheduler = MultiStepLR(optimizer, [5, 10])
	print('\n\nInitial')
	print('-' * 10)
	trainable_resnet_model.eval()
	init_loss, init_metric = training.pass_epoch(
		trainable_resnet_model,
		loss_fn,
		aligned_faces,
		batch_metrics=metrics,
		show_running=True,
		device=device,
	)
	for epoch in range(epochs):
		print(f'\nEpoch {epoch + 1}/{epochs}')
		print('-' * 10)

		trainable_resnet_model.train()
		train_loss, train_metrics = training.pass_epoch(
			trainable_resnet_model,
			loss_fn,
			aligned_faces,
			optimizer,
			scheduler,
			batch_metrics=metrics,
			show_running=True,
			device=device,
		)

		trainable_resnet_model.eval()
		test_loss, test_metrics = training.pass_epoch(
			trainable_resnet_model,
			loss_fn,
			aligned_faces,
			batch_metrics=metrics,
			show_running=True,
			device=device,
		)
	assert init_metric['acc'] < train_metrics['acc']
	assert init_metric['acc'] < test_metrics['acc']
	pass
