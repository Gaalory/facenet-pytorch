import torch
from fixture import (
	aligned_faces,
	classify_resnet_model,
	dataset_images,
	embedding_resnet_model,
	mtcnn_model,
	setup_and_teardown,
)

from models.inception_resnet_v1 import (
	BasicConv2d,
	Block8,
	Block17,
	Block35,
	Mixed_6a,
	Mixed_7a,
)


def test_basic_conv2d():
	ttest = torch.rand((7, 3, 160, 160))
	model = BasicConv2d(3, 32, kernel_size=3, stride=1)
	output = model(ttest)
	assert output.shape == torch.Size([7, 32, 158, 158])


def test_basic_block8():
	ttest = torch.rand((7, 1792, 8, 8))
	model = Block8(scale=0.1)
	output = model(ttest)
	assert output.shape == torch.Size([7, 1792, 8, 8])


def test_basic_block17():
	ttest = torch.rand((7, 896, 17, 17))
	model = Block17(scale=0.1)
	output = model(ttest)
	assert output.shape == torch.Size([7, 896, 17, 17])


def test_basic_block35():
	ttest = torch.rand((7, 256, 35, 35))
	model = Block35(scale=0.1)
	output = model(ttest)
	assert output.shape == torch.Size([7, 256, 35, 35])


def test_basic_mixed6a():
	ttest = torch.rand((7, 256, 35, 35))
	model = Mixed_6a()
	output = model(ttest)
	assert output.shape == torch.Size([7, 896, 17, 17])


def test_basic_mixed7a():
	ttest = torch.rand((7, 896, 17, 17))
	model = Mixed_7a()
	output = model(ttest)
	assert output.shape == torch.Size([7, 1792, 8, 8])


def test_resnet_embeddings(
	setup_and_teardown, embedding_resnet_model, aligned_faces
):
	for x, _y in aligned_faces:
		embs = embedding_resnet_model(x)
		assert embs is not None
		assert embs.shape == torch.Size([5, 512]) or embs.shape == torch.Size(
			[5, 1536]
		)


def test_resnet_classification(
	setup_and_teardown, classify_resnet_model, aligned_faces
):
	for x, _y in aligned_faces:
		prob = classify_resnet_model(x)
		assert prob is not None
		assert prob.shape == torch.Size([5, 5])
