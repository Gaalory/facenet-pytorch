import pytest
import torch

from fixture import (
	aligned_faces,
	mtcnn_model,
	dataset_images,
	classify_resnet_model,
	embedding_resnet_model,
	pretrained_embedding_resnet_model,
	setup_and_teardown,
)

from facenet.models.inception_resnet_v1 import (
	BasicConv2d,
	Block8,
	Block17,
	Block35,
	InceptionResnetV1,
	Mixed_6a,
	Mixed_7a,
)


def test_errors():
	with pytest.raises(Exception):
		model = InceptionResnetV1(
			pretrained=None, classify=True, num_classes=None
		)
	with pytest.raises(ValueError):
		model = InceptionResnetV1(pretrained='gachimuchi')


def test_device():
	model = InceptionResnetV1(device='cpu')
	assert model.device == 'cpu'


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


def test_resnet_load_weights(
	setup_and_teardown, pretrained_embedding_resnet_model, aligned_faces
):
	for x, _y in aligned_faces:
		prob = pretrained_embedding_resnet_model(x)
		assert prob is not None
		assert prob.shape == torch.Size([5, 7451])

	ttest = torch.rand((7, 3, 160, 160))
	m = InceptionResnetV1(num_classes=5, classify=True, device='cpu')
	ret1 = m(ttest)
	m.change_num_classes(new_num_classes=3)
	ret2 = m(ttest)
	m.change_num_classes(10)
	ret3 = m(ttest)
	assert list(ret1.size()) == [7, 5]
	assert list(ret2.size()) == [7, 3]
	assert list(ret3.size()) == [7, 10]
