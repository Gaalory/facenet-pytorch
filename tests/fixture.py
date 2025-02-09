import os
import shutil
from os.path import join as pjoin

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from models.inception_resnet_v1 import InceptionResnetV1
from models.inception_resnet_v2 import InceptionResnetV2
from models.mtcnn import MTCNN, fixed_image_standardization

tmp_path: str = pjoin('tests', 'tmp')
data_path: str = 'data'


@pytest.fixture(scope='module', autouse=True)
def setup_and_teardown():
	if os.path.exists(tmp_path):
		shutil.rmtree(tmp_path)
	os.makedirs(tmp_path, exist_ok=True)

	yield
	shutil.rmtree(tmp_path)


@pytest.fixture
def mtcnn_model():
	return MTCNN(image_size=299, device=torch.device('cpu'))


@pytest.fixture(params=[1, 2])
def classify_resnet_model(request):
	if request.param == 1:
		return InceptionResnetV1(classify=True, num_classes=5).eval()
	else:
		return InceptionResnetV2(classify=True, num_classes=5).eval()


@pytest.fixture(params=[1, 2])
def embedding_resnet_model(request):
	if request.param == 1:
		return InceptionResnetV1(classify=False).eval()
	else:
		return InceptionResnetV2(classify=False).eval()


@pytest.fixture
def dataset_images():
	trans = transforms.Compose([transforms.Resize(512)])

	dataset = datasets.ImageFolder(
		pjoin(data_path, 'test_images'), transform=trans
	)
	dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}
	return dataset


@pytest.fixture
def aligned_faces(dataset_images, mtcnn_model):
	for img, idx in dataset_images:
		name = dataset_images.idx_to_class[idx]
		mtcnn_model(
			img,
			save_path=pjoin(tmp_path, f'test_images_aligned/{name}/1.png'),
		)

	trans = transforms.Compose(
		[
			np.float32,
			transforms.ToTensor(),
			fixed_image_standardization,
			transforms.Resize(299),
		]
	)
	ds = ImageFolder(pjoin(tmp_path, 'test_images_aligned'), transform=trans)
	return DataLoader(ds, batch_size=5, shuffle=True)
