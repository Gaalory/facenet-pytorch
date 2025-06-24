from os.path import join as pjoin

import numpy as np
import torch
from fixture import dataset_images, mtcnn_model, current_dir
from PIL import Image


def test_mtcnn_multi_face(mtcnn_model):
	cur_d = current_dir()
	img = Image.open(pjoin(cur_d,"data", 'multiface.jpg'))
	boxes, probs = mtcnn_model.detect(img)
	assert boxes.shape == (6, 4)


def test_mtcnn_no_face(mtcnn_model):
	img = Image.new('RGB', (512, 512))
	faces = mtcnn_model(img)
	assert faces is None


def test_mtcnn_multi_image(mtcnn_model):
	cur_d = current_dir()
	img = [
		Image.open(pjoin(cur_d,"data", 'multiface.jpg')),
		Image.open(pjoin(cur_d,"data", 'multiface.jpg')),
	]
	batch_boxes, batch_probs = mtcnn_model.detect(img)
	assert batch_boxes.shape == (2, 6, 4)


def test_mtcnn_batch_and_types(mtcnn_model, dataset_images):
	for img, _idx in dataset_images:
		img_box = mtcnn_model.detect(img)[0]
		assert (img_box - mtcnn_model.detect(np.array(img))[0]).sum() < 1e-2
		assert (
			img_box - mtcnn_model.detect(torch.as_tensor(np.array(img)))[0]
		).sum() < 1e-2
		assert (img_box - mtcnn_model.detect([img, img])[0]).sum() < 1e-2
		assert (
			img_box
			- mtcnn_model.detect(np.array([np.array(img), np.array(img)]))[0]
		).sum() < 1e-2
		assert (
			img_box
			- mtcnn_model.detect(
				torch.as_tensor(np.array([np.array(img), np.array(img)]))
			)[0]
		).sum() < 1e-2


def test_mtcnn_selection_methods(mtcnn_model):
	cur_d = current_dir()
	img = Image.open(pjoin(cur_d,'data', 'multiface.jpg'))
	mtcnn_model.selection_method = 'probability'
	assert mtcnn_model.detect(img) is not None

	mtcnn_model.selection_method = 'largest_over_theshold'
	assert mtcnn_model.detect(img) is not None

	mtcnn_model.selection_method = 'center_weighted_size'
	assert mtcnn_model.detect(img) is not None
