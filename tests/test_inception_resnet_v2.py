import torch

from facenet.models.inception_resnet_v2 import (
	InceptionResnetV2,
	block_A_v2,
	block_B_v2,
	block_C_v2,
	mixed_ab,
	mixed_bc,
	stem,
)


def test_stem():
	tensor_test = torch.rand((7, 3, 299, 299))
	st = stem()
	ret = st(tensor_test)
	assert list(ret.size()) == [7, 384, 35, 35]


def test_block_a():
	ttest = torch.rand((7, 384, 35, 35))
	mod = block_A_v2(input=384)
	ret = mod(ttest)
	# 384,35,35
	assert list(ret.size()) == [7, 384, 35, 35]
	pass


def test_block_b():
	ttest = torch.rand((7, 1152, 17, 17))
	mod = block_B_v2(input=1152)
	ret = mod(ttest)
	# 1157,17,17

	assert list(ret.size()) == [7, 1152, 17, 17]
	pass


def test_block_c():
	ttest = torch.rand((7, 2144, 8, 8))
	mod = block_C_v2(input=2144)
	ret = mod(ttest)

	assert list(ret.size()) == [7, 2144, 8, 8]
	pass


def test_reduc_ab():
	ttest = torch.rand((7, 384, 35, 35))
	mod = mixed_ab(input=384)
	ret = mod(ttest)
	# 1152,17,17
	assert list(ret.size()) == [7, 1152, 17, 17]
	pass


def test_reduc_bc():
	ttest = torch.rand((7, 1152, 17, 17))
	mod = mixed_bc(input=1152)
	ret = mod(ttest)
	# 2144,8,8
	assert list(ret.size()) == [7, 2144, 8, 8]
	pass


def test_model():
	ttest = torch.rand((7, 3, 299, 299))
	m = InceptionResnetV2(num_classes=5, device='cpu')
	ret = m(ttest)

	assert list(ret.size()) == [7, 1536]
	pass


def test_model_change_size():
	ttest = torch.rand((7, 3, 299, 299))
	m = InceptionResnetV2(num_classes=5, classify=True, device='cpu')
	ret1 = m(ttest)
	m.change_num_classes(new_num_classes=3)
	ret2 = m(ttest)
	m.change_num_classes(10)
	ret3 = m(ttest)
	assert list(ret1.size()) == [7, 5]
	assert list(ret2.size()) == [7, 3]
	assert list(ret3.size()) == [7, 10]
