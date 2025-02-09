import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
	def __init__(
		self,
		in_planes: int,
		out_planes: int,
		kernel_size: int | tuple[int, int],
		stride: int | tuple[int, int],
		padding: int | tuple[int, int] = 0,
	) -> None:
		super().__init__()
		self.conv = nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			bias=False,
		)  # verify bias false
		self.bn = nn.BatchNorm2d(
			out_planes,
			eps=0.001,  # value found in tensorflow
			momentum=0.1,  # default pytorch value
			affine=True,
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class stem(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
		self.conv_2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
		self.conv_3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.pool_4a = nn.MaxPool2d(3, 2)
		self.conv_4b = BasicConv2d(64, 96, kernel_size=3, stride=2)
		# filter concat
		self.branch_5_1 = nn.Sequential(
			BasicConv2d(160, 64, kernel_size=1, stride=1),
			BasicConv2d(64, 96, kernel_size=3, stride=1),
		)
		self.branch_5_2 = nn.Sequential(
			BasicConv2d(160, 64, kernel_size=1, stride=1),
			BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
			BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
			BasicConv2d(64, 96, kernel_size=3, stride=1),
		)
		# filter concat
		self.pool_6a = nn.MaxPool2d(3, 2)
		self.conv_6b = BasicConv2d(192, 192, kernel_size=3, stride=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv_1(x)
		x = self.conv_2(x)
		x = self.conv_3(x)
		x0 = self.pool_4a(x)
		x1 = self.conv_4b(x)
		x = torch.cat((x0, x1), dim=1)
		x0 = self.branch_5_1(x)
		x1 = self.branch_5_2(x)
		x = torch.cat((x0, x1), dim=1)
		x0 = self.pool_6a(x)
		x1 = self.conv_6b(x)
		x = torch.cat((x0, x1), dim=1)
		return x


class block_A_v2(nn.Module):
	def __init__(self, input: int, scale: float = 1.0) -> None:
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(input, 32, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(input, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(input, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
			BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1),
		)

		self.conv2d = nn.Conv2d(128, input, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class block_B_v2(nn.Module):
	def __init__(self, input: int, scale: float = 1.0) -> None:
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(input, 192, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(input, 128, kernel_size=1, stride=1),
			BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
			BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
		)

		self.conv2d = nn.Conv2d(384, input, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class block_C_v2(nn.Module):
	def __init__(self, input: int, scale: float = 1.0, noReLU: bool = False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(input, 192, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(input, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
			BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)),
		)

		self.conv2d = nn.Conv2d(448, input, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


class mixed_ab(nn.Module):
	def __init__(self, input: int) -> None:
		super().__init__()

		self.branch0 = BasicConv2d(input, 384, kernel_size=3, stride=2)

		self.branch1 = nn.Sequential(
			BasicConv2d(input, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BasicConv2d(256, 384, kernel_size=3, stride=2),
		)

		self.branch2 = nn.MaxPool2d(3, stride=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		return out


class mixed_bc(nn.Module):
	def __init__(self, input: int) -> None:
		super().__init__()

		self.branch0 = nn.Sequential(
			BasicConv2d(input, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 384, kernel_size=3, stride=2),
		)

		self.branch1 = nn.Sequential(
			BasicConv2d(input, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 288, kernel_size=3, stride=2),
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(input, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
			BasicConv2d(288, 320, kernel_size=3, stride=2),
		)

		self.branch3 = nn.MaxPool2d(3, stride=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		out = torch.cat((x0, x1, x2, x3), 1)
		return out


class InceptionResnetV2(nn.Module):
	"""Inception Resnet V1 model with optional loading of pretrained weights.

	Model parameters can be loaded based on pretraining on the VGGFace2 or
	CASIA-Webface datasets. Pretrained state_dicts are automatically
	downloaded on model instantiation if requested and cached in the torch
	cache. Subsequent instantiations use the cache rather than redownloading.

	Keyword Arguments:
	    pretrained {str} -- Optional pretraining dataset. Either 'vggface2'
	        or 'casia-webface'.
	        (default: {None})
	    classify {bool} -- Whether the model should output classification
	        probabilities or feature embeddings.
	        (default: {False})
	    num_classes {int} -- Number of output classes. If 'pretrained' is
	        set and num_classes not equal to that used for the pretrained
	        model, the final linear layer will be randomly initialized.
	        (default: {None})
	    dropout_prob {float} -- Dropout probability. (default: {0.6})
	"""

	def __init__(
		self,
		pretrained: str | None = None,
		classify: bool = False,
		num_classes: int | None = None,
		dropout_prob: float = 0.6,
		device: None | torch.device = None,
	):
		super().__init__()

		# Set simple attributes
		self.pretrained = pretrained
		self.classify = classify
		self.num_classes = num_classes

		# TODO: pretrained with VGGface2 and Casia

		# Define layers
		self.stem = stem()  # n,384,35,35

		self.repeat_a = nn.Sequential(
			block_A_v2(input=384, scale=0.17),
			block_A_v2(input=384, scale=0.17),
			block_A_v2(input=384, scale=0.17),
			block_A_v2(input=384, scale=0.17),
			block_A_v2(input=384, scale=0.17),
		)
		self.mixed_ab = mixed_ab(input=384)  # n,1152,17,17
		self.repeat_b = nn.Sequential(
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
			block_B_v2(input=1152, scale=0.10),
		)
		self.mixed_bc = mixed_bc(input=1152)  # n,2144,8,8
		self.repeat_c = nn.Sequential(
			block_C_v2(input=2144, scale=0.20),
			block_C_v2(input=2144, scale=0.20),
			block_C_v2(input=2144, scale=0.20),
			block_C_v2(input=2144, scale=0.20),
			block_C_v2(input=2144, scale=0.20),
		)
		self.block8 = block_C_v2(input=2144, noReLU=True)
		self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(dropout_prob)
		self.last_linear = nn.Linear(2144, 1536, bias=False)
		self.last_bn = nn.BatchNorm1d(
			1536, eps=0.001, momentum=0.1, affine=True
		)

		self.flatten = nn.Flatten(start_dim=1)

		if self.classify and self.num_classes is not None:
			self.logits = nn.Linear(1536, self.num_classes)

		self.device = torch.device('cpu')
		if device is not None:
			self.device = device
			self.to(device)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Calculate embeddings or logits given a batch of input image tensors.

		Arguments:
		    x {torch.tensor} -- Batch of image tensors representing faces.

		Returns:
		    torch.tensor -- Batch of embedding vectors or multinomial logits.
		"""
		x = self.stem(x)
		x = self.repeat_a(x)
		x = self.mixed_ab(x)
		x = self.repeat_b(x)
		x = self.mixed_bc(x)
		x = self.repeat_c(x)
		x = self.block8(x)
		x = self.avgpool_1a(x)
		x = self.dropout(x)
		x = self.flatten(x)
		x = self.last_linear(x)
		x = self.last_bn(x)
		x = self.logits(x) if self.classify else F.normalize(x, p=2, dim=1)
		return x
