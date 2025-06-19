import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
from brevitas.quant import Int8Bias

class model_masked(nn.Module):
	def __init__(self,masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']

		self.m4 = masks['fc4']

		super(model_masked,self).__init__()
		self.quantized_model = False
		self.input_shape = 1024

		self.fc1 = nn.Linear(self.input_shape,1024)
		self.fc2 = nn.Linear(1024,1024)
		self.fc3 = nn.Linear(1024,1024)

		self.fc4 = nn.Linear(1024,10)


		self.act1 = nn.ReLU()
		self.act2 = nn.ReLU()
		self.act3 = nn.ReLU()

		self.softmax  = nn.Softmax(0)

	def update_masks(self, masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']

		self.m4 = masks['fc4']

	def mask_to_device(self, device):
		self.m1 = self.m1.to(device)
		self.m2 = self.m2.to(device)
		self.m3 = self.m3.to(device)
		
		self.m4 = self.m4.to(device)

	def force_mask_apply(self):
		self.fc1.weight.data.mul_(self.m1)
		self.fc2.weight.data.mul_(self.m2)
		self.fc3.weight.data.mul_(self.m3)

		self.fc4.weight.data.mul_(self.m4)

	def forward(self, x):
		x = self.act1(self.fc1(x))
		self.fc1.weight.data.mul_(self.m1)
		x = self.act2(self.fc2(x))
		self.fc2.weight.data.mul_(self.m2)
		x = self.act3(self.fc3(x))
		self.fc3.weight.data.mul_(self.m3)

		out = self.fc4(x)
		self.fc4.weight.data.mul_(self.m4)
		return out

class model_bv_masked(nn.Module):
	def __init__(self, masks, precision = 4):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		
		self.m4 = masks['fc4']

		self.weight_precision = precision

		super(model_bv_masked, self).__init__()
		self.input_shape = int(1024)
		self.quant_inp = qnn.QuantIdentity(bit_width=self.weight_precision,weight_quant_type=QuantType.INT, return_quant_tensor=True)
		self.quantized_model = True

		self.fc1 = qnn.QuantLinear(self.input_shape, int(1024),bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc2 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc3 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		
		self.fc4 = qnn.QuantLinear(1024,10,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)


		self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)

		#self.softmax  = nn.Softmax(0)

	def update_masks(self, masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']

		self.m4 = masks['fc4']

	def mask_to_device(self, device):
		self.m1 = self.m1.to(device)
		self.m2 = self.m2.to(device)
		self.m3 = self.m3.to(device)
		
		self.m4 = self.m4.to(device)

	def force_mask_apply(self):
		self.fc1.weight.data.mul_(self.m1)
		self.fc2.weight.data.mul_(self.m2)
		self.fc3.weight.data.mul_(self.m3)

		self.fc4.weight.data.mul_(self.m4)

	def forward(self, x):
		x = self.quant_inp(x)
		x = self.act1(self.fc1(x))
		self.fc1.weight.data.mul_(self.m1)
		x = self.act2(self.fc2(x))
		self.fc2.weight.data.mul_(self.m2)
		x = self.act3(self.fc3(x))
		self.fc3.weight.data.mul_(self.m3)

		#softmax_out = self.softmax(self.fc7(x))
		out = self.fc4(x)
		self.fc4.weight.data.mul_(self.m4)
		return out



