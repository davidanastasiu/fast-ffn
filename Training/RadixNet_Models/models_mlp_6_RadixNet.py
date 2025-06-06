import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
from brevitas.quant import Int8Bias
#from brevitas.quant import Int4Bias

class model_masked(nn.Module):
	def __init__(self,masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

		super(model_masked,self).__init__()
		self.quantized_model = False
		self.input_shape = 1024

		self.fc1 = nn.Linear(self.input_shape,1024)
		self.fc2 = nn.Linear(1024,1024)
		self.fc3 = nn.Linear(1024,1024)
		self.fc4 = nn.Linear(1024,1024)
		self.fc5 = nn.Linear(1024,1024)
		self.fc6 = nn.Linear(1024,1024)

		self.fc7 = nn.Linear(1024,10)


		self.act1 = nn.ReLU()
		self.act2 = nn.ReLU()
		self.act3 = nn.ReLU()
		self.act4 = nn.ReLU()
		self.act5 = nn.ReLU()
		self.act6 = nn.ReLU()

		self.softmax  = nn.Softmax(0)

	def update_masks(self, masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

	def mask_to_device(self, device):
		self.m1 = self.m1.to(device)
		self.m2 = self.m2.to(device)
		self.m3 = self.m3.to(device)
		self.m4 = self.m4.to(device)
		self.m5 = self.m5.to(device)
		self.m6 = self.m6.to(device)

		self.m7 = self.m7.to(device)

	def force_mask_apply(self):
		self.fc1.weight.data.mul_(self.m1)
		self.fc2.weight.data.mul_(self.m2)
		self.fc3.weight.data.mul_(self.m3)
		self.fc4.weight.data.mul_(self.m4)
		self.fc5.weight.data.mul_(self.m5)
		self.fc6.weight.data.mul_(self.m6)

		self.fc7.weight.data.mul_(self.m7)

	def forward(self, x):
		x = self.act1(self.fc1(x))
		self.fc1.weight.data.mul_(self.m1)
		x = self.act2(self.fc2(x))
		self.fc2.weight.data.mul_(self.m2)
		x = self.act3(self.fc3(x))
		self.fc3.weight.data.mul_(self.m3)
		x = self.act4(self.fc4(x))
		self.fc4.weight.data.mul_(self.m4)
		x = self.act5(self.fc5(x))
		self.fc5.weight.data.mul_(self.m5)
		x = self.act6(self.fc6(x))
		self.fc6.weight.data.mul_(self.m6)

		out = self.fc7(x)
		self.fc7.weight.data.mul_(self.m7)
		return out
	
'''
#######INT8b bias###########
class onetwenty_layer_model_bv_masked(nn.Module):
	def __init__(self, masks, precision = 4):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

		self.weight_precision = precision

		super(onetwenty_layer_model_bv_masked, self).__init__()
		self.input_shape = int(1024)
		self.quant_inp = qnn.QuantIdentity(bit_width=self.weight_precision,weight_quant_type=QuantType.INT, return_quant_tensor=True)
		self.quantized_model = True

		self.fc1 = qnn.QuantLinear(self.input_shape, int(1024),bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc2 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc3 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc4 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc5 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc6 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)

		self.fc7 = qnn.QuantLinear(1024,10,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)


		self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act5 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act6 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)

		#self.softmax  = nn.Softmax(0)

	def update_masks(self, masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

	def mask_to_device(self, device):
		self.m1 = self.m1.to(device)
		self.m2 = self.m2.to(device)
		self.m3 = self.m3.to(device)
		self.m4 = self.m4.to(device)
		self.m5 = self.m5.to(device)
		self.m6 = self.m6.to(device)

		self.m7 = self.m7.to(device)

	def force_mask_apply(self):
		self.fc1.weight.data.mul_(self.m1)
		self.fc2.weight.data.mul_(self.m2)
		self.fc3.weight.data.mul_(self.m3)
		self.fc4.weight.data.mul_(self.m4)
		self.fc5.weight.data.mul_(self.m5)
		self.fc6.weight.data.mul_(self.m6)

		self.fc7.weight.data.mul_(self.m7)

	def forward(self, x):
		x = self.quant_inp(x)
		x = self.act1(self.fc1(x))
		self.fc1.weight.data.mul_(self.m1)
		x = self.act2(self.fc2(x))
		self.fc2.weight.data.mul_(self.m2)
		x = self.act3(self.fc3(x))
		self.fc3.weight.data.mul_(self.m3)
		x = self.act4(self.fc4(x))
		self.fc4.weight.data.mul_(self.m4)
		x = self.act5(self.fc5(x))
		self.fc5.weight.data.mul_(self.m5)
		x = self.act6(self.fc6(x))
		self.fc6.weight.data.mul_(self.m6)

		#softmax_out = self.softmax(self.fc7(x))
		out = self.fc7(x)
		self.fc7.weight.data.mul_(self.m7)
		return out


'''
#######INT8b bias###########
class model_bv_masked(nn.Module):
	def __init__(self, masks, precision = 4):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

		self.weight_precision = precision

		super(model_bv_masked, self).__init__()
		self.input_shape = int(1024)
		self.quant_inp = qnn.QuantIdentity(bit_width=self.weight_precision,weight_quant_type=QuantType.INT, return_quant_tensor=True)
		self.quantized_model = True

		self.fc1 = qnn.QuantLinear(self.input_shape, int(1024),bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc2 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc3 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc4 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc5 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)
		self.fc6 = qnn.QuantLinear(1024,1024,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)

		self.fc7 = qnn.QuantLinear(1024,10,bias=True,weight_quant_type=QuantType.INT,weight_bit_width=self.weight_precision,bias_quant=Int8Bias)


		self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act5 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)
		self.act6 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, return_quant_tensor=True)

		#self.softmax  = nn.Softmax(0)

	def update_masks(self, masks):
		self.m1 = masks['fc1']
		self.m2 = masks['fc2']
		self.m3 = masks['fc3']
		self.m4 = masks['fc4']
		self.m5 = masks['fc5']
		self.m6 = masks['fc6']

		self.m7 = masks['fc7']

	def mask_to_device(self, device):
		self.m1 = self.m1.to(device)
		self.m2 = self.m2.to(device)
		self.m3 = self.m3.to(device)
		self.m4 = self.m4.to(device)
		self.m5 = self.m5.to(device)
		self.m6 = self.m6.to(device)

		self.m7 = self.m7.to(device)

	def force_mask_apply(self):
		self.fc1.weight.data.mul_(self.m1)
		self.fc2.weight.data.mul_(self.m2)
		self.fc3.weight.data.mul_(self.m3)
		self.fc4.weight.data.mul_(self.m4)
		self.fc5.weight.data.mul_(self.m5)
		self.fc6.weight.data.mul_(self.m6)

		self.fc7.weight.data.mul_(self.m7)

	def forward(self, x):
		x = self.quant_inp(x)
		x = self.act1(self.fc1(x))
		self.fc1.weight.data.mul_(self.m1)
		x = self.act2(self.fc2(x))
		self.fc2.weight.data.mul_(self.m2)
		x = self.act3(self.fc3(x))
		self.fc3.weight.data.mul_(self.m3)
		x = self.act4(self.fc4(x))
		self.fc4.weight.data.mul_(self.m4)
		x = self.act5(self.fc5(x))
		self.fc5.weight.data.mul_(self.m5)
		x = self.act6(self.fc6(x))
		self.fc6.weight.data.mul_(self.m6)

		#softmax_out = self.softmax(self.fc7(x))
		out = self.fc7(x)
		self.fc7.weight.data.mul_(self.m7)
		return out


'''
class three_layer_model_masked(nn.Module):
        def __init__(self,masks):
            #Model with <16,64,32,32,5> Behavior
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']
            #self.m4 = masks['fc4']
            #self.m5 = masks['fc5']
            #self.m6 = masks['fc6']
            #self.m7 = masks['fc7']
            #self.m8 = masks['fc8']
            #self.m9 = masks['fc9']
            #self.m10 = masks['fc10']

            #self.m11 = masks['fc11']
            #self.m12 = masks['fc12']
            #self.m13 = masks['fc13']
            #self.m14 = masks['fc14']
            #self.m15 = masks['fc15']
            #self.m16 = masks['fc16']
            #self.m17 = masks['fc17']
            #self.m18 = masks['fc18']
            #self.m19 = masks['fc19']
            #self.m20 = masks['fc20']

            #self.m21 = masks['fc21']
            #self.m22 = masks['fc22']
            #self.m23 = masks['fc23']
            #self.m24 = masks['fc24']
            #self.m25 = masks['fc25']
            #self.m26 = masks['fc26']
            #self.m27 = masks['fc27']
            #self.m28 = masks['fc28']
            #self.m29 = masks['fc29']
            #self.m30 = masks['fc30']

            #self.m21 = masks['fc31']
            #self.m22 = masks['fc32']
            #self.m23 = masks['fc33']
            #self.m24 = masks['fc34']
            #self.m25 = masks['fc35']
            #self.m26 = masks['fc36']
            #self.m27 = masks['fc37']
            #self.m28 = masks['fc38']
            #self.m29 = masks['fc39']
            #self.m30 = masks['fc40']

            self.m120 = masks['fc120']

            super(three_layer_model_masked,self).__init__()
            self.quantized_model = False
            self.input_shape = 1024 #(16,)

            self.fc1 = nn.Linear(self.input_shape,1024)
            self.fc2 = nn.Linear(1024,1024)
            self.fc3 = nn.Linear(1024,1024)
            #self.fc4 = nn.Linear(1024,1024)
            #self.fc5 = nn.Linear(1024,1024)
            #self.fc6 = nn.Linear(1024,1024)
            #self.fc7 = nn.Linear(1024,1024)
            #self.fc8 = nn.Linear(1024,1024)
            #self.fc9 = nn.Linear(1024,1024)
            #self.fc10 = nn.Linear(1024,1024)

            #self.fc11 = nn.Linear(1024,1024)
            #self.fc12 = nn.Linear(1024,1024)
            #self.fc13 = nn.Linear(1024,1024)
            #self.fc14 = nn.Linear(1024,1024)
            #self.fc15 = nn.Linear(1024,1024)
            #self.fc16 = nn.Linear(1024,1024)
            #self.fc17 = nn.Linear(1024,1024)
            #self.fc18 = nn.Linear(1024,1024)
            #self.fc19 = nn.Linear(1024,1024)
            #self.fc20 = nn.Linear(1024,1024)

            #self.fc21 = nn.Linear(1024,1024)
            #self.fc22 = nn.Linear(1024,1024)
            #self.fc23 = nn.Linear(1024,1024)
            #self.fc24 = nn.Linear(1024,1024)
            #self.fc25 = nn.Linear(1024,1024)
            #self.fc26 = nn.Linear(1024,1024)
            #self.fc27 = nn.Linear(1024,1024)
            #self.fc28 = nn.Linear(1024,1024)
            #self.fc29 = nn.Linear(1024,1024)
            #self.fc30 = nn.Linear(1024,1024)

            #self.fc31 = nn.Linear(1024,1024)
            #self.fc32 = nn.Linear(1024,1024)
            #self.fc33 = nn.Linear(1024,1024)
            #self.fc34 = nn.Linear(1024,1024)
            #self.fc35 = nn.Linear(1024,1024)
            #self.fc36 = nn.Linear(1024,1024)
            #self.fc37 = nn.Linear(1024,1024)
            #self.fc38 = nn.Linear(1024,1024)
            #self.fc39 = nn.Linear(1024,1024)
            #self.fc40 = nn.Linear(1024,1024)

            self.fc120 = nn.Linear(1024,10)

            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            #self.act4 = nn.ReLU()
            #self.act5 = nn.ReLU()
            #self.act6 = nn.ReLU()
            #self.act7 = nn.ReLU()
            #self.act8 = nn.ReLU()
            #self.act9 = nn.ReLU()
            #self.act10 = nn.ReLU()

            #self.act11 = nn.ReLU()
            #self.act12 = nn.ReLU()
            #self.act13 = nn.ReLU()
            #self.act14 = nn.ReLU()
            #self.act15 = nn.ReLU()
            #self.act16 = nn.ReLU()
            #self.act17 = nn.ReLU()
            #self.act18 = nn.ReLU()
            #self.act19 = nn.ReLU()
            #self.act20 = nn.ReLU()

            #self.act21 = nn.ReLU()
            #self.act22 = nn.ReLU()
            #self.act23 = nn.ReLU()
            #self.act24 = nn.ReLU()
            #self.act25 = nn.ReLU()
            #self.act26 = nn.ReLU()
            #self.act27 = nn.ReLU()
            #self.act28 = nn.ReLU()
            #self.act29 = nn.ReLU()
            #self.act30 = nn.ReLU()

            #self.act31 = nn.ReLU()
            #self.act32 = nn.ReLU()
            #self.act33 = nn.ReLU()
            #self.act34 = nn.ReLU()
            #self.act35 = nn.ReLU()
            #self.act36 = nn.ReLU()
            #self.act37 = nn.ReLU()
            #self.act38 = nn.ReLU()
            #self.act39 = nn.ReLU()
            #self.act40 = nn.ReLU()

            self.softmax = nn.Softmax(0)

        def update_masks(self, masks):
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']
            #self.m4 = masks['fc4']
            #self.m5 = masks['fc5']
            #self.m6 = masks['fc6']
            #self.m7 = masks['fc7']
            #self.m8 = masks['fc8']
            #self.m9 = masks['fc9']
            #self.m10 = masks['fc10']

            #self.m11 = masks['fc11']
            #self.m12 = masks['fc12']
            #self.m13 = masks['fc13']
            #self.m14 = masks['fc14']
            #self.m15 = masks['fc15']
            #self.m16 = masks['fc16']
            #self.m17 = masks['fc17']
            #self.m18 = masks['fc18']
            #self.m19 = masks['fc19']
            #self.m20 = masks['fc20']

            #self.m21 = masks['fc21']
            #self.m22 = masks['fc22']
            #self.m23 = masks['fc23']
            #self.m24 = masks['fc24']
            #self.m25 = masks['fc25']
            #self.m26 = masks['fc26']
            #self.m27 = masks['fc27']
            #self.m28 = masks['fc28']
            #self.m29 = masks['fc29']
            #self.m30 = masks['fc30']

            #self.m31 = masks['fc31']
            #self.m32 = masks['fc32']
            #self.m33 = masks['fc33']
            #self.m34 = masks['fc34']
            #self.m35 = masks['fc35']
            #self.m36 = masks['fc36']
            #self.m37 = masks['fc37']
            #self.m38 = masks['fc38']
            #self.m39 = masks['fc39']
            #self.m40 = masks['fc40']

            self.m120 = masks['fc120']


        def mask_to_device(self, device):
            self.m1 = self.m1.to(device)
            self.m2 = self.m2.to(device)
            self.m3 = self.m3.to(device)
            #self.m4 = self.m4.to(device)
            #self.m5 = self.m5.to(device)
            #self.m6 = self.m6.to(device)
            #self.m7 = self.m7.to(device)
            #self.m8 = self.m8.to(device)
            #self.m9 = self.m9.to(device)
            #self.m10 = self.m10.to(device)

            #self.m11 = self.m11.to(device)
            #self.m12 = self.m12.to(device)
            #self.m13 = self.m13.to(device)
            #self.m14 = self.m14.to(device)
            #self.m15 = self.m15.to(device)
            #self.m16 = self.m16.to(device)
            #self.m17 = self.m17.to(device)
            #self.m18 = self.m18.to(device)
            #self.m19 = self.m19.to(device)
            #self.m20 = self.m20.to(device)

            #self.m21 = self.m21.to(device)
            #self.m22 = self.m22.to(device)
            #self.m23 = self.m23.to(device)
            #self.m24 = self.m24.to(device)
            #self.m25 = self.m25.to(device)
            #self.m26 = self.m26.to(device)
            #self.m27 = self.m27.to(device)
            #self.m28 = self.m28.to(device)
            #self.m29 = self.m29.to(device)
            #self.m30 = self.m30.to(device)

            #self.m31 = self.m31.to(device)
            #self.m32 = self.m32.to(device)
            #self.m33 = self.m33.to(device)
            #self.m34 = self.m34.to(device)
            #self.m35 = self.m35.to(device)
            #self.m36 = self.m36.to(device)
            #self.m37 = self.m37.to(device)
            #self.m38 = self.m38.to(device)
            #self.m39 = self.m39.to(device)
            #self.m40 = self.m40.to(device)

            self.m120 = self.m120.to(device)




        def force_mask_apply(self):
            self.fc1.weight.data.mul_(self.m1)
            self.fc2.weight.data.mul_(self.m2)
            self.fc3.weight.data.mul_(self.m3)
            #self.fc4.weight.data.mul_(self.m4)
            #self.fc5.weight.data.mul_(self.m5)
            #self.fc6.weight.data.mul_(self.m6)
            #self.fc7.weight.data.mul_(self.m7)
            #self.fc8.weight.data.mul_(self.m8)
            #self.fc9.weight.data.mul_(self.m9)
            #self.fc10.weight.data.mul_(self.m10)

            #self.fc11.weight.data.mul_(self.m11)
            ##self.fc12.weight.data.mul_(self.m12)
            #self.fc13.weight.data.mul_(self.m13)
            #self.fc14.weight.data.mul_(self.m14)
            #self.fc15.weight.data.mul_(self.m15)
            #self.fc16.weight.data.mul_(self.m16)
            #self.fc17.weight.data.mul_(self.m17)
            #self.fc18.weight.data.mul_(self.m18)
            #self.fc19.weight.data.mul_(self.m19)
            #self.fc20.weight.data.mul_(self.m20)

            #self.fc21.weight.data.mul_(self.m21)
            #self.fc22.weight.data.mul_(self.m22)
            #self.fc23.weight.data.mul_(self.m23)
            #self.fc24.weight.data.mul_(self.m24)
            #self.fc25.weight.data.mul_(self.m25)
            #self.fc26.weight.data.mul_(self.m26)
            #self.fc27.weight.data.mul_(self.m27)
            #self.fc28.weight.data.mul_(self.m28)
            #self.fc29.weight.data.mul_(self.m29)
            #self.fc30.weight.data.mul_(self.m30)

            #self.fc31.weight.data.mul_(self.m31)
            #self.fc32.weight.data.mul_(self.m32)
            #self.fc33.weight.data.mul_(self.m33)
            #self.fc34.weight.data.mul_(self.m34)
            #self.fc35.weight.data.mul_(self.m35)
            #self.fc36.weight.data.mul_(self.m36)
            #self.fc37.weight.data.mul_(self.m37)
            #self.fc38.weight.data.mul_(self.m38)
            #self.fc39.weight.data.mul_(self.m39)
            #self.fc40.weight.data.mul_(self.m40)

            self.fc120.weight.data.mul_(self.m120)



        def forward(self, x):
            x = self.act1(self.fc1(x))
            self.fc1.weight.data.mul_(self.m1)
            x = self.act2(self.fc2(x))
            self.fc2.weight.data.mul_(self.m2)
            x = self.act3(self.fc3(x))
            self.fc3.weight.data.mul_(self.m3)
            #x = self.act4(self.fc4(x))
            #self.fc4.weight.data.mul_(self.m4)
            #x = self.act5(self.fc5(x))
            #self.fc5.weight.data.mul_(self.m5)
            #x = self.act6(self.fc6(x))
            #self.fc6.weight.data.mul_(self.m6)
            #x = self.act7(self.fc7(x))
            #self.fc7.weight.data.mul_(self.m7)
            #x = self.act8(self.fc8(x))
            #self.fc8.weight.data.mul_(self.m8)
            #x = self.act9(self.fc9(x))
            #self.fc9.weight.data.mul_(self.m9)
            #x = self.act10(self.fc10(x))
            #self.fc10.weight.data.mul_(self.m10)

            #x = self.act11(self.fc11(x))
            #self.fc11.weight.data.mul_(self.m11)
            #x = self.act12(self.fc12(x))
            #self.fc12.weight.data.mul_(self.m12)
            #x = self.act13(self.fc13(x))
            #self.fc13.weight.data.mul_(self.m13)
            #x = self.act14(self.fc14(x))
            #self.fc14.weight.data.mul_(self.m14)
            #x = self.act15(self.fc15(x))
            #self.fc15.weight.data.mul_(self.m15)
            #x = self.act16(self.fc16(x))
            #self.fc16.weight.data.mul_(self.m16)
            #x = self.act17(self.fc17(x))
            #self.fc17.weight.data.mul_(self.m17)
            #x = self.act18(self.fc18(x))
            #self.fc18.weight.data.mul_(self.m18)
            #x = self.act19(self.fc19(x))
            #self.fc19.weight.data.mul_(self.m19)
            #x = self.act20(self.fc20(x))
            #self.fc20.weight.data.mul_(self.m20)

            #x = self.act21(self.fc21(x))
            #self.fc21.weight.data.mul_(self.m21)
            #x = self.act22(self.fc22(x))
            #self.fc22.weight.data.mul_(self.m22)
            #x = self.act23(self.fc23(x))
            #self.fc23.weight.data.mul_(self.m23)
            #x = self.act24(self.fc24(x))
            #self.fc24.weight.data.mul_(self.m24)
            #x = self.act25(self.fc25(x))
            #self.fc25.weight.data.mul_(self.m25)
            #x = self.act26(self.fc26(x))
            #self.fc26.weight.data.mul_(self.m26)
            #x = self.act27(self.fc27(x))
            #self.fc27.weight.data.mul_(self.m27)
            #x = self.act28(self.fc28(x))
            #self.fc28.weight.data.mul_(self.m28)
            #x = self.act29(self.fc29(x))
            #self.fc29.weight.data.mul_(self.m29)
            #x = self.act30(self.fc30(x))
            #self.fc30.weight.data.mul_(self.m30)

            #x = self.act31(self.fc31(x))
            #self.fc31.weight.data.mul_(self.m31)
            #x = self.act32(self.fc32(x))
            #self.fc32.weight.data.mul_(self.m32)
            #x = self.act33(self.fc33(x))
            #self.fc33.weight.data.mul_(self.m33)
            #x = self.act34(self.fc34(x))
            #self.fc34.weight.data.mul_(self.m34)
            #x = self.act35(self.fc35(x))
            #self.fc35.weight.data.mul_(self.m35)
            #x = self.act36(self.fc36(x))
            #self.fc36.weight.data.mul_(self.m36)
            #x = self.act37(self.fc37(x))
            #self.fc37.weight.data.mul_(self.m37)
            #x = self.act38(self.fc38(x))
            #self.fc38.weight.data.mul_(self.m38)
            #x = self.act39(self.fc39(x))
            #self.fc39.weight.data.mul_(self.m39)
            #x = self.act40(self.fc40(x))
            #self.fc40.weight.data.mul_(self.m40)

            softmax_out = self.softmax(self.fc120(x))
            self.fc120.weight.data.mul_(self.m120)

            return softmax_out



class onetwenty_layer_model_masked(nn.Module):
        def __init__(self,masks):
            #Model with <1024...,1024,10> Behavior
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']
            self.m4 = masks['fc4']
            self.m5 = masks['fc5']
            self.m6 = masks['fc6']
            self.m7 = masks['fc7']
            self.m8 = masks['fc8']
            super(onetwenty_layer_model_masked,self).__init__()
            self.quantized_model = False
            self.input_shape = 784 #(16,)

            self.fc1 = nn.Linear(self.input_shape,1024)
            self.fc2 = nn.Linear(1024,1024)
            self.fc3 = nn.Linear(1024,1024)
            self.fc4 = nn.Linear(1024,1024)
            self.fc5 = nn.Linear(1024,1024)
            self.fc6 = nn.Linear(1024,1024)
            self.fc7 = nn.Linear(1024,1024)
            self.fc8 = nn.Linear(1024,10)

            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.act4 = nn.ReLU()
            self.act5 = nn.ReLU()
            self.act6 = nn.ReLU()
            self.act7 = nn.ReLU()
            self.softmax = nn.Softmax(0)

        def update_masks(self, masks):
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']
            self.m4 = masks['fc4']
            self.m5 = masks['fc5']
            self.m6 = masks['fc6']
            self.m7 = masks['fc7']
            self.m8 = masks['fc8']

        def mask_to_device(self, device):
            self.m1 = self.m1.to(device)
            self.m2 = self.m2.to(device)
            self.m3 = self.m3.to(device)
            self.m4 = self.m4.to(device)
            self.m5 = self.m5.to(device)
            self.m6 = self.m6.to(device)
            self.m7 = self.m7.to(device)
            self.m8 = self.m8.to(device)

        def force_mask_apply(self):
            self.fc1.weight.data.mul_(self.m1)
            self.fc2.weight.data.mul_(self.m2)
            self.fc3.weight.data.mul_(self.m3)
            self.fc4.weight.data.mul_(self.m4)
            self.fc5.weight.data.mul_(self.m5)
            self.fc6.weight.data.mul_(self.m6)
            self.fc7.weight.data.mul_(self.m7)
            self.fc8.weight.data.mul_(self.m8)


        def forward(self, x):
            x = self.act1(self.fc1(x))
            self.fc1.weight.data.mul_(self.m1)
            x = self.act2(self.fc2(x))
            self.fc2.weight.data.mul_(self.m2)
            x = self.act3(self.fc3(x))
            self.fc3.weight.data.mul_(self.m3)
            x = self.act4(self.fc4(x))
            self.fc4.weight.data.mul_(self.m4)
            x = self.act5(self.fc5(x))
            self.fc5.weight.data.mul_(self.m5)
            x = self.act6(self.fc6(x))
            self.fc6.weight.data.mul_(self.m6)
            x = self.act7(self.fc7(x))
            self.fc7.weight.data.mul_(self.m7)  
            softmax_out = self.softmax(self.fc8(x))
            self.fc8.weight.data.mul_(self.m8)

            return softmax_out

class onetwenty_layer_model_bv_masked(nn.Module):
    def __init__(self, masks, precision = 8):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        self.m4 = masks['fc4']
        self.weight_precision = precision
        # Model with <1024...,1024,10> Behavior
        super(onetwenty_layer_model_bv_masked, self).__init__()
        self.input_shape = int(1024)  # (1024,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.fc1 = qnn.QuantLinear(self.input_shape, int(1024),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(1024, 1024,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(1024, 1024,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(1024, 10,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6) #TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.softmax = nn.Softmax(0)

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
        test = self.fc1(x)
        x = self.act1(test)
        self.fc1.weight.data.mul_(self.m1)
        x = self.act2(self.fc2(x))
        self.fc2.weight.data.mul_(self.m2)
        x = self.act3(self.fc3(x))
        self.fc3.weight.data.mul_(self.m3)
        softmax_out = self.softmax(self.fc4(x))
        self.fc4.weight.data.mul_(self.m4)
        return softmax_out
'''