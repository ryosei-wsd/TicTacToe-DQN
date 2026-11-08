#!/usr/bin/env python3

# 最適化手法 (GD / Momentum / RMSprop / Adam)。
#   apply_gradients(weight, bias, dw, db) で、ウェイトとバイアスをその場で更新する。
#   Momentum / RMSprop / Adam が持つ内部状態 (勢い・二乗平均など) は、更新対象の行列の id をキーに保持する。

from abc import ABC, abstractmethod
from enum import Enum	# StrEnum は 3.11 以降なので Enum を使う
import matrix_util as mat


class Function(Enum):
	GD			= 'gd'
	MOMENTUM	= 'momentum'
	RMSPROP		= 'rmsprop'
	ADAM		= 'adam'


def get_instance(name:str, **kwargs):
	"""名前 (文字列 or Function) から最適化手法のインスタンスを作る。kwargs は learning_rate など。"""
	if name is None:
		name = 'none'
	elif isinstance(name, Function):
		name = name.value
	else:	# should be str
		name = name.lower()

	if name in ['none', 'default', 'gd', 'sgd', 'gradientdescent', 'gradient_descent']:
		return GD(**kwargs)
	elif name == 'momentum':
		return Momentum(**kwargs)
	elif name == 'rmsprop':
		return RMSprop(**kwargs)
	elif name == 'adam':
		return Adam(**kwargs)
	else:
		raise NotImplementedError(f"'{name}' is not implemented yet.")


class Optimizer(ABC):
	'''	最適化手法の抽象基底クラス
	'''
	def __init__(self, learning_rate=0.1, **kwargs):
		self.learning_rate  = learning_rate
		self.set_params(**kwargs)
	
	def set_params(self, **kwargs):
		for key, value in kwargs.items():
			exec(f'self.{key} = {value}')
	
	def get_params(self):
		return self.__dict__
	
	@abstractmethod
	def apply_gradient(self, weight, bias, weight_gradient, bias_gradient):
		raise NotImplementedError()
	
	def apply_gradients(self, weight, bias, weight_gradient, bias_gradient):
		# 1 つの層のウェイトとバイアスを更新する
		self.apply_gradient(weight, weight_gradient)
		self.apply_gradient(bias  ,   bias_gradient)
	
	def __repr__(self):
		opt = []
		for k, v in self.__dict__.items():
			if k[0] != '_':
				opt.append(f'{k}={v}')
		opt = ', '.join(opt)
		return f'{self.__class__.__name__}({opt})'


class GD(Optimizer):
	'''	勾配降下法
		v = v - learning_rate * gradient
	'''
	def apply_gradient(self, variable, gradient):
		lr = self.learning_rate
		mat.assign(variable, mat.add(variable, mat.mul(gradient, -lr)))


class Momentum(Optimizer):
	'''	Momentum
		dv = eta * dv - lr * gradient
		v = v + dv
	'''
	def __init__(self, learning_rate=0.1, eta=0.9, **kwargs):
		self.__momentum = {}
		super().__init__(learning_rate=learning_rate, eta=eta, **kwargs)
	
	def apply_gradient(self, variable, gradient):
		momentum_id = id(variable)
		if momentum_id in self.__momentum.keys():
			momentum = self.__momentum[momentum_id]
		else:
			momentum = [[0.0 for _ in row] for row in variable]
			self.__momentum.update({momentum_id: momentum})
		
		lr  = self.learning_rate
		eta = self.eta
		
		mat.assign(momentum, mat.sub(mat.mul(momentum, eta), mat.mul(gradient, lr)))
		mat.assign(variable, mat.add(variable, momentum))
	

class RMSprop(Optimizer):
	'''	RMSprop
		rms = rho * rms + (1 - rho) * gradient**2
		w = w - lr * gradient / sqrt(rms + epsilon)
	'''
	def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7, **kwargs):
		self.__rms = {}
		super().__init__(learning_rate=learning_rate, rho=rho, epsilon=epsilon, **kwargs)
	
	def apply_gradient(self, variable, gradient):
		rms_id = id(variable)
		if rms_id in self.__rms.keys():
			rms = self.__rms[rms_id]
		else:
			rms = [[0.0 for _ in row] for row in variable]
			self.__rms.update({rms_id: rms})
		
		lr  = self.learning_rate
		rho = self.rho
		eps = self.epsilon
		
		mat.assign(rms, mat.add(mat.mul(rms, rho), mat.mul(mat.pow(gradient, 2), 1.0 - rho)))
		mat.assign(variable, mat.sub(variable, mat.mul(mat.hadamard_product(gradient, mat.pow(mat.add(rms, eps), -0.5)), lr)))
	

class Adam(Optimizer):
	'''	Adam
		mu = beta_1 * mu + (1 - beta_1) * gradient
		nu = beta_2 * nu + (1 - beta_2) * gradient**2
		
		muhat = mu / (1 - beta_1**t)		(t: その変数に対する更新回数)
		nuhat = nu / (1 - beta_2**t)
		
		v = v - lr * muhat / [sqrt(nuhat) + epsilon]
	'''
	def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
		self.__momentum = {}
		self.__rms = {}
		self.__step = {}
		super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs)
	
	def apply_gradient(self, variable, gradient):
		var_id = id(variable)
		if var_id in self.__momentum.keys():
			momentum = self.__momentum[var_id]
			rms      = self.__rms[var_id]
		else:
			momentum = [[0.0 for _ in row] for row in variable]
			rms      = [[0.0 for _ in row] for row in variable]
			self.__momentum.update({var_id: momentum})
			self.__rms.update({var_id: rms})
		
		# 更新回数（バイアス補正用）
		t = self.__step.get(var_id, 0) + 1
		self.__step[var_id] = t
		
		lr  = self.learning_rate
		b1  = self.beta_1
		b2  = self.beta_2
		eps = self.epsilon
		
		mat.assign(momentum, mat.add(mat.mul(momentum, b1), mat.mul(gradient, 1-b1)))
		mat.assign(rms, mat.add(mat.mul(rms, b2), mat.mul(mat.pow(gradient, 2), 1-b2)))
		
		muhat = mat.mul(momentum, 1/(1-b1**t))
		nuhat = mat.mul(rms     , 1/(1-b2**t))
		
		dv = mat.mul(mat.hadamard_product(muhat, mat.pow(mat.add(mat.pow(nuhat, 0.5), eps), -1)), lr)
		mat.assign(variable, mat.sub(variable, dv))


if __name__ == '__main__':
	# 動作確認: 1 入力・1 出力のモデル y = w*x + b (w=10, b=7) を、x=1 のとき y=1 になるように
	# 各最適化手法で 3 回更新して、ウェイトと内部状態を表示する (末尾のコメントが出力例)
	from copy import deepcopy
	import nn_loss
	
	INPUT = [[1]]
	IDEAL = INPUT

	INI_W = [[10]]
	INI_B = [[7]]
	
	loss_fn = nn_loss.get_instance(nn_loss.Function.MSE, reduction=nn_loss.Reduction.NONE)
	
	for optimizer in	[get_instance(Function.GD,			learning_rate=0.1), 
						 get_instance(Function.MOMENTUM,	learning_rate=0.1,	eta=0.9), 
						 get_instance(Function.RMSPROP,		learning_rate=0.001, rho=0.9, epsilon=1e-07), 
						 get_instance(Function.ADAM,		learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)]:
		print(f'\033[1;4mOptimizer: {optimizer}\033[0m')
	
		w = deepcopy(INI_W)
		b = deepcopy(INI_B)
		
		mu_name = f'_{optimizer.__class__.__name__}__momentum'
		nu_name = f'_{optimizer.__class__.__name__}__rms'
		
		print(f' W&B #0: [{w[0][0]}, {b[0][0]}]')
		
		for i in range(1, 4):
			pred = mat.add(mat.mul(INPUT, w), b)
			loss = loss_fn(IDEAL, pred)
			g = loss_fn.gradient
			optimizer.apply_gradients(w, b, g, g)
			print(f'LOSS&GRAD: {loss}, {g}')
			print(f' W&B #{i}: [{w[0][0]}, {b[0][0]}]')
			
			if hasattr(optimizer, mu_name):
				d = optimizer.__dict__[mu_name]
				print(f'     mu: {[v[0][0] for k, v in d.items()]}')
			if hasattr(optimizer, nu_name):
				d = optimizer.__dict__[nu_name]
				print(f'     nu: {[v[0][0] for k, v in d.items()]}')
		
		print()


'''
Optimizer: GD(learning_rate=0.1)
 W&B #0: [10, 7]
LOSS&GRAD: [256.0], [[32.0]]
 W&B #1: [6.8, 3.8]
LOSS&GRAD: [92.16], [[19.2]]
 W&B #2: [4.88, 1.88]
LOSS&GRAD: [33.1776], [[11.52]]
 W&B #3: [3.7279999999999998, 0.728]

Optimizer: Momentum(learning_rate=0.1, eta=0.9)
 W&B #0: [10, 7]
LOSS&GRAD: [256.0], [[32.0]]
 W&B #1: [6.8, 3.8]
     mu: [-3.2, -3.2]
LOSS&GRAD: [92.16], [[19.2]]
 W&B #2: [1.9999999999999991, -1.0000000000000009]
     mu: [-4.800000000000001, -4.800000000000001]
LOSS&GRAD: [3.1554436208840472e-30], [[-3.552713678800501e-15]]
 W&B #3: [-2.320000000000002, -5.320000000000002]
     mu: [-4.320000000000001, -4.320000000000001]

Optimizer: RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
 W&B #0: [10, 7]
LOSS&GRAD: [256.0], [[32.0]]
 W&B #1: [9.996837722341375, 6.996837722341375]
     nu: [102.39999999999998, 102.39999999999998]
LOSS&GRAD: [255.79765422984798], [[31.9873508893655]]
 W&B #2: [9.994543994695645, 6.994543994695645]
     nu: [194.47906169193914, 194.47906169193914]
LOSS&GRAD: [255.6509347324968], [[31.97817597878258]]
 W&B #3: [9.992623622601048, 6.992623622601048]
     nu: [277.2915294157439, 277.2915294157439]

Optimizer: Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
 W&B #0: [10, 7]
LOSS&GRAD: [256.0], [[32.0]]
 W&B #1: [9.999000000003125, 6.999000000003125]
     mu: [3.1999999999999993, 3.1999999999999993]
     nu: [1.024000000000001, 1.024000000000001]
LOSS&GRAD: [255.93600400019997], [[31.9960000000125]]
 W&B #2: [9.998000003266615, 6.998000003266616]
     mu: [6.079600000001249, 6.079600000001249]
     nu: [2.046720016000802, 2.046720016000802]
LOSS&GRAD: [255.87201620901106], [[31.99200001306646]]
 W&B #3: [9.997000011956484, 6.997000011956485]
     mu: [8.67084000130777, 8.67084000130777]
     nu: [3.068161360820847, 3.068161360820847]
'''
