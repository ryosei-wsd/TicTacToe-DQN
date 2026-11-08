#!/usr/bin/env python3

# 活性化関数 (linear / ReLU / sigmoid)。
#   呼び出すと出力を返し、同時にその入力での微分値を .gradient に保持する
#   (逆伝播で「層の出力の勾配 x 活性化関数の微分」を計算するときに使う)。

import math
from enum import Enum	# StrEnum は 3.11 以降なので Enum を使う
from abc import ABC, abstractmethod

class Function(Enum):
	NONE	= 'none'
	THRU	= 'thru'
	LINEAR	= 'linear'
	SIGMOID	= 'sigmoid'
	RELU	= 'relu'


def get_instance(name:str, **kwargs):
	"""名前 (文字列 or Function) から活性化関数のインスタンスを作る。None / 'linear' / 'thru' は恒等関数。"""
	if name is None:
		name = 'none'
	elif isinstance(name, Function):
		name = name.value
	else:	# should be str
		name = name.lower()

	if name in ['none', 'linear', 'thru']:
		return linear(**kwargs)
	elif name == 'sigmoid':
		return sigmoid(**kwargs)
	elif name == 'relu':
		return relu(**kwargs)
	else:
		raise NotImplementedError(f"'{name}' is not implemented yet.")


class Activator(ABC):
	"""活性化関数の抽象基底クラス。__call__(行列) -> 出力、.gradient -> 直前の入力での微分。"""
	def __init__(self, **kwargs):
		self._gradient = []
		self.set_params(**kwargs)
	
	def set_params(self, **kwargs):
		# 追加のパラメーターをそのまま属性にする (例: sigmoid の gain)
		for key, value in kwargs.items():
			exec(f'self.{key} = {value}')
	
	def get_params(self):
		return self.__dict__
	
	def __repr__(self):
		return f'{self.__class__.__name__}'
	
	@abstractmethod
	def __call__(self, m):
		raise NotImplementedError()
	
	@property
	def gradient(self):
		return self._gradient
	
	@gradient.setter
	def gradient(self, m):
		raise NotImplementedError('you can\'t change this value.')


class linear(Activator):
	"""恒等関数  f(x) = x,  f'(x) = 1"""

	def __call__(self, m):
		self._gradient = [[1   for val in rows ] for rows in m]
		return 			 [[val for val in rows ] for rows in m]


class relu(Activator):
	"""ReLU  f(x) = max(x, 0),  f'(x) = 1 (x >= 0) / 0 (x < 0)"""

	def __call__(self, m):
		self._gradient = [[float(val >= 0.0) for val in rows ] for rows in m]
		return 			 [[  max(val,   0.0) for val in rows ] for rows in m]


class sigmoid(Activator):
	"""シグモイド  f(x) = 1 / (1 + exp(-gain * x)),  f'(x) = gain * f(x) * (1 - f(x))  (gain の既定は 1.0)"""

	def __call__(self, m):
		a = self.gain if 'gain' in self.__dict__ else 1.0
		s = [[(1 + math.exp(-a * val))**-1 for val in rows ] for rows in m]
		self._gradient = [[a * val * (1.0 - val) for val in rows ] for rows in s]
		return 			 s
	
	def __repr__(self):
		if 'gain' in self.__dict__:
			return f'{self.__class__.__name__}(gain={self.gain})'
		else:
			return f'{self.__class__.__name__}'


if __name__ == '__main__':
	# 動作確認: 各関数の出力と微分を表示する (右のコメントが期待される値)
	m = [[1.0, 0.5, 0.0, -0.5, -1.0]]
	
	print(f'  INPUT: {m}')
	
	f = get_instance(Function.NONE)
	a = f(m)									#	[[ 1.0,  0.5,  0.0, -0.5, -1.0]]
	print(f'{str(f):>18}: {a}\t{f.gradient}')	#	[[ 1.0,  1.0,  1.0,  1.0,  1.0]]
	
	f = get_instance(Function.RELU)
	a = f(m)									#	[[ 1.0,  0.5,  0.0,  0.0,  0.0]]
	print(f'{str(f):>18}: {a}\t{f.gradient}')	#	[[ 1.0,  1.0,  1.0,  0.0,  0.0]]
	
	f = get_instance(Function.SIGMOID)
	a = f(m)									#	[[ 0.7310585786, 0.6224593312, 0.5 , 0.3775406688, 0.2689414214]]
	print(f'{str(f):>18}: {a}\t{f.gradient}')	#	[[ 0.1966119332, 0.2350037122, 0.25, 0.2350037122, 0.1966119332]]
	
	f = get_instance(Function.SIGMOID, gain=2.0)
	a = f(m)									#	[[0.8807970779778823, 0.7310585786300049, 0.5, 0.2689414213699951, 0.11920292202211755]]
	print(f'{str(f):>18}: {a}\t{f.gradient}')	#	[[0.20998717080701323, 0.3932238664829637, 0.5, 0.3932238664829637, 0.209987170807013]]
	
