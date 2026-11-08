#!/usr/bin/env python3

# 損失関数 (平均二乗誤差 MSE / Huber)。
#   __call__(正解, 予測) で損失を返し、同時に予測値に対する勾配 dL/dy_pred を .gradient に保持する
#   (逆伝播の出発点)。reduction で戻り値の集約方法を選ぶ:
#     NONE: サンプルごとの損失のリスト / SUM: その合計 / AVERAGE: サンプル平均
#     (AVERAGE のときは勾配も 1/サンプル数 される。ミニバッチ学習用)

import math
from enum import Enum	# StrEnum は 3.11 以降なので Enum を使う

from abc import ABC, abstractmethod
import matrix_util as mat


class Function(Enum):
	MSE		= 'mse'
	HUBER	= 'huber'
	
class Reduction(Enum):
	NONE    = 'none'
	SUM     = 'sum'
	AVERAGE = 'average'	# ミニバッチ用
	

def get_instance(name:str, **kwargs):
	"""名前 (文字列 or Function) から損失関数のインスタンスを作る。kwargs は reduction や delta など。"""
	if name is None:
		name = 'none'
	elif isinstance(name, Function):
		name = name.value
	else:	# should be str
		name = name.lower()

	if name in ['none', 'default', 'mse', 'meansquarederror']:
		return MeanSquaredError(**kwargs)
	elif name == 'huber':
		return Huber(**kwargs)
	else:
		raise NotImplementedError(f"'{name}' is not implemented yet.")


class Loss(ABC):
	"""	損失関数の抽象基底クラス
	"""
	def __init__(self, reduction=Reduction.NONE, **kwargs):
		self._gradient = []
		self.reduction = reduction
		self.r_none = None
		self.r_sum  = None
		self.r_avg  = None
		self.set_params(**kwargs)
	
	def set_params(self, **kwargs):
		for key, value in kwargs.items():
			exec(f'self.{key} = {value}')
	
	def get_params(self):
		return self.__dict__
	
	@abstractmethod
	def __call__(self, y_truevalue, y_predicted):
		raise NotImplementedError()
	
	@property
	def gradient(self):
		return self._gradient


class MeanSquaredError(Loss):
	"""	平均二乗誤差
		L  = 1/N * (y_true - y_pred)**2
		L' = -2/N * (y_true - y_pred)
	
	Args:
		None
	Attrbutes:
		gradient: 最後の入力の勾配を保持する。
	Returns:
		平均二乗誤差。
	"""
	def __call__(self, y_truevalue, y_predicted):
		m = len(y_truevalue)
		n = len(y_truevalue[0])
		
		diff = mat.sub(y_truevalue, y_predicted)
		self._gradient = mat.mul(diff, -2/n)
		
		self.err_none = [sum(row) / n for row in mat.pow(diff, 2)]
		self.err_sum  = sum(self.err_none)
		self.err_avg  = self.err_sum / m
		
		if self.reduction == Reduction.AVERAGE:
			self._gradient = mat.mul(self._gradient, 1/m)
			return self.err_avg
		elif self.reduction == Reduction.SUM:
			return self.err_sum
		else:
			return self.err_none

class Huber(Loss):
	"""	Huber Loss
		a = y_true - y_pred
		L  = 1/2 * a**2		for |a| <= δ
		   = δ(|a| - δ/2)	otherwise
		L' = -a				for |a| <= δ
		   = -δ				for a > 0
		   = δ				otherwise
		(損失値は出力ノード数 n で割った平均だが、勾配は n で割らず、要素ごとの L' をそのまま返す。
		 DQN では、選んだ行動の TD 誤差を [-δ, δ] に切り詰めた値がそのまま勾配になる)
	
	Args:
		delta: デフォルト 1.0
	Attrbutes:
		gradient: 最後の入力の勾配を保持する。
	Returns:
		フーバー損失。
	"""
	def __init__(self, delta=1.0, **kwargs):
		self.delta = delta
		super().__init__(**kwargs)
		
	def __call__(self, y_truevalue, y_predicted):
		delta  = self.delta
		deltah = 0.5 * delta
		
		m = len(y_truevalue)
		n = len(y_truevalue[0])
		
		diff = mat.sub(y_truevalue, y_predicted)
		self.err_none  = [0 for _ in diff]
		self._gradient = [[0 for _ in r] for r in diff]
		
		for i in range(m):
			loss = 0.0
			for j in range(n):
				a  = diff[i][j]
				aa = a * (-1 if a < 0 else 1)
				if aa <= delta:
					loss += 0.5 * a * a
					self._gradient[i][j] = -a
				else:
					loss += delta * (aa - deltah)
					self._gradient[i][j] = delta * (1 if a < 0 else -1)
			
			self.err_none[i]  = loss / n
	
		self.err_sum  = sum(self.err_none)
		self.err_avg  = self.err_sum / m
		
		if self.reduction == Reduction.AVERAGE:
			self._gradient = mat.mul(self._gradient, 1/m)
			return self.err_avg
		elif self.reduction == Reduction.SUM:
			return self.err_sum
		else:
			return self.err_none



if __name__ == '__main__':
	import nn_layers
	
	# 動作確認: 手計算した期待値 (下のコメント) と実際の出力を比べる
	y_true = [[0., 1., 0., 0.], [0., 0., 0.,  0.], [1., 0., 2.,  0.]]
	y_inpt = [[1., 1., 0., 0.], [1., 1., 2., -1.], [2., 3., 0., -2.]]
	
	w = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
	b = [[17, 18, 19, 20]]
	
	nn = nn_layers.Layers((4, 4), precision='d')
	nn.layers[0].weight = w
	nn.layers[0].bias   = b
	
	print(nn.layers[0].bias)
	
	y_pred = [mat.add(mat.mul([x], w), b)[0] for x in y_inpt]
	# [[23.0, 26.0, 29.0, 32.0]], [[28.0, 32.0, 36.0, 40.0]], [[8.0, 12.0, 16.0, 20.0]]
	print(y_pred)
	
	# 手計算による期待値
	'''
	True Value:			Predicted:			Diff:
	[[ 0, 1, 0, 0 ],	[[ 23, 26, 29, 32 ],	[[ -23, -25, -29, -32 ],
	 [ 0, 0, 0, 0 ],  -  [ 28, 32, 36, 40 ], =   [ -28, -32, -36, -40 ],
	 [ 1, 0, 2, 0 ]]	 [  8, 12, 16, 20 ]]	 [  -7, -12, -14, -20 ]]
	
	MSE:	[[ (529 +  625 +  841 + 1024) / 4 ],  [[ 3019/4 ],	 [[  754.75 ],		grad:	[[ 11.5, 12.5, 14.5, 16 ],		
			 [ (784 + 1024 + 1296 + 1600) / 4 ], = [ 4704/4 ], =  [ 1176.0  ],				 [ 14  , 16  , 18  , 20 ],		
			 [ ( 49 +  144 +  196 +  400) / 4 ]]   [  789/4 ]]	  [  197.25 ]]				 [  3.5,  6  ,  7  , 10 ]]		
	
	Huber:	[[ (22.5 + 24.5 + 28.5 + 31.5) / 4 ],  [[ 107/4 ],	 [[ 26.75 ],		grad:	[[ 1, 1, 1, 1 ],		
	δ = 1	 [ (27.5 + 31.5 + 35.5 + 39.5) / 4 ], = [ 134/4 ], =  [ 33.5  ],				 [ 1, 1, 1, 1 ],		
			 [ ( 6.5 + 11.5 + 13.5 + 19.5) / 4 ]]   [  51/4 ]]	  [ 12.75 ]]				 [ 1, 1, 1, 1 ]]		
	'''
	
	print('\033[1;4mINPUT\033[0m')
	mat.mprint(y_true, format='g', precision=1, title='\033[1mTrue Value\033[0m', enable_ansi_escape=False)
	mat.mprint(y_pred, format='g', precision=6, title='\033[1mPredicted\033[0m',  enable_ansi_escape=False)
	mat.mprint(mat.sub(y_true, y_pred), format='g', precision=6, title='\033[1mDiff\033[0m',  enable_ansi_escape=False)
	
	print('\033[1;4mMSE (Reduction=None)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.NONE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	'''
	# calc dL/dW & dL/dB
	LOSS = []
	dLdY = []
	dLdX = []
	dLdW = []
	dLdB = []
	for x, t in zip(y_inpt, y_true):
		y = nn.fprop([x])
		loss = lf([t], y)
		LOSS.append(loss[0])
		dLdY.append(lf.gradient[0])
		dLdX.append(nn.bprop(lf.gradient))
		dLdW.append(nn.layers[0].dw)
		dLdB.append(nn.layers[0].db)
	
	dLdW_SUM = mat.mul(dLdW[0], 0)
	for dldw in dLdW:
		dLdW_SUM = mat.add(dLdW_SUM, dldw)
	dLdB_SUM = mat.mul(dLdB[0], 0)
	for dldb in dLdB:
		dLdB_SUM = mat.add(dLdB_SUM, dldb)
	
	print(f'LOSS: {LOSS}')
	print(f'dLdY: {dLdY}')
	print(f'dLdX: {dLdX}')
	print(f'dLdW: {dLdW}')
	print(f'dLdB: {dLdB}')
	
	print()
	print(f'ΣdLdW: {dLdW_SUM}')
	print(f'ΣdLdB: {dLdB_SUM}')
	
	exit()
	'''
	
	print('\033[1;4mMSE (Reduction=SUM)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.SUM)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mMSE (Reduction=AVERAGE)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.AVERAGE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=None)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.NONE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=SUM)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.SUM)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=AVERAGE)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.AVERAGE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	y_true = [y_true[0]]
	y_pred = [y_pred[0]]
	
	print('\033[1;4mINPUT\033[0m')
	mat.mprint(y_true, format='g', precision=1, title='\033[1mTrue Value\033[0m', enable_ansi_escape=False)
	mat.mprint(y_pred, format='g', precision=6, title='\033[1mPredicted\033[0m',  enable_ansi_escape=False)
	mat.mprint(mat.sub(y_true, y_pred), format='g', precision=6, title='\033[1mDiff\033[0m',  enable_ansi_escape=False)
	
	print('\033[1;4mMSE (Reduction=None)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.NONE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mMSE (Reduction=SUM)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.SUM)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mMSE (Reduction=AVERAGE)\033[0m')
	lf = get_instance(Function.MSE, reduction=Reduction.AVERAGE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	
	print('\033[1;4mHuber Loss (Reduction=None)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.NONE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=SUM)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.SUM)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=AVERAGE)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.AVERAGE)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	
	y_true = [[0. , 1. ], [0. , 0. ]]
	y_pred = [[0.6, 0.4], [0.4, 0.6]]
	
	mat.mprint(y_true, format='g', precision=1, title='\033[1mTrue Value\033[0m', enable_ansi_escape=False)
	mat.mprint(y_pred, format='g', precision=1, title='\033[1mPredicted\033[0m',  enable_ansi_escape=False)
	mat.mprint(mat.sub(y_true, y_pred), format='g', precision=1, title='\033[1mDiff\033[0m',  enable_ansi_escape=False)
	
	print('\033[1;4mHuber Loss (Reduction=None, delta=1.2)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.NONE, delta=1.2)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=SUM, delta=1.2)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.SUM, delta=1.2)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' =', enable_ansi_escape=False, title_break=False)
	
	print('\033[1;4mHuber Loss (Reduction=AVERAGE, delta=1.2)\033[0m')
	lf = get_instance(Function.HUBER, reduction=Reduction.AVERAGE, delta=1.2)
	loss = lf(y_true, y_pred)
	print(f'  LOSS  = {loss}')
	mat.mprint(lf.gradient, format='g', precision=6, title='dL/dy', title_separator=' = ', enable_ansi_escape=False, title_break=False)
	
	
