#!/usr/bin/env python3

# 全結合 (Dense) 層 Layer と、それを積み重ねた多層ネットワーク Layers (順伝播・逆伝播つき)。
#   * ミニバッチ (行 = サンプル) をそのまま流せる。バイアスは全サンプルに加え、勾配はサンプル方向に合計する。
#   * 活性化関数は nn_activator、損失は nn_loss、ウェイトの更新は nn_optimizer が受け持つ。
#   * 動作確認として、逆伝播の勾配を数値微分と比べる gradient_check() を持つ (このファイルを実行すると走る)。

import json
import math
import os
from array import array
from random import random
import nn_activator
import matrix_util as mat


class Layer():
	"""全結合層 1 枚。 出力 = 活性化関数(入力 x ウェイト + バイアス)。 fprop で順伝播、bprop で逆伝播 (dw, db を保持)。"""

	def __new__(cls, in_node_count:int, out_node_count:int, activator:str=None, precision:str='f', initializer:str='uniform'):
		if (not isinstance(in_node_count, int)) or in_node_count < 1:
			raise ValueError('in_node_count must be an integer greater than 0.')
		if (not isinstance(out_node_count, int)) or out_node_count < 1:
			raise ValueError('out_node_count must be an integer greater than 0.')
		if precision not in ['f', 'd']:
			raise NotImplementedError(f'precision type \'{precision}\' is not supported.')
		if initializer not in ['uniform', 'xavier', 'he']:
			raise ValueError(f'unknown initializer \'{initializer}\'.')
		
		if activator is None:
			pass
		elif isinstance(activator, str):
			if not activator.lower() in nn_activator.Function._value2member_map_:
				raise ValueError(f'unknown activator function \'{activator}\'.')
		elif isinstance(activator, (list, tuple)) and len(activator)==2:
			if not activator[0].lower() in nn_activator.Function._value2member_map_:
				raise ValueError(f'unknown activator function \'{activator[0]}\'.')
			if not nn_activator.get_instance(activator[0], **activator[1]):
				raise ValueError(f'unknown activator option \'{activator[1]}\'.')
		else:
			raise ValueError(f'unknown activator format \'{activator}\'.')
		
		return super().__new__(cls)
	
	def __init__(self, in_node_count:int, out_node_count:int, activator:str=None, precision:str='f', initializer:str='uniform'):
		self.__inodes = in_node_count
		self.__onodes = out_node_count
		self.__precision = precision
		
		# 活性化関数
		if isinstance(activator, str):
			self.__activator = nn_activator.get_instance(activator)
		elif isinstance(activator, (list, tuple)) and len(activator)==2:
			self.__activator = nn_activator.get_instance(activator[0], **activator[1])
		else:
			self.__activator = nn_activator.get_instance(nn_activator.Function.NONE)
		
		# ウェイト、バイアスの初期化
		#   uniform: [-1, 1] の一様乱数（既定）
		#   xavier : [-a, a], a = sqrt(6 / (in + out))   … sigmoid / linear 向け
		#   he     : [-a, a], a = sqrt(6 / in)           … ReLU 向け
		if initializer == 'xavier':
			limit = math.sqrt(6.0 / (self.__inodes + self.__onodes))
		elif initializer == 'he':
			limit = math.sqrt(6.0 / self.__inodes)
		else:
			limit = 1.0
		
		self.__weight = []
		for _ in range(self.inodes):
			self.__weight.append(array(self.precision, [(random()*2-1) * limit for _ in range(self.__onodes)]))
		
		self.__bias = [array(self.precision, [0.0 for _ in range(self.__onodes)])]
		
		# バックプロパゲーション用
		self.__last_input = None	# 最後の入力
		self.__dw = None			# ウェイトの勾配
		self.__db = None			# バイアスの勾配
	
	def __repr__(self):
		return f'Layer({self.__inodes}, {self.__onodes}, activator={self.__activator}, precision=\'{self.__precision}\')'
		
	@property
	def inodes(self):
		return self.__inodes
	@inodes.setter
	def inodes(self, n):
		raise NotImplementedError('you can\'t change input node count.')
	
	@property
	def onodes(self):
		return self.__onodes
	@onodes.setter
	def onodes(self, n):
		raise NotImplementedError('you can\'t change output node count.')
	
	@property
	def activator(self):
		return self.__activator
	@activator.setter
	def activator(self, n):
		raise NotImplementedError('you can\'t change activator function.')
	
	@property
	def precision(self):
		return self.__precision
	@precision.setter
	def precision(self, s):
		raise NotImplementedError('you can\'t change the precision.')
	
	@property
	def weight(self):	# array からリストに変換して返す。
		return [self.__weight[i].tolist() for i in range(self.__inodes)]
	
	def get_raw_weight(self):	# 実体を返す。i.e. _Layer__weight
		return self.__weight
	
	@weight.setter
	def weight(self, w):
		if len(w) != self.__inodes:
			raise ValueError(f'dimension mismatch. dimension must be {self.inodes}x{self.onodes}.')
		
		for i in range(self.__inodes):
			if len(w[i]) != self.__onodes:
				raise ValueError(f'dimension mismatch. dimension must be {self.inodes}x{self.onodes}.')
		
		# 将来的に RMSProp 等の実装を考慮して、IN-PLACE で更新する。
		for i in range(self.__inodes):
			for j in range(self.__onodes):
				self.__weight[i][j] = w[i][j]
	
	@property
	def bias(self):
		return [self.__bias[0].tolist()]
	
	def get_raw_bias(self):	# 実体を返す。i.e. _Layer__bias
		return self.__bias
	
	@bias.setter
	def bias(self, b):
		if len(b) != 1:
			raise ValueError(f'dimension mismatch. dimension must be 1x{self.onodes}.')
		
		if len(b[0]) != self.__onodes:
			raise ValueError(f'dimension mismatch. dimension must be 1x{self.onodes}.')
		
		for j in range(self.__onodes):
			self.__bias[0][j] = b[0][j]
	
	@property
	def dw(self):
		return self.__dw
	@dw.setter
	def dw(self, s):
		raise NotImplementedError('you can\'t change this value.')
	
	@property
	def db(self):
		return self.__db
	@db.setter
	def db(self, s):
		raise NotImplementedError('you can\'t change this value.')
	
	def fprop(self, in_state):
		# in_state: (バッチ数 x in) の行列。バッチ数は 1 以上で良い。
		# 呼び出し側が後から in_state を書き換えても影響が出ないよう、行だけコピーして保持する。
		self.__last_input = [list(row) for row in in_state]
		
		# バイアス (1 x out) は全てのバッチ行に加える（ブロードキャスト）
		bias = self.__bias[0]
		z = [[v + b for v, b in zip(row, bias)] for row in mat.mul(self.__last_input, self.__weight)]
		
		return self.__activator(z)
	
	def bprop(self, gradient, input_gradient=True):
		# gradient: (バッチ数 x out)。dw はバッチ全体の和、db も同様にバッチ方向に合計して (1 x out) にする。
		# input_gradient=False のときは dx の計算を省略する（最初の層では不要なので）。
		da        = mat.hadamard_product(self.__activator.gradient, gradient)
		self.__dw = mat.mul(mat.transpose(self.__last_input), da)
		self.__db = [[sum(col) for col in zip(*da)]]
		
		if not input_gradient:
			return None
		
		return mat.mul(da, mat.transpose(self.__weight))
		
	def binwb(self):
		# 未実装
		
		raise NotImplementedError()
	

class Layers():
	"""多層ネットワーク。node_counts=[入力, 隠れ層..., 出力] と、層ごとの活性化関数を指定する。
		例: Layers([9, 64, 9], ['relu', 'linear'], 'd', 'he')
	"""

	def __new__(cls, node_counts:list, activators=None, precision='f', initializer='uniform'):
		# node_counts
		if not isinstance(node_counts, (list, tuple))		\
		or not set([type(v) for v in node_counts]) == {int}	\
		or sum([-1 if i<=0 else 0 for i in node_counts]) < 0:
			raise ValueError('node_counts must be a list/tuple with at least 2 integers greater than 0.')
		
		# activators, precision, initializer
		# layer will take care of it.
		
		return super().__new__(cls)
	
	
	def __init__(self, node_counts:list, activators=None, precision='f', initializer='uniform'):
		# インスタンス生成、引数は __new__() でチェック済み
		# node_counts
		self.layer_count = len(node_counts) - 1
		self.node_counts  = node_counts
		
		# activators
		if activators is None:
			activators = [None for _ in range(self.layer_count)]
		elif isinstance(activators, str):
			activators = [activators for _ in range(self.layer_count)]
		
		self.activators  = activators
		self.precision   = precision
		self.initializer = initializer
		
		# setup layers
		self.layers = []
		for i in range(self.layer_count):
			self.layers.append(
				Layer(	in_node_count	= self.node_counts[i],
						out_node_count	= self.node_counts[i+1],
						activator		= self.activators[i],
						precision		= self.precision,
						initializer		= self.initializer	))
		
	def __repr__(self):
		return f'Layers({self.node_counts}, activators={self.activators}, precision=\'{self.precision}\')'
		
	def __str__(self):
		return self.__repr__()
	
	def fprop(self, in_state):
		# 各層が入力を内部でコピーするので、ここでは deepcopy しない（deepcopy は遅い）。
		res = in_state
		
		for nn in self.layers:
			res = nn.fprop(res)
		
		return res
	
	def bprop(self, gradient, input_gradient=True):
		# input_gradient=False なら、入力に対する勾配 dL/dX は計算しない（None を返す）。
		# 学習だけが目的なら False にした方が速い。
		res = gradient
		for i in reversed(range(self.layer_count)):
			res = self.layers[i].bprop(res, input_gradient or i > 0)
		
		return res
	
	def gradient_check(self, x, y_true, loss, h=1e-5, rtol=1e-4, atol=1e-7, gradient_scale=1.0, verbose=False):
		'''	逆伝播で求めた勾配を、数値微分 (中心差分) と比較して、bprop が正しいかを確認する。
			
				dL/dp ≒ [ L(p + h) - L(p - h) ] / 2h  を、全てのウェイト・バイアス・入力 x の要素 p について計算し、
				bprop() の結果と比べる。 |bprop - 数値微分| <= atol + rtol * max(|bprop|, |数値微分|) なら一致とみなす。
			
			Args:
				x      : 入力 (バッチ数 x 入力ノード数)
				y_true : 正解 (バッチ数 x 出力ノード数)
				loss   : nn_loss の損失関数。reduction は AVERAGE か SUM (NONE は不可)
				h, rtol, atol : 数値微分の刻み幅と、一致とみなす許容誤差
				gradient_scale : loss.gradient が、loss の戻り値の微分の何倍かを表す係数 (既定 1.0 = 一致)。
				      nn_loss.Huber は、損失値は出力ノード数 n で割るが、勾配は L' = -a (n で割らない) と定義されているので、
				      Huber を使うときは gradient_scale=n を指定する。
			
			Returns:
				dict: 'passed' (全て一致したか), 'max_abs_error', 'max_rel_error', 'checked' (比べた個数),
				      'failed' (一致しなかった個数), 'details' ({'W1': (最大絶対誤差, 最大相対誤差), ..., 'X': ...})
			
				* このネットワーク自体は変更しない (内部で倍精度 'd' のコピーを使うので precision='f' でも正しく検査できる)。
				* ReLU の折れ目 (入力が 0 の点) の近くでは、数値微分が合わないことがある。
		'''
		net = Layers(list(self.node_counts), self.activators, 'd', self.initializer)
		net.copy_from(self)
		
		x = [[float(v) for v in row] for row in x]
		
		def value():
			v = loss(y_true, net.fprop(x))
			if isinstance(v, list):
				raise ValueError('loss reduction must be AVERAGE or SUM (not NONE).')
			return v
		
		# 逆伝播による勾配
		value()
		dx = net.bprop(loss.gradient, True)
		analytic = {}
		for i, layer in enumerate(net.layers):
			analytic[f'W{i+1}'] = [list(row) for row in layer.dw]
			analytic[f'b{i+1}'] = [list(row) for row in layer.db]
		analytic['X'] = [list(row) for row in dx]
		
		# 数値微分と比較する対象 (名前, 行列の実体, 逆伝播の勾配)
		targets = []
		for i, layer in enumerate(net.layers):
			targets.append((f'W{i+1}', layer.get_raw_weight()))
			targets.append((f'b{i+1}', layer.get_raw_bias()))
		targets.append(('X', x))
		
		details = {}
		max_abs = max_rel = 0.0
		checked = failed = 0
		
		for name, mtx in targets:
			g_abs = g_rel = 0.0
			
			for r in range(len(mtx)):
				for c in range(len(mtx[0])):
					org = mtx[r][c]
					mtx[r][c] = org + h
					lp = value()
					mtx[r][c] = org - h
					lm = value()
					mtx[r][c] = org
					
					numeric = (lp - lm) / (2 * h) * gradient_scale
					bp      = analytic[name][r][c]
					err     = abs(bp - numeric)
					scale   = max(abs(bp), abs(numeric))
					rel     = err / scale if scale > 1e-8 else 0.0
					
					checked += 1
					if err > atol + rtol * scale:
						failed += 1
					
					g_abs = max(g_abs, err)
					g_rel = max(g_rel, rel)
			
			details[name] = (g_abs, g_rel)
			max_abs = max(max_abs, g_abs)
			max_rel = max(max_rel, g_rel)
		
		result = {'passed': failed == 0, 'max_abs_error': max_abs, 'max_rel_error': max_rel,
					'checked': checked, 'failed': failed, 'details': details}
		
		if verbose:
			for name, (a, r) in details.items():
				print(f'    {name:<3} max abs error {a:.3e}   max rel error {r:.3e}')
			print(f'    -> {"OK" if result["passed"] else "NG"}: {checked - failed}/{checked} 一致')
		
		return result
	
	def copy_from(self, other):
		'''	other のウェイトとバイアスを、この Layers にコピーする（DQN のターゲットネットワーク同期用）。
		'''
		if list(self.node_counts) != list(other.node_counts):
			raise ValueError('network structure mismatch.')
		
		for dst, src in zip(self.layers, other.layers):
			dst.weight = src.weight
			dst.bias   = src.bias
	
	def save(self, filepath):
		'''	ウェイトとバイアスを JSON ファイルに保存する。
		'''
		data = {
			'node_counts': list(self.node_counts),
			'precision'  : self.precision,
			'weights'    : [layer.weight for layer in self.layers],
			'biases'     : [layer.bias   for layer in self.layers],
		}
		
		dpath = os.path.dirname(filepath)
		if dpath:
			os.makedirs(dpath, exist_ok=True)
		
		with open(filepath, 'w', encoding='utf-8') as f:
			json.dump(data, f)
	
	def load(self, filepath):
		'''	save() で保存したウェイトとバイアスを読み込む。ネットワーク構造が同じであること。
		'''
		with open(filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
		
		if list(data['node_counts']) != list(self.node_counts):
			raise ValueError(f'network structure mismatch. file: {data["node_counts"]}, this: {list(self.node_counts)}')
		
		for layer, w, b in zip(self.layers, data['weights'], data['biases']):
			layer.weight = w
			layer.bias   = b
	
	

if __name__ == '__main__':
	# 動作確認 (1): 固定のウェイトで、3-5-4 の 2 層ネットワークの順伝播と逆伝播を活性化関数ごとに表示する
	import nn_loss
	
	s0 = [[2, 1, 3]]
	t0 = [[1, 2, 3, 4]]
	w1 = [[-1.01,  1.03, -1.05,  1.07, -1.09],
		  [ 1.11, -1.13,  1.15, -1.17,  1.19],
		  [-1.21,  1.23, -1.25,  1.27, -1.29]]
	b1 = [[ 1.31, -1.33,  1.35, -1.37,  1.39]]
	w2 = [[-2.02,  2.04, -2.06,  2.08],
		  [-2.1 ,  2.12, -2.14,  2.16],
		  [-2.18,  2.2 , -2.22,  2.24],
		  [-2.26,  2.28, -2.3 ,  2.32],
		  [-2.34,  2.36, -2.38,  2.4 ]]
	b2 = [[-2.42,  2.44, -2.46,  2.48]]
	
	mat.mprint(s0, format='g', title='INPUT STATE')
	mat.mprint(w1, format='g', title='W1')
	mat.mprint(b1, format='g', title='B1')
	mat.mprint(w2, format='g', title='W2')
	mat.mprint(b2, format='g', title='B2')
	mat.mprint(t0, format='g', title='IDEAL OUTPUT')
	
	for af in ['linear', 'relu', 'sigmoid']:
		print(f"\nLayers((3, 5, 4), '{af}', 'f')")
		my_nn = Layers((3, 5, 4), af, 'f')
		
		my_nn.layers[0].weight = w1
		my_nn.layers[0].bias   = b1
		my_nn.layers[1].weight = w2
		my_nn.layers[1].bias   = b2
		
		fprop = my_nn.fprop(s0)
		mat.mprint(fprop, format='g', title='FPROP')
		
		#loss = sum(mat.pow(mat.sub(t0, fprop), 2)[0]) / 4
		#dldy = mat.mul(mat.sub(t0, fprop), -2/4)
		
		loss_fn = nn_loss.MeanSquaredError(reduction=nn_loss.Reduction.SUM)
		loss = loss_fn(t0, fprop)
		dldy = loss_fn.gradient
		
		mat.mprint([[loss]], format='g', title='LOSS')
		mat.mprint(dldy, format='g', title='dL/dY')
		
		dldx = my_nn.bprop(dldy)
		
		for i in reversed(range(len(my_nn.layers))):
			mat.mprint(my_nn.layers[i].dw, format='g', title=f'dL/dW{i+1}')
			mat.mprint(my_nn.layers[i].db, format='g', title=f'dL/dB{i+1}')
		
		mat.mprint(dldx, format='g', title='dL/dX')

	
	
	
	# ------------------------------------------------------------------
	#  勾配チェック: 逆伝播の勾配を数値微分と比較する
	# ------------------------------------------------------------------
	import random as rnd	# (モジュール先頭の 'from random import random' を上書きしないよう別名にする)
	
	rnd.seed(0)
	all_passed = True
	
	cases = [
		# (名前, ノード数, 活性化関数, 損失関数, 損失の引数, gradient_scale)
		#   Huber の勾配は損失値の微分の n 倍 (nn_loss の仕様: L' = -a)。n = 出力ノード数
		('ReLU + linear / MSE (SUM)',        [5, 7, 4],    ['relu', 'linear'],             nn_loss.Function.MSE,   dict(reduction=nn_loss.Reduction.SUM), 1),
		('sigmoid + linear / MSE (AVERAGE)', [4, 6, 3],    ['sigmoid', 'linear'],          nn_loss.Function.MSE,   dict(reduction=nn_loss.Reduction.AVERAGE), 1),
		('ReLU x2 + linear / Huber',         [6, 8, 8, 3], ['relu', 'relu', 'linear'],     nn_loss.Function.HUBER, dict(reduction=nn_loss.Reduction.AVERAGE, delta=0.5), 3),
		('DQN 構成 9-64-9 / Huber (float32)',[9, 64, 9],   ['relu', 'linear'],             nn_loss.Function.HUBER, dict(reduction=nn_loss.Reduction.AVERAGE), 9),
	]
	
	print('\n\n=== 勾配チェック (数値微分との比較, バッチ数 8) ===')
	
	for name, nodes, acts, lf_name, lf_args, gscale in cases:
		precision = 'f' if 'float32' in name else 'd'
		net = Layers(nodes, acts, precision, 'he')
		lf  = nn_loss.get_instance(lf_name, **lf_args)
		x   = [[rnd.uniform(-1, 1) for _ in range(nodes[0])]  for _ in range(8)]
		y   = [[rnd.uniform(-2, 2) for _ in range(nodes[-1])] for _ in range(8)]
		
		print(f'[{name}]')
		res = net.gradient_check(x, y, lf, gradient_scale=gscale, verbose=True)
		all_passed = all_passed and res['passed']
	
	# 対照実験: Huber で gradient_scale を指定しないと、不一致として検出されること (チェック自体が働いている確認)
	net = Layers([6, 8, 3], ['relu', 'linear'], 'd', 'he')
	lf  = nn_loss.get_instance(nn_loss.Function.HUBER, reduction=nn_loss.Reduction.AVERAGE)
	x   = [[rnd.uniform(-1, 1) for _ in range(6)] for _ in range(8)]
	y   = [[rnd.uniform(-2, 2) for _ in range(3)] for _ in range(8)]
	res = net.gradient_check(x, y, lf)
	print(f'[対照] Huber で gradient_scale を指定しない: {"一致 (想定外)" if res["passed"] else "不一致を検出 (想定どおり)"}'
			f'  (最大相対誤差 {res["max_rel_error"]:.3f}, 1 - 1/n = {1 - 1/3:.3f})')
	all_passed = all_passed and not res['passed']
	
	print('\n勾配チェック:', 'ALL OK' if all_passed else 'FAILED')
