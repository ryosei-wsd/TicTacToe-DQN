#!/usr/bin/env python3

# 行列演算ユーティリティ (純 Python)。
#   行列は「リストのリスト」(行 x 列)。NumPy は使わず、ニューラルネットワークの計算に必要な
#   最低限の演算 (積・和差・転置・アダマール積・累乗・表示) だけを自前で実装している。

import math
import os
from operator import mul as _mul
os.system('')	# Windows のターミナルで ANSI エスケープシーケンスを有効にする


def mul(m1, m2):
	"""行列の積 (m2 がスカラーならスカラー倍)。学習時間のほとんどがここなので、内積は sum(map()) で高速化している。"""
	
	if isinstance(m2, float) or isinstance(m2, int):	# スカラー倍
		return [[v * m2 for v in row] for row in m1]
	else:												# 行列の積
		if len(m1[0]) != len(m2):
			raise ValueError('Invalid matrix size')
		
		# m2 を列ごとに取り出しておき、m1 の各行との内積を求める
		cols = list(zip(*m2))
		return [[sum(map(_mul, row, col)) for col in cols] for row in m1]
	
	

def add(m1, m2):
	"""要素ごとの和 (m2 がスカラーなら全要素に加算)。"""
	
	if isinstance(m2, float) or isinstance(m2, int):	# スカラーを足す
		return [[v + m2 for v in row] for row in m1]
	else:												# 行列の和
		if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
			raise ValueError('Invalid matrix size')
		
		return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))] 

def sub(m1, m2):
	"""要素ごとの差 (m2 がスカラーなら全要素から減算)。"""
	
	if isinstance(m2, float) or isinstance(m2, int):	# スカラーを減ず
		return [[v - m2 for v in row] for row in m1]
	else:												# 行列の差
		if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
			raise ValueError('Invalid matrix size')
		
		return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))] 

def pow(m, e):
	"""要素ごとの累乗。"""
	
	return [[v**e for v in row] for row in m]
	
	
def transpose(m):
	"""転置。"""
	
	return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))] 

def hadamard_product(m1, m2):
	"""要素ごとの積 (アダマール積)。"""
	
	return [[m1[i][j] * m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))] 
	

def assign(m1, m2):
	"""m2 の値を m1 に書き込む (m1 の実体を保ったまま更新。ウェイトの更新に使う)。"""
	
	for i in range(len(m1)):
		for j in range(len(m1[0])):
			m1[i][j] = m2[i][j]

def validate(m):
	"""NaN / inf を含まなければ True。"""
	
	for row in m:
		for v in row:
			if math.isnan(v) or math.isinf(v):
				return False
	return True


# ---- 表示 (デバッグ・動作確認用): 小数点の位置を揃えて行列を整形する ----

def mprint(m, format='f', precision=6, title=None, indent=4, title_indent=2, title_separator=': ', title_break=True, enable_ansi_escape=True):
	
	print(mformat(m, format=format, precision=precision, title=title, indent=indent, title_indent=title_indent, title_separator=title_separator, title_break=title_break, enable_ansi_escape=enable_ansi_escape))
	
def mformat(m, format='f', precision=6, title=None, indent=4, title_indent=2, title_separator=': ', title_break=True, enable_ansi_escape=False):
	
	outstr = ''
	fmt = '{:' + f'.{precision}{format}' + '}'
	indent2 = 0
	
	s = [[fmt.format(v) for v in row] for row in m]
	l = [[[v.find('.'), len(v)] for v in row] for row in s]
	l = [[[u if u >= 0 else v, v-u if u >= 0 else 0] for u, v in row] for row in l]
	
	w = [[0, 0] for _ in m[0]]
	for j in range(len(m[0])):
		mi = mf = 0
		for i in range(len(m)):
			if mi < l[i][j][0]:
				mi = l[i][j][0]
			if mf < l[i][j][1]:
				mf = l[i][j][1]
		w[j][0] = mi
		w[j][1] = mf
	
	# タイトル
	if title:
		outstr += ' ' * (title_indent)
		if enable_ansi_escape:
			outstr += f'\033[1;4m{title}\033[0m'
		else:
			outstr += f'{title}'
		outstr += title_separator
		if title_break:
			outstr += '\n'
		else:
			indent = 0
			indent2 = title_indent + len(title) + len(title_separator)
	
	# 行列本体
	for i in range(len(m)):
		outstr += ' ' * indent
		if i == 0:
			outstr += '[[ '
		else:
			outstr += ' [ '
			
		for j in range(len(m[0])):
			outstr += ' ' * (w[j][0] - l[i][j][0])
			outstr += s[i][j]
			outstr += ' ' * (w[j][1] - l[i][j][1])
			
			if j < len(m[0]) - 1:
				outstr += ', '
				
		if i < len(m) - 1:
			outstr += ' ],\n'
		else:
			outstr += ' ]]\n'
	
	if indent2:	# タイトルを改行せず同じ行に置いたとき、2 行目以降の字下げを揃える
		pad = ' ' * indent2
		lines = outstr.split('\n')
		for i in range(1, len(lines)):
			lines[i] = pad + lines[i]
		outstr = '\n'.join(lines)
		
	return outstr
