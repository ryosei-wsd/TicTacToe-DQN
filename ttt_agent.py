#!/usr/bin/env python3

# TicTacToe エージェントの共通インターフェース (抽象基底クラス)。
#   random / Q 学習 / DQN の各エージェントはこれを継承し、同じ呼び出し方で対戦できる。
#     action(board_state, player_id)           : 盤面を見て、打つマス (0 .. n*n-1) を返す
#     feedback(board_state, player_id, reward) : 自分の手の結果を受け取る (学習するエージェントだけが使う)

from abc import ABC, abstractmethod

class TTTAgent(ABC):
	"""n: 盤面サイズ (n x n, n 目並べ)、trainable: 学習するか (False なら学習も探索もしない)。"""
	def __new__(cls, n:int=3, trainable:bool=False):
		if type(n) != int:
			raise TypeError('board size must be an integer')
		
		if n < 3:
			raise ValueError('minimum board size is 3')
		
		if type(trainable) != bool:
			raise TypeError('trainable must be a bool')
		
		return super().__new__(cls)
	
	def __init__(self, n:int=3, trainable:bool=False):
		self.__size = n
		self.trainable	= trainable
	
	@property
	def size(self):
		return self.__size
		
	@size.setter
	def size(self, *args):
		raise PermissionError("You CAN'T change!!")
	
	@abstractmethod
	def action(self, board_state:list, player_id:int) -> int:
		pass
	
	def _regulate(self, board_state:list, player_id:int, opponent:int=-1):
		# 盤面を自分視点に直す (自分 = 1, 空き = 0, 相手 = opponent)。先手・後手を区別せずに扱える。
		return [1 if c == player_id else 0 if c == 0 else opponent for c in board_state]
	def feedback(self, board_state:list, player_id:int, reward:float):
		# 既定では何もしない。ゲーム継続中は reward=None、終局時は勝ち +1 / 負け -1 / 引き分け 0。
		pass

