#!/usr/bin/env python3

# ランダムエージェントと Q 学習エージェントの対戦 (表形式 Q 学習のベースライン)。
#   3 つのルール (QL 後手 / QL 先手 / 勝者が後手) で、未学習の QL エージェントを対戦しながら学習させ、
#   引き分け・ランダムの勝ち・QL の勝ちの割合を、盤面をアニメーション表示しながら数える。
#   使い方: python3 match_rnd_vs_ql.py [N] [M] [保存するなら何か]   (N: 盤面サイズ, M: 対戦数)

import os
from tictactoe import TicTacToe
from ttt_agent_random import TTTAgentRandom
from ttt_agent_QL import TTTAgentQL
from copy import deepcopy

os.system('')	# Windows 上で、ANSI エスケープシーケンスを有効にする

class Statistician():
	"""対戦を回して結果 (割合の推移と学習後の Q テーブル) を集計する。"""
	
	def __new__(cls, n:int=3, match_count:int=100000):
		if type(n) != int:
			raise TypeError('board size must be an integer')
		
		if n < 3:
			raise ValueError('minimum board size is 3')
		
		if type(match_count) != int:
			raise TypeError('match count must be an integer')
		
		if match_count < 1:
			raise ValueError('match count must be greater than 0')
		
		return super().__new__(cls)
	
	def __init__(self, n:int=3, match_count:int=100000):
		self.__size = n
		self.__match_count = match_count
	
	def doit(self, save:bool=False):
		"""3 つのルールで対戦する。{'RULE #k': {'h': 推移, 'q': Q テーブル}} を返す。
		h は 各局の終了時点での累積数 ÷ 総対戦数 (引分・ランダム勝ち・QL 勝ちの 3 本)。"""
		
		ret = {}
		line_height = (self.__size << 1) + 4	# 進行状況表示用
		count_width = len(str(self.__match_count - 1))
		
		print('結果の順番は [引分, ランダムエージェントの勝ち, QL エージェントの勝ち]')
		print('\033[?25l', end='', flush=False)	# カーソルを隠す
		
		try:	# 中断を含む例外発生時にカーソルを表示したいので...
			for rule in range(1, 4):
				print(f'RULE #{rule}: {["QL 後手", "QL 先手", "勝者が後手"][rule-1]}')
				
				result  = [0, 0, 0]	# [draw, rndwin. qlwin]
				history = [[], [], []]
				
				# 各ルールにおいて、QL エージェントを未学習の状態から開始したいので、ここで生成する
				players = [TTTAgentRandom(self.__size), TTTAgentQL(self.__size)]
				swapped = False
				
				if rule == 2:	# ルール2の時だけ、QL エージェントを先手に設定（ルール3の時に先手でも構わない）
					players.reverse()
					swapped = True
				
				for i in range(1, self.__match_count+1):
					game = TicTacToe(self.__size)
					
					qlm = 1 if swapped else 2
					qlp = players[qlm - 1]
					req_fdbk = False
					
					print(f'先手: {["Rnd", "QL "][1 if swapped else 0]}, 後手: {["Rnd", "QL "][0 if swapped else 1]}')
					
					while game.state==TicTacToe.GameState.ONGOING:
						next_player = players[game.next - 1]
						
						if next_player == qlp:	# 次の指し手が QL なら、
							req_fdbk = True		# フィードバックを要請
						
						next_action = next_player.action(game.board, game.next)
						game.play(game.next, next_action)
						
						if game.state == TicTacToe.GameState.DRAW:
							result[0] += 1
							qlp.feedback(game.board, qlm, 0)
						elif game.state == TicTacToe.GameState.OVER:
							winner_idx = 3 - game.winner if swapped else game.winner
							result[winner_idx] += 1
							
							reward = 1 if winner_idx==2 else -1
							qlp.feedback(game.board, qlm, reward)
							
							if rule == 3 and game.winner == 1:
								players.reverse()
								swapped = not swapped
						else:									# ゲームがまだ進行中で、
							if req_fdbk and next_player != qlp:	# 指し手がランダムだったなら...
								qlp.feedback(game.board, qlm, None)
								req_fdbk = False
						
						# 進行状況を表したいので、ここでは盤面とこれまでの結果を表示する。プログレスバー等もあり。
						game.print_board(stat=True)
						print(f'\033[K', end='', flush=False)	# 行消去
						print(f'#{i:{"0" + str(count_width+1)}}/{self.__match_count}', end='', flush=False)	# 回数
						print(f'\tRESULT: {result}', end='', flush=False)	# 結果、勝ち数
						print(f' [{result[0]/i:.3f}, {result[1]/i:.3f}, {result[2]/i:.3f}]', end='', flush=False)	# 勝率
						print(f'\033[{line_height}F', end='', flush=True)	# カーソル移動
					
					for j in range(3):
						history[j].append(result[j]/self.__match_count)
					
					print(f'\033[{1}F', end='', flush=True)	# カーソル移動
			
				print(f'\033[{line_height+1}E', end='', flush=False)
				print()
				
				ret.update({f'RULE #{rule}': {'h': history, 'q': deepcopy(qlp.q_table)}})
				
				if save:
					qlp.save(comment=f'RULE #{rule}')
				
		except KeyboardInterrupt:
			print(f'\033[{line_height+1}E', end='', flush=False)
		except Exception as e:
			print(f'\033[{line_height+1}E', end='', flush=False)
			raise e
		
		print('\033[?25h', end='', flush=False)	# カーソルを表示
			
		return ret
		
					
if __name__ == '__main__':
	import sys
	
	def print_usage():
		print(f'usage: {sys.argv[0]} N M\n\tN: board size (int) >= 3 (optional)\n\tM: match count (int) > 0 (optional)')
	
	N = 3
	M = 100000
	save2file = False
	
	if len(sys.argv) > 1:
		try:
			N = int(sys.argv[1])
		except:
			print_usage()
			exit()
	if len(sys.argv) > 2:
		try:
			M = int(sys.argv[2])
		except:
			print_usage()
			exit()
	if len(sys.argv) > 3:
		save2file = bool(sys.argv[3])
		
	if len(sys.argv) > 4:
			print_usage()
			print(f"ignoring '{' '.join(sys.argv[3:])}'...")
	
	
	s = Statistician(n=N, match_count=M)
	ret = s.doit(save=save2file)
	
	# 割合の推移をバイナリで保存 (ヘッダー: 盤面サイズ, 対戦数 / 本体: ルールごとに 引分・ランダム勝ち・QL 勝ち)
	from array import array
	from datetime import datetime
	
	if save2file:
		dt = datetime.now()
		filename = f'QL_match_result_{N}x{N}-{M}-{dt.strftime("%Y%m%d_%H%M%S.%f")}.history'
	
		with open(filename, 'wb') as f:
			f.write(array('B', [N]).tobytes())
			f.write(array('Q', [M]).tobytes())
		
			for k in ret.keys():
				f.write(array('f', ret[k]['h'][0]).tobytes())	# tie
				f.write(array('f', ret[k]['h'][1]).tobytes())	# rnd
				f.write(array('f', ret[k]['h'][2]).tobytes())	# QL
	
	
