#!/usr/bin/env python3

# ランダムエージェント: 空いているマスから一様ランダムに選ぶ。対戦相手 (ベースライン) として使う。

from random import randint
from ttt_agent import TTTAgent

class TTTAgentRandom(TTTAgent):

	def action(self, board_state:list, player_id:int) -> int:
		empty_cells = [i for i, v in enumerate(board_state) if v==0]
		
		if len(empty_cells)==0:
			raise NotImplementedError("no space left!!")
			
		return empty_cells[randint(0, len(empty_cells)-1)]
