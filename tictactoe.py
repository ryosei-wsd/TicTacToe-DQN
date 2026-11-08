#!/usr/bin/env python3

# TicTacToe (n x n の盤面で、縦・横・斜めに n 個そろえた方が勝ち = n 目並べ) の環境。
#   エージェントは「盤面 (list) を見て打つマスを返す」だけで対戦できる。
#   盤面は 1 次元の list (長さ n*n)。 index = x + y * n。 0: 空き, 1: 先手, 2: 後手。

from enum import IntEnum
from random import choice


class TicTacToe:
    """ゲームの状態 (盤面・手番・勝敗) を管理する。play() で 1 手打つと、勝敗を判定して手番を進める。"""

    Mark = ' ox'   # 表示用の記号 (空き, 先手, 後手)

    class Player(IntEnum):
        FIRST = 1
        SECOND = 2

    class GameState(IntEnum):
        ONGOING = 0
        DRAW = 1
        OVER = 2

    class CellState(IntEnum):
        EMPTY = 0
        FIRST = 1
        SECOND = 2

    class LineState(IntEnum):
        PENDING = 0
        DRAW = 1
        FIXED = 2

    class Direction(IntEnum):
        COLUMN = 0
        ROW = 1
        DIAGONAL = 2

    class Diagonal(IntEnum):
        TL2BR = 0
        TR2BL = 1

    # ----------------------------------
    # Initialization
    # ----------------------------------

    def __init__(self, n: int = 3):

        if type(n) != int:
            raise TypeError('board size must be an integer')

        if n < 3:
            raise ValueError('minimum board size is 3')

        self.__size = n
        self.__next = self.Player.FIRST

        self.__board = [
            self.CellState.EMPTY
            for _ in range(n * n)
        ]

        self.__state = self.GameState.ONGOING
        self.__winner = None

        # 縦・横・斜めの全ライン (2n+2 本) の状態。
        #   PENDING: 未確定 / DRAW: 両者の石があり、もう誰も揃えられない / FIXED: 揃って勝負が決まった
        self.__judge = {
            self.Direction.COLUMN: [
                self.LineState.PENDING
            ] * n,

            self.Direction.ROW: [
                self.LineState.PENDING
            ] * n,

            self.Direction.DIAGONAL: [
                self.LineState.PENDING,
                self.LineState.PENDING
            ]
        }

    # ----------------------------------
    # Properties
    # ----------------------------------

    @property
    def size(self):
        return self.__size

    @property
    def next(self):
        return self.__next

    @property
    def board(self):
        return [int(cell) for cell in self.__board]

    @property
    def judge(self):
        return {
            direction: states.copy()
            for direction, states in self.__judge.items()
        }

    @property
    def state(self):
        return self.__state

    @property
    def winner(self):
        return self.__winner

    # ----------------------------------
    # Cell operations
    # ----------------------------------

    def is_vacant(self, x: int, y: int = None):
        """そのマス (index または x, y) に打てるか (ゲーム継続中で、かつ空きか)。"""

        if y is None:
            if x not in range(self.size * self.size):
                return False

            index = x

        else:
            if x not in range(self.size) or y not in range(self.size):
                return False

            index = x + y * self.size

        return (
            self.state == self.GameState.ONGOING
            and self.__board[index] == self.CellState.EMPTY
        )

    # ----------------------------------
    # Play
    # ----------------------------------

    def play(self, player: 'TicTacToe.Player',
             x: int, y: int = None):
        """
        player が 1 手打つ。マスは index (x だけ) か座標 (x, y) で指定する。
        手番違い・埋まっているマス・範囲外は例外。 戻り値は (ゲームの状態, 勝者)。
        """

        if self.state != self.GameState.ONGOING:
            raise PermissionError('GAME IS OVER!!')

        if player not in list(self.Player):
            raise PermissionError('you are NOT a player.')

        player = self.Player(player)

        if player != self.__next:
            raise PermissionError(
                f'next turn is player {self.__next}'
            )

        if type(x) != int or (y is not None and type(y) != int):
            raise TypeError('coordinates must be integers')

        # Linear index
        if y is None:

            if x not in range(self.size * self.size):
                raise IndexError(f'index {x} is out of range.')

            index = x

        # Two-dimensional coordinates
        else:

            if x not in range(self.size) or y not in range(self.size):
                raise IndexError(f'({x}, {y}) is out of range.')

            index = x + y * self.size

        if not self.is_vacant(index):
            raise PermissionError('cell is NOT vacant')

        self.__board[index] = self.CellState(player)

        ret = self._judge()

        if self.state == self.GameState.ONGOING:

            self.__next = (
                self.Player.SECOND
                if player == self.Player.FIRST
                else self.Player.FIRST
            )

        else:
            self.__next = None

        return ret

    # ----------------------------------
    # Judge
    # ----------------------------------

    def _judge(self):
        """勝敗の判定。まだ決まっていない (PENDING の) ラインだけを調べ、全ラインが決着済みなら引き分け。"""

        if self.state != self.GameState.ONGOING:
            return self.state, self.winner

        # Check rows and columns
        for direction in (
            self.Direction.COLUMN,
            self.Direction.ROW
        ):

            for p in range(self.size):

                if self.__judge[direction][p] != self.LineState.PENDING:
                    continue

                if direction == self.Direction.COLUMN:
                    indexes = [
                        p + i * self.size
                        for i in range(self.size)
                    ]

                else:
                    indexes = [
                        i + p * self.size
                        for i in range(self.size)
                    ]

                self._check_line(direction, p, indexes)

                if self.state == self.GameState.OVER:
                    return self.state, self.winner

        # Check diagonals
        for diagonal in self.Diagonal:

            if self.__judge[self.Direction.DIAGONAL][diagonal] != self.LineState.PENDING:
                continue

            if diagonal == self.Diagonal.TL2BR:

                indexes = [
                    i + i * self.size
                    for i in range(self.size)
                ]

            else:

                indexes = [
                    (self.size - 1 - i) + i * self.size
                    for i in range(self.size)
                ]

            self._check_line(
                self.Direction.DIAGONAL,
                diagonal,
                indexes
            )

            if self.state == self.GameState.OVER:
                return self.state, self.winner

        # Check draw
        if all(
            state != self.LineState.PENDING
            for states in self.__judge.values()
            for state in states
        ):

            self.__state = self.GameState.DRAW

        return self.state, self.winner

    # ----------------------------------
    # Check individual line
    # ----------------------------------

    def _check_line(self, direction, position, indexes):
        """1 本のラインを調べる: 両者の石があれば DRAW、片方が n 個そろえば勝ち (FIXED)。"""

        cells = [
            self.__board[i]
            for i in indexes
        ]

        first = cells.count(self.CellState.FIRST)
        second = cells.count(self.CellState.SECOND)

        # Both players exist in the same line
        if first > 0 and second > 0:

            self.__judge[direction][position] = self.LineState.DRAW

        # Player FIRST wins
        elif first == self.size:

            self.__judge[direction][position] = self.LineState.FIXED
            self.__state = self.GameState.OVER
            self.__winner = self.Player.FIRST

        # Player SECOND wins
        elif second == self.size:

            self.__judge[direction][position] = self.LineState.FIXED
            self.__state = self.GameState.OVER
            self.__winner = self.Player.SECOND

    # ----------------------------------
    # Print board
    # ----------------------------------

    def print_board(self, stat: bool = False):
        """盤面を表示する。stat=True ならラインごとの状態 (judge) も添える。"""

        hdiv = '    +' + '---+' * self.size

        print()

        # Column and diagonal states
        if stat:

            print('    ', end='')

            print(
                '%d' % self.__judge[
                    self.Direction.DIAGONAL
                ][self.Diagonal.TL2BR],
                end=''
            )

            for x in range(self.size):

                print(
                    '   %d' % self.__judge[
                        self.Direction.COLUMN
                    ][x],
                    end=''
                )

            print(
                '   %d' % self.__judge[
                    self.Direction.DIAGONAL
                ][self.Diagonal.TR2BL]
            )

        # Board
        for y in range(self.size):

            print(hdiv)
            print('    |', end='')

            for x in range(self.size):

                mark = self.Mark[
                    self.__board[x + y * self.size]
                ]

                print(f' {mark} |', end='')

            # Row state
            if stat:

                print(
                    ' %d' % self.__judge[
                        self.Direction.ROW
                    ][y],
                    end=''
                )

            print()

        print(hdiv)
        print()


# ==================================
# Main
# ==================================

if __name__ == '__main__':

    # デモ: ランダムに打って 1 ゲーム進め、盤面を表示する。終わったら何目並べかを聞かれる (3 未満で終了)
    n = 3

    while n >= 3:

        game = TicTacToe(n)

        nn = n ** 2

        for i in range(nn):

            # Find vacant cells
            empty = [
                idx
                for idx, val in enumerate(game.board)
                if val == 0
            ]

            if not empty:
                break

            # Random play
            c = choice(empty)

            player = game.next

            print(
                f'#{i + 1}: '
                f'{TicTacToe.Mark[player]} = {c + 1}'
            )

            game.play(player, c)

            game.print_board(stat=True)

            if game.state == game.GameState.DRAW:

                print('Result: DRAW')
                break

            elif game.state == game.GameState.OVER:

                print(
                    'Result: Player %d (%.1s) WON'
                    % (
                        game.winner,
                        TicTacToe.Mark[game.winner]
                    )
                )

                break

        # Next game
        while True:

            try:

                n = int(
                    input(
                        '\n次は何目並べにしますか?（3未満で終了）: '
                    )
                )

                break

            except ValueError:

                print('整数を入力してください。')