#!/usr/bin/env python3

# 表形式の Q 学習エージェント (DQN の比較対象)。
#   状態 = 盤面を 3 進数で表した整数、Q テーブル = [状態][マス] の表。 ε-greedy で行動を選び、
#   feedback() で Q(s,a) += α (r + γ max Q(s',・) - Q(s,a)) と更新する。
#   盤面ごとに表の行が要るので、n が大きくなるとテーブルが急に大きくなる (DQN はこれをネットワークで近似する)。

from random import random, randint
from ttt_agent import TTTAgent

import sys
import os
from datetime import datetime
from array import array


class TTTAgentQL(TTTAgent):
    """
    learning_rate: 学習率 α / discount_factor: 割引率 γ / greediness: ランダムに打つ確率 ε
    set_termimnal_q_to_reward: 終局の状態の Q を報酬にする / update_per_episode: 1 手ごとでなく終局後にまとめて更新
    filepath: 保存済みのエージェントを読み込む
    """

    def __new__(
        cls,
        n: int = 3,
        trainable: bool = True,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        greediness: float = 0.1,
        set_termimnal_q_to_reward: bool = True,
        update_per_episode: bool = False,
        filepath=None
    ):
        if learning_rate < 0.0 or learning_rate > 1.0:
            raise ValueError('alpha MUST BE between 0.0 and 1.0')

        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError('discount_factor MUST BE between 0.0 and 1.0')

        if greediness < 0.0 or greediness > 1.0:
            raise ValueError('greediness MUST BE between 0.0 and 1.0')

        if filepath and not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        return super(TTTAgentQL, cls).__new__(cls, n, trainable)

    def __init__(
        self,
        n: int = 3,
        trainable: bool = True,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        greediness: float = 0.1,
        set_termimnal_q_to_reward: bool = True,
        update_per_episode: bool = False,
        filepath=None
    ):
        super().__init__(n, trainable)

        if filepath:
            self.__file = filepath
            self.load(filepath)

        else:
            self.__learning_rate = learning_rate
            self.__discount_factor = discount_factor
            self.__greediness = greediness
            self.__r2q = set_termimnal_q_to_reward
            self.__upe = update_per_episode

            self.__last_state_and_action = None
            self.__episode = []
            self.__update_count = 0
            self.__match_count = 0

            # Q table array
            self.__q_table = []

            nn = n**2
            p2 = nn >> 1
            p1 = nn - p2

            # 盤面コードの上限。全部で 3**nn 通りだが、石の数の制約を使って配列を少し小さくしている
            bmax = sum([2 * 3**n for n in range(nn - 1, p1 - 2, -1)])
            bmax += sum([3**n for n in range(p1)])

            self.__q_table = [[0.0 for _ in range(nn)] for _ in range(bmax)]

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def discount_factor(self):
        return self.__discount_factor

    @discount_factor.setter
    def discount_factor(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def greediness(self):
        return self.__greediness

    @greediness.setter
    def greediness(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def last_state_and_action(self):
        return self.__last_state_and_action

    @last_state_and_action.setter
    def last_state_and_action(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def set_termimnal_q_to_reward(self):
        return self.__r2q

    @set_termimnal_q_to_reward.setter
    def set_termimnal_q_to_reward(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def update_per_episode(self):
        return self.__upe

    @update_per_episode.setter
    def update_per_episode(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def episode(self):
        return self.__episode

    @episode.setter
    def episode(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    @property
    def q_table(self):
        return self.__q_table

    @q_table.setter
    def q_table(self, *args):
        raise PermissionError('sorry... you can\'t change...')

    def action(self, board_state: list, player_id: int) -> int:

        empty_cells = [i for i, v in enumerate(board_state) if v == 0]

        if len(empty_cells) == 0:
            raise NotImplementedError("no space left!!")

        state = self._board2state(board_state, player_id)

        # ε-greedy: 確率 greediness でランダム、それ以外は Q 値が最大のマス
        if random() < self.__greediness:
            action = empty_cells[randint(0, len(empty_cells) - 1)]

        else:
            qall = [self.q_table[state][i] for i in empty_cells]
            qmax = max(qall)

            if qmax == 0.0:
                # 未学習なので何でもいい
                action = empty_cells[randint(0, len(empty_cells) - 1)]

            else:
                # Q値による選択
                action = empty_cells[qall.index(qmax)]

        self.__last_state_and_action = (state, action)

        return action

    def feedback(self, board_state: list, player_id: int, reward: float):

        if not self.trainable:
            self.__last_state_and_action = None
            return

        if not self.__last_state_and_action:
            return

        s0, a = self.__last_state_and_action
        self.__last_state_and_action = None
        s1 = self._board2state(board_state, player_id)

        if not self.__upe:
            # 行動選択毎に更新

            if self.__r2q:
                # エピソード終端のQを報酬に設定
                if reward is not None:
                    self.__match_count += 1

                    for i in range(self.size * self.size):
                        self.__q_table[s1][i] = reward

                    self.__update_count += 1

            qs0a = self.__q_table[s0][a]
            qs1max = max(self.__q_table[s1])

            reward = 0 if not reward else reward

            self.__q_table[s0][a] += self.__learning_rate * (
                reward + self.__discount_factor * qs1max - qs0a
            )

            self.__update_count += 1

        else:
            # エピソード単位の更新
            self.__episode.append([s0, a, s1, reward])

            if reward is not None:
                self.__match_count += 1

                if self.__r2q:
                    # エピソード終端のQを報酬に設定
                    for i in range(self.size * self.size):
                        self.__q_table[s1][i] = reward

                    self.__update_count += 1

                for s0, a, s1, r in reversed(self.__episode):
                    qs0a = self.__q_table[s0][a]
                    qs1max = max(self.__q_table[s1])

                    r = 0 if r is None else r

                    self.__q_table[s0][a] += self.__learning_rate * (
                        r + self.__discount_factor * qs1max - qs0a
                    )

                    self.__update_count += 1

                self.__episode.clear()

    def _board2state(self, board_state: list, player_id: int) -> int:
        # 盤面を 3 進数として読んだ整数 (player_id は使わず、盤面の値 1: 先手 / 2: 後手 をそのまま使う)

        b = board_state
        s = 0

        for i in range(len(b)):
            s += b[i] * 3**i

        return s

    # ---- 学習結果 (Q テーブル) のファイル保存・読み込み ----

    def save(self, filepath: str = '', comment: str = 'None'):
        """
        Qテーブルやパラメーター等をファイルに保存する。
        """

        # デフォルトファイル名
        dt = datetime.now()

        default_filename = (
            f'Q-table_{self.size}x{self.size}-'
            f'{self.__match_count}-'
            f'{dt.strftime("%Y%m%d_%H%M%S.%f")}.qtab'
        )

        if filepath:
            dpath = os.path.dirname(filepath)
            fname = os.path.basename(filepath)
        else:
            dpath = ''
            fname = ''

        # ディレクトリーの有無を確認
        if dpath and not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)

        # ファイルパス
        filepath = dpath if dpath else ''
        filepath += '/' + fname if fname else default_filename

        # ファイルの有無の確認と調整
        base, ext = filepath.rsplit('.', 1)

        i = 0
        suffix = ''

        while os.path.exists(base + suffix + '.' + ext):
            i += 1
            suffix = f'-{i}'

        filepath = base + suffix + '.' + ext

        # ファイルに書き込む
        byteorder = 0 if sys.byteorder[0] == 'l' else 1
        precision = 1

        spec = array('B', [byteorder << 7 | precision << 6])
        tstamp = array('d', [dt.timestamp()])
        size = array('B', [self.size])
        lrate = array('d', [self.__learning_rate])
        drate = array('d', [self.__discount_factor])
        greed = array('d', [self.__greediness])
        lopt = array('B', [self.__r2q << 7 | self.__upe << 6])
        mcount = array('Q', [self.__match_count])
        ucount = array('Q', [self.__update_count])

        if comment:
            comm = array('u', [c for c in comment]).tobytes()
            coml = len(comm)

            while coml > 256:
                comment = comment[:-1]
                comm = array('u', [c for c in comment]).tobytes()
                coml = len(comm)

            coml -= 1

        bytes_wriiten = 0

        with open(filepath, 'wb') as f:
            bytes_wriiten += f.write(b'QTAB')
            bytes_wriiten += f.write(b'SPEC' + spec.tobytes())

            bytes_wriiten += f.write(b'TIME' + tstamp.tobytes())
            bytes_wriiten += f.write(b'SIZE' + size.tobytes())
            bytes_wriiten += f.write(b'LrnR' + lrate.tobytes())
            bytes_wriiten += f.write(b'DisF' + drate.tobytes())
            bytes_wriiten += f.write(b'Grdy' + greed.tobytes())
            bytes_wriiten += f.write(b'Optn' + lopt.tobytes())
            bytes_wriiten += f.write(b'MCNT' + mcount.tobytes())
            bytes_wriiten += f.write(b'UCNT' + ucount.tobytes())

            if comment:
                bytes_wriiten += f.write(b'COMM')
                bytes_wriiten += f.write(array('B', [coml]))
                bytes_wriiten += f.write(comm)

            # Q-Table
            bytes_wriiten += f.write(b'QTab')

            for qrow in self.__q_table:
                bytes_wriiten += f.write(array('d', qrow).tobytes())

        return bytes_wriiten

    def load(self, filepath: str):
        """
        Qテーブルやパラメーター等をファイルから読み込む。
        """

        fdate = None
        comm = ''
        fltfmt = None

        byteorder = 0 if sys.byteorder[0] == 'l' else 1
        swapbytes = False
        precision = 0

        n = 0
        learning_rate = 0
        discount_factor = 0
        greediness = 0
        r2q = False
        upe = False
        match_count = 0
        update_count = 0
        q_table = []

        with open(filepath, 'rb') as f:

            # ファイル読み込み関数
            def readbytes(typecode, num):
                b = array(typecode)

                try:
                    b.fromfile(f, num)

                except EOFError:
                    return

                except Exception as e:
                    raise e

                return b

            # ファイルヘッダー確認
            file_header = readbytes('B', 4)

            if file_header.tobytes() != b'QTAB':
                raise RuntimeError('フォーマットが違います。')

            # 残りを読み込む
            while True:

                chunk_header = readbytes('B', 4)

                if not chunk_header:
                    # EOF
                    break

                elif chunk_header.tobytes() == b'SPEC':
                    spec = readbytes('B', 1)[0]

                    fbyteorder = spec >> 7
                    precision = spec >> 6 & 1

                    if fbyteorder != byteorder:
                        swapbytes = True

                    fltfmt = (precision, fbyteorder)

                elif chunk_header.tobytes() == b'TIME':
                    timestamp = readbytes('d', 1)[0]

                    if swapbytes:
                        timestamp.byteswap()

                    fdate = datetime.fromtimestamp(timestamp)

                elif chunk_header.tobytes() == b'SIZE':
                    n = readbytes('B', 1)[0]

                elif chunk_header.tobytes() == b'LrnR':
                    v = readbytes('d', 1) if precision else readbytes('f', 1)

                    if swapbytes:
                        v.byteswap()

                    learning_rate = v[0]

                elif chunk_header.tobytes() == b'DisF':
                    v = readbytes('d', 1) if precision else readbytes('f', 1)

                    if swapbytes:
                        v.byteswap()

                    discount_factor = v[0]

                elif chunk_header.tobytes() == b'Grdy':
                    v = readbytes('d', 1) if precision else readbytes('f', 1)

                    if swapbytes:
                        v.byteswap()

                    greediness = v[0]

                elif chunk_header.tobytes() == b'Optn':
                    lopt = readbytes('B', 1)[0]

                    r2q = lopt >> 7
                    upe = lopt >> 6 & 1

                elif chunk_header.tobytes() == b'MCNT':
                    mcount = readbytes('Q', 1)

                    if swapbytes:
                        mcount.byteswap()

                    match_count = mcount[0]

                elif chunk_header.tobytes() == b'UCNT':
                    ucount = readbytes('Q', 1)

                    if swapbytes:
                        ucount.byteswap()

                    update_count = ucount[0]

                elif chunk_header.tobytes() == b'COMM':
                    coml = readbytes('B', 1)[0] + 1
                    comb = readbytes('B', coml)

                    comm = array('u')
                    comm.frombytes(comb.tobytes())
                    comm = ''.join(comm)

                elif chunk_header.tobytes() == b'QTab':

                    # __init__() と同じ
                    nn = n**2
                    p2 = nn >> 1
                    p1 = nn - p2

                    bmax = sum([
                        2 * 3**n for n in range(nn - 1, p1 - 2, -1)
                    ])
                    bmax += sum([3**n for n in range(p1)])

                    typecode = 'd' if precision else 'f'

                    if swapbytes:
                        for _ in range(bmax):
                            qrow = readbytes(typecode, nn)
                            qrow.byteswap()
                            q_table.append(qrow.tolist())

                    else:
                        for _ in range(bmax):
                            qrow = readbytes(typecode, nn)
                            q_table.append(qrow.tolist())

                else:
                    print(
                        f'未知のデータが見つかりました。'
                        f'{chunk_header.tobytes()}'
                    )

        print(f'　　ファイル: {filepath}')

        if fdate:
            print(
                f'　　作成日時: '
                f'{fdate.strftime("%Y/%m/%d %H:%M:%S.%f")}'
            )

        if fltfmt:
            print(
                f'フォーマット: '
                f'{["32bit", "64bit"][fltfmt[0]]} float '
                f'({["Little", "Big"][fltfmt[1]]} Endian)'
            )

        if comm:
            print(f'　　コメント: {comm}')

        if match_count:
            print(f'　　対戦回数: {match_count}')

        if update_count:
            print(f'　　更新回数: {update_count}')

        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__greediness = greediness
        self.__r2q = r2q
        self.__upe = upe
        self.__match_count = match_count
        self.__update_count = update_count
        self.__q_table = q_table