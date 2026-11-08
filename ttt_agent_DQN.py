#!/usr/bin/env python3

# TicTacToe 用の DQN (Deep Q-Network) エージェント。
#
#   表形式の Q 学習 (ttt_agent_QL.py) の「Q テーブル」を、盤面 -> 各マスの Q 値 を出すニューラルネットワークに
#   置き換えたもの。ネットワークは自作の nn_layers / nn_loss / nn_optimizer / matrix_util だけで構成している
#   (NumPy や深層学習フレームワークは不使用)。盤面サイズ n (>= 3) は任意で、入出力のノード数は n*n。
#
#   構成要素
#     Q ネットワーク          入力: 盤面 (自分 +1 / 相手 -1 / 空き 0) -> 出力: 各マスの Q 値。
#                             自分視点に直して入れるので、先手でも後手でも同じネットワークで扱える。
#     経験再生 (replay)       (s, a, r, s', 終局か) を貯め、ランダムに取り出して学習する (連続する手の相関を弱める)。
#     ターゲットネットワーク  一定回数の更新ごとに Q ネットワークをコピーし、TD ターゲットの計算に使う (学習の安定化)。
#     ε-greedy               ランダムに打つ確率 ε を epsilon_start から epsilon_end へ線形に下げる。
#     損失 Huber / 最適化 Adam
#     合法手マスク            埋まっているマスは、行動選択でも max Q の計算でも除外する。
#
#   エージェントの規約 (TTTAgent)
#     action(board, player_id)            : 打つマスを返す
#     feedback(board, player_id, reward)  : 自分の手の結果を受け取って学習する
#                                           (ゲーム継続中は reward=None、終局時は 勝ち +1 / 負け -1 / 引き分け 0)
#     trainable=False にすると ε=0 の貪欲な行動になり、学習もしない (評価用)。
#
#   保存と復元
#     save(path)                  : ウェイトと、path + '.hyper.json' にハイパーパラメーター・学習の進み具合を保存
#     TTTAgentDQN.from_file(path) : 保存したファイルから、同じ設定のエージェントを作る
#     load(path)                  : 既存のエージェントに読み込む (ネットワークの形が同じであること)
#     ※ 経験再生バッファー・Adam の内部状態・乱数の状態は保存しない (再開後の学習はビット単位では再現されない)

import json
import os
import warnings
from collections import deque
from random import random, randint, sample

import nn_layers
import nn_loss
import nn_optimizer
from ttt_agent import TTTAgent


class TTTAgentDQN(TTTAgent):
    """
    主なハイパーパラメーター:
        hidden_nodes           隠れ層のノード数 (タプル)         learning_rate     Adam の学習率
        discount_factor        割引率 γ                          batch_size        1 回の更新に使う経験の数
        epsilon_start / _end   ε の初期値 / 下限                 memory_size       経験再生バッファーの容量
        epsilon_decay_episodes ε を下限まで下げる対戦数         warmup            学習を始める経験の数
        target_update_interval ターゲット同期の間隔 (更新回数)   train_interval    何手ごとに更新するか
        encoding               盤面の入力形式 ('signed': +1/0/-1、'planes': 自分・相手の 2 面)
    """

    def __new__(
        cls,
        n: int = 3,
        trainable: bool = True,
        hidden_nodes=(32,),
        learning_rate: float = 0.001,
        discount_factor: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,
        batch_size: int = 32,
        memory_size: int = 10000,
        warmup: int = 200,
        train_interval: int = 1,
        target_update_interval: int = 200,
        optimizer: str = 'adam',
        loss: str = 'huber',
        encoding: str = 'signed',
        precision: str = 'd',
        filepath=None
    ):
        cls._validate_hyperparameters(
            hidden_nodes, learning_rate, discount_factor, epsilon_start, epsilon_end,
            epsilon_decay_episodes, batch_size, memory_size, warmup, train_interval,
            target_update_interval, encoding
        )

        if filepath and not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        return super(TTTAgentDQN, cls).__new__(cls, n, trainable)

    @staticmethod
    def _validate_hyperparameters(hidden_nodes, learning_rate, discount_factor, epsilon_start, epsilon_end,
                                  epsilon_decay_episodes, batch_size, memory_size, warmup, train_interval,
                                  target_update_interval, encoding):
        """ハイパーパラメーターを検証する (コンストラクターと load() で共通)。"""
        if not isinstance(hidden_nodes, (list, tuple)) or len(hidden_nodes) < 1 \
                or any(type(h) != int or h < 1 for h in hidden_nodes):
            raise ValueError('hidden_nodes MUST BE a list/tuple of integers greater than 0')

        if learning_rate <= 0.0:
            raise ValueError('learning_rate MUST BE greater than 0.0')

        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError('discount_factor MUST BE between 0.0 and 1.0')

        if not (0.0 <= epsilon_end <= epsilon_start <= 1.0):
            raise ValueError('0.0 <= epsilon_end <= epsilon_start <= 1.0 MUST BE satisfied')

        if type(epsilon_decay_episodes) != int or epsilon_decay_episodes < 1:
            raise ValueError('epsilon_decay_episodes MUST BE an integer greater than 0')

        for name, value in (('batch_size', batch_size),
                            ('memory_size', memory_size),
                            ('train_interval', train_interval),
                            ('target_update_interval', target_update_interval)):
            if type(value) != int or value < 1:
                raise ValueError(f'{name} MUST BE an integer greater than 0')

        if type(warmup) != int or warmup < 0:
            raise ValueError('warmup MUST BE an integer greater than or equal to 0')

        if memory_size < batch_size:
            raise ValueError('memory_size MUST BE greater than or equal to batch_size')

        if encoding not in ('signed', 'planes'):
            raise ValueError("encoding MUST BE 'signed' or 'planes'")

    def __init__(
        self,
        n: int = 3,
        trainable: bool = True,
        hidden_nodes=(32,),
        learning_rate: float = 0.001,
        discount_factor: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,
        batch_size: int = 32,
        memory_size: int = 10000,
        warmup: int = 200,
        train_interval: int = 1,
        target_update_interval: int = 200,
        optimizer: str = 'adam',
        loss: str = 'huber',
        encoding: str = 'signed',
        precision: str = 'd',
        filepath=None
    ):
        super().__init__(n, trainable)

        # ---- ハイパーパラメーター ----
        self.__hidden_nodes = tuple(hidden_nodes)
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__epsilon_start = epsilon_start
        self.__epsilon_end = epsilon_end
        self.__epsilon_decay_episodes = epsilon_decay_episodes
        self.__batch_size = batch_size
        self.__memory_size = memory_size
        self.__warmup = max(warmup, batch_size)
        self.__train_interval = train_interval
        self.__target_update_interval = target_update_interval
        self.__optimizer_name = optimizer
        self.__loss_name = loss
        self.__encoding = encoding
        self.__precision = precision

        # ---- ネットワーク ----
        nn = n ** 2
        in_nodes = nn if encoding == 'signed' else 2 * nn
        node_counts = [in_nodes] + list(self.__hidden_nodes) + [nn]
        activators = ['relu'] * len(self.__hidden_nodes) + ['linear']

        # 学習する Q ネットワークと、TD ターゲット計算用のターゲットネットワーク
        self.__net = nn_layers.Layers(node_counts, activators, precision, 'he')
        self.__target = nn_layers.Layers(node_counts, activators, precision, 'he')
        self.__target.copy_from(self.__net)

        self.__loss = nn_loss.get_instance(loss, reduction=nn_loss.Reduction.AVERAGE)
        self.__optimizer = nn_optimizer.get_instance(optimizer, learning_rate=learning_rate)

        # ---- 学習の状態 ----
        self.__memory = deque(maxlen=memory_size)   # 経験再生バッファー
        self.__last_state_and_action = None
        self.__step_count = 0       # feedback() で経験を受け取った回数
        self.__match_count = 0      # 終局まで進んだ対戦数 (学習中のみ数える)
        self.__update_count = 0     # ネットワークを更新した回数
        self.__last_loss = None

        if filepath:
            self.load(filepath)

    # ------------------------------------------------------------------
    #  プロパティ (読み取り専用)
    # ------------------------------------------------------------------

    @property
    def hyperparameters(self):
        return {
            'n': self.size,
            'hidden_nodes': self.__hidden_nodes,
            'learning_rate': self.__learning_rate,
            'discount_factor': self.__discount_factor,
            'epsilon_start': self.__epsilon_start,
            'epsilon_end': self.__epsilon_end,
            'epsilon_decay_episodes': self.__epsilon_decay_episodes,
            'batch_size': self.__batch_size,
            'memory_size': self.__memory_size,
            'warmup': self.__warmup,
            'train_interval': self.__train_interval,
            'target_update_interval': self.__target_update_interval,
            'optimizer': self.__optimizer_name,
            'loss': self.__loss_name,
            'encoding': self.__encoding,
            'precision': self.__precision,
        }

    @property
    def epsilon(self):
        """現在の ε (ランダムに打つ確率)。学習しない (trainable=False) 時は 0。"""
        if not self.trainable:
            return 0.0

        r = min(1.0, self.__match_count / self.__epsilon_decay_episodes)

        return self.__epsilon_start + (self.__epsilon_end - self.__epsilon_start) * r

    @property
    def match_count(self):
        return self.__match_count

    @property
    def update_count(self):
        return self.__update_count

    @property
    def step_count(self):
        return self.__step_count

    @property
    def last_loss(self):
        return self.__last_loss

    @property
    def memory_length(self):
        return len(self.__memory)

    @property
    def last_state_and_action(self):
        return self.__last_state_and_action

    # ------------------------------------------------------------------
    #  盤面の符号化
    # ------------------------------------------------------------------

    def _encode(self, board_state: list, player_id: int) -> list:
        """盤面を、自分視点のネットワーク入力 (list of float) に変換する。"""

        # self._regulate(): 自分 = 1, 空き = 0, 相手 = -1
        b = self._regulate(board_state, player_id)

        if self.__encoding == 'signed':
            return [float(v) for v in b]

        # 'planes': [自分の石があるか] + [相手の石があるか]
        return [1.0 if v == 1 else 0.0 for v in b] + [1.0 if v == -1 else 0.0 for v in b]

    # ------------------------------------------------------------------
    #  行動選択
    # ------------------------------------------------------------------

    def action(self, board_state: list, player_id: int) -> int:

        empty_cells = [i for i, v in enumerate(board_state) if v == 0]

        if len(empty_cells) == 0:
            raise NotImplementedError("no space left!!")

        state = self._encode(board_state, player_id)

        if self.trainable and random() < self.epsilon:
            # ε-greedy: ランダムに打つ
            action = empty_cells[randint(0, len(empty_cells) - 1)]

        else:
            # 貪欲: 空きマスの中で Q 値が最大のマス (同値ならランダム)
            q = self.__net.fprop([state])[0]
            qmax = max(q[i] for i in empty_cells)
            best = [i for i in empty_cells if q[i] == qmax]
            action = best[randint(0, len(best) - 1)]

        self.__last_state_and_action = (state, action)

        return action

    # ------------------------------------------------------------------
    #  学習
    # ------------------------------------------------------------------

    def feedback(self, board_state: list, player_id: int, reward: float):

        if not self.trainable:
            self.__last_state_and_action = None
            return

        if not self.__last_state_and_action:
            return

        s0, a = self.__last_state_and_action
        self.__last_state_and_action = None

        done = reward is not None
        r = float(reward) if done else 0.0
        s1 = self._encode(board_state, player_id)

        # 終了していなければ、次に打てるマスを覚えておく (max Q の計算を空きマスだけに制限するため)
        legal1 = () if done else tuple(i for i, v in enumerate(board_state) if v == 0)

        self.__memory.append((s0, a, r, s1, done, legal1))
        self.__step_count += 1

        if done:
            self.__match_count += 1

        if len(self.__memory) >= self.__warmup and self.__step_count % self.__train_interval == 0:
            self._learn()

    def _learn(self):
        """経験再生バッファーからミニバッチを取り出し、Q ネットワークを 1 回更新する。"""

        batch = sample(self.__memory, self.__batch_size)
        gamma = self.__discount_factor

        states = [b[0] for b in batch]
        next_states = [b[3] for b in batch]

        q_pred = self.__net.fprop(states)               # B x n**2
        q_next = self.__target.fprop(next_states)       # B x n**2 (ターゲットネットワーク)

        # 教師信号: 選んだ行動の Q だけ TD ターゲットに置き換え、他は予測値のまま (= 誤差 0)
        #   y = r                              (終局)
        #   y = r + γ * max_{a' が空き} Q'(s', a')   (継続)
        y_true = [list(row) for row in q_pred]

        for i, (s0, a, r, s1, done, legal1) in enumerate(batch):
            if done or not legal1:
                y = r
            else:
                y = r + gamma * max(q_next[i][j] for j in legal1)

            y_true[i][a] = y

        loss = self.__loss(y_true, q_pred)
        self.__net.bprop(self.__loss.gradient, input_gradient=False)

        for layer in self.__net.layers:
            self.__optimizer.apply_gradients(
                layer.get_raw_weight(), layer.get_raw_bias(), layer.dw, layer.db
            )

        self.__last_loss = loss
        self.__update_count += 1

        if self.__update_count % self.__target_update_interval == 0:
            self.__target.copy_from(self.__net)

    # ------------------------------------------------------------------
    #  デバッグ・保存
    # ------------------------------------------------------------------

    def q_values(self, board_state: list, player_id: int) -> list:
        """現在の盤面に対する各マスの Q 値を返す (確認用)。"""
        return self.__net.fprop([self._encode(board_state, player_id)])[0]

    # ハイパーパラメーターのうち、ネットワークの形を決めるもの (保存時と同じでなければ読み込めない)
    _STRUCTURE_KEYS = ('n', 'hidden_nodes', 'encoding', 'precision')

    # 読み込み時に復元するもの
    _RESTORE_KEYS = ('learning_rate', 'discount_factor', 'epsilon_start', 'epsilon_end', 'epsilon_decay_episodes',
                     'batch_size', 'memory_size', 'warmup', 'train_interval', 'target_update_interval',
                     'optimizer', 'loss')

    @staticmethod
    def _hyper_path(filepath: str) -> str:
        return filepath + '.hyper.json'

    def save(self, filepath: str):
        """
        ウェイトを filepath に、ハイパーパラメーターと学習の進み具合を filepath + '.hyper.json' に保存する。
          保存するもの : ウェイト, ハイパーパラメーター, match_count, update_count, step_count
          保存しないもの: 経験再生バッファー, Adam の内部状態 (勢い・二乗平均), 乱数の状態
                         -> 読み込み後に学習を再開すると、ここまでの学習とビット単位で同じ結果にはならない。
        """
        self.__net.save(filepath)

        with open(self._hyper_path(filepath), 'w', encoding='utf-8') as f:
            json.dump({**self.hyperparameters,
                       'match_count': self.__match_count,
                       'update_count': self.__update_count,
                       'step_count': self.__step_count}, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str, restore_state: bool = True) -> bool:
        """
        save() で保存した内容を読み込む。

        restore_state=True (既定) のとき、ウェイトに加えて次を復元する。
          * ハイパーパラメーター (学習率, 割引率, ε, バッチ, 経験再生の容量, 更新間隔, 最適化手法 など)
          * 学習回数 match_count (ε の減少の進み具合もこれで決まる)、更新回数 update_count、step_count
        ネットワークの形 (n, hidden_nodes, encoding, precision) は、このエージェントと保存時で同じであること
        (違えば ValueError)。形ごと復元したいときは from_file() を使う。
        経験再生バッファーは空に戻り、Adam の内部状態も初期化される。

        Returns:
            ハイパーパラメーターと学習回数も復元できたら True。
            .hyper.json が無い (ウェイトだけの) 場合は警告を出してウェイトだけを読み込み、False を返す。
        """
        saved = None

        if restore_state:
            hyper_path = self._hyper_path(filepath)

            if os.path.exists(hyper_path):
                with open(hyper_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)

                self._check_structure(saved)
                hp = self._merged_hyperparameters(saved)       # ここで値を検証 (失敗しても何も変更しない)
                counts = self._read_counts(saved)
            else:
                warnings.warn(f'{hyper_path} が見つからないため、ウェイトだけを読み込みます。')

        self.__net.load(filepath)
        self.__target.copy_from(self.__net)
        self.__last_state_and_action = None

        if saved is None:
            return False

        self.__apply_hyperparameters(hp)
        self.__match_count, self.__update_count, self.__step_count = counts

        return True

    @classmethod
    def from_file(cls, filepath: str, trainable: bool = True):
        """save() で保存したファイルから、ネットワークの形も含めて同じ設定のエージェントを作る。"""
        hyper_path = cls._hyper_path(filepath)

        for path in (filepath, hyper_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        with open(hyper_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)

        kwargs = {k: saved[k] for k in (*cls._STRUCTURE_KEYS, *cls._RESTORE_KEYS) if k in saved}
        kwargs['hidden_nodes'] = tuple(kwargs['hidden_nodes'])

        agent = cls(trainable=trainable, **kwargs)
        agent.load(filepath)

        return agent

    # ---- load() の下請け ----

    def _check_structure(self, saved: dict):
        current = self.hyperparameters

        for key in self._STRUCTURE_KEYS:
            if key not in saved:
                continue

            value = tuple(saved[key]) if key == 'hidden_nodes' else saved[key]

            if value != current[key]:
                raise ValueError(f"'{key}' mismatch. file: {saved[key]}, this agent: {current[key]}")

    def _merged_hyperparameters(self, saved: dict) -> dict:
        hp = self.hyperparameters

        for key in self._RESTORE_KEYS:
            if key in saved:
                hp[key] = saved[key]

        self._validate_hyperparameters(
            hp['hidden_nodes'], hp['learning_rate'], hp['discount_factor'], hp['epsilon_start'], hp['epsilon_end'],
            hp['epsilon_decay_episodes'], hp['batch_size'], hp['memory_size'], hp['warmup'], hp['train_interval'],
            hp['target_update_interval'], hp['encoding']
        )

        return hp

    @staticmethod
    def _read_counts(saved: dict):
        counts = []

        for key in ('match_count', 'update_count', 'step_count'):
            value = saved.get(key, 0)

            if type(value) != int or value < 0:
                raise ValueError(f'{key} MUST BE an integer greater than or equal to 0')

            counts.append(value)

        return tuple(counts)

    def __apply_hyperparameters(self, hp: dict):
        self.__learning_rate = hp['learning_rate']
        self.__discount_factor = hp['discount_factor']
        self.__epsilon_start = hp['epsilon_start']
        self.__epsilon_end = hp['epsilon_end']
        self.__epsilon_decay_episodes = hp['epsilon_decay_episodes']
        self.__batch_size = hp['batch_size']
        self.__memory_size = hp['memory_size']
        self.__warmup = hp['warmup']
        self.__train_interval = hp['train_interval']
        self.__target_update_interval = hp['target_update_interval']
        self.__optimizer_name = hp['optimizer']
        self.__loss_name = hp['loss']

        # 容量が変わったときは、中身を保ったまま作り直す。最適化手法・損失関数は新しい設定で作り直す
        self.__memory = deque(self.__memory, maxlen=self.__memory_size)
        self.__loss = nn_loss.get_instance(self.__loss_name, reduction=nn_loss.Reduction.AVERAGE)
        self.__optimizer = nn_optimizer.get_instance(self.__optimizer_name, learning_rate=self.__learning_rate)
