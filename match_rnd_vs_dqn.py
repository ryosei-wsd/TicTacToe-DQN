#!/usr/bin/env python3

# ======================================================================
#  ランダムエージェントと DQN エージェントの対戦実験 (TicTacToe)
#
#  ■ 何をするスクリプトか
#    未学習の DQN (ttt_agent_DQN.py) を、ランダムに打つ相手と対戦させながら学習させ、
#    勝ち・引き分け・負けの割合がどう推移するかを記録する。ルールごとに DQN を作り直して 1 から学習する。
#    記録するのは次の 2 つ (どちらも dqn_results/ に CSV とグラフで保存する)。
#      * 学習中の成績      : ε-greedy でランダムな手も混ざる、実際に対戦した結果
#      * 学習を止めた成績  : 途中経過の DQN を ε=0 の貪欲な手だけで、先手・後手それぞれ対戦させた結果 (実力の指標)
#    n=3 のときは、全盤面の探索で求めた「対ランダムの理論上の最適成績」も参考に出す。
#
#  ■ 使い方
#      python3 match_rnd_vs_dqn.py            # 端末で「何目並べか」「試行回数」「ルール」を質問される (Enter で既定値: 3, 5000, 1 2 3)
#      python3 match_rnd_vs_dqn.py N M        # N 目並べ (盤面 N x N)、各ルール M 局 (指定した項目は質問されない)
#      python3 match_rnd_vs_dqn.py --help     # 他のオプション (--rules, --seed, --hp KEY=VALUE, --no-prompt ...)
#      python3 tune_dqn.py                    # ハイパーパラメーター調整 (約 20 分。同様に端末で入力できる)
#    matplotlib が無ければグラフは作らない。それ以外は標準ライブラリと自作モジュールだけで動く。
#
#  ■ 関連ファイル
#      ttt_agent_DQN.py     DQN エージェント           tune_dqn.py       ハイパーパラメーター調整
#      experiment_util.py   シード管理・端末入力       tictactoe.py      ゲーム環境
#      ttt_agent.py / ttt_agent_random.py / ttt_agent_QL.py   エージェントの共通部分 / ランダム / 表形式 Q 学習
#      nn_layers.py / nn_loss.py / nn_optimizer.py / nn_activator.py / matrix_util.py   自作のニューラルネットワーク
#    (python3 nn_layers.py を実行すると、逆伝播の勾配を数値微分と比べる勾配チェックも走る。全ての構成で一致を確認済み)
#
#  ■ 乱数シード (experiment_util.py で一元管理)
#      1 つのエージェントの学習を始める直前に set_seed(seed) を呼ぶ。seed = 基準シード*10000 + オフセット + 番号。
#      本スクリプトは seed_for(基準シード, 'rule', ルール番号) (基準 0 なら RULE 1, 2, 3 -> 1, 2, 3)、
#      tune_dqn.py は seed_for(基準シード, 'tune', 繰り返し番号) (基準 0 なら 1000, 1001, ...)。
#      条件 (ハイパーパラメーター) を比べるときは、条件が違っても同じ番号 = 同じシードを使う。
#      同じシード・同じ設定なら、同じ Python のバージョンで同じ結果になる。使ったシードは CSV の seed 列にも残る。
#
#  ■ 対戦ルール (盤面 3x3, 3 目並べ。報酬は 勝ち +1, 負け -1, 引き分け 0, 途中は 0)
#      RULE 1: DQN は常に後手        RULE 2: DQN は常に先手
#      RULE 3: 勝者が次の対戦で後手 (引き分けなら変更なし。最初は DQN が後手)
#
#  ■ ハイパーパラメーター (採用値。DEFAULT_HYPERPARAMETERS)
#      ネットワーク 入力 9 - 隠れ層 64 (ReLU) - 出力 9 (線形), He 初期化, 入力は自分視点 (自分 +1 / 相手 -1 / 空き 0)
#      最適化 Adam (学習率 0.001), 損失 Huber (delta=1), 割引率 0.9, バッチ 32, 経験再生 10000,
#      学習開始 200 経験後, 毎手 1 回更新, ターゲット同期 200 更新ごと,
#      ε: 1.0 -> 0.05 を 1000 局で線形に減少。置けないマスは行動選択・max Q から除外。
#
#  ■ 調整の経緯 (tune_dqn.py: RULE 1, 学習 2000 局, 3 シード, 先手・後手 各 1000 局の平均勝率 [平均±標準偏差])
#      基準 (隠れ層 32 ほか上記)           0.910±0.015
#      隠れ層 16 / 64 / 32-32              0.892 / 0.934 / 0.899   <- 64 が最良 (負け 4.0%, ただし実行時間 約 2 倍)
#      学習率 0.0003 / 0.003               0.861 / 0.893           (0.0003 は学習が遅い)
#      割引率 0.5 / 0.99                   0.904 / 0.904           (0.99 は途中経過が悪い)
#      ε 減少 300 / 2000 局                0.904 / 0.918
#      ターゲット同期 50 / 1000            0.915 / 0.910           (1000 は途中経過が悪い)
#      バッチ 16 / 64                      0.904 / 0.894
#      経験再生 1000                       0.892
#      入力 2 面 (自分/相手) 符号化        0.878
#      更新間隔 2 手                       0.882                   (速いが学習が遅い)
#    別シードでの確認 (3 シード): 基準 0.916±0.017, 隠れ層 64 0.927±0.013, 64 + ε 2000 局 0.921±0.012
#    -> 隠れ層 64 を採用。他の差は 3 シードでは誤差の範囲で、有意とは言えない。
#
#  ■ 結果 (基準シード 0, 5000 局/ルール。学習を止めた (ε=0) DQN と、相手 3000 局ずつの最終評価)
#                       DQN 先手 (勝/分/負)      DQN 後手 (勝/分/負)      平均勝率
#      RULE 1           0.979 / 0.006 / 0.014    0.890 / 0.041 / 0.068    0.935
#      RULE 2           0.964 / 0.011 / 0.025    0.781 / 0.029 / 0.191    0.872
#      RULE 3           0.976 / 0.010 / 0.014    0.890 / 0.036 / 0.073    0.933
#      参考: 理論上の最適方策 (対ランダム, 全盤面の探索で計算)
#                       先手 0.995 / 0.005 / 0.000   後手 0.934 / 0.056 / 0.010
#    学習中の割合の推移 (250 局ごとの区間)  学習中 W/D/L -> 貪欲な DQN (先手後手平均) W/D/L
#      RULE 1  250局 .27/.16/.56 -> .61/.06/.33   1000局 .77/.08/.16 -> .93/.02/.05   5000局 .86/.06/.08 -> .93/.03/.05
#      RULE 2  250局 .60/.12/.28 -> .54/.07/.39   1000局 .92/.04/.04 -> .90/.03/.08   5000局 .97/.02/.02 -> .86/.02/.12
#      RULE 3  250局 .45/.13/.42 -> .48/.04/.47   1000局 .78/.05/.17 -> .92/.04/.05   5000局 .89/.02/.09 -> .92/.03/.05
#
#  ■ 対戦回数について
#    成績 (先手後手の平均勝率) が最終値の ±0.03 以内に入り続ける最初の記録点は RULE 1: 1000 局, RULE 2: 2250 局,
#    RULE 3: 1000 局。その 2 倍以上の 5000 局を学習させ、以降に改善が無いことを確認した (基準設定で 12000 局まで
#    学習させても 2000 局以降は改善なし)。評価 3000 局での勝率の標準誤差は 0.5% 前後 (p=0.9 のとき)。
#
#  ■ 考察
#    * 勝率は 1000 局前後で急上昇し、その後は頭打ちになる。学習中の割合が貪欲な成績より低いのは ε=0.05 の
#      ランダムな手のため。 先手は最適値 (99.5%) に近い 96〜98%。後手は 89% で最適値 (93.4%) にあと一歩。
#    * RULE 2 は先手でしか学習しないため、後手の成績が低い (78%)。盤面を自分視点に正規化していても、
#      後手の局面を経験しないと転用しきれない。先手と後手の両方を経験する RULE 1, 3 の方が平均で高い。
#    * 後手は理論上でも負けが残るため、勝率の上限は先手より低い。
#    * 結果は乱数 (シード) とハードウェアで多少変わる (数値は Python 3.12.3 での 1 回の実行)。
#    * n=4 以上も動作するが (--hp hidden_nodes="(64,)" 等で調整可)、未調整・未収束で本結果には含めない。
# ======================================================================

import argparse
import ast
import csv
import os
import random
import sys
import time
from statistics import mean

import experiment_util as eu
from tictactoe import TicTacToe
from ttt_agent_random import TTTAgentRandom
from ttt_agent_DQN import TTTAgentDQN


# ----------------------------------------------------------------------
#  対戦ルール (match_rnd_vs_ql.py と同じ 3 つ)
# ----------------------------------------------------------------------
RULES = {
    1: 'DQN は常に後手',
    2: 'DQN は常に先手',
    3: '勝者が次の対戦で後手 (引き分けなら変更なし。最初は DQN が後手)',
}

# ----------------------------------------------------------------------
#  最終的に採用したハイパーパラメーター (n=3)
#  (TTTAgentDQN のコンストラクターの引数と同じ。調整の経緯は冒頭のコメントを参照)
# ----------------------------------------------------------------------
DEFAULT_HYPERPARAMETERS = {
    'hidden_nodes': (64,),
    'learning_rate': 0.001,
    'discount_factor': 0.9,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 1000,
    'batch_size': 32,
    'memory_size': 10000,
    'warmup': 200,
    'train_interval': 1,
    'target_update_interval': 200,
    'optimizer': 'adam',
    'loss': 'huber',
    'encoding': 'signed',
}


# ----------------------------------------------------------------------
#  1 局の対戦
# ----------------------------------------------------------------------
def play_game(n: int, dqn: TTTAgentDQN, opponent, dqn_first: bool):
    """
    1 局対戦する。DQN エージェントへのフィードバックは match_rnd_vs_ql.py と同じ方式。
      - 相手が打ってゲームが続くとき : feedback(盤面, id, None)
      - ゲーム終了時                 : feedback(盤面, id, +1 / -1 / 0)

    Returns:
        (結果, 勝者)  結果は DQN から見た 'W' (勝ち) / 'D' (引き分け) / 'L' (負け)。
        勝者は TicTacToe.Player (引き分けなら None)。
    """
    game = TicTacToe(n)
    dqn_id = 1 if dqn_first else 2
    result = None

    while game.state == TicTacToe.GameState.ONGOING:
        turn = game.next
        mover = dqn if turn == dqn_id else opponent

        game.play(turn, mover.action(game.board, turn))

        if game.state == TicTacToe.GameState.OVER:
            result = 'W' if game.winner == dqn_id else 'L'
            dqn.feedback(game.board, dqn_id, 1 if result == 'W' else -1)

        elif game.state == TicTacToe.GameState.DRAW:
            result = 'D'
            dqn.feedback(game.board, dqn_id, 0)

        elif mover is opponent:
            dqn.feedback(game.board, dqn_id, None)

    return result, game.winner


def evaluate(n: int, dqn: TTTAgentDQN, opponent, dqn_first: bool, games: int):
    """
    学習を止め (trainable=False: ε=0 の貪欲な行動、更新なし)、games 局対戦して
    (勝率, 引分率, 負け率) を返す。学習側の乱数列に影響しないよう、乱数の状態を保存・復元する。
    """
    trainable = dqn.trainable
    dqn.trainable = False
    rstate = random.getstate()

    count = {'W': 0, 'D': 0, 'L': 0}

    try:
        for _ in range(games):
            count[play_game(n, dqn, opponent, dqn_first)[0]] += 1
    finally:
        dqn.trainable = trainable
        random.setstate(rstate)

    return count['W'] / games, count['D'] / games, count['L'] / games


# ----------------------------------------------------------------------
#  ルールごとの先手・後手の決め方
# ----------------------------------------------------------------------
def initial_role(rule: int) -> bool:
    """最初の対戦で DQN が先手なら True。"""
    return rule == 2


def next_role(rule: int, dqn_first: bool, winner) -> bool:
    """次の対戦で DQN が先手なら True。"""
    if rule == 3 and winner == TicTacToe.Player.FIRST:
        return not dqn_first   # 先手が勝ったら入れ替える (= 勝者が後手になる)

    return dqn_first


# ----------------------------------------------------------------------
#  参考値: 3x3 で、ランダムな相手に対する理論上の最適方策の成績
#    DQN の手番では「期待値 (勝ち - 負け) が最大」の手を選び、相手の手番は全ての空きマスの平均を取る
#    (同点なら勝ちが多い方)。全盤面を調べるので n=3 のみ。
# ----------------------------------------------------------------------
def optimal_vs_random_3x3():
    """Returns: {'first': (勝, 分, 負), 'second': (勝, 分, 負)}"""
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    cache = {}

    def winner(b):
        for i, j, k in lines:
            if b[i] and b[i] == b[j] == b[k]:
                return b[i]
        return 0

    def value(b, turn, me):
        key = (b, turn, me)

        if key in cache:
            return cache[key]

        w = winner(b)

        if w:
            res = (1.0, 0.0, 0.0) if w == me else (0.0, 0.0, 1.0)
        elif 0 not in b:
            res = (0.0, 1.0, 0.0)
        else:
            cands = [value(b[:i] + (turn,) + b[i + 1:], 3 - turn, me) for i in range(9) if b[i] == 0]

            if turn == me:
                res = max(cands, key=lambda t: (t[0] - t[2], t[0]))
            else:
                res = tuple(sum(t[j] for t in cands) / len(cands) for j in range(3))

        cache[key] = res

        return res

    empty = (0,) * 9

    return {'first': value(empty, 1, 1), 'second': value(empty, 1, 2)}


# ----------------------------------------------------------------------
#  1 つのルールで、未学習の DQN を学習させながら対戦させる
# ----------------------------------------------------------------------
def run_rule(n: int, rule: int, games: int, hyper: dict = None, seed: int = 0,
             interval: int = 250, eval_games: int = 500, final_eval_games: int = 3000,
             verbose: bool = True):
    """
    Args:
        n                : 盤面サイズ (n >= 3)
        rule             : 対戦ルール (RULES 参照)
        games            : 学習中の対戦数
        hyper            : TTTAgentDQN に渡すハイパーパラメーター (省略時は DEFAULT_HYPERPARAMETERS)
        seed             : 乱数のシード (experiment_util.seed_for() で決めた値を渡す)
        interval         : この対戦数ごとに、(a) 学習中の勝ち・引分・負けの割合 と
                           (b) 学習を止めた貪欲な DQN の成績 (先手・後手それぞれ eval_games 局) を記録する
        final_eval_games : 学習終了後の最終評価 (先手・後手それぞれ) の対戦数
    Returns:
        dict: 'rows' (記録), 'final' (最終評価), 'agent', 'seconds', 'interrupted'
    """
    hp = dict(DEFAULT_HYPERPARAMETERS)
    if hyper:
        hp.update(hyper)

    eu.set_seed(seed)   # この実験 (1 つのエージェントの学習) の乱数をここで初期化する

    dqn = TTTAgentDQN(n, **hp)
    rnd = TTTAgentRandom(n)

    dqn_first = initial_role(rule)

    rows = []
    block = {'W': 0, 'D': 0, 'L': 0}
    block_first = 0
    interrupted = False
    t0 = time.time()
    played = 0

    try:
        for g in range(1, games + 1):
            first_now = dqn_first
            result, winner = play_game(n, dqn, rnd, dqn_first)
            dqn_first = next_role(rule, dqn_first, winner)

            played = g
            block[result] += 1
            block_first += first_now

            if g % interval == 0 or g == games:
                m = sum(block.values())

                ef = evaluate(n, dqn, rnd, True, eval_games)
                es = evaluate(n, dqn, rnd, False, eval_games)

                row = {
                    'game': g,
                    'seed': seed,
                    'epsilon': dqn.epsilon,
                    'loss': dqn.last_loss if dqn.last_loss is not None else float('nan'),
                    'train_win': block['W'] / m, 'train_draw': block['D'] / m, 'train_lose': block['L'] / m,
                    'train_first_ratio': block_first / m,
                    'first_win': ef[0], 'first_draw': ef[1], 'first_lose': ef[2],
                    'second_win': es[0], 'second_draw': es[1], 'second_lose': es[2],
                    'all_win': (ef[0] + es[0]) / 2, 'all_draw': (ef[1] + es[1]) / 2, 'all_lose': (ef[2] + es[2]) / 2,
                }
                rows.append(row)

                block = {'W': 0, 'D': 0, 'L': 0}
                block_first = 0

                if verbose:
                    print(
                        f'[RULE {rule}] {g:>6}/{games}  eps={row["epsilon"]:.2f}  loss={row["loss"]:.4f} | '
                        f'train W{row["train_win"]:.3f} D{row["train_draw"]:.3f} L{row["train_lose"]:.3f} | '
                        f'greedy 1st W{ef[0]:.3f} D{ef[1]:.3f} L{ef[2]:.3f}  '
                        f'2nd W{es[0]:.3f} D{es[1]:.3f} L{es[2]:.3f} | {time.time() - t0:.0f}s',
                        flush=True
                    )

    except KeyboardInterrupt:
        interrupted = True
        print('\n中断されました。ここまでの結果で最終評価します。')

    # 最終評価 (学習を止めて、先手・後手それぞれで対戦)
    ff = evaluate(n, dqn, rnd, True, final_eval_games)
    fs = evaluate(n, dqn, rnd, False, final_eval_games)

    final = {
        'first': ff, 'second': fs,
        'all': tuple((a + b) / 2 for a, b in zip(ff, fs)),
        'games': played, 'eval_games': final_eval_games,
    }

    return {'rule': rule, 'seed': seed, 'rows': rows, 'final': final, 'agent': dqn,
            'seconds': time.time() - t0, 'interrupted': interrupted, 'hyper': dqn.hyperparameters}


def find_convergence(rows, key='all_win', tail=5, tol=0.03):
    """
    収束した (成績が頭打ちになった) 対戦数を返す。
    最後の tail 回の記録の平均を最終成績とし、それ以降の全ての記録が最終成績の ±tol 以内に
    収まっている最初の記録点の対戦数を返す。記録が少なければ None。
    """
    if len(rows) <= tail:
        return None

    final = mean(r[key] for r in rows[-tail:])

    for i in range(len(rows)):
        if all(abs(r[key] - final) <= tol for r in rows[i:]):
            return rows[i]['game']

    return None


# ----------------------------------------------------------------------
#  CSV と グラフ
# ----------------------------------------------------------------------
def save_csv(rows, path):
    if not rows:
        return

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_results(n, results, path, show=False):
    """ルールごとに、左: 学習中の割合の推移、右: 貪欲な DQN の成績の推移 を描く。"""
    try:
        import matplotlib

        if not show:
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt

    except ImportError:
        print('matplotlib が見つからないのでグラフは作成しません。')
        return

    ref = optimal_vs_random_3x3() if n == 3 else None

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.8 * len(results)), squeeze=False)

    for r, res in enumerate(results):
        rows = res['rows']
        x = [row['game'] for row in rows]

        ax = axes[r][0]
        ax.plot(x, [row['train_win'] for row in rows], label='win', color='tab:blue')
        ax.plot(x, [row['train_draw'] for row in rows], label='draw', color='tab:green')
        ax.plot(x, [row['train_lose'] for row in rows], label='lose', color='tab:red')
        ax.set_title(f'RULE {res["rule"]}: during training (epsilon-greedy), per {x[0] if x else 0} games')
        ax.set_xlabel('games')
        ax.set_ylabel('ratio')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center right')

        ax = axes[r][1]
        ax.plot(x, [row['first_win'] for row in rows], label='win (DQN first)', color='tab:blue')
        ax.plot(x, [row['second_win'] for row in rows], label='win (DQN second)', color='tab:orange')
        ax.plot(x, [row['all_draw'] for row in rows], label='draw (avg)', color='tab:green')
        ax.plot(x, [row['all_lose'] for row in rows], label='lose (avg)', color='tab:red')

        if ref:
            ax.axhline(ref['first'][0], color='tab:blue', linestyle=':', linewidth=1)
            ax.axhline(ref['second'][0], color='tab:orange', linestyle=':', linewidth=1)

        ax.set_title(f'RULE {res["rule"]}: greedy DQN (no learning) vs random' + (' [dotted: optimal play]' if ref else ''))
        ax.set_xlabel('games (training)')
        ax.set_ylabel('ratio')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center right', fontsize=8)

    fig.suptitle(f'DQN vs Random Agent ({n}x{n})')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'グラフを保存しました: {path}')

    if show:
        plt.show()

    plt.close(fig)


def print_summary(n, results):
    ref = optimal_vs_random_3x3() if n == 3 else None

    print()
    print(f'=== SUMMARY ({n}x{n}) ===  (W, D, L = 勝ち, 引き分け, 負け の割合。学習を止めた DQN の最終評価)')

    for res in results:
        f = res['final']
        conv = find_convergence(res['rows'])

        print(f'RULE {res["rule"]} ({RULES[res["rule"]]})  学習 {f["games"]} 局, 評価 各 {f["eval_games"]} 局, seed={res["seed"]}, {res["seconds"]:.0f} 秒')
        print(f'    DQN 先手: W {f["first"][0]:.3f}  D {f["first"][1]:.3f}  L {f["first"][2]:.3f}')
        print(f'    DQN 後手: W {f["second"][0]:.3f}  D {f["second"][1]:.3f}  L {f["second"][2]:.3f}')
        print(f'    平均    : W {f["all"][0]:.3f}  D {f["all"][1]:.3f}  L {f["all"][2]:.3f}')
        print(f'    成績が頭打ちになった対戦数 (平均勝率が最終値の±0.03 以内に入り続ける最初の記録点): {conv}')

    if ref:
        print('参考: 理論上の最適方策 (対ランダム)')
        print(f'    先手: W {ref["first"][0]:.3f}  D {ref["first"][1]:.3f}  L {ref["first"][2]:.3f}')
        print(f'    後手: W {ref["second"][0]:.3f}  D {ref["second"][1]:.3f}  L {ref["second"][2]:.3f}')


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------
def parse_hp(items):
    """--hp key=value を辞書にする。値は Python のリテラルとして解釈する。例: --hp hidden_nodes="(64,)" """
    hp = {}

    for item in items or []:
        if '=' not in item:
            raise ValueError(f'--hp は key=value の形で指定してください: {item}')

        k, v = item.split('=', 1)

        try:
            hp[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            hp[k.strip()] = v.strip()

    return hp


if __name__ == '__main__':

    DEFAULT_SIZE, DEFAULT_GAMES, DEFAULT_RULES = 3, 5000, [1, 2, 3]

    parser = argparse.ArgumentParser(
        description='ランダムエージェントと DQN エージェントの対戦 (TicTacToe)。'
                    '盤面サイズ・試行回数・ルールを指定しなければ、端末から対話的に入力できる。')
    parser.add_argument('size', nargs='?', type=int, default=None, help=f'何目並べか = 盤面サイズ n (>=3, 既定 {DEFAULT_SIZE})')
    parser.add_argument('games', nargs='?', type=int, default=None, help=f'ルールごとの試行回数 = 学習しながらの対戦数 (既定 {DEFAULT_GAMES})')
    parser.add_argument('--rules', type=int, nargs='+', default=None, choices=sorted(RULES), help='実行するルール (既定 1 2 3)')
    parser.add_argument('--seed', type=int, default=0, help='基準の乱数シード (既定 0)。ルールごとのシードは seed_for(seed, "rule", ルール番号)')
    parser.add_argument('--no-prompt', action='store_true', help='質問せず、指定されなかった項目は既定値にする')
    parser.add_argument('--interval', type=int, default=250, help='記録・評価の間隔 (対戦数, 既定 250)')
    parser.add_argument('--eval-games', type=int, default=500, help='途中評価の対戦数 (先手・後手それぞれ, 既定 500)')
    parser.add_argument('--final-eval-games', type=int, default=3000, help='最終評価の対戦数 (先手・後手それぞれ, 既定 3000)')
    parser.add_argument('--hp', action='append', metavar='KEY=VALUE', help='ハイパーパラメーターの上書き (複数指定可)')
    parser.add_argument('--outdir', default='dqn_results', help='CSV・グラフの保存先 (既定 dqn_results)')
    parser.add_argument('--save', action='store_true', help='学習済みのウェイトも保存する')
    parser.add_argument('--no-plot', action='store_true', help='グラフを作らない')
    parser.add_argument('--show', action='store_true', help='グラフを画面に表示する')
    args = parser.parse_args()

    # ---- 指定されていない項目は、端末から入力してもらう (端末でなければ既定値) ----
    interactive = eu.is_interactive() and not args.no_prompt

    if interactive and (args.size is None or args.games is None or args.rules is None):
        print('設定を入力してください (Enter だけなら [ ] の既定値になります)')

    size = args.size
    if size is None:
        size = eu.ask_int('何目並べ (盤面は n x n。3 なら通常の三目並べ)', DEFAULT_SIZE, minimum=3) if interactive else DEFAULT_SIZE

    games = args.games
    if games is None:
        games = eu.ask_int('ルールごとの試行回数 (学習しながらの対戦数)', DEFAULT_GAMES, minimum=1) if interactive else DEFAULT_GAMES

    rules = args.rules
    if rules is None:
        rules = eu.ask_int_list('実行するルール (1: DQN 後手固定, 2: DQN 先手固定, 3: 勝者が後手)', DEFAULT_RULES, RULES) if interactive else DEFAULT_RULES

    if size < 3:
        parser.error('size は 3 以上にしてください')

    if games < 1 or args.interval < 1 or args.eval_games < 1 or args.final_eval_games < 1:
        parser.error('対戦数は 1 以上にしてください')

    hyper = parse_hp(args.hp)

    os.makedirs(args.outdir, exist_ok=True)

    print(f'\n{size}目並べ (盤面 {size}x{size}), ルール {rules}, ルールごとに {games} 局, 基準シード={args.seed}')
    print('ハイパーパラメーター:', {**DEFAULT_HYPERPARAMETERS, **hyper})

    if size != 3 and not hyper:
        print('注意: 既定のハイパーパラメーターは n=3 で調整したものです。n が大きいときは試行回数を増やす・--hp で調整してください。')

    results = []

    for rule in rules:
        seed = eu.seed_for(args.seed, 'rule', rule)

        print(f'\nRULE #{rule}: {RULES[rule]}  (seed={seed})')

        res = run_rule(size, rule, games, hyper, seed=seed,
                       interval=args.interval, eval_games=args.eval_games,
                       final_eval_games=args.final_eval_games)
        results.append(res)

        save_csv(res['rows'], os.path.join(args.outdir, f'dqn_{size}x{size}_rule{rule}.csv'))

        if args.save:
            res['agent'].save(os.path.join(args.outdir, f'dqn_{size}x{size}_rule{rule}.json'))

        if res['interrupted']:
            break

    print_summary(size, results)

    if not args.no_plot:
        plot_results(size, results, os.path.join(args.outdir, f'dqn_{size}x{size}.png'), show=args.show)
