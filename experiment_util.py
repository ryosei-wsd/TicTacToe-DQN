#!/usr/bin/env python3

# ======================================================================
#  実験スクリプト (match_rnd_vs_dqn.py, tune_dqn.py) で共通に使う道具
#
#  1. 乱数シードの管理
#       このプロジェクトの乱数は全て Python 標準の random モジュールのグローバルな状態を使う
#       (nn_layers の初期化, ttt_agent_random, ttt_agent_DQN の ε-greedy・経験再生)。
#       そのため、1 回の実験 (= 1 つのエージェントの学習) を始める直前に set_seed() を 1 回呼べば、
#       その実験の結果は再現できる。
#
#       シードは seed_for(基準シード, 種別, 番号) で決める。
#           seed = 基準シード * 10000 + (種別ごとのオフセット) + 番号
#         'rule' : match_rnd_vs_dqn.py のルール番号 (1, 2, 3)         -> 基準シード 0 なら 1, 2, 3
#         'tune' : tune_dqn.py の繰り返し番号 (0, 1, 2, ...)          -> 基準シード 0 なら 1000, 1001, ...
#       複数の条件 (ハイパーパラメーター) を比較するときは、条件ごとに同じ番号 = 同じシードを使う
#       (条件が違っても同じ乱数列から始まるので、差が乱数の運だけで生じにくい)。
#
#  2. ターミナルからの入力
#       ask_int(), ask_int_list() : 数値を対話的に入力させる (Enter で既定値)。
#       is_interactive()          : 標準入力が端末のときだけ True。リダイレクトやバックグラウンド実行では
#                                   質問せずに既定値を使う。
# ======================================================================

import random
import sys

SEED_STRIDE = 10000

# 種別ごとの (オフセット, 番号の上限)
SEED_KINDS = {
    'rule': (0, 1000),
    'tune': (1000, 9000),
}


def seed_for(base_seed: int, kind: str, index: int) -> int:
    """基準シード・種別・番号から、実験用のシードを決める。"""
    if kind not in SEED_KINDS:
        raise ValueError(f"unknown seed kind '{kind}'. choose from {list(SEED_KINDS)}")

    if type(base_seed) != int or base_seed < 0:
        raise ValueError('base_seed MUST BE an integer greater than or equal to 0')

    offset, limit = SEED_KINDS[kind]

    if type(index) != int or not 0 <= index < limit:
        raise ValueError(f"index for '{kind}' MUST BE an integer between 0 and {limit - 1}")

    return base_seed * SEED_STRIDE + offset + index


def set_seed(seed: int) -> int:
    """乱数を初期化する (Python 標準の random)。設定したシードを返す。"""
    random.seed(seed)

    return seed


def is_interactive() -> bool:
    """端末から対話的に実行されているか。"""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def ask_int(message: str, default: int, minimum: int = None, maximum: int = None, input_func=input) -> int:
    """
    整数を入力させる。Enter だけなら default。不正な入力のときは入力し直させる。
    (全角数字も int() が解釈できるので、そのまま使える)
    """
    while True:
        try:
            text = input_func(f'{message} [{default}]: ').strip()
        except EOFError:
            print()
            return default

        if text == '':
            return default

        try:
            value = int(text)
        except ValueError:
            print('  整数で入力してください。')
            continue

        if minimum is not None and value < minimum:
            print(f'  {minimum} 以上で入力してください。')
            continue

        if maximum is not None and value > maximum:
            print(f'  {maximum} 以下で入力してください。')
            continue

        return value


def ask_int_list(message: str, default: list, choices, input_func=input) -> list:
    """整数のリストを入力させる (空白かカンマ区切り)。Enter だけなら default。choices にある値だけ受け付ける。"""
    default_text = ' '.join(str(v) for v in default)

    while True:
        try:
            text = input_func(f'{message} [{default_text}]: ').strip()
        except EOFError:
            print()
            return list(default)

        if text == '':
            return list(default)

        try:
            values = [int(t) for t in text.replace(',', ' ').split()]
        except ValueError:
            print('  整数を空白かカンマで区切って入力してください。')
            continue

        if not values or any(v not in choices for v in values):
            print(f'  {sorted(choices)} の中から選んでください。')
            continue

        # 重複は除く (順番は保つ)
        return list(dict.fromkeys(values))
