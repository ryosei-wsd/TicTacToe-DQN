#!/usr/bin/env python3

# ======================================================================
#  DQN エージェントのハイパーパラメーター調整
#
#  基準の設定 (BASELINE) から、ハイパーパラメーターを 1 つずつ変えて (one-factor-at-a-time)
#  ランダムエージェントと対戦させ、成績を比較する。乱数の影響を減らすため、各設定を複数のシードで
#  繰り返し、平均と標準偏差を出す。
#
#  評価指標 (大きいほど良い)
#    final : 学習を終えた DQN (貪欲・学習なし) の勝率。先手・後手それぞれ final_eval_games 局の平均。
#    auc   : 学習の途中経過 (interval ごとの評価) の勝率の平均。早く強くなるほど大きい (学習の速さ)。
#    lose  : 最終評価での負けの割合 (小さいほど良い)
#
#  使い方:
#    python3 tune_dqn.py                       # 端末で「何目並べか」「試行回数」「繰り返し回数」を質問される (Enter で既定値)
#    python3 tune_dqn.py 3 2000 --seeds 3 --rule 1   # 指定した項目は質問されない
#    python3 tune_dqn.py --only learning_rate discount_factor
#
#  乱数シード: 繰り返し i 回目は seed_for(基準シード, 'tune', i)。どの設定でも同じ i なら同じシード (experiment_util.py 参照)。
# ======================================================================

import argparse
import csv
import os
import time
from statistics import mean, pstdev

import experiment_util as eu
import match_rnd_vs_dqn as match


# 調整の基準 (他のパラメーターは match_rnd_vs_dqn.DEFAULT_HYPERPARAMETERS のまま)
BASELINE = {
    'hidden_nodes': (32,),
    'learning_rate': 0.001,
    'discount_factor': 0.9,
    'epsilon_decay_episodes': 1000,
    'target_update_interval': 200,
    'batch_size': 32,
    'memory_size': 10000,
    'encoding': 'signed',
    'train_interval': 1,
}

# 試す値 (基準の値も含めてあるが、基準は 1 回だけ実行する)
SWEEP = {
    'hidden_nodes': [(16,), (32,), (64,), (32, 32)],
    'learning_rate': [0.0003, 0.001, 0.003],
    'discount_factor': [0.5, 0.9, 0.99],
    'epsilon_decay_episodes': [300, 1000, 2000],
    'target_update_interval': [50, 200, 1000],
    'batch_size': [16, 32, 64],
    'memory_size': [1000, 10000],
    'encoding': ['signed', 'planes'],
    'train_interval': [1, 2],
}


def run_config(n, hyper, games, seeds, rule, interval, eval_games, final_eval_games, base_seed=0):
    """1 つの設定を seeds の数だけ繰り返し、(final, auc, lose, 秒) の平均と標準偏差を返す。"""
    finals, aucs, loses, secs = [], [], [], []

    for seed in range(seeds):
        res = match.run_rule(n, rule, games, hyper, seed=eu.seed_for(base_seed, 'tune', seed), interval=interval,
                             eval_games=eval_games, final_eval_games=final_eval_games, verbose=False)
        finals.append(res['final']['all'][0])
        loses.append(res['final']['all'][2])
        aucs.append(mean(r['all_win'] for r in res['rows']))
        secs.append(res['seconds'])

    return {
        'final': mean(finals), 'final_sd': pstdev(finals),
        'auc': mean(aucs), 'auc_sd': pstdev(aucs),
        'lose': mean(loses), 'sec': mean(secs),
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN のハイパーパラメーター調整。指定しなかった項目は端末から入力できる。')
    parser.add_argument('size', nargs='?', type=int, default=None, help='何目並べか = 盤面サイズ n (>=3, 既定 3)')
    parser.add_argument('games', nargs='?', type=int, default=None, help='1 回の学習の試行回数 = 対戦数 (既定 2000)')
    parser.add_argument('--seeds', type=int, default=None, help='各設定の繰り返し回数 (既定 3)')
    parser.add_argument('--seed', type=int, default=0, help='基準の乱数シード (既定 0)')
    parser.add_argument('--no-prompt', action='store_true', help='質問せず、指定されなかった項目は既定値にする')
    parser.add_argument('--rule', type=int, default=1, choices=sorted(match.RULES), help='対戦ルール (既定 1)')
    parser.add_argument('--interval', type=int, default=500, help='途中評価の間隔 (既定 500)')
    parser.add_argument('--eval-games', type=int, default=200, help='途中評価の対戦数 (先手・後手それぞれ, 既定 200)')
    parser.add_argument('--final-eval-games', type=int, default=1000, help='最終評価の対戦数 (先手・後手それぞれ, 既定 1000)')
    parser.add_argument('--only', nargs='+', choices=sorted(SWEEP), help='調べるハイパーパラメーターを限定する')
    parser.add_argument('--out', default='dqn_results/tuning.csv', help='結果の CSV (既定 dqn_results/tuning.csv)')
    args = parser.parse_args()

    interactive = eu.is_interactive() and not args.no_prompt

    if interactive and None in (args.size, args.games, args.seeds):
        print('設定を入力してください (Enter だけなら [ ] の既定値になります)')

    size = args.size if args.size is not None else (eu.ask_int('何目並べ (盤面は n x n。3 なら通常の三目並べ)', 3, minimum=3) if interactive else 3)
    games = args.games if args.games is not None else (eu.ask_int('1 回の学習の試行回数 (対戦数)', 2000, minimum=1) if interactive else 2000)
    seeds = args.seeds if args.seeds is not None else (eu.ask_int('各設定の繰り返し回数 (シードの数)', 3, minimum=1, maximum=9000) if interactive else 3)

    if size < 3 or games < 1 or seeds < 1:
        parser.error('size は 3 以上、games と seeds は 1 以上にしてください')

    names = args.only if args.only else list(SWEEP)

    print(f'{size}目並べ, RULE {args.rule}: {match.RULES[args.rule]}, 学習 {games} 局, {seeds} 回 (基準シード {args.seed}), '
          f'評価 途中 {args.eval_games} / 最終 {args.final_eval_games} 局 (先手・後手それぞれ)')
    print('基準:', BASELINE)

    cache = {}       # 同じ設定を二度実行しない
    table = []
    t_start = time.time()

    def evaluate_config(hyper):
        key = repr(sorted(hyper.items()))

        if key not in cache:
            cache[key] = run_config(size, hyper, games, seeds, args.rule, args.interval,
                                    args.eval_games, args.final_eval_games, args.seed)

        return cache[key]

    # 基準
    base = evaluate_config(dict(BASELINE))
    print(f'{"baseline":<26} final {base["final"]:.3f}±{base["final_sd"]:.3f}  auc {base["auc"]:.3f}±{base["auc_sd"]:.3f}  '
          f'lose {base["lose"]:.3f}  {base["sec"]:.0f}s/run', flush=True)

    for name in names:
        for value in SWEEP[name]:
            hyper = dict(BASELINE)
            hyper[name] = value

            r = evaluate_config(hyper)
            mark = ' (baseline)' if value == BASELINE[name] else ''
            label = f'{name}={value}{mark}'

            print(f'{label:<26} final {r["final"]:.3f}±{r["final_sd"]:.3f}  auc {r["auc"]:.3f}±{r["auc_sd"]:.3f}  '
                  f'lose {r["lose"]:.3f}  {r["sec"]:.0f}s/run   [{time.time() - t_start:.0f}s elapsed]', flush=True)

            table.append({'parameter': name, 'value': str(value), **r})

    outdir = os.path.dirname(args.out)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f'保存しました: {args.out}')
