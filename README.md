TicTacToe DQN — Pure Python Reinforcement Learning

Overview

Pythonの標準ライブラリを中心に、ニューラルネットワークからDeep Q-Network（DQN）までをスクラッチ実装した強化学習プロジェクトです。

NumPy・PyTorch・TensorFlowなどの機械学習ライブラリに依存せず、行列演算、ニューラルネットワーク、誤差逆伝播法、最適化アルゴリズム、DQNの学習処理を実装しました。

自作したDQNエージェントをTicTacToe（n目並べ）環境で学習させ、ランダムエージェントとの対戦を通じて性能を評価しました。

Results

3×3のTicTacToeにおいて、各ルールで5,000局の学習を行い、学習を停止したDQNエージェントとランダムエージェントを各3,000局対戦させました。

Evaluation Results

Rule	DQN First (Win / Draw / Loss)	DQN Second (Win / Draw / Loss)	Average Win Rate
Rule 1	0.979 / 0.006 / 0.014	0.890 / 0.041 / 0.068	0.935
Rule 2	0.964 / 0.011 / 0.025	0.781 / 0.029 / 0.191	0.872
Rule 3	0.976 / 0.010 / 0.014	0.890 / 0.036 / 0.073	0.933

学習済みモデルを評価する際はε-greedyによる探索を停止し、貪欲な行動選択を行っています。

また、理論上の最適な行動選択を行った場合の結果とも比較し、DQNの性能を評価しました。

※結果はPython 3.12.3での実行例です。乱数シードを固定することで再現可能です。

Implementation

1. Neural Network from Scratch

Pythonの標準ライブラリを用いて、ニューラルネットワークの基本的な構造を実装しました。

* 全結合層（Affine Layer）
* 順伝播・逆伝播
* ミニバッチ学習
* 活性化関数（ReLU、Sigmoid、Linear）
* 損失関数（MSE、Huber）
* 最適化アルゴリズム（Gradient Descent、Momentum、RMSprop、Adam）

2. Deep Q-Network (DQN)

自作ニューラルネットワークを利用して、DQNエージェントを実装しました。

* Experience Replay
* Target Network
* ε-greedy
* 合法手マスク
* Q値に基づく行動選択
* 学習済みモデルの保存・復元

3. Reinforcement Learning Environment

TicTacToeのゲーム環境と複数のエージェントを実装しました。

* 任意の盤面サイズに対応したn目並べ
* 勝敗判定
* ランダムエージェント
* 表形式Q-Learningエージェント
* DQNエージェント

4. Experiment and Evaluation

学習結果を定量的に評価するための実験環境を実装しました。

* ランダムエージェントとの対戦
* Q-LearningとDQNの比較
* 先手・後手別の勝率評価
* ハイパーパラメータ調整
* 複数の乱数シードを用いた実験
* CSV・グラフによる実験結果の記録

Technical Highlights

Gradient Check

ニューラルネットワークの逆伝播処理が正しく実装されているかを検証するため、解析的に求めた勾配と数値微分による勾配を比較する仕組みを実装しました。

重み・バイアス・入力に対する勾配を比較し、MSE・Huberおよび各活性化関数を組み合わせた構成で検証しています。

Mini-batch Learning

ミニバッチ学習に対応するため、バイアスのブロードキャストやバッチ方向の勾配集約を実装しました。

Reproducibility

乱数シードを一元管理し、異なるハイパーパラメータや学習条件を比較する際の再現性を確保しました。

Evaluation Strategy

学習中の探索を含む成績と、学習終了後に探索を停止したエージェントの成績を分けて評価しました。

また、先手・後手による性能差も測定することで、異なる対戦条件における学習結果を分析しました。

Project Structure

Component	File	Description
Experiment	match_rnd_vs_dqn.py	Random Agent vs DQNの対戦実験
	tune_dqn.py	ハイパーパラメータ調整
	experiment_util.py	乱数シード管理など
Agent	ttt_agent_DQN.py	DQNエージェント
	ttt_agent_QL.py	Q-Learningエージェント
	match_rnd_vs_ql.py	Q-Learningの対戦実験
	ttt_agent_random.py	ランダムエージェント
	ttt_agent.py	エージェント共通インターフェース
Environment	tictactoe.py	TicTacToeゲーム環境
Neural Network	nn_layers.py	全結合層・ニューラルネットワーク
	nn_loss.py	損失関数
	nn_optimizer.py	最適化アルゴリズム
	nn_activator.py	活性化関数
	matrix_util.py	行列演算

Requirements

* Python 3.12.3
* Matplotlib（グラフ描画用・任意）

※ニューラルネットワークやDQNの実装にNumPy・PyTorch・TensorFlowは使用していません。

How to Run

1. DQN Training and Evaluation

python3 match_rnd_vs_dqn.py

盤面サイズ、学習回数、対戦ルールなどを端末から指定できます。

引数による実行も可能です。

python3 match_rnd_vs_dqn.py 3 5000

2. Neural Network Test

python3 nn_layers.py

自作ニューラルネットワークの動作確認と勾配チェックを実行します。

3. Hyperparameter Tuning

python3 tune_dqn.py

複数の乱数シードを用いて、ハイパーパラメータの調整・比較を行います。

実験結果は dqn_results/ に保存されます。

Technologies

* Python
* Standard Library
* Matplotlib

Learning Objectives

本プロジェクトでは、強化学習の理論と実装の対応関係を理解することを目的としました。

特に、ニューラルネットワークの順伝播・逆伝播からDQNの学習アルゴリズムまでを一貫して実装することで、エージェントが環境との相互作用を通じて行動価値を学習する仕組みへの理解を深めました。

また、Q-Learningとの比較やハイパーパラメータの調整、複数条件での実験を通じて、学習結果を定量的に評価することにも取り組みました。

References

大学の授業で扱った強化学習・ニューラルネットワークの内容を学習の出発点とし、それを基に実装を進めました。

授業で学んだ内容を発展させ、ニューラルネットワークのスクラッチ実装、DQNの各種アルゴリズム、実験・評価環境の構築に取り組みました。
