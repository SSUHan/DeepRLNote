import pickle
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False)
args = parser.parse_args()

MODEL_HISTORY_PATH = os.path.join(".", "pong_historys")

historys = pickle.load(open(os.path.join(MODEL_HISTORY_PATH, 'history.p'), 'rb'))

print(type(historys))
print(len(historys))
print(historys)
plt.plot(historys['episode_numbers'], historys['episode_rewards'], "r*", historys['episode_numbers'], historys['running_means'], 'b^')
if not args.save:
	plt.show()
else:
	plt.savefig('pong_history.png')