import pickle
import os
import matplotlib.pyplot as plt

MODEL_HISTORY_PATH = os.path.join(".", "pong_historys")

historys = pickle.load(open(os.path.join(MODEL_HISTORY_PATH, 'history.p'), 'rb'))

print(type(historys))
print(len(historys))
print(historys)

plt.plot(historys['episode_numbers'], historys['episode_rewards'], "r*", historys['episode_numbers'], historys['running_means'], 'b^')
plt.show()