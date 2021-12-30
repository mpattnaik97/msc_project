import matplotlib.pyplot as plt
import pandas as pd

# Read the results from log file
loss = pd.read_csv('../results/another_ssd.log')

# Plot and save the results
plt.figure()

plt.plot(loss['epoch'], loss['loss'], label = 'training loss')
plt.plot(loss['epoch'], loss['val_loss'], label = 'validation loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
