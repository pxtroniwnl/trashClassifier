import matplotlib.pyplot as plt
import pandas as pd

resultsPath="C:/Users/Alejandro/Desktop/Trash/runs/classify/train4/results.csv"
results = pd.read_csv(resultsPath)

plt.figure()
plt.plot(results['epoch'], results['train/loss'], label='train loss')
plt.plot(results['epoch'], results['val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss vs epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()


plt.figure()
plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')

plt.show()

print("Graficos correctamente puestos")