import matplotlib.pyplot as plt

trainLoss = list(range(10))
testLoss = list(range(10))

plt.plot(trainLoss,label="train")
plt.plot(testLoss,label="test")
plt.legend()
plt.show()

