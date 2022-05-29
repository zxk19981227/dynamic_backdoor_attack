from matplotlib import pyplot as plt

x = [1, 2, 3, 4, 5]
# x = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
#      ]
olid = [71.39, 67.79, 67.79, 65.81, 64.88]
# olid = [62.32, 69.76, 71.39, 69.76, 69.76, 69.76, 69.76, 69.76, 69.76, 69.76
#         ]
agnews = [82.61, 81.18, 81.05, 81.81, 80.01]
# agnews = [85.44, 85.51, 83.87, 82.63, 82.07, 81.91, 81.84, 81.81, 81.81, 81.81
#           ]
sst = [77.1, 72.04, 69.68, 68.42, 68.09]
# sst = [83.96, 82.53, 78.91, 77.1, 76.44, 76.6, 76.55, 76.55, 76.55, 76.55
#        ]

plt.plot(x, agnews, color='blue', marker='x', linestyle='-.', label=f"AG's News ASR")
plt.plot(x, olid, color='green', marker='o', linestyle=':', label=f"OLID ASR")
plt.plot(x, sst, color='orange', marker='+', linestyle='--', label=f"SST-2 ASR")
plt.xlabel('Beam Size')
plt.ylabel('ASR')
plt.legend()
plt.show()
