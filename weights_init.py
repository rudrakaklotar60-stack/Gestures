import numpy as np

WI = np.random.randn(39, 128) * (np.sqrt( 2/ 39))
WH = np.random.randn(128, 128) * (np.sqrt( 2/ 128))
WH2 = np.random.randn(128, 64) * (np.sqrt( 2/ 128))
WH3 = np.random.randn(64,32) * (np.sqrt( 2/ 64))
WO = np.random.randn(32, 10) * (np.sqrt( 2/ 32))

BI = np.zeros(128)
BH = np.zeros(128)
BH2 = np.zeros(64)
BH3 = np.zeros(32)
BO = np.zeros(10)

np.savez("Wights.npz", 
        WI = WI,
        WH = WH,
        WH2 = WH2,
        WH3 = WH3,
        WO = WO,

        BI = BI,
        BH = BH,
        BH2 = BH2,
        BH3 = BH3, 
        BO = BO
        
        )

print("SAVED!!")

