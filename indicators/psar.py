import pandas as pd
import numpy as np

def getParabolicSAR(history, step=0.02, max_step=0.2):
    high = history['High'].values
    low = history['Low'].values
    close = history['Close'].values
    length = len(history)

    sar = np.zeros(length)
    up_trend = np.ones(length, dtype=bool)
    af = step * np.ones(length)
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, length):
        prev_sar = sar[i-1]
        prev_af = af[i-1]
        prev_ep = ep
        prev_trend = up_trend[i-1]

        # Calcula SAR base
        if prev_trend:
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            # Checa reversão com Close: se Close < SAR ou Low < SAR
            if close[i] < sar[i] or low[i] < sar[i]:
                up_trend[i] = False
                sar[i] = prev_ep
                ep = low[i]
                af[i] = step
            else:
                up_trend[i] = True
                ep = max(prev_ep, high[i])
                if ep > prev_ep:
                    af[i] = min(prev_af + step, max_step)
                else:
                    af[i] = prev_af
        else:
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            # Checa reversão com Close: se Close > SAR ou High > SAR
            if close[i] > sar[i] or high[i] > sar[i]:
                up_trend[i] = True
                sar[i] = prev_ep
                ep = high[i]
                af[i] = step
            else:
                up_trend[i] = False
                ep = min(prev_ep, low[i])
                if ep < prev_ep:
                    af[i] = min(prev_af + step, max_step)
                else:
                    af[i] = prev_af

    history['Parabolic_SAR'] = sar
