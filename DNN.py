# This is final version of DNN model about stock market prediction
# 2020-02-13 by fiona_youkyung

# import package
import sqlite3 as sq
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import random
import math
from scipy.stats import norm
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.utils import shuffle

class DNN_stock() :
    def __init__(self):
        # stock_code : 005930등 6자리 숫자
        # db_name : 파일명 (.db 없이 입력 하기)
        self.stock_code = None
        self.db_name = None
        self.df = None
        self.data = None
        self.u = None
        self.d = None
        self.period = None
        self.ft = None
        self.Train_X = None
        self.Train_Y = None
        self.Test_X = None
        self.Test_Y = None
        self.history = None
        self.batch_size = None
        self.epoch = None
        self.n = None
        self.model = None

    def open_stock(self,stock_code,db_name) :
        """

        :param stock_code: 005930등 6자리 숫자
        :param db_name: 파일명 (.db 없이 입력 하기)
        :return:db에 저장된 파일로 부터 파일 가져오기
        """
        if not len(stock_code) == 6 :
            raise ValueError("종목 코드 형식이 잘못 되었습니다.")
        self.stock_code = stock_code
        self.db_name = db_name
        con = sq.connect(f'{self.db_name}.db')
        print('Connect to sqlite3')

        cur = con.cursor()
        df = pd.read_sql(f'select * from {self.db_name} where Code = "{self.stock_code}.KS"', con, index_col = 'index')
        print(f'"{self.stock_code}.KS"  DATA selected from {df.index[0]} to {df.index[-1]}')
        df.sort_index()

        con.close()
        self.df = df
        return df



    def get_featureSet(self,data,u,d,period):
        """
        OHLDV 데이터로부터 다음과 같은 Fearureset을 구성한다.
        MACD : 이동평균 수렴 확산지수 (12일 이동 평균선과, 26일 이동평균선의 차이를 나타낸 값)
        RSI : 상대 강도 지수로 가격의 상승 압력과 하락 압력간의 상대적인 강도를 나타내는 지표
        obv : 주가가 전일에 비해 상승하였을 때의 거래량 누계에서 하락하였을 때는 거래량 누계를 차감한 지표
        liquidity : 주식 유동성
        parkinson : 범위 변동성으로 장중 High-Price와 Low Price로 장중 변동성을 계산함
        volatility : 주식 휘발성
        :param data: OPEN, High, Low, Close, Volumn 다섯개의 컬럼으로 이루어짐
        :param u: 목표 수익률 표준편차
        :param d: 손절률 표준편차
        :param period:holding 기간
        :return:[MACD],[RSI],[obv],[liquidity],[parkinson],[volatility ],[class]
        return : 0 - 주가 횡보, 1 - 주가 하락, 2 - 주가 상승
        """
        # Feature value를 계산한 후 Z-score Normalization 한다
        self.data = data
        self.u = u
        self.d = d
        self.period = period
        fmacd = self.scale(self.MACD(self.data, 12, 26, 9))
        frsi = self.scale(self.RSI(self.data, 40))
        fobv = self.scale(self.OBV(self.data, ext=True))
        fliquidity = self.scale(self.Liquidity(self.data))
        fparkinson = self.scale(self.ParkinsonVol(self.data, 10))
        fvol = self.scale(self.CloseVol(self.data, 10))

        ft = pd.DataFrame()
        ft['macd'] = fmacd
        ft['rsi'] = frsi
        ft['obv'] = fobv
        ft['liquidity'] = fliquidity
        ft['parkinson'] = fparkinson
        ft['volatility'] = fvol
        ft['class'] = self.getUpDnClass(self.data, self.u, self.d, self.period)
        ft = ft.dropna()
        self.ft = ft

        # Feature들의 value (수준) 보다는 방향 (up, down)을 분석하는 것이 의미가 있어 보임.
        # 방향을 어떻게 검출할 지는 향후 연구 과제로 한다

        return ft


    # Supervised Learning을 위한 class를 부여한다
    #
    # up : 목표 수익률 표준편차
    # dn : 손절률 표준편차
    # period : holding 기간
    # return : 0 - 주가 횡보, 1 - 주가 하락, 2 - 주가 상승
    # ---------------------------------------------------
    def getUpDnClass(self,data, up, dn, period):
        # 주가 수익률의 표준편차를 측정한다
        r = []
        for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
            if math.isnan(prev.Close):
                continue
            r.append(np.log(curr.Close / prev.Close))
        s = np.std(r)

        # 목표 수익률과 손절률을 계산한다
        uLimit = up * s * np.sqrt(period)
        dLimit = dn * s * np.sqrt(period)

        # 가상 Trading을 통해 미래 주가 방향에 대한 Class를 결정한다
        rclass = []
        for i in range(len(data) - 1):
            # 매수 포지션을 취한다
            buyPrc = data.iloc[i].Close
            y = np.nan

            # 매수 포지션 이후 청산 지점을 결정한다
            for k in range(i + 1, len(data)):
                sellPrc = data.iloc[k].Close

                # 수익률을 계산한다
                rtn = np.log(sellPrc / buyPrc)

                # 목표 수익률이나 손절률에 도달하면 루프를 종료한다
                if k > i + period:
                    # hoding 기간 동안 목표 수익률이나 손절률에 도달하지 못했음
                    # 주가가 횡보한 것임
                    y = 0
                    break
                else:
                    if rtn > uLimit:
                        y = 2  # 수익
                        break
                    elif rtn < dLimit:
                        y = 1  # 손실
                        break

            rclass.append(y)

        rclass.append(np.nan)
        return pd.DataFrame(rclass, index=data.index)

    # MACD 지표를 계산한다
    # MACD Line : 12-day EMA - 26-day EMA
    # Signal Line : 9-day EMA of MACD line
    # MACD oscilator : MACD Line - Signal Line
    # ----------------------------------------
    def MACD(self,ohlc, nFast=12, nSlow=26, nSig=9, percent=True):
        self.ema1 = self.EMA(ohlc.Close, nFast)
        self.ema2 = self.EMA(ohlc.Close, nSlow)

        if percent:
            self.macdLine = 100 * (self.ema1 - self.ema2) / self.ema2
        else:
            self.macdLine = self.ema1 - self.ema2
        self.signalLine = self.EMA(self.macdLine, nSig)

        return pd.DataFrame(self.macdLine - self.signalLine, index=ohlc.index)

    # 지수이동평균을 계산한다
    # data : Series
    def EMA(self,data, n):
        ma = []

        # data 첫 부분에 na 가 있으면 skip한다
        x = 0
        while True:
            if math.isnan(data[x]):
                ma.append(data[x])
            else:
                break;
            x += 1

        # x ~ n - 1 기간까지는 na를 assign 한다
        for i in range(x, x + n - 1):
            ma.append(np.nan)

        # x + n - 1 기간은 x ~ x + n - 1 까지의 평균을 적용한다
        sma = np.mean(data[x:(x + n)])
        ma.append(sma)

        # x + n 기간 부터는 EMA를 적용한다
        k = 2 / (n + 1)

        for i in range(x + n, len(data)):
            # print(i, data[i])
            ma.append(ma[-1] + k * (data[i] - ma[-1]))

        return pd.Series(ma, index=data.index)

    # RSI 지표를 계산한다. (Momentum indicator)
    # U : Gain, D : Loss, AU : Average Gain, AD : Average Loss
    # smoothed RS는 고려하지 않았음.
    # --------------------------------------------------------
    def RSI(self,ohlc, n=14):
        self.closePrice = pd.DataFrame(ohlc.Close)
        U = np.where(self.closePrice.diff(1) > 0, self.closePrice.diff(1), 0)
        D = np.where(self.closePrice.diff(1) < 0, self.closePrice.diff(1) * (-1), 0)

        U = pd.DataFrame(U, index=ohlc.index)
        D = pd.DataFrame(D, index=ohlc.index)

        AU = U.rolling(window=n).mean()
        AD = D.rolling(window=n).mean()

        return 100 * AU / (AU + AD)

    # On Balance Volume (OBV) : buying and selling pressure
    # ext = False : 기존의 OBV
    # ext = True  : Extended OBV. 가격 변화를 이용하여 거래량을 매수수량, 매도수량으로 분해하여 매집량 누적
    # -------------------------------------------------------------------------------------------------
    def OBV(self,ohlcv, ext=True):
        obv = [0]

        # 기존의 OBV
        if ext == False:
            # 기술적 지표인 OBV를 계산한다
            for curr, prev in zip(ohlcv.itertuples(), ohlcv.shift(1).itertuples()):
                if math.isnan(prev.Volume):
                    continue

                if curr.Close > prev.Close:
                    obv.append(obv[-1] + curr.Volume)
                if curr.Close < prev.Close:
                    obv.append(obv[-1] - curr.Volume)
                if curr.Close == prev.Close:
                    obv.append(obv[-1])
        # Extendedd OBV
        else:
            # 가격 변화를 측정한다. 가격 변화 = 금일 종가 - 전일 종가
            deltaClose = ohlcv['Close'].diff(1)
            deltaClose = deltaClose.dropna(axis=0)

            # 가격 변화의 표준편차를 측정한다
            stdev = np.std(deltaClose)

            for curr, prev in zip(ohlcv.itertuples(), ohlcv.shift(1).itertuples()):
                if math.isnan(prev.Close):
                    continue

                buy = curr.Volume * norm.cdf((curr.Close - prev.Close) / stdev)
                sell = curr.Volume - buy
                bs = abs(buy - sell)

                if curr.Close > prev.Close:
                    obv.append(obv[-1] + bs)
                if curr.Close < prev.Close:
                    obv.append(obv[-1] - bs)
                if curr.Close == prev.Close:
                    obv.append(obv[-1])

        return pd.DataFrame(obv, index=ohlcv.index)

    # 유동성 척도를 계산한다
    def Liquidity(self,ohlcv):
        k = []

        i = 0
        for curr in ohlcv.itertuples():
            dp = abs(curr.High - curr.Low)
            if dp == 0:
                if i == 0:
                    k = [np.nan]
                else:
                    # dp = 0 이면 유동성은 매우 큰 것이지만, 계산이 불가하므로 이전의 유동성을 유지한다
                    k.append(k[-1])
            else:
                k.append(np.log(curr.Volume) / dp)
            i += 1

        return pd.DataFrame(k, index=ohlcv.index)

    # 전일 Close price와 금일 Close price를 이용하여 변동성을 계산한다
    def CloseVol(self,ohlc, n):
        rtn = pd.DataFrame(ohlc['Close']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
        vol = pd.DataFrame(rtn).rolling(window=n).std()

        return pd.DataFrame(vol, index=ohlc.index)

    # 당일의 High price와 Low price를 이용하여 Parkinson 변동성 (장 중 변동성)을 계산한다.
    def ParkinsonVol(self,ohlc, n):
        vol = []
        for i in range(n - 1):
            vol.append(np.nan)

        for i in range(n - 1, len(ohlc)):
            sigma = 0
            for k in range(0, n):
                sigma += np.log(ohlc.iloc[i - k].High / ohlc.iloc[i - k].Low) ** 2
            vol.append(np.sqrt(sigma / (n * 4 * np.log(2))))

        return pd.DataFrame(vol, index=ohlc.index)

    # Z-score normalization
    def scale(self,data):
        col = data.columns[0]
        return (data[col] - data[col].mean()) / data[col].std()

    # 시계열을 평활화한다
    def smooth(self,data, s=5):
        y = data[data.columns[0]].values
        w = np.isnan(y)
        y[w] = 0.
        sm = ndimage.gaussian_filter1d(y, s)
        return pd.DataFrame(sm)

    def get_hist(self,ft):
        """
        Class들의 분포를 살펴 볼 수 있는 histogram을 알려줌
        :param ft: fetured data
        :return: histogram, u,d,holding period
        """
        plt.hist(self.ft['class'])
        plt.show()
        print(f'u : {self.u}, d : {self.d}, period : {self.period}')

    def train_test_split(self,data,train_set_rate):
        """
        전체 데이터 셋에서 train_set_rate대로 Test set과, trainset을 생성해줌
        :param data : ft data set
        :param train_set_rate: 0~ 1
        :return: Train_X, Train_Y, Test_X, Test_Y
        """
        data = shuffle(data)
        nlen = len(data)
        n = int(nlen*train_set_rate)-1
        self.n = n
        Train_X = data.iloc[0:n,0:6].values
        Train_Y = np_utils.to_categorical(data.iloc[0:n,6].values)
        Test_X = data.iloc[n:(nlen-1), 0:6].values
        Test_Y = np_utils.to_categorical(data.iloc[n:(nlen-1),6].values)

        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y

        return Train_X,Train_Y,Test_X,Test_Y

    def Model_Create(self,learining_rate,batch_size,e):
        """
        DNN 모델 생성
        :param optimizer:'adam'으로 고정
        :param learining_rate: 학습률
        :param batch_size: 배치 사이즈
        :param e: 에포크
        :return: 학습 완료 모델 (history)
        """
        self.model = Sequential()
        self.model.add(Dense(30,input_dim=6,activation='relu'))
        self.model.add(Dense(30,activation='relu'))
        self.model.add(Dense(30,activation='relu'))
        self.model.add(Dense(3,activation='sigmoid'))
        adam = optimizers.Adam(lr = learining_rate)
        self.model.compile(loss = 'mse', optimizer=adam,metrics=['accuracy'])

        history = self.model.fit(self.Train_X,self.Train_Y, batch_size=batch_size,epochs=e,\
                            validation_data=(self.Test_X,self.Test_Y))

        self.history = history
        self.batch_size = batch_size
        self.epoch = e
        return history


    def get_graph_traindata(self):
        """
        Train_data의 성능 그래프
        :return: Performance of Training Data
        """
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.history.history['accuracy'], color = 'red')
        ax2.plot(self.history.history['loss'], color = 'blue')
        ax1.set_xlabel("Epoch")
        ax1.set_title(f'Performance of Training Data epoch = {self.epoch},batch = {self.batch_size}')
        ax1.set_ylabel("Accuracy", color = 'red')
        ax2.set_ylabel("Loss", color = 'blue')
        plt.show()

    def get_graph_testdata(self):
        """
        Test_data의 성능 그래프
        :return: Performance of test Data
        """
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.history.history['val_accuracy'], color = 'red')
        ax2.plot(self.history.history['val_loss'], color = 'blue')
        ax1.set_xlabel("Epoch")
        ax1.set_title(f'Performance of Test Data epoch = {self.epoch},batch = {self.batch_size}')
        ax1.set_ylabel("Accuracy", color='red')
        ax2.set_ylabel("Loss", color='blue')
        plt.show()
    def today_prediction(self,today_date):
        """
        학습된 모델을 가지고, 내일의 주식 가격 예측

        :param today_date: '2020-02-07 00:0000'
        :return:
        """
        todayX =self.ft[self.ft.index == 'today_date']
        print(todayX)
        today_final_X = todayX.iloc[0:self.n, 0:6].values
        today_final_Y = np_utils.to_categorical(todayX.iloc[0:self.n, 6].values)
        predY = self.model.predict(today_final_X)
        predClass = np.argmax(predY)
        print()
        if predClass == 0:
            print("* 향후 주가는 횡보할 것으로 예상됨.")
        elif predClass == 1:
            print("* 향후 주가는 하락할 것으로 예상됨.")
        else:
            print("* 향후 주가는 상승할 것으로 예상됨.")

        prob = predY / predY.sum()
        np.set_printoptions(precision=4)
        print("* 주가 횡보 확률 = %.2f %s" % (prob[0][0] * 100, '%'))
        print("* 주가 하락 확률 = %.2f %s" % (prob[0][1] * 100, '%'))
        print("* 주가 상승 확률 = %.2f %s" % (prob[0][2] * 100, '%'))


if __name__ == '__main__' :
    stock = DNN_stock()
    df = stock.open_stock('005930','stock_10')
    print(df)
    final = stock.get_featureSet(df,0.8,-0.8,1)
    print(final)
    stock.get_hist(final)
    Train_X, Train_Y, Test_X, Test_Y = stock.train_test_split(final,0.8)
    history = stock.Model_Create(0.005,50,150)
    stock.get_graph_traindata()
    stock.get_graph_testdata()
    stock.today_prediction('2020-02-06 00:00:00')






