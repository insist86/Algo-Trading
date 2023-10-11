import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class Backtest:
    """
    Backtest è una classe per fare il backtest di una strategia di trading.

    Questa classe è utilizzata per eseguire un backtest di una data strategia di trading su dati storici. 
    Consente di calcolare varie metriche di trading, come i rendimenti cumulativi, il drawdown e altre statistiche. 
    È inoltre in grado di visualizzare i risultati del backtest.

    Parameters
    ----------
    data : DataFrame
        I dati storici su cui backtestare la strategia di trading. Il DataFrame dovrebbe essere indicizzato per tempo
        e dovrebbe contenere almeno i dati sui prezz

    TradingStrategy : object
        La strategia di trading da backtestare. Dovrebbe essere un'istanza di una classe che implementa
        metodi get_entry_signal e get_exit_signal.

    parameters : dict
        I parametri della strategia da utilizzare durante il backtest

    run_directly : bool, default False
        Se True, il backtest viene eseguito durante l'inizializzazione. Altrimenti, il metodo run deve essere
        chiamato esplicitamente.

    title : str, default None
        Il titolo del grafico del backtest. Se None, verrà utilizzato un titolo predefinito.
    """

    # costruttore della classe e serve per inizializzare tutte le variabili membro dell'oggetto Backtest
    # e prepara il DataFrame data per l'esecuzione del backtest
    def __init__(self, data, TradingStrategy, parameters, run_directly=False, title=None):
        # imposto i paramtri:
        # - viene inizializzata la strategia passando i dati e i parametri all'oggetto TradingStrategy
        self.TradingStrategy = TradingStrategy(data, parameters) 
        # - self.start_date_backtest ottiene la data di inizio del backtest dalla strategia di trading
        self.start_date_backtest = self.TradingStrategy.start_date_backtest
        # - self.data contiene il subset dei dati a partire dalla data di inizio del backtest.
        self.data = data.loc[self.start_date_backtest:]

        # - Se le colonne "returns", "duration", "buy_count", e "sell_count" non sono presenti nel 
        # DataFrame self.data, vengono aggiunte con valori iniziali di 0
        if "returns" not in self.data.columns:
            self.data["returns"] = 0
        if "duration" not in self.data.columns:
            self.data["duration"] = 0
        if "buy_count" not in self.data.columns:
            self.data["buy_count"] = 0
        if "sell_count" not in self.data.columns:
            self.data["sell_count"] = 0

        # - self.count_buy e self.count_sell sono inizializzati a 0 
        # e terranno traccia del numero di operazioni di acquisto e vendita effettuate.
        self.count_buy, self.count_sell = 0, 0
        # - self.entry_trade_time e self.exit_trade_time sono inizializzati a None 
        # e verranno utilizzati per tracciare i tempi di ingresso e uscita del trade.
        self.entry_trade_time, self.exit_trade_time = None, None

        if run_directly:
            self.run()
            self.display_metrics()
            self.display_graphs(title)

    # Il metodo run(self) esegue il backtest della strategia di trading sui dati storici, registrando i risultati del trade, 
    # come i segnali di acquisto e vendita, i ritorni di posizione e la durata del trade, direttamente nel DataFrame self.data.
    def run(self):

        for current_time in self.data.index:

            # 1. Segnali di Ingresso e Apertura di una Posizione:
            # - Questa linea chiama il metodo get_entry_signal della strategia di trading con il tempo corrente come parametro, 
            # ottenendo così un segnale di ingresso e il tempo di ingresso nel trade.
            entry_signal, self.entry_trade_time = self.TradingStrategy.get_entry_signal(current_time)
            
            # Queste linee aggiornano il DataFrame self.data, impostando "buy_count" a 1 se il segnale di ingresso è 1 (acquisto) 
            # e "sell_count" a 1 se il segnale di ingresso è -1 (vendita).
            self.data.loc[current_time, "buy_count"] = 1 if entry_signal == 1 else 0
            self.data.loc[current_time, "sell_count"] = 1 if entry_signal == -1 else 0

            # 2. Segnali di Uscita e Chiusura di una Posizione:
            # - Questa linea chiama il metodo get_exit_signal della strategia di trading con il tempo corrente come parametro, 
            # ottenendo così un ritorno di posizione e il tempo di uscita dal trade.
            position_return, self.exit_trade_time = self.TradingStrategy.get_exit_signal(current_time)

            # Archiviazione del Ritorno di Posizione e Durata quando si Chiude un Trade
            if position_return != 0:
                self.data.loc[current_time, "returns"] = position_return
                self.data.loc[current_time, "duration"] = (self.exit_trade_time - self.entry_trade_time).total_seconds()

    # Questo metodo calcola diverse metriche di trading come i rendimenti cumulativi, il drawdown, e altre statistiche.
    def get_vector_metrics(self):
        # calcolo dei Rendimenti Cumulativi:
        self.data["cumulative_returns"] = (self.data["returns"]).cumsum()

        # calcola il massimo cumulativo dei rendimenti cumulativi(accumulate max) # (1,3,5,3,1) --> (1,3,5,5,5) - 0.01 --> 1.01
        running_max = np.maximum.accumulate(self.data["cumulative_returns"] + 1)

        # calcola il drawdown
        self.data["drawdown"] = (self.data["cumulative_returns"] + 1) / running_max - 1

    # Questo metodo si occupa della visualizzazione grafica dei rendimenti cumulativi e del drawdown della strategia di trading.
    def display_graphs(self, title=None):

        # calcola i rendimenti cumulativi e il drawdown
        self.get_vector_metrics()

        # che poi vengono estratti da self.data e memorizzati nelle seguenti variabili
        cum_ret = self.data["cumulative_returns"]
        drawdown = self.data["drawdown"]

        # imposta il font style
        plt.rc('font', weight='bold', size=12)

        # aggiunge 2 subplots : rendimenti e drawdawn
        fig, (cum, dra) = plt.subplots(2, 1, figsize=(15, 7))
        plt.setp(cum.spines.values(), color="#ffffff")
        plt.setp(dra.spines.values(), color="#ffffff")

        # imposta il titolo del grafico
        if title is None:
            fig.suptitle("Overview of the Strategy", size=18, fontweight='bold')
        else:
            fig.suptitle(title, size=18, fontweight='bold')

        # crea il grafico dei rendimenti cumulativi
        cum.plot(cum_ret*100, color="#569878",linewidth=1.5)
        cum.fill_between(cum_ret.index, cum_ret * 100, 0,
                         cum_ret >= 0, color="#569878", alpha=0.30)
        cum.axhline(0, color="#569878")
        cum.grid(axis="y", color='#505050', linestyle='--', linewidth=1, alpha=0.5)
        cum.set_ylabel("Cumulative Return (%)", size=15, fontweight='bold')

        # crea il grafico del drawdown
        dra.plot(drawdown.index, drawdown * 100, color="#C04E4E", alpha=0.50, linewidth=0.5)
        dra.fill_between(drawdown.index, drawdown * 100, 0,
                         drawdown * 100 <= 0, color="#C04E4E", alpha=0.30)
        dra.grid(axis="y", color='#505050', linestyle='--', linewidth=1, alpha=0.5)
        dra.set_ylabel("Drawdown (%)", size=15, fontweight='bold')

        # visualizza il grafico
        plt.show()

    # Stampa diverse metriche di trading, incluse la durata media degli scambi, il numero di acquisti e vendite, 
    # il rendimento sul periodo,il drawdown massimo, e altre statistiche relative al trading mensile.
    def display_metrics(self):
        # calcolo dei rendimenti cumulativi
        self.get_vector_metrics()

        # Durata media delle operazioni
        try:
            seconds = self.data.loc[self.data["duration"] != 0]["duration"].mean()
            minutes = seconds // 60
            minutes_left = int(minutes % 60)
            hours = minutes // 60
            hours_left = int(hours % 24)
            days = int(hours / 24)
        except:
            minutes_left = 0
            hours_left = 0
            days = 0

        # Buy&Sell count
        buy_count = self.data["buy_count"].sum()
        sell_count = self.data["sell_count"].sum()

        # Rendimento del periodo
        return_over_period = self.data["cumulative_returns"].iloc[-1] * 100

        # Calcolo del drawdown max
        dd_max = -self.data["drawdown"].min() * 100

        # HIT ratio ovvero la percentuale di successo dei trades.
        nb_trade_positive = len(self.data.loc[self.data["returns"] > 0])
        nb_trade_negative = len(self.data.loc[self.data["returns"] < 0])
        hit = nb_trade_positive * 100 / (nb_trade_positive + nb_trade_negative)

        # Risk reward ratio - rapporto rischio rendimento
        average_winning_value = self.data.loc[self.data["returns"] > 0]["returns"].mean()
        average_losing_value = self.data.loc[self.data["returns"] < 0]["returns"].mean()

        rr_ratio = -average_winning_value / average_losing_value

        # Metric ret/month
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]

        # calcolo del ritorno mensile
        ben_month = []

        for month in months:
            for year in years:
                try:
                    information = self.data.loc[f"{year}-{month}"]
                    cum = information["returns"].sum()
                    ben_month.append(cum)
                except:
                    pass

        sr = pd.Series(ben_month, name="returns")

        pct_winning_month = (1-(len(sr[sr <= 0]) / len(sr)))*100
        best_month_return = np.max(ben_month) * 100
        worse_month_return = np.min(ben_month) * 100

        # ritorno medio mensile
        cmgr = np.mean(ben_month) * 100

        print("------------------------------------------------------------------------------------------------------------------")
        print(f" AVERAGE TRADE LIFETIME: {days}D  {hours_left}H  {minutes_left}M \t Nb BUY: {buy_count} \t Nb SELL: {sell_count} ")
        print("                                                                                                                  ")
        print(f" Return (period): {'%.2f' % return_over_period}% \t\t\t\t Maximum drawdown: {'%.2f' % dd_max}%")
        print(f" HIT ratio: {'%.2f' % hit}% \t\t\t\t\t\t R ratio: {'%.2f' % rr_ratio}")
        print(f" Best month return: {'%.2f' % best_month_return}% \t\t\t\t Worse month return: {'%.2f' % worse_month_return}%")
        print(f" Average ret/month: {'%.2f' % cmgr}% \t\t\t\t Profitable months: {'%.2f' % pct_winning_month}%")
        print("------------------------------------------------------------------------------------------------------------------")

    # Restituisce il rendimento sul periodo e il drawdown massimo
    def get_ret_dd(self):
        
        # calcola delle metriche
        self.get_vector_metrics()

        # Calcolo del Rendimento nel Periodo
        return_over_period = self.data["cumulative_returns"].iloc[-1] * 100

        # Calcolo del Drawdown Massimo
        dd_max = self.data["drawdown"].min() * 100

        return return_over_period, dd_max

    # Chiama i metodi display_metrics e display_graphs per visualizzare le metriche e i grafici.
    def display(self, title=None):
        self.display_metrics()
        self.display_graphs(title)
