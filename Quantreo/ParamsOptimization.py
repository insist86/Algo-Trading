"""
La classe ParamsOptimization è progettata per eseguire un'ottimizzazione dei parametri di una strategia di trading su un set di dati storici. 
Utilizza diverse combinazioni di parametri variabili e parametri fissi per trovare la combinazione ottimale che massimizza un certo criterio,che è il ritorno sul drawdown massimo. 

Questa classe potrebbe essere un componente chiave per lo sviluppo di strategie di trading algoritmico e l'analisi delle loro prestazioni.
"""

import pandas as pd
from Quantreo.Backtest import *
import itertools


class ParamsOptimization:

    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range):

        self.data = data # dati storici da utilizzare
        self.TradingStrategy = TradingStrategy # Parametri fissi della strategia
        self.parameters_range = parameters_range # Range dei parametri variabili della strategia
        self.fixed_parameters = fixed_parameters # Parametri fissi della strategia

        self.dictionaries = None # Sarà utilizzato per memorizzare tutte le possibili combinazioni di parametri
        self.get_combinations()

        self.BT, self.criterion = None, None # Saranno utilizzati per eseguire il backtest e memorizzare il criterio ottimizzato

        self.best_params_sample_df, self.best_params_sample = None, None # Saranno utilizzati per memorizzare i migliori parametri trovati durante l'ottimizzazione.

        # columns Saranno utilizzate per creare un DataFrame per memorizzare i risultati dell'ottimizzazione.
        self.columns = list(self.parameters_range.keys()) # 
        self.columns.append("criterion")

    # Questo metodo crea una lista di dizionari con tutte le possibili combinazioni di parametri variabili
    # e aggiunge i parametri fissi a ciascun dizionario di combinazione
    def get_combinations(self):
        # crea una lista di dizionari con tutte le possibili combinazioni (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # aggiunge i parametri fissi a ciascun dizionario di combinazione
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)
    
    # Questo metodo inizializza un backtest con un set specifico di parametri e calcola il ritorno e il drawdown massimo della strategia con questi parametri. 
    # Successivamente, calcola un criterio (ritorno / drawdown) basato su questi valori
    def get_criterion(self, sample, params):
        # inizializza un backtest con un set specifico di parametri
        self.BT = Backtest(data=sample, TradingStrategy=self.TradingStrategy, parameters=params)

        # calcola i ritorni della strategia (su questo specifico datasets e con questi  parameters)
        self.BT.run()

        # calcola e archivia i criteri (ritorno e maximum drawdown del periodo)
        ret, dd = self.BT.get_ret_dd()
        self.criterion = ret / dd

    # Questo metodo esegue l'ottimizzazione dei parametri sulla base del criterio calcolato e memorizza i migliori parametri trovati. 
    # Crea anche un DataFrame con tutte le combinazioni di parametri e i loro criteri associati. 
    # Infine, estrae la combinazione di parametri con il miglior criterio e la memorizza    
    def get_best_params_train_set(self):
        # Store of the possible parameters combinations with the associated criterion
        # Qui mettiamo il miglior criterio sul set di addestramento per trovare i migliori parametri, MA lo sostituiremo.
        # con il miglior criterio sul set di test per avvicinarsi il più possibile alla realtà.
        storage_values_params = []

        for self.params_item in self.dictionaries:
            # Estrarre i parametri variabili dal dizionario
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            # Compute the criterion and add it to the list of params
            self.get_criterion(self.data, self.params_item)
            current_params.append(self.criterion)

            # Aggiungiamo la lista current_params ai valori di storage_params al fine di creare un dataframe.
            storage_values_params.append(current_params)

        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)

        # Estrai la riga del dataframe con i migliori parametri
        self.best_params_sample_df = df_find_params.sort_values(by="criterion", ascending=False).iloc[0:1, :]

        # Crea un dizionario con i migliori parametri sul set di addestramento per testarli successivamente sul set di test
        self.best_params_sample = dict(df_find_params.sort_values(by="criterion", ascending=False).iloc[0, :-1])
        self.best_params_sample.update(self.fixed_parameters)

