import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from Quantreo.Backtest import *
import itertools
import warnings
warnings.filterwarnings("ignore")


class CombinatorialPurgedCV:
    """
    Classe per Combinatorial Purged Cross Validation (CPCV). Questa classe suddivide un dataset
    in N partizioni per l'addestramento e il testing, applicando una purga per evitare l'overfitting
    based on temporal information.


    Parameters
    --------------------
    data : DataFrame
        Il DataFrame contenente i dati da suddividere per l'addestramento e il testing
    TradingStrategy : class
        La classe che definisce la strategia di trading da valutare
    fixed_parameters : dict
        Un dizionario con i parametri della TradingStrategy che rimarranno fissi durante la cross-validation.
    parameters_range : dict
        Un dizionario con i parametri della TradingStrategy che saranno ottimizzati. 
        Ciascun parametro dovrebbe essere associato a un intervallo o a un elenco di valori possibili.
    N : int, optional
        Il numero di partizioni da creare. Il valore predefinito è 10.
    k : int, optional
        Il numero di partizioni da utilizzare per il testing. Il valore predefinito è 2.
    purge_pct : float, optional
        La percentuale dei dati da purgare tra i set di addestramento e di test. Il valore predefinito è 0,10.

    """
    # Inizializza variabili e parametri come il DataFrame dei dati, la classe della strategia di trading, i parametri fissi e variabili, il numero di partizioni, la percentuale di dati da "purge", e altre variabili necessarie per calcolare e memorizzare i criteri e i risultati dell'ottimizzazione
    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range, N=10, k=2, purge_pct=0.10):
        # Imposto i parametri iniziali
        self.data = data
        self.TradingStrategy = TradingStrategy
        self.fixed_parameters = fixed_parameters
        self.parameters_range = parameters_range
        self.N = N
        self.k = k
        self.purge_pct = purge_pct

        # Variabili necessarie per calcolare e salvare i nostri criteri
        self.BT, self.criterion = None, None
        self.dictionaries = None
        self.best_params_sample_df, self.best_params_sample = None, None
        self.dfs_list_pbo = []
        self.smooth_result = pd.DataFrame()
        self.best_params_smoothed = list()
        self.counter = 1
        self.lambdas = []
        self.train_sample, self.test_sample, self.output_params = None, None, None
        self.lists, self.df_lists = None, None
        self.lmb_series, self.pbo = None, None

        # Crea il dataframe che conterrà i parametri ottimali (ranging parameters + criteria) 
        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion_train")
        self.columns.append("criterion_test")
        self.df_results = pd.DataFrame(columns=self.columns)
        self.train_df_list, self.test_df_list = None, None
        self.plots = {}

    # questo metodo crea una lista di dizionari con tutte le possibili combinazioni di parametri variabili
    # e aggiunge i parametri fissi a ciascun dizionario di combinazione
    def get_combinations(self):
        # # crea una lista di dizionari con tutte le possibili combinazioni (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # aggiunge i parametri fissi a ciascun dizionario di combinazione
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)

    # questi 2 metodi creano insiemi di dati di addestramento e di test, applicando un "purge" tra gli insiemi di addestramento e di test per prevenire l'overfitting basato su informazioni temporali.
    def get_index_samples(self):
        # Indici dei campioni
        nb_set = list(range(self.N))

        # Generare tutte le combinazioni di k tra N.
        combinations_test = list(combinations(nb_set, self.k))

        # Generare il complemento delle combinazioni (set di addestramento - train set)
        combinations_train = [list(set(nb_set) - set(combinaisons_test)) for combinaisons_test in combinations_test]
        self.lists = []

        # Creare una lista con l'indice del test e l'indice dell'addestramento.
        for i in range(len(combinations_test)):
            self.lists.append([list(combinations_test[i]), combinations_train[i]])
    def get_sub_samples(self):
        # Creare una divisione equa dei dati (N campioni).
        split_data = np.array_split(self.data, self.N)

        # Elencare le coppie di dataframe di addestramento e test
        self.df_lists = []

        # STEP 1: Riorganizzare e aggiungere purge ai set
        for i in range(len(self.lists)):
            # Estrarre l'indice per ciascuna coppia di sottocampioni
            list_sets = self.lists[i]
            test_idx = list_sets[0]
            train_idx = list_sets[1]

            # Creare una lista contenente i sottocampioni per ciascun periodo
            test_sets = [split_data[i] for i in test_idx]
            train_sets = []

            # Creare i periodi di Purge & embargo 
            for j in train_idx:
                train_df_ind = split_data[j]

                # Rimuovere l'inizio o la fine se l'insieme di addestramento attraversa un set di test.
                if (j - 1 in test_idx) and (j + 1 in test_idx):
                    split_embargo = 2 * int(len(train_df_ind) * self.purge_pct)
                    split_purge = int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[split_embargo:-split_purge, :]
                elif j + 1 in test_idx:
                    split_purge = int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[:-split_purge, :]
                elif j - 1 in test_idx:
                    split_embargo = 2 * int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[split_embargo:, :]

                # Aggiungere purged set alla lista
                train_sets.append(train_df_ind)

            self.df_lists.append([train_sets, test_sets])

        # STEP 2: Concatenare gli insiemi consecutivi
        # Creare liste vuote che li conterranno
        train_output_list = []
        test_output_list = []

        # Analizziamo per ogni coppia (train, test) se possiamo concatenare gli insiemi consecutivi.
        for j in range(len(self.df_lists)):
            new_list_train = [self.df_lists[j][0][0]]
            new_list_test = [self.df_lists[j][1][0]]

            # Verifichiamo separatamente se è necessario concatenare alcuni insiemi per gli insiemi di addestramento e quelli di test,
            # poiché ciò non aumenterà molto i calcoli (1 secondo), ma diminuirà notevolmente la complessità del codice.

            # Estrai ogni lista di insiemi di addestramento per verificare se è necessaria una concatenazione
            for i in range(1, len(self.df_lists[j][0])):
                # Estrai l'indice per l'ultimo valore dell'insieme precedente e il primo valore dell'insieme corrente
                idx_end = self.df_lists[j][0][i - 1].index[-1]
                idx_start = self.df_lists[j][0][i].index[0]

                # Verifica se l'indice dell'insieme successivo è lo stesso dell'indice che segue l'ultimo indice dell'insieme precedente
                # (utilizzando la variabile data che è l'input dei dati)
                sub_data = self.data.loc[idx_end:]
                normal_start_idx = sub_data.index[1]

                # Verifica se l'indice dell'insieme successivo è lo stesso dell'indice che segue l'ultimo indice dell'insieme precedente
                # (utilizzando la variabile data che è l'input dei dati)
                if idx_start == normal_start_idx:
                    # Prendi l'ultimo insieme
                    current_df = new_list_train[-1]

                    # Aggiungi questo insieme a quello precedente perché sono consecutivi
                    current_df_updated = pd.concat((current_df, self.df_lists[j][0][i]), axis=0)

                    # Sostituisci l'insieme con l'insieme aggiornato
                    new_list_train[-1] = current_df_updated
                else:
                    # # Quando l'ultimo indice dell'insieme precedente non è il primo indice di questo insieme, 
                    # iniziamo un altro insieme
                    new_list_train.append(self.df_lists[j][0][i])

            # Aggiungi la lista di insiemi di addestramento a una lista (perché la lista di insiemi di addestramento è solo un percorso)
            train_output_list.append(new_list_train)

            # Estrai ogni lista di insiemi di test per verificare se è necessaria una concatenazione
            for i in range(1, len(self.df_lists[j][1])):
                # Estrai l'indice per l'ultimo valore dell'insieme precedente e il primo valore dell'insieme corrente
                idx_end = self.df_lists[j][1][i - 1].index[-1]
                idx_start = self.df_lists[j][1][i].index[0]

                # Utilizza i dati dall'inizio (la variabile data) per verificare se idx_start è davvero l'indice successivo
                # Se sì, significa che i due insiemi sono consecutivi
                sub_data = self.data.loc[idx_end:]
                normal_start_idx = sub_data.index[1]

                # Verifica se l'indice dell'insieme successivo è lo stesso dell'indice che segue l'ultimo indice dell'insieme precedente
                # (utilizzando la variabile data che è l'input dei dati)
                if idx_start == normal_start_idx:
                    # Prendi l'ultimo insieme
                    current_df = new_list_test[-1]

                    # Aggiungi questo insieme a quello precedente perché sono consecutivi
                    current_df_updated = pd.concat((current_df, self.df_lists[j][1][i]), axis=0)

                    # Sostituisci l'insieme con l'insieme aggiornato
                    new_list_test[-1] = current_df_updated
                else:
                    # Quando l'ultimo indice dell'insieme precedente non è il primo indice di questo insieme, iniziamo un altro insieme
                    new_list_test.append(self.df_lists[j][1][i])

            # Aggiungi la lista di insiemi di test a una lista (perché la lista di insiemi di test è solo un percorso)
            test_output_list.append(new_list_test)

        # Sostituiamo la lista di insiemi di addestramento e di test per ogni coppia con lo stesso insieme ma con una concatenazione
        # quando è possibile.
        for j in range(len(self.df_lists)):
            self.df_lists[j][0] = train_output_list[j]
            self.df_lists[j][1] = test_output_list[j]

    # questo metodo concatena ciascun insieme di addestramento, prepara alcuni pesi se necessario, calcola i ritorni per ciascun insieme di dati e calcola il criterio (il rapporto di Calmar)
    def get_returns(self, train=True):
        #  # Concatenare ogni insieme di addestramento per addestrare i pesi (SOLO) con il maggior numero possibile di dati
        self.train_sample = pd.concat(self.train_df_list, axis=0)

        # Preparare alcuni pesi (se necessario)
        Strategy = self.TradingStrategy(self.train_sample, self.params_item)

        # Estrarre i pesi che ci permetteranno di eseguire il nostro algoritmo (specialmente per strategie che richiedono un addestramento come ML)
        self.output_params = Strategy.output_dictionary

        # Creare una lista vuota per memorizzare i rendimenti (e concatenarli)
        list_return = []

        # Scegliere l'insieme giusto a seconda della modalità Addestramento o Test (Train or Test)
        if train:
            df_list = self.train_df_list
        else:
            df_list = self.test_df_list

        for tsample in df_list:
            # Calcola i ritorni
            self.BT = Backtest(data=tsample, TradingStrategy=self.TradingStrategy, parameters=self.output_params)
            self.BT.run()

            # Aggiungi i ritorni nelle liste
            list_return.append(self.BT.data)

        # Concatena la lista per avere l'intero backtest (dobbiamo farlo per evitare valori aberranti nel gap)
        sets = pd.concat(list_return, axis=0)

        # Iniziamo nuovamente la classe Backtest (senza eseguire il backtest) solo per calcolare il criterio
        self.BT = Backtest(data=sets, TradingStrategy=self.TradingStrategy, parameters=self.params_item)

        # Calcolo e memorizzazione del criterio (Rendimento nel periodo diviso per il massimo drawdown)
        ret, dd = self.BT.get_ret_dd()

        # Utilizziamo il rapporto Calmar come criterio
        self.criterion = ret / np.abs(dd)

    # questo metodo Trova e memorizza i migliori parametri per ciascun set di addestramento e di test basandosi sul criterio calcolato
    def get_best_params_set(self):
        storage_values_params = []

        for self.params_item in self.dictionaries:
            # Estrarre i parametri variabili dal dizionario.
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            # Calcolare il criterio e aggiungerlo alla lista dei parametri (criterion train)
            self.get_returns(train=True)
            current_params.append(self.criterion)

            # Calcolare il criterio e aggiungerlo alla lista dei parametri (criterion test)
            self.get_returns(train=False)
            current_params.append(self.criterion)

            storage_values_params.append(current_params)

        # Estrarre la riga del dataframe con i migliori parametri.
        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)
        self.dfs_list_pbo.append(df_find_params)

        # Estrarre la riga del dataframe con i migliori parametri.
        self.best_params_sample_df = df_find_params.sort_values(by="criterion_train", ascending=False).iloc[0:1, :]

        # !! Mettiamo l'ultimo valore dell'indice come indice.
        # Perché SENZA di ciò, quando si sostituisce il valore del criterio in seguito, si sostituiranno tutti i valori con lo stesso indice.
        self.best_params_sample_df.index = [self.counter]

        # Aggiungiamo i migliori parametri al dataframe che contiene tutti i migliori parametri per ciascun periodo.
        self.df_results = pd.concat((self.df_results, self.best_params_sample_df), axis=0)

        # Crea un dizionario con i migliori parametri sull'insieme di addestramento per testarli successivamente sull'insieme di test.
        self.best_params_sample = dict(df_find_params.sort_values(by="criterion_train", ascending=False).iloc[0, :-2])
        self.best_params_sample.update(self.fixed_parameters)

    # questo metodo Esegue l'ottimizzazione sui diversi insiemi di dati di addestramento e di test, trovando i migliori parametri per ciascun set.
    def run_optimization(self):
        # Crea il sottocampione sub-samples
        self.get_sub_samples()
        self.get_combinations()

        # Esegui l'ottimizzazione
        for couple_list in tqdm(self.df_lists):
            # Estratte i set train and test dal campione
            self.train_df_list, self.test_df_list = couple_list[0], couple_list[1]

            self.get_best_params_set()
            self.counter += 1

    
    def get_combination_graph(self, ax):
        # Iterare su ciascuna coppia di insiemi per visualizzare......
        for i in range(len(self.df_lists)):
            # Estrarre un campione
            list_couple = self.df_lists[i]

            # Estrarre i sets train e test in questa coppia
            train_df_list, test_df_list = list_couple[0], list_couple[1]

            # Concatenare ciascun insieme in veri periodi di addestramento e di test.
            df_test = pd.concat(test_df_list, axis=0)
            df_train = pd.concat(train_df_list, axis=0)

            # Tracciare i periodi per ciascuna coppia (aggiungiamo i alla prima serie per poter visualizzare facilmente ciascun insieme).
            ax.plot(df_train.index, np.ones(len(df_train)) + i, "o", color='#6F9FCA', linewidth=1)
            ax.plot(df_test.index, np.ones(len(df_test)) + i, "o", color='#CA7F6F', linewidth=1)

        # Alcune impostazioni di layout.
        ax.set_title(f"Nb tests: {len(self.df_lists)}")
        plt.legend(["TRAIN", "TEST"], loc="upper left")
        plt.show()

    # questo metodo Calcola la Probabilità di Sovraottimizzazione (Probability of Overfitting - PBO) basandosi sui logit dei ranghi.
    def get_pbo(self):
        for ind_df in self.dfs_list_pbo:
            # Ordina il df usando la colonna criterion test
            dfp_ordered = ind_df.sort_values(by="criterion_test", ascending=False)

            # Reindirizza il df
            dfp_ordered.index = [len(dfp_ordered) + 1 - i for i in range(1, len(dfp_ordered) + 1)]

            # Ordina nuovamente il df in base alla colonna del criterion train 
            dfp_rank = dfp_ordered.sort_values(by="criterion_train", ascending=False)

            # Estrai il rango nel test della migliore combinazione nell'insieme di addestramento.
            rank = dfp_rank.index[0]

            # Create the relative rank of the OOS performance
            wcb = rank / (len(dfp_ordered) + 1)

            # Creare il logit
            lambda_c = np.log(wcb / (1 - wcb))

            # Aggiungilo alla lista dei logit.
            self.lambdas.append(lambda_c)
            print(dfp_rank.index[0], lambda_c)

        # Crea una serie con la lista di lambda.
        self.lmb_series = pd.Series(self.lambdas)

        # Calcola la probabilità di overfitting
        self.pbo = 100 * len(self.lmb_series[self.lmb_series < 0]) / len(self.lmb_series)

    def get_pbo_graph(self, ax):
        ax.hist(self.lmb_series, color="#6F9FCA", bins=10, edgecolor='black', density=True)
        sns.kdeplot(self.lmb_series, color="#CA6F6F", ax=ax)
        ax.text(0.95, 0.90, f'Probability of Overfitting: {self.pbo:.2f} %', horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.9'))
        ax.set_title("Hist of Rank Logits")
        ax.set_xlabel("Logits")
        ax.set_ylabel("Frequency")

    def get_degration_graph(self, ax):
        x = self.df_results["criterion_train"]
        y = self.df_results["criterion_test"]

        # Linea di Regressione
        coeffs = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = coeffs[1] + coeffs[0] * line_x

        # P(SR_OOS < 0)
        ct_oos = self.df_results["criterion_test"]
        p_oos_pos = 100*len(ct_oos[ct_oos>0]) / len(ct_oos)

        ax.scatter(x, y)
        ax.plot(line_x, line_y, color='#CA6F6F')
        ax.text(0.95, 0.90, f'P(SR[00S] > 0): {p_oos_pos:.2f} %', horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.9'))
        ax.set_title(f"Criterion TEST = {coeffs[1]:.2f} + TRAIN * {coeffs[0]:.2f} + epsilon")
        ax.set_xlabel("Criterion Train")
        ax.set_ylabel("Criterion Test")

    # Visualizza vari grafici, inclusi il grafico PBO, il grafico della degradazione e il grafico delle combinazioni, per aiutare a interpretare i risultati dell'ottimizzazione.
    def display_all_graph(self):
        fig, axes = plt.subplot_mosaic('AB;CC', figsize=(15, 8))

        self.get_pbo_graph(axes["A"])
        self.get_degration_graph(axes["B"])
        self.get_combination_graph(axes["C"])

        fig.subplots_adjust(hspace=0.9)

        plt.show()git 