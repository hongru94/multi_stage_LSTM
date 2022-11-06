import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import trange
from datetime import date, timedelta
import pmdarima as pm

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from sklearn.cluster import KMeans
import random
import torch.nn.functional as F
pd.options.mode.chained_assignment = None

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

target = []
for i in perdelta(date(2021, 12, 4), date(2022, 1, 2), timedelta(days=7)):
   t = i.strftime('%Y-%m-%d')
   target.append(t)


params = OrderedDict(
    day_id = ['2021-12-25'],
    type = ['cases'],
    seq_length = [21],
    output_size = [7],
    num_pred_features = [16],
    date_range_main = [40],
    date_range_features = [40],
    input_size = [19],
    main_hidden_layer_size = [256],
    features_hidden_layer_size = [328],
    main_num_layers = [1],
    features_num_layers = [1],
    sample_size = [10],
    cluster = [False],
    num_future_weeks = [4],
    dropout = [0.8],
    Rt = [False]
)

#data_file = 
#main_file =
#features_file =
#prediction_file =


demo = pd.read_csv('/Users/hongru/Projects/Covid_projection/data/age_US_state.csv',
                  dtype = {'FIPS' : np.str_}).set_index('FIPS')
# Clustering Based on population

def K_means(df, number_of_clusters):
    '''
    Add a cluster column to df
    '''
    kmeans = KMeans(n_clusters=number_of_clusters)
    y_kmeans = kmeans.fit_predict(df['total_pop'].to_numpy().reshape(-1, 1))

    df['cluster'] = y_kmeans
    return df

def quantile(df, quantile):
    '''
    Add a cluster column to df
    '''
    cluster = []
    for i in range(len(quantile)-1):
        cluster.append(i)
    df.loc[:, 'cluster'] = pd.qcut(df['total_pop'],q = quantile, duplicates='raise', labels = cluster).to_numpy()
    return df

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_rate):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_rate)


        self.linear = nn.Linear(hidden_layer_size, 1000)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(1000, output_size)

    def forward(self, input_seq):
        h = (torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size),
             torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size))

        lstm_out, self.hidden_cell = self.lstm(input_seq, h)

        # only return the results for last sequence
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)
        predictions = F.relu(predictions)
        predictions = self.dropout(predictions)
        predictions = self.linear2(predictions)

        return predictions



def predict(model, data, sequence_length, num_layers, input_size, hidden_layer_size):
    pred = []

    with torch.no_grad():
        for seq in data:
            seq = torch.tensor(seq).reshape(-1, sequence_length, input_size)
            model.hidden_cell = (torch.zeros(num_layers, seq.size(0), hidden_layer_size),
                                 torch.zeros(num_layers, seq.size(0), hidden_layer_size))
            prediction = model(seq.float())
            pred.append(prediction)

    return pred


def get_new_predict_data(pred_features, pred, predict_data, df, state_ordered, run):
    """
    pred_featues: get new predicted times series features

    pred: add new data to pred, remove extra rows
    """
    j = 0

    output_data = []
    for i in state_ordered:
        constant = (df.iloc[df.index.get_level_values('FIPS') == i].iloc[:run.output_size, -2:]).to_numpy()
        # -2 here is the number of constant features

        cases_pred = pred[j]
        features_pred = pred_features[j].reshape(run.output_size, run.num_pred_features)
        # 3 here is the number of time series features, not include cases

        new_data = np.concatenate((cases_pred.T, features_pred), axis=1)
        new_data = np.concatenate((new_data, constant), axis=1)

        nxt_round_data = np.concatenate((predict_data[j][run.output_size:], new_data), axis=0)

        output_data.append(nxt_round_data)
        j += 1

    return output_data

def Arima_rt(raw_df, df, run):


    df_rt = raw_df[['Date', 'FIPS', 'Rt_median']]
    df_rt = df_rt[(raw_df['Date'] <= (pd.to_datetime(run.day_id) + timedelta(-7)))]

    for state in df['FIPS'].unique():

        fit = pm.auto_arima(df_rt[df_rt['FIPS'] == state]['Rt_median'][-80:])
        forecasts = pd.Series(fit.predict(7))
        df.loc[df[(df['FIPS'] == state)][-7:].index, 'Rt_median'] = forecasts.values

    return df


class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

def make_preidctions(runs):

    for run in runs:
        raw_df = pd.read_pickle(data_file)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        df = raw_df[(raw_df['Date'] <= pd.to_datetime(run.day_id))]

        if run.Rt == True:

            print(df.loc[df[(df['FIPS'] == '36')][-7:].index, 'Rt_median'])
            df = Arima_rt(raw_df, df, run)
            print(df.loc[df[(df['FIPS'] == '36')][-7:].index, 'Rt_median'])
        df = df.set_index(['FIPS', 'Date'])
        #incident rate
        df[run.type] = (df[run.type] / df['total_pop']) * 10000

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df.iloc[:, 1:])
        train_features_normalized = scaler.transform(df.iloc[:, 1:])
        scaler_cases = MinMaxScaler(feature_range=(0, 1))
        scaler_cases.fit(np.asarray(df.iloc[:, 0]).reshape(-1, 1))
        train_cases_normalized = scaler_cases.transform(np.asarray(df.iloc[:, 0]).reshape(-1, 1))
        df.iloc[:, 1:] = train_features_normalized
        df[run.type] = train_cases_normalized

        if run.cluster == True:
            df_output = pd.DataFrame(columns=['FIPS', 'Date', 'Predicted_Cases', 'Week', 'Run', 'cluster'])
            df_cluster = quantile(df, [0, 0.25, 0.75, 1])
            df_cluster.to_csv('/Users/hongru/Projects/Covid_projection/LSTM_Model/outputs/cluster_df'
                              '/df_cluster_quantile4.csv')
            for cluster in df_cluster['cluster'].unique():
                predict_data = []
                state_ordered = []
                df = df_cluster[df_cluster['cluster'] == cluster]
                df = df.drop(columns=['cluster'])
                for state in df.index.get_level_values('FIPS').unique():
                    df_state = df.iloc[(df.index.get_level_values('FIPS') == state) \
                                                & (df.index.get_level_values('Date') <= \
                                                   pd.to_datetime(run.day_id))][-(run.seq_length):]
                    predict_data.append(df_state.to_numpy())
                    state_ordered.append(state)
                '''
                Load model
                '''

                m_state_dict_features = torch.load(features_file + 'cluster' + str(cluster) + '_' +
                                                   str(run.features_num_layers) + '_' +
                                                   str(run.features_hidden_layer_size) + '_'
                                                   + (pd.to_datetime(run.day_id)).strftime('%Y-%m-%d')
                                                   +  '_LSTM_weights.pt')

                m_state_dict_main = torch.load(main_file + 'cluster' + str(cluster) + '_'
                                               + str(run.main_num_layers) + '_' +
                                                str(run.main_hidden_layer_size) + '_'
                                                + (pd.to_datetime(run.day_id)).strftime('%Y-%m-%d')
                                                 + '_LSTM_weights.pt')

                for sample in trange(run.sample_size):
                    dropout_rate = random.uniform(0, 1)

                    model_main = LSTM(run.input_size, run.main_hidden_layer_size, run.main_num_layers, run.output_size,
                                      dropout_rate)
                    model_main.load_state_dict(m_state_dict_main)

                    model_features = LSTM(run.input_size, run.features_hidden_layer_size, run.main_num_layers,
                                          run.output_size * run.num_pred_features,
                                          dropout_rate)
                    model_features.load_state_dict(m_state_dict_features)

                    '''
                    Make predictions for week 1
                    '''

                    new_cases = predict(model_main, predict_data, run.seq_length, run.main_num_layers,
                                        run.input_size, run.main_hidden_layer_size)
                    new_features = predict(model_features, predict_data, run.seq_length, run.features_num_layers,
                                           run.input_size, run.features_hidden_layer_size)

                    start_date = (pd.to_datetime(run.day_id) + timedelta(1)).strftime('%Y-%m-%d')
                    end_date = (pd.to_datetime(run.day_id) + timedelta(7)).strftime('%Y-%m-%d')

                    x = pd.date_range(start_date, periods=7, freq='D')

                    j = 0
                    for i in state_ordered:
                        predicted = new_cases[j]
                        incidence = scaler_cases.inverse_transform(np.asarray(predicted).reshape(-1, 1))
                        #pred_cases = incidence
                        pred_cases = (incidence / 10000) * demo.loc[i]['total_pop']

                        for num in range(len(x)):
                            dic = {
                                'FIPS': i,
                                'Date': x[num],
                                'Predicted_Cases': pred_cases[num].item(),
                                'Week': 'Week1',
                                'Run': sample,
                                'cluster': cluster
                            }
                            df_output = df_output.append(dic, ignore_index=True)
                        j += 1

                    combined_df = get_new_predict_data(new_features, new_cases, predict_data, df,
                                                       state_ordered, run)

                    for week_id in range(2, run.num_future_weeks+1):

                        new_cases = predict(model_main, combined_df, run.seq_length, run.main_num_layers,
                                            run.input_size, run.main_hidden_layer_size)
                        new_features = predict(model_features, combined_df, run.seq_length, run.features_num_layers,
                                               run.input_size, run.features_hidden_layer_size)

                        start_date_update = (pd.to_datetime(start_date) + timedelta(7 * (week_id - 1))).strftime(
                            '%Y-%m-%d')
                        end_date_update = (pd.to_datetime(end_date) + timedelta(7 * (week_id - 1))).strftime('%Y-%m-%d')

                        x_update = pd.date_range(start_date_update, periods=7, freq='D')

                        for i in range(len(state_ordered)):
                            predicted_update = new_cases[i]
                            incidence_update = scaler_cases.inverse_transform(
                                np.asarray(predicted_update).reshape(-1, 1))
                            # pred_cases_update = incidence_update
                            pred_cases_update = (incidence_update / 10000) * demo.loc[state_ordered[i]]['total_pop']
                            for num in range(len(x_update)):
                                dic = {
                                    'FIPS': state_ordered[i],
                                    'Date': x_update[num],
                                    'Predicted_Cases': pred_cases_update[num].item(),
                                    'Week': 'Week' + str(week_id),
                                    'Run': sample,
                                    'cluster': cluster
                                }
                                df_output = df_output.append(dic, ignore_index=True)

                        combined_df = get_new_predict_data(new_features, new_cases, combined_df, df,
                                                           state_ordered, run)



        else:

            df_output = pd.DataFrame(columns=['FIPS', 'Predicted_Cases', 'Week', 'Run'])

            predict_data = []
            state_ordered = []

            for state in df.index.get_level_values('FIPS').unique():
                    df_state = df.iloc[(df.index.get_level_values('FIPS') == state) \
                                                & (df.index.get_level_values('Date') <= \
                                                   pd.to_datetime(run.day_id))][-(run.seq_length):]
                    predict_data.append(df_state.to_numpy())
                    state_ordered.append(state)

            '''
            Load model
            '''

            m_state_dict_features = torch.load(features_file + str(run.date_range_features) + '_' +
                                               str(run.features_hidden_layer_size) + '_'
                                               + (pd.to_datetime(run.day_id)).strftime('%Y-%m-%d')
                                               + '_LSTM_weights.pt')

            m_state_dict_main = torch.load(main_file + str(run.date_range_main) + '_' +
                                           str(run.main_hidden_layer_size) + '_'
                                           + (pd.to_datetime(run.day_id)).strftime('%Y-%m-%d')
                                           + '_LSTM_weights.pt')

            for sample in trange(run.sample_size):

                #dropout_rate = random.uniform(0, 1)
                dropout_rate = run.dropout

                model_main = LSTM(run.input_size, run.main_hidden_layer_size, run.main_num_layers, run.output_size,
                                  dropout_rate)
                model_main.load_state_dict(m_state_dict_main)

                model_features = LSTM(run.input_size, run.features_hidden_layer_size, run.features_num_layers,
                                      run.output_size * run.num_pred_features,
                                      dropout_rate)
                model_features.load_state_dict(m_state_dict_features)



                '''
                Make predictions
                '''
                new_cases = predict(model_main, predict_data, run.seq_length, run.main_num_layers,
                                run.input_size, run.main_hidden_layer_size)
                new_features = predict(model_features, predict_data, run.seq_length, run.features_num_layers,
                                   run.input_size, run.features_hidden_layer_size)

                start_date = (pd.to_datetime(run.day_id) + timedelta(1)).strftime('%Y-%m-%d')
                end_date = (pd.to_datetime(run.day_id) + timedelta(7)).strftime('%Y-%m-%d')

                x = pd.date_range(start_date, periods=7, freq='D')

                j = 0
                for i in state_ordered:

                    predicted = new_cases[j]
                    incidence = scaler_cases.inverse_transform(np.asarray(predicted).reshape(-1, 1))
                    pred_cases = incidence
                    pred_cases = (incidence / 10000) * demo.loc[i]['total_pop']
                    # pred_cases = (incidence**2) * demo.loc[i]['total_pop']
                    # pred_cases = np.exp(incidence) * demo.loc[i]['total_pop']

                    #
                    dic = {
                        'FIPS': i,
                        'Predicted_Cases': np.sum(pred_cases),
                        'Week': 'Week1',
                        'Run': sample
                    }
                    df_output = df_output.append(dic, ignore_index=True)
                    j += 1

                combined_df = get_new_predict_data(new_features, new_cases, predict_data, df,
                                                       state_ordered, run)

                for week_id in range(2, run.num_future_weeks+1):

                    new_cases = predict(model_main, combined_df, run.seq_length, run.main_num_layers,
                                        run.input_size, run.main_hidden_layer_size)
                    new_features = predict(model_features, combined_df, run.seq_length, run.features_num_layers,
                                           run.input_size, run.features_hidden_layer_size)

                    start_date_update = (pd.to_datetime(start_date) + timedelta(7 * (week_id - 1))).strftime('%Y-%m-%d')
                    end_date_update = (pd.to_datetime(end_date) + timedelta(7 * (week_id - 1))).strftime('%Y-%m-%d')

                    x_update = pd.date_range(start_date_update, periods=7, freq='D')

                    for i in range(len(state_ordered)):
                        predicted_update = new_cases[i]
                        incidence_update = scaler_cases.inverse_transform(np.asarray(predicted_update).reshape(-1, 1))
                        pred_cases_update = incidence_update
                        pred_cases_update = (incidence_update / 10000) * demo.loc[state_ordered[i]]['total_pop']


                        dic = {
                            'FIPS': state_ordered[i],
                            'Predicted_Cases': np.sum(pred_cases_update),
                            'Week': 'Week' + str(week_id),
                            'Run': sample
                        }
                        df_output = df_output.append(dic, ignore_index=True)

                    combined_df = get_new_predict_data(new_features, new_cases, combined_df, df, state_ordered, run)

        df_output['Predicted_Cases'][df_output['Predicted_Cases'] < 0] = 0
        df_output.to_csv(prediction_file + run.day_id + '_'
                     + str(run.date_range_main) + '_' + str(run.main_hidden_layer_size) + '_'
                     + str(run.date_range_features) + '_' + str(run.features_hidden_layer_size) + '.csv',
                     index=False)
        print(run.day_id)

runs = RunBuilder.get_runs(params)
make_preidctions(runs)









