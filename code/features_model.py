import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import trange
from datetime import date, timedelta

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from sklearn.cluster import KMeans
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
    lr = [0.0005],
    batch_size = [5],
    seq_length = [21],
    output_size = [7],
    num_pred_features = [16],
    date_range = [40],
    input_size = [19],
    hidden_layer_size = [328],
    num_layers = [1],
    ratio = [0.7],
    num_epochs = [50],
    dropout_rate = [0.8],
    lossfunc = [nn.SmoothL1Loss(beta=0.01, reduction = 'sum')],
    cluster = [False],
    RT=[False]
)


#data_file =
#model_file =


class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def prepare_data_features_model(df, seq_length, output_size, date_range):
    '''
    df: pandas df contain all the data
    seq_length: number of days consider as input
    output_size: number of days to predict
    date_range: length of history to consider
    Output: prepared data and state list
    '''
    full_data = []
    state_ordered = []

    if run.cluster == True:
       df = df.drop(columns=['cluster'])

    for state in df.index.get_level_values('FIPS').unique():
        df_state = df.iloc[df.index.get_level_values('FIPS') == state]

        if len(df_state) <= date_range:
            L = len(df_state.to_numpy())
            train_state = []
            for i in range(L - seq_length - output_size + 1):
                train_seq = df_state.to_numpy()[i:i + seq_length]
                train_label = df_state.to_numpy()[i:i + seq_length + output_size][seq_length:seq_length + output_size,
                              1:-2]
                train_state.append((train_seq, train_label))

            for x in train_state:
                full_data.append(x)
            state_ordered.append(state)
        else:
            df_state = df.iloc[df.index.get_level_values('FIPS') == state][-date_range:]

            train_state = []

            L = len(df_state.to_numpy())
            for i in range(L - seq_length - output_size + 1):
                train_seq = df_state.to_numpy()[i:i + seq_length]
                train_label = df_state.to_numpy()[i:i + seq_length + output_size][seq_length:seq_length + output_size,
                              1:-2]
                train_state.append((train_seq, train_label))

            for x in train_state:
                full_data.append(x)
            state_ordered.append(state)
    return full_data, state_ordered


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_rate):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)


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


def splitdata(full_data, ratio, batch_size):
    train_size = int(ratio * len(full_data))
    test_size = len(full_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, test_loader



runs = RunBuilder.get_runs(params)

for run in RunBuilder.get_runs(params):
    raw_df = pd.read_pickle(data_file)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    df = raw_df[(raw_df['Date'] <= pd.to_datetime(run.day_id))]
    df = df[~df['FIPS'].isin(['05', '46'])]

    if run.RT == True:
       print(df.loc[df[(df['FIPS'] == '36')][-7:].index, 'Rt_median'])
       df = Arima_rt(raw_df, df, run)
       print(df.loc[df[(df['FIPS'] == '36')][-7:].index, 'Rt_median'])

    df = df.set_index(['FIPS', 'Date'])

    #convert to incident
    df[run.type] = (df[run.type] / df['total_pop']) * 10000
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_normalized = scaler.fit_transform(df.iloc[:, 1:])
    scaler_cases = MinMaxScaler(feature_range=(0, 1))
    train_cases_normalized = scaler_cases.fit_transform(np.asarray(df.iloc[:, 0]).reshape(-1, 1))
    df.iloc[:, 1:] = train_features_normalized
    df[run.type] = train_cases_normalized

    select_df = df.loc[df.index.get_level_values('Date') <= run.day_id]


    '''
    Train model for each cluster
    '''


    full_data_main, state_ordered = prepare_data_features_model(select_df, run.seq_length,
                                                                run.output_size, run.date_range)

    model_main = LSTM(run.input_size, run.hidden_layer_size, run.num_layers, run.output_size*run.num_pred_features,
                          run.dropout_rate)

    train_loader_main, test_loader_main = splitdata(full_data_main, run.ratio, run.batch_size)

    loss_function = run.lossfunc

    optimizer_main = torch.optim.Adam(model_main.parameters(), lr=run.lr)

    track_loss_train = []
    track_loss_test = []
    best_loss = 100000

    for i in trange(run.num_epochs):

        model_main.train()
        epoch_loss_train = 0

        for i, (seq, labels) in enumerate(train_loader_main):
            optimizer_main.zero_grad()
            seq = torch.as_tensor(seq).reshape(-1, run.seq_length, run.input_size)
            model_main.hidden_cell = (torch.zeros(run.num_layers, seq.size()[0], run.hidden_layer_size),
                                          torch.zeros(run.num_layers, seq.size()[0], run.hidden_layer_size))
            y_pred = model_main(seq.float())

            single_loss = loss_function(y_pred.reshape(-1, run.output_size, run.num_pred_features), torch.as_tensor(labels).float())
            single_loss.backward()
            optimizer_main.step()

            epoch_loss_train += single_loss.item()

        track_loss_train.append(epoch_loss_train)

        with torch.no_grad():
            epoch_loss_test = 0

            for i, (seq, labels) in enumerate(test_loader_main):
                seq = torch.as_tensor(seq).reshape(-1, run.seq_length, run.input_size)
                model_main.hidden_cell = (torch.zeros(run.num_layers, seq.size()[0], run.hidden_layer_size),
                                              torch.zeros(run.num_layers, seq.size()[0], run.hidden_layer_size))
                y_pred = model_main(seq.float())

                single_loss = loss_function(y_pred.reshape(-1, run.output_size, run.num_pred_features), torch.as_tensor(labels).float())
                epoch_loss_test += single_loss.item()

            track_loss_test.append(epoch_loss_test)


        if epoch_loss_test  < best_loss:

            best_loss = epoch_loss_test
            print('Train Loss: ', epoch_loss_train)
            print('Test Loss: ', epoch_loss_test)
            es = 0
            torch.save(model_main.state_dict(),
                           model_file + str(run.date_range) + '_' + str(run.hidden_layer_size)
                           + '_' + (pd.to_datetime(run.day_id)).strftime('%Y-%m-%d') + '_LSTM_weights.pt')
        else:
            es += 1
            print("Counter {} of 5".format(es))
            print('Train Loss: ', epoch_loss_train)
            print('Test Loss: ', epoch_loss_test)


        if es > 2:
            print("Early stopping with best_loss: ", best_loss, "and test_loss for this epoch: ",
                      epoch_loss_test,
                      "...")

            break
print(run.day_id)


