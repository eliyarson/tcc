# %%
import tensorflow as tf
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
csv_path = '/mnt/c/Users/eli_t/tcc/microdados/basica2018-01.txt'

#df = pd.read_csv(csv_path, encoding='latin-1', sep=';')

df_master = pd.DataFrame()
anos = ['2008', '2009', '2010', '2011', '2012',
        '2013', '2014', '2015', '2016', '2017', '2018']
mes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for i in anos:
    print(i)
    for j in mes:
        print(j)
        path = '/mnt/c/Users/eli_t/tcc/microdados/basica/'
        csv_path = f'{path}basica{i}-{j}.txt'
        df = pd.read_csv(csv_path, encoding='latin-1', sep=';')
        df = df[df['sg_iata_origem'] == 'GRU']
        data = df.groupby('dt_partida_real').agg(flights=('id_basica', 'nunique'),
                                                 paid_passengers=(
            'nr_passag_pagos', 'sum')
            # ,companies=('id_empresa', 'nunique')
        )
        data = data[data['flights'] > 10]
        df_master = df_master.append(data)
        print(data.shape)


# %%

df_master.plot(subplots=True)

avg_flights = df_master['flights'].mean().round()
avg_pp = df_master['paid_passengers'].mean().round()
#avg_companies = df_master['companies'].mean().round()

df_master['flights'] = df_master['flights'].apply(
    lambda x: avg_flights if x <= 50 else x)
df_master['paid_passengers'] = df_master['paid_passengers'].apply(
    lambda x: avg_pp if x <= 10000 else x)
#df_master['companies'] = df_master['companies'].apply(lambda x: avg_companies if x<=20 else x)


df_master.plot(subplots=True)
# REMOVENDO RUIDOS
# %%
TRAIN_SPLIT = 3600
past_history = 90
future_target = 90
EPOCHS = 5
EVALUATION_INTERVAL = 3000
tf.random.set_seed(13)
BATCH_SIZE = 1000
BUFFER_SIZE = 1000
STEP = 1


dataset = df_master.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in tqdm(range(start_index, end_index)):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target months of flight to predict : {}'.format(
    y_train_multi[0].shape))


train_data_multi = tf.data.Dataset.from_tensor_slices(
    (x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(64,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(64,
                                          return_sequences=True))
multi_step_model.add(tf.keras.layers.LSTM(64,
                                          return_sequences=True))
multi_step_model.add(tf.keras.layers.LSTM(64, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# %%
for x, y in val_data_multi.take(1):
    print(multi_step_model.predict(x).shape)


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50,
                                          callbacks=[cp_callback])

plot_train_history(multi_step_history,
                   'Multi-Step Training and validation loss')
for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
# %%
