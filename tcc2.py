# %%
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn import preprocessing


tf.random.set_seed(13)

plt.rcParams["figure.figsize"] = [8, 6]

# read csv data
csv_path = '/mnt/c/Users/eli_t/tcc/dados_agregados.csv'
df = pd.read_csv(csv_path, encoding='latin-1', sep=',')

plt.figure()
df.plot(subplots=True, title='Dados brutos')
plt.xlabel("Dias decorridos")
plt.ylabel("Número de Passageiros         -        Número de Vôos", loc='bottom')


avg_flights = df['flights'].mean()
avg_pp = df['paid_passengers'].mean()

df['flights'] = df['flights'].apply(
    lambda x: avg_flights if x <= 50 else x)
df['paid_passengers'] = df['paid_passengers'].apply(
    lambda x: avg_pp if x <= 10000 else x)

df.plot(subplots=True)
plot_tcc = df[df['dt_partida_real'] >
              '2019-01-01'].set_index('dt_partida_real')['flights']


def standardize(df, columns):
    for column in columns:
        column_avg = df[column].mean()
        column_std = df[column].std()
        df[f'{column}_standard'] = df[column].apply(
            lambda x: (x-column_avg)/column_std)
    return df


columns = ['flights', 'paid_passengers']
df = df.pipe(standardize, columns=columns)

df['yearmonth'] = df['dt_partida_real'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))


def transform_output(array, avg, std):
    array = (array*std)+avg
    return array


def create_time_steps(length):
    return list(range(0, length))


df.plot(subplots=True)


train = df[(df['dt_partida_real'] >= '2013-01-01') &
           (df['dt_partida_real'] <= '2018-12-31')]
validate = df[df['dt_partida_real'] >= '2019-01-01']


df_agg = df.groupby('yearmonth').agg(flights=('flights', 'sum'),
                                     paid_passengers=('paid_passengers', 'sum')).reset_index()
df_agg = df_agg.pipe(standardize, columns=columns)
train_agg = df_agg[(df_agg['yearmonth'] >= '2010-01') &
                   (df_agg['yearmonth'] <= '2018-12')]
validate_agg = df_agg[df_agg['yearmonth'] >= '2018-01']

# %%


print("Vanilla LSTM, univariate")
# split a univariate sequence into samples


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
seq = np.array(data['flights'])
print('Define the input:\n', seq)
# choose a number of time steps
n_steps = 3
print('Timesteps: ', n_steps)
# split into samples
X, y = split_sequence(seq, n_steps)
# summarize the data
print('\nSummarize the data:\n')
for i in range(len(X)):
    print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(
    '\nReshape from [samples, timesteps] into [samples, timesteps, features]')
print('\nBefore:')
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print('After:')
print(X.shape)
# define model
print('Model type: Sequential')
print('50 LSTM Units in the Hidden Layer')
print('ADAM algorithm, MSE loss.')
model = tf.keras.models.Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
epochs = 1000
steps_per_epoch = 1
model.fit(X, y, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)
# demonstrate prediction
x_input = X[0]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print('Prediction: ', yhat)


# %%

print("Stacked LSTM, univariate single output")
# split a univariate sequence into samples


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print('Define the input:\n', raw_seq)
# choose a number of time steps
n_steps = 3
print('Timesteps: ', n_steps)
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
print('\nSummarize the data:\n')
for i in range(len(X)):
    print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(
    '\nReshape from [samples, timesteps] into [samples, timesteps, features]')
print('\nBefore:')
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print('After:')
print(X.shape)
# define model
print('Model type: Sequential')
print('50 LSTM Units in the Hidden Layer')
print('ADAM algorithm, MSE loss.')
model = tf.keras.models.Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True,
               input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
epochs = 1000
steps_per_epoch = 1
model.fit(X, y, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)

# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print('Prediction: ', yhat)


# %%

print("Vanilla LSTM, univariate, multiple steps, vector output model")
# split a univariate sequence into samples


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
seq = np.array(train['flights'])
print('Define the input:\n', seq)
# choose a number of time steps
n_steps_in = 180
n_steps_out = 180
print('Timesteps: ', n_steps_in)
# split into samples
X, y = split_sequence(seq, n_steps_in, n_steps_out)
# summarize the data
print('\nSummarize the data:\n')
for i in range(len(X)):
    print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(
    '\nReshape from [samples, timesteps] into [samples, timesteps, features]')
print('\nBefore:')
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print('After:')
print(X.shape)
# define model
print('Model type: Sequential')
print('100 LSTM Units in the Hidden Layer')
print('ADAM algorithm, MSE loss.')
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# fit model
epochs = X.shape[0]
steps_per_epoch = 1
model.fit(X, y, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

# %%

print("Stacked LSTM, univariate, multiple steps, vector output model")
# split a univariate sequence into samples


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
seq = np.array(train['flights'])
print('Define the input:\n', seq)
# choose a number of time steps
n_steps_in = 100
n_steps_out = 100
print('Timesteps: ', n_steps_in)
# split into samples
X, y = split_sequence(seq, n_steps_in, n_steps_out)
# summarize the data
print('\nSummarize the data:\n')
for i in range(len(X)):
    print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(
    '\nReshape from [samples, timesteps] into [samples, timesteps, features]')
print('\nBefore:')
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print('After:')
print(X.shape)
# define model
print('Model type: Sequential')
print('100 LSTM Units in each Hidden Layer')
print('ADAM algorithm, MSE loss.')
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True,
               input_shape=(n_steps_in, n_features)))
model.add(LSTM(25, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# fit model
epochs = X.shape[0]
steps_per_epoch = 1
model.fit(X, y, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

# %%
print("Bidirectional LSTM, univariate, multiple steps, vector output model")
# split a univariate sequence into samples

plt.rcParams["figure.figsize"] = [16, 9]


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
seq = np.array(train_agg['flights_standard'])
print('Define the input:\n', seq)
# choose a number of time steps
n_steps_in = 12
n_steps_out = n_steps_in
print('Timesteps: ', n_steps_in)
# split into samples
X, y = split_sequence(seq, n_steps_in, n_steps_out)
X_val, y_val = split_sequence(
    np.array(validate_agg['flights_standard']), n_steps_in, n_steps_out)
# summarize the data
print('\nSummarize the data:\n')
for i in range(len(X)):
    print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(
    '\nReshape from [samples, timesteps] into [samples, timesteps, features]')
print('\nBefore:')
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
print('After:')
print(X.shape)
# define model
print('Model type: Bidirectional LSTM')
print('50 LSTM Units in the Hidden Layer')
print('ADAM algorithm, MSE loss.')
#epochs = [1000, 2000, 3000, 5000, 10000]
epochs = range(1000, 10000, 1000)
linwidth = range(len(epochs))
train_fig = plt.figure(1)
lr = 0.00005
val_fig = plt.figure(2)
for epoch, linewidth in zip(epochs, linwidth):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu',
                                 input_shape=(n_steps_in, n_features))))
    model.add(Dense(n_steps_out))
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')

    # fit model"
    print(f'Starting training on {epoch} epochs.')
    steps_per_epoch = 1
    history = model.fit(X, y, validation_data=(X_val, y_val), epochs=epoch,
                        steps_per_epoch=steps_per_epoch, verbose=0)
    train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.figure(1)
    plt.title(f'Mean Squared Error')
    plt.plot(history.history['loss'], label=f'train_{epoch}')
    plt.plot(history.history['val_loss'], label=f'test_{epoch}')

    model.save_weights(f'saves/train_{epoch}_epochs_{train_time}')
    # demonstrate prediction
    predict_input = train_agg[train_agg['yearmonth']
                              >= '2018-01']['flights_standard'].values[-n_steps_in:]
    x_input = predict_input
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    validate_flights = validate_agg['flights_standard'].values[-n_steps_out:]
    predict = yhat
    steps = create_time_steps(yhat.shape[1])
    steps_2 = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai',
               'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    plt.figure(2)
    plt.plot(steps_2, predict[0], label=f'Predição - {epoch} Épocas',
             color='blue', linestyle='dashed', linewidth=linewidth/5)
    mse = np.sqrt((np.square(validate_flights-predict)).mean())
    print(mse)
    print('Done.')
plt.figure(1)
plt.legend()
plt.savefig(
    f"train_images/learning_rate_{lr}_mse_{epoch}_epochs_loss_{train_time}")
plt.figure(2)
plt.plot(steps_2, validate_flights, label='Real')
plt.legend()
plt.savefig(f"train_images/true_val_{epoch}_epochs_loss_{train_time}")
# %%
plt.rcParams["figure.figsize"] = [16, 9]


def plot_tcc2(plot_tcc):
    ax = plot_tcc.plot()
    ax.set_ylabel('Vôos por Dia')
    ax.set_xlabel('Data')
    plt.savefig('plot_tcc.png')


plot_tcc2(plot_tcc)
# %%
# %%
# fazer -> loss por epoch (fit history)
# -> enviar email poley


# %%


# %%


# %%


# %%
# %%
