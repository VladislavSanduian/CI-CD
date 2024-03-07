import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima_model(data_path, order=(1, 0, 1)):
    # Загрузка предобработанных данных
    data = pd.read_csv(data_path)

    # Вывод имен столбцов
    print("Column names in the dataset:", data.columns)

    # Выбор временного ряда для обучения модели ARIMA
    # Убедитесь, что выбираете правильный столбец, содержащий временной ряд

    # Пример: time_series = data['Your_Column_Name']
    time_series = data[data.columns[1]]  # Выберите второй столбец (индекс 1), содержащий временной ряд

    # Создание и обучение модели ARIMA
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()

    return fitted_model

def plot_predictions(model, data, title="ARIMA Model Predictions"):
    # Построение предсказаний модели
    predictions = model.predict(start=0, end=len(data)-1)

    # Визуализация данных и предсказаний
    plt.plot(data.index, data[data.columns[1]], label='Actual Data')  # Выберите второй столбец (индекс 1), содержащий временной ряд
    plt.plot(data.index, predictions, label='ARIMA Predictions')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def process_files_in_train_folder():
    # Получение текущей директории
    current_directory = os.getcwd()

    # Путь к папке "train"
    train_folder_path = os.path.join(current_directory, 'train')

    # Поиск файлов, начинающихся с "preprocessed_" в папке 'train'
    train_files = [file for file in os.listdir(train_folder_path) if file.startswith('preprocessed_')]

    if not train_files:
        print("No preprocessed files found in the 'train' folder.")
        return

    for train_file in train_files:
        # Выбор текущего файла
        train_data_file = os.path.join(train_folder_path, train_file)

        # Обучение модели ARIMA
        arima_model = train_arima_model(train_data_file)

        # Визуализация предсказаний на обучающих данных
        plot_predictions(arima_model, pd.read_csv(train_data_file))

def main():
    # Обработка всех файлов в папке "train"
    process_files_in_train_folder()

if __name__ == "__main__":
    main()