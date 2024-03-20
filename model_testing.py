import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_preparation import train_arima_model, plot_predictions

def test_arima_model(model, test_data_path):
    # Загрузка тестовых данных
    test_data = pd.read_csv(test_data_path)

    # Вывод имен столбцов
    print("Column names in the test dataset:", test_data.columns)

    # Выбор временного ряда для тестирования
    # Убедитесь, что выбираете правильный столбец, содержащий временной ряд
    # Пример: test_series = test_data['Your_Column_Name']
    test_series = test_data[test_data.columns[1]]  # Выберите второй столбец (индекс 1), содержащий временной ряд

    # Построение предсказаний модели ARIMA
    predictions = model.predict(start=0, end=len(test_series)-1)

    # Вычисление метрик
    mse = mean_squared_error(test_series, predictions)
    mae = mean_absolute_error(test_series, predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

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

        # Тестирование модели ARIMA на данных из папки "test"
        test_arima_model(arima_model, train_data_file.replace('train', 'test'))

def main():
    # Обработка всех файлов в папке "train"
    process_files_in_train_folder()

if __name__ == "__main__":
    main()
