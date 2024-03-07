import os
import numpy as np
import pandas as pd

def generate_time_series(length=365, anomalies=False, noise=False, seasonality=True, cycles=True):
    time = pd.date_range(start='2002-01-01', periods=length, freq='D')

    # Генерация стационарных временных рядов для мяса, молока и яблок
    meat_prices = np.random.normal(loc=10, scale=2, size=length)
    milk_prices = np.random.normal(loc=5, scale=1, size=length)
    apple_prices = np.random.normal(loc=2, scale=0.5, size=length)

    # Добавление сезонности и циклов
    if seasonality:
        time_of_year = np.sin(2 * np.pi * np.arange(length) / 365)
        meat_prices += 2 * time_of_year
        milk_prices += 1 * time_of_year
        apple_prices += 0.5 * time_of_year

    if cycles:
        time_cycle = np.sin(2 * np.pi * np.arange(length) / 30)  # Пример цикла продолжительностью в 30 дней
        meat_prices += 1 * time_cycle
        milk_prices += 0.5 * time_cycle
        apple_prices += 0.2 * time_cycle

    if anomalies:
        # Внесем аномалии в цены на продукты
        anomaly_indices = np.random.choice(length, size=int(0.05 * length), replace=False)
        meat_prices[anomaly_indices] += np.random.normal(loc=5, scale=2, size=len(anomaly_indices))
        milk_prices[anomaly_indices] += np.random.normal(loc=2, scale=1, size=len(anomaly_indices))
        apple_prices[anomaly_indices] += np.random.normal(loc=1, scale=0.5, size=len(anomaly_indices))

    if noise:
        # Добавим случайный шум к ценам на продукты
        meat_prices += np.random.normal(loc=0, scale=0.5, size=length)
        milk_prices += np.random.normal(loc=0, scale=0.2, size=length)
        apple_prices += np.random.normal(loc=0, scale=0.1, size=length)

    return pd.DataFrame({'Date': time, 'Meat_Price': meat_prices, 'Milk_Price': milk_prices, 'Apple_Price': apple_prices})

def save_dataset(dataset, root_folder, split_ratio=0.7):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    # Определение индекса разделения для train и test
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset.iloc[:split_index]
    test_data = dataset.iloc[split_index:]

    # Определение проектов
    projects = ['Meat', 'Milk', 'Apple']

    # Создание папок train и test
    train_folder = os.path.join(root_folder, 'train')
    test_folder = os.path.join(root_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Сохранение данных в соответствующие папки для каждого проекта
    for project in projects:
        project_train_data = train_data[['Date', f'{project}_Price']]
        project_test_data = test_data[['Date', f'{project}_Price']]

        project_train_data.to_csv(os.path.join(train_folder, f'{project.lower()}_train_data.csv'), index=False)
        project_test_data.to_csv(os.path.join(test_folder, f'{project.lower()}_test_data.csv'), index=False)

def main():
    # Создадим и сохраним данные
    generated_data = generate_time_series(length=3650, anomalies=True, noise=True, seasonality=True, cycles=True)
    save_dataset(generated_data, '.', split_ratio=0.7)

if __name__ == "__main__":
    main()