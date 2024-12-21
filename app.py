import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt

def load_data(file_path):
    """Загружает данные о температуре из CSV файла."""
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def smooth_temperature(data, window=30):
    """Вычисляет скользящее среднее температуры."""
    data['temperature'] = data['temperature'].rolling(window=window).mean()
    return data

def calculate_seasonal_statistics(data):
    """Рассчитывает среднюю температуру и стандартное отклонение для каждого сезона в каждом городе."""
    return data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

def find_anomalies(data):
    """Находит аномалии в данных, где температура выходит за пределы среднего ± 2 стандартных отклонения."""
    anomalies = []
    for _, group in data.groupby(['city', 'season']):
        mean = group['temperature'].mean()
        std = group['temperature'].std()
        anomaly_condition = (group['temperature'] < mean - 2 * std) | (group['temperature'] > mean + 2 * std)
        anomalies.append(group[anomaly_condition])
    return pd.concat(anomalies)

def get_current_temperature(city, api_key):
    """Получает текущую температуру для указанного города через OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['main']['temp'], None
    else:
        return None, response.json()

def analyze_temperature_data(file_path):
    """Основная функция для анализа исторических данных о температуре."""
    data = load_data(file_path)
    data = smooth_temperature(data)
    seasonal_stats = calculate_seasonal_statistics(data)
    anomalies = find_anomalies(data)

    return data, seasonal_stats, anomalies

def display_results(data, seasonal_stats, anomalies, current_temp, is_normal):
    """Отображает результаты анализа и текущую температуру в Streamlit."""
    
    st.title("Анализ температурных данных")
    st.subheader("Описательная статистика")
    st.write(data.describe())

    st.subheader("Временной ряд температур с аномалиями")
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'].values, data['temperature'].values, label='Температура', color='blue')
    plt.scatter(anomalies['timestamp'].values, anomalies['temperature'].values, color='red', label='Аномалии')
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.title('Температура с выделением аномалий')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Сезонный профиль")
    plt.figure(figsize=(8, 4))
    plt.bar(seasonal_stats['season'].values, seasonal_stats['mean'].values, yerr=seasonal_stats['std'].values, capsize=5)
    plt.title('Сезонный профиль')
    plt.xlabel('Сезон')
    plt.ylabel('Средняя температура (°C)')
    st.pyplot(plt)

    st.subheader("Текущая температура")
    if current_temp is not None:
        st.write(f"Текущая температура: {current_temp} °C")
        status = "нормальна" if is_normal else "аномальна"
        st.write(f"Температура является {status} для текущего сезона.")
    else:
        st.write("Не удалось получить текущую температуру.")

def main():
    """Основная функция приложения Streamlit."""
    
    st.title("Анализ температурных данных")
    uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV)", type="csv")
    
    if uploaded_file is not None:
        data, seasonal_stats, anomalies = analyze_temperature_data(uploaded_file)
        city = st.selectbox("Выберите город", data['city'].unique())
        api_key = st.text_input("Введите ваш API ключ OpenWeatherMap")
        
        data = data[
            (data['city'] == city)
        ]
        
        seasonal_stats = seasonal_stats[
            (seasonal_stats['city'] == city)
        ]
        
        anomalies = anomalies[
            (anomalies['city'] == city)
        ]

        if api_key:
            current_temp, error_message = get_current_temperature(city, api_key)
            if current_temp is not None:
                current_season = data[data['timestamp'] == data['timestamp'].max()]['season'].values[0]
                seasonal_mean = seasonal_stats[seasonal_stats['season'] == current_season]['mean'].values[0]
                seasonal_std = seasonal_stats[seasonal_stats['season'] == current_season]['std'].values[0]
                is_normal = abs(current_temp - seasonal_mean) <= 2 * seasonal_std
                
                display_results(data, seasonal_stats, anomalies, current_temp, is_normal)
            else:
                st.error(f"Ошибка: {error_message['message']}")
    
if __name__ == "__main__":
    main()
