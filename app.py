import streamlit as st
import joblib
import numpy as np

# Загрузка модели и обработчиков
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title('Прогноз мировых продаж видеоигр')

# Выбор характеристик игры
genre = st.selectbox('Выберите жанр', label_encoders['Genre'].classes_)
platform = st.selectbox('Выберите платформу', label_encoders['Platform'].classes_)
publisher = st.selectbox('Выберите издателя', label_encoders['Publisher'].classes_)
year = st.number_input('Год выпуска игры', min_value=1980, max_value=2025, value=2015)

# Средние значения продаж
na_sales_mean = 0.26485
eu_sales_mean = 0.14671
jp_sales_mean = 0.07783
other_sales_mean = 0.04809

# Кнопка предсказания
if st.button('Предсказать продажи'):
    # Кодирование категорий
    genre_encoded = label_encoders['Genre'].transform([genre])[0]
    platform_encoded = label_encoders['Platform'].transform([platform])[0]
    publisher_encoded = label_encoders['Publisher'].transform([publisher])[0]

    # Расчёт возраста игры
    age = 2017 - year

    # Формируем входной массив
    input_data = np.array([[genre_encoded, platform_encoded, publisher_encoded,
                            year, age, na_sales_mean, eu_sales_mean, jp_sales_mean, other_sales_mean]])

    # Стандартизируем числовые признаки
    input_data[:, [3, 5, 6, 7, 8]] = scaler.transform(input_data[:, [3, 5, 6, 7, 8]])

    # Предсказание
    prediction = model.predict(input_data)[0]

    st.success(f'Прогноз мировых продаж: {prediction:.2f} млн копий')

    st.info('Внимание: прогноз делается на основе средних продаж по регионам.')
