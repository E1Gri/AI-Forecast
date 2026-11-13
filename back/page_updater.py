
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape

import numpy as np
import pandas as pd
import requests

from functions import encode_data, add_lags

SESSION = requests.Session()

CITIES = {
    "msk": {
        "name": "Москва",
        "latitude": 55.7522,
        "longitude": 37.6156,
    },
    "spb": {
        "name": "Санкт-Петербург",
        "latitude": 59.9386,
        "longitude": 30.3141,
    },
}

TIMEZONE = "Europe/Moscow"

HOURLY = [
    "temperature_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "rain",
]

TEMP = "temperature_2m_mean (°C)"
PRESSURE = "surface_pressure_mean (hPa)"
SPEED = "wind_speed_10m_mean (km/h)"
DIRECTION = "wind_direction_10m_dominant (°)"
RAIN = "rain_sum (mm)"


# Загрузка моделей

BASE = Path(__file__).resolve().parent
FRONT = BASE.parent / "front"
MODELS = BASE.parent / "models"

FRONT.mkdir(exist_ok=True)

with open(MODELS / "msk_model.pkl", "rb") as f:
    msk_model = pickle.load(f)

with open(MODELS / "spb_model.pkl", "rb") as f:
    spb_model = pickle.load(f)


def request_weather(url, params):
    response = SESSION.get(url, params=params, timeout=45)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise ValueError(data.get("reason", "Ошибка Open-Meteo"))

    return data


def daily_aggregation(hourly):
    """
    Преобразование почасовой погоды в дневные показатели.
    Направление ветра рассчитывается как векторное среднее.
    """

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()

    for column in HOURLY:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[HOURLY].isna().any().any():
        raise ValueError("Есть пропуски в почасовой погоде")

    result = []

    for date, group in df.groupby("date"):
        angles = np.deg2rad(
            group["wind_direction_10m"].to_numpy()
        )

        speeds = group["wind_speed_10m"].to_numpy()

        x = np.sum(np.sin(angles) * speeds)
        y = np.sum(np.cos(angles) * speeds)

        direction = (
            np.degrees(np.arctan2(x, y)) % 360
            if np.hypot(x, y) > 0
            else 0.0
        )

        result.append({
            "time": date,
            TEMP: group["temperature_2m"].mean(),
            PRESSURE: group["surface_pressure"].mean(),
            SPEED: group["wind_speed_10m"].mean(),
            DIRECTION: direction,
            RAIN: group["rain"].sum(),
        })

    return pd.DataFrame(result).sort_values("time")


def getWeather(city, number_of_lags):
    """
    Возвращает:
    current - текущая погода;
    history - завершённые предыдущие дни;
    today_data - суточные показатели сегодня
                 (частично прогнозные).
    """

    config = CITIES[city]

    today = datetime.now(
        ZoneInfo(TIMEZONE)
    ).date()

    yesterday = today - timedelta(days=1)

    # Берём достаточно истории для модели.
    start = today - timedelta(days=number_of_lags + 7)

    common = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
    }

    # Историческая погода: только прошедшие дни.
    archive_params = {
        **common,
        "start_date": start.isoformat(),
        "end_date": yesterday.isoformat(),
        "hourly": ",".join(HOURLY),
    }

    archive = request_weather(
        "http://archive-api.open-meteo.com/v1/archive",
        archive_params,
    )

    history = daily_aggregation(archive["hourly"])

    # Текущая погода + прогноз по часам на сегодня.
    forecast_params = {
        **common,
        "current": ",".join(HOURLY),
        "hourly": ",".join(HOURLY),
        "forecast_days": 1,
    }

    forecast = request_weather(
        "https://api.open-meteo.com/v1/forecast",
        forecast_params,
    )

    current = forecast["current"]

    today_data = daily_aggregation(forecast["hourly"])

    today_data = today_data[
        today_data["time"].dt.date == today
    ].copy()

    if today_data.empty:
        raise ValueError("Нет суточных данных за сегодня")

    return current, history, today_data


def prepare_prediction(history, today_data, model_info):
    """
    Создание признаков через функции из functions.py.
    """

    n = int(model_info["number_of_lags"])

    if "drop_time" not in model_info:
        raise ValueError(
            "В словаре модели отсутствует drop_time. "
            "Добавь его при сохранении модели."
        )

    drop_time = model_info["drop_time"]

    # История + сегодняшний день.
    df = pd.concat(
        [history, today_data],
        ignore_index=True,
    )

    df = df.sort_values("time").tail(n).reset_index(drop=True)

    if len(df) != n:
        raise ValueError(
            f"Недостаточно данных: нужно {n} дней"
        )

    # Категоризация только для категориальной модели.
    if model_info["method"] == "Категориальный":
        df, _ = encode_data(df, model_info["means"])

    # Создаём строку для завтрашней даты.
    tomorrow = df["time"].iloc[-1] + pd.Timedelta(days=1)

    # Эти значения временные.
    # Они не попадут в X после удаления target
    # и исходных погодных колонок.
    next_day = {
        "time": tomorrow,
        TEMP: 0,
        PRESSURE: 0,
        SPEED: 0,
        DIRECTION: 0,
        RAIN: 0,
    }

    df = pd.concat(
        [df, pd.DataFrame([next_day])],
        ignore_index=True,
    )

    # Используем оригинальную функцию из ноутбука.
    df = add_lags(df, n, drop_time)

    if len(df) != 1:
        raise ValueError(
            "Не удалось сформировать строку прогноза"
        )

    X = df.drop(columns=["target"])

    return X


def predict_weather(model_info, history, today_data):
    X = prepare_prediction(
        history,
        today_data,
        model_info,
    )

    prediction = model_info["Model"].predict(X)

    probability = float(prediction.data[0, 0])

    if not 0 <= probability <= 1:
        raise ValueError("Некорректный результат модели")

    return probability


def update_page(filename, current, history, probability, city_name):
    """
    Генерирует HTML:
    - текущая погода;
    - вероятность дождя завтра;
    - предыдущие 7 дней.
    """

    temperature = float(current["temperature_2m"])
    pressure = float(current["surface_pressure"])
    wind_speed = float(current["wind_speed_10m"])
    wind_direction = float(current["wind_direction_10m"])
    rain = float(current["rain"]) > 0

    probability_percent = round(probability * 100)

    status = (
        "🌧️ Сейчас идёт дождь"
        if rain
        else "🌤️ Сейчас дождя нет"
    )

    updated_at = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%d.%m.%Y, %H:%M:%S")

    rows = ""

    last_week = history.tail(7).iloc[::-1]

    for _, day in last_week.iterrows():
        rows += f"""
        <tr>
            <td>{day["time"].strftime("%d.%m.%Y")}</td>
            <td>{day[TEMP]:.1f} °C</td>
            <td>{day[PRESSURE]:.0f} гПа</td>
            <td>{day[SPEED]:.1f} км/ч</td>
            <td>{day[RAIN]:.1f} мм</td>
        </tr>
        """

    city = escape(city_name)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForecastAI — {city}</title>
<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    min-height: 100vh;
    font-family: Arial, sans-serif;
    color: white;
    background:
        radial-gradient(circle at 15% 10%, #245ca0, transparent 40%),
        radial-gradient(circle at 85% 90%, #5246a5, transparent 40%),
        linear-gradient(135deg, #071329, #102b50);
}}

header {{
    padding: 25px 7%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ffffff20;
}}

.logo {{
    font-size: 24px;
    font-weight: bold;
}}

.logo span {{
    color: #65caff;
}}

a {{
    color: #a9dfff;
    text-decoration: none;
}}

main {{
    width: min(1050px, 92%);
    margin: 55px auto;
}}

h1 {{
    font-size: clamp(32px, 5vw, 48px);
    margin-bottom: 10px;
}}

.subtitle {{
    color: #c6d7ee;
    margin-bottom: 30px;
}}

.card {{
    padding: 30px;
    margin-bottom: 25px;
    border: 1px solid #ffffff30;
    border-radius: 24px;
    background: #ffffff12;
    box-shadow: 0 20px 60px #00000025;
}}

.status {{
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 24px;
}}

.probability {{
    color: #8bdcff;
    font-size: 64px;
    font-weight: bold;
}}

.probability p {{
    font-size: 15px;
    font-weight: normal;
    color: #c6d7ee;
}}

.stats {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-top: 28px;
}}

.stat {{
    padding: 20px;
    border-radius: 16px;
    background: #07132955;
    border: 1px solid #ffffff18;
}}

.stat span {{
    display: block;
    color: #b7cbe8;
    font-size: 13px;
    margin-bottom: 10px;
}}

.stat strong {{
    font-size: 22px;
}}

.table-wrap {{
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}

th, td {{
    padding: 15px 10px;
    text-align: left;
    border-bottom: 1px solid #ffffff25;
}}

th {{
    color: #a9dfff;
}}

.updated {{
    margin-top: 28px;
    padding-top: 20px;
    border-top: 1px solid #ffffff25;
    color: #c6d7ee;
    font-size: 14px;
    line-height: 1.7;
}}

footer {{
    text-align: center;
    color: #9db5d8;
    font-size: 13px;
    padding: 25px;
}}

@media(max-width: 600px) {{
    header {{ padding: 20px; }}
    main {{ margin: 35px auto; }}
    .card {{ padding: 22px; }}
    .stats {{ grid-template-columns: 1fr; }}
    .status {{ font-size: 22px; }}
    th, td {{ font-size: 12px; padding: 10px 6px; }}
}}
</style>
</head>
<body>

<header>
    <div class="logo">☁ Forecast<span>AI</span></div>
    <a href="index.html">← Выбор города</a>
</header>

<main>
    <h1>📍 {city}</h1>
    <p class="subtitle">
        Текущая погода, история и прогноз дождя
    </p>

    <section class="card">
        <div class="status">{status}</div>

        <h2>🌧️ Прогноз на завтра</h2>
        <div class="probability">
            {probability_percent}%
            <p>Вероятность дождя по ML-модели</p>
        </div>

        <h2>Погода сейчас</h2>

        <div class="stats">
            <div class="stat">
                <span>🌡 Температура</span>
                <strong>{temperature:.1f} °C</strong>
            </div>
            <div class="stat">
                <span>🧭 Давление</span>
                <strong>{pressure:.1f} гПа</strong>
            </div>
            <div class="stat">
                <span>💨 Скорость ветра</span>
                <strong>{wind_speed:.1f} км/ч</strong>
            </div>
            <div class="stat">
                <span>🧭 Направление ветра</span>
                <strong>{wind_direction:.0f}°</strong>
            </div>
        </div>

        <div class="updated">
            🕒 Обновлено:
            <strong>{updated_at}</strong>
        </div>
    </section>

    <section class="card">
        <h2>📊 Предыдущие 7 дней</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Дата</th>
                        <th>Температура</th>
                        <th>Давление</th>
                        <th>Ветер</th>
                        <th>Осадки</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </section>

</main>

<footer>
    ForecastAI · Анализ погоды с помощью машинного обучения
</footer>

</body>
</html>
"""

    Path(filename).write_text(html, encoding="utf-8")


def create_index():
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForecastAI</title>
<style>
body {
    min-height: 100vh;
    margin: 0;
    font-family: Arial, sans-serif;
    color: white;
    background: linear-gradient(135deg, #071329, #234b82);
    display: grid;
    place-items: center;
}
main {
    text-align: center;
    padding: 25px;
}
h1 { font-size: 48px; }
p { color: #c6d7ee; }
.links {
    display: flex;
    gap: 20px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 35px;
}
a {
    display: block;
    padding: 25px 40px;
    border-radius: 18px;
    background: #ffffff18;
    border: 1px solid #ffffff35;
    color: white;
    font-size: 22px;
    text-decoration: none;
}
a:hover { background: #ffffff30; }
</style>
</head>
<body>
<main>
    <h1>☁ ForecastAI</h1>
    <p>Прогноз дождя с помощью машинного обучения</p>
    <div class="links">
        <a href="msk.html">📍 Москва</a>
        <a href="spb.html">📍 Санкт-Петербург</a>
    </div>
</main>
</body>
</html>
"""
    (FRONT / "index.html").write_text(html, encoding="utf-8")


def update_city(city, model_info):
    current, history, today_data = getWeather(
        city,
        model_info["number_of_lags"],
    )

    probability = predict_weather(
        model_info,
        history,
        today_data,
    )

    update_page(
        FRONT / f"{city}.html",
        current,
        history,
        probability,
        CITIES[city]["name"],
    )

    print(
        f"{CITIES[city]['name']}: "
        f"вероятность дождя {probability:.1%}"
    )


def main():
    create_index()

    while True:
        for city, model_info in [
            ("msk", msk_model),
            ("spb", spb_model),
        ]:
            try:
                update_city(city, model_info)
            except Exception as error:
                print(f"Ошибка {city}: {error}")

        print("Следующее обновление через 10 минут")
        time.sleep(600)


if __name__ == "__main__":
    main()
