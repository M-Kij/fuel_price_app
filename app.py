import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from pycaret.classification import load_model, predict_model

MIN_DATE = datetime.date(2004,1,1)
MAX_DATE = datetime.date(2024,12,31)
DIESEL_MODEL = load_model('diesel_model')
SUPER95_MODEL = load_model('super95_model')

# Utworzenie górnej granicy zakresu dat
def max_date_choice(new_date):
    if new_date <= MAX_DATE:
            end_date = new_date
    else:
        end_date = MAX_DATE
        st.info('dane kończą się 31 grudnia 2024 roku')
    return(end_date)

# Zamiana opisu parametru na nazwę kolumny w df
def get_option_key(option):
    options_dict = {
        "Ceny ropy naftowej na giełdzie (USD za baryłkę)": "ropa naftowa",
        "Kurs wymiany USD/PLN": "USD/PLN",
        "Ceny hurtowe netto oleju napędowego Ekodiesel w Orlen (PLN/litr)": "diesel",
        "Ceny hurtowe netto benzyny bezołowiowej - Eurosuper 95 w Orlen (PLN/litr)": "super95",
        "Przewidywane ceny hurtowe netto oleju napędowego Ekodiesel (PLN/litr)" : "diesel predykcja",
        "Przewidywane ceny hurtowe netto benzyny bezołowiowej Eurosuper 95 (PLN/litr)" : "super95 predykcja",
        "Ceny ropy naftowej na giełdzie (USDx100 za baryłkę)": "ropa naftowa"
    }
    return options_dict.get(option, None)

# Wczytanie i przetworzenie danych
df = pd.read_csv("fuel_prediction.csv", sep=";", index_col=0)
df = df.rename_axis('data')
df[['diesel', 'super95', 'super95 predykcja', 'diesel predykcja']] = df[['diesel',
                         'super95','super95 predykcja', 'diesel predykcja']] / 1000
df = df[['ropa naftowa', 'USD/PLN', 'diesel', 'super95', 'diesel predykcja', 'super95 predykcja']]
df.index = pd.to_datetime(df.index)

# START
st.title('Ceny hurtowe paliw w Polsce')

# Utworzenie zakładek
tabs = st.tabs(['Dane', 'Analizy', 'Predykcje'])

with tabs[0]:
    # Utworzenie pojedynczego wyboru opcji
    option = st.radio("Aplikacja pozwala analizować od 2004 do 2025 roku:",
                    options=["Ceny hurtowe netto oleju napędowego Ekodiesel w Orlen (PLN/litr)",
                            "Ceny hurtowe netto benzyny bezołowiowej - Eurosuper 95 w Orlen (PLN/litr)",
                            "Ceny ropy naftowej na giełdzie (USD za baryłkę)",
                            "Kurs wymiany USD/PLN"], key="general", index=0)

    if option:
        opt = get_option_key(option)
  
    # Utworzenie kolumn z wyborami okresu
    st.write('Wybór zakresu dat:')
    col1, col2 = st.columns(2)
    with col1:
        start_date=st.date_input('Wybierz początek okresu', value=MIN_DATE,
                                  min_value=MIN_DATE, max_value=MAX_DATE, key="data")
    with col2:
        end_choice=st.selectbox('Wybierz długość okresu', 
                                ['tydzień', 'miesiąc', 'kwartał', 
                                'rok', '5 lat', 'maksymalny'], key="1")  
        
        if end_choice == 'tydzień':
            new_date = start_date + datetime.timedelta(days=7)
            end_date = max_date_choice(new_date)
        if end_choice == 'miesiąc':
            new_date = start_date + relativedelta(months=1)
            end_date = max_date_choice(new_date)
        if end_choice == 'kwartał':
            new_date = start_date + relativedelta(months=3)
            end_date = max_date_choice(new_date)
        if end_choice == 'rok':
            new_date = start_date + relativedelta(years=1)
            end_date = max_date_choice(new_date)
        if end_choice == '5 lat':
            new_date = start_date + relativedelta(years=5)
            end_date = max_date_choice(new_date)
        if end_choice == 'maksymalny':
            end_date = MAX_DATE

    # Utworzenie suwaka wyboru zakresu dat
    if start_date and end_date:
        start, end = st.slider(
            'Zwiększenie dokładności wyboru dat',
            min_value=start_date,
            max_value=end_date,
            value=(start_date, end_date),
            format="YYYY-MM-DD", key="s1"
    )
    # Generowanie wykresu
    if st.button('Wykres i dane', key="b1"):
        if start != end:
            # Utworzenie df z wybranym okresem
            date_df = df.loc[start:end]
            # Generowanie wykresu
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=date_df.index, y=date_df[opt], mode='lines'))
            fig.update_traces(line=dict(color='red', width=1))
            fig.update_layout(
                title=f'{option}<br>w okresie od {start} do {end}',
                xaxis_title="daty",
                yaxis_title="cena")
            st.plotly_chart(fig)

            date_df.index = date_df.index.strftime('%Y-%m-%d')
            date_df = date_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            st.dataframe(date_df, use_container_width=True)
        else:
            st.warning("Zwiększ zakres dat!")
    else:
        st.write('Kliknij przycisk, aby wygenerować wykres i zobaczyć dane')

with tabs[1]:
   
    st.write("Wybór parametrów do porównania:")
    options_ext = ["Ceny hurtowe netto oleju napędowego Ekodiesel w Orlen (PLN/litr)",
                    "Ceny hurtowe netto benzyny bezołowiowej - Eurosuper 95 w Orlen (PLN/litr)",
                    "Ceny ropy naftowej na giełdzie (USDx100 za baryłkę)",
                    "Kurs wymiany USD/PLN", 
                    "Przewidywane ceny hurtowe netto oleju napędowego Ekodiesel (PLN/litr)",
                    "Przewidywane ceny hurtowe netto benzyny bezołowiowej Eurosuper 95 (PLN/litr)"]
    c1, c2 = st.columns(2)
    with c1:
        # Utworzenie pojedynczego wyboru pierwszego parametru     
        option1 = st.radio("Pierwszy parametr:", options_ext, key="first", index=0)

    with c2:
        # Utworzenie pojedynczego wyboru drugiego parametru
        option2 = st.radio("Drugi parametr:", options_ext, key="second", index=1)
    if option1 == option2:
        st.warning("Wybierz różne parametry!")
    if option1:
        opt1 = get_option_key(option1)
    if option2:
        opt2 = get_option_key(option2)
    
    text = f'Wybrane parametry, to:<br>1. {option1}<br>2. {option2}'
    
    st.markdown(f"""
        <div style="border: 2px solid red; padding: 10px; border-radius: 5px;">
            <p>{text}</p>
        </div>
    """, unsafe_allow_html=True)

    # Utworzenie kolumn z wyborami okresu
    st.write('Wybór zakresu dat:')
    col1, col2 = st.columns(2)
    with col1:
        start_date=st.date_input('Wybierz początek okresu', value=MIN_DATE,
                                  min_value=MIN_DATE, max_value=MAX_DATE, key="analysis")
    with col2:
        end_choice=st.selectbox('Wybierz długość okresu', 
                                ['tydzień', 'miesiąc', 'kwartał', 
                                'rok', '5 lat', 'maksymalny'], key="2")  
        
        if end_choice == 'tydzień':
            new_date = start_date + datetime.timedelta(days=7)
            end_date = max_date_choice(new_date)
        if end_choice == 'miesiąc':
            new_date = start_date + relativedelta(months=1)
            end_date = max_date_choice(new_date)
        if end_choice == 'kwartał':
            new_date = start_date + relativedelta(months=3)
            end_date = max_date_choice(new_date)
        if end_choice == 'rok':
            new_date = start_date + relativedelta(years=1)
            end_date = max_date_choice(new_date)
        if end_choice == '5 lat':
            new_date = start_date + relativedelta(years=5)
            end_date = max_date_choice(new_date)
        if end_choice == 'maksymalny':
            end_date = MAX_DATE

    # Utworzenie suwaka wyboru zakresu dat
    if start_date and end_date:
        start, end = st.slider(
            'Zwiększenie dokładności wyboru dat',
            min_value=start_date,
            max_value=end_date,
            value=(start_date, end_date),
            format="YYYY-MM-DD", key="s2")
    if st.button('Wykres i dane', key="b2"):
        if start != end:
            # Utworzenie df z wybranym okresem
            date_df = df.loc[start:end]
            date_df_100 = date_df.copy()
            date_df_100['ropa naftowa'] = date_df_100['ropa naftowa'] / 100
            # Generowanie wykresu
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=date_df_100.index, y=date_df_100[opt1],
                                      mode='lines', name="Parametr 1"))
            fig.add_trace(go.Scatter(x=date_df_100.index, y=date_df_100[opt2],
                                      mode='lines', name="Parametr 2"))

            fig.update_layout(
                title=f'Parametr 1: {option1}<br>Parametr 2: {option2}<br>w okresie od {start} do {end}',
                xaxis_title="daty",
                yaxis_title="cena",
                template="presentation")
            st.plotly_chart(fig)
            # Pokazanie danych w df
            date_df.index = date_df.index.strftime('%Y-%m-%d')
            date_df = date_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            st.dataframe(date_df, use_container_width=True)
        else:
            st.warning("Zwiększ zakres dat!")
    else:
        st.write('Kliknij przycisk, aby wygenerować wykres i zobaczyć dane')   
with tabs[2]:         
    st.subheader("Przewidywanie cen hurtowych paliw w zależności od ceny ropy naftowej i kursu wymiany USD/PLN")
    st.write("")
    ca, cb, cc = st.columns([5,1,5])
    with ca:
        # Utworzenie suwaka do określenia ceny ropy naftowej
        oil_price = st.slider('Cena ropy naftowej (USD/baryłkę)', 0, 150, 75)
                
    with cc:
        # Utworzenie suwaka do określenia kursu wymiany USD/PLN
        exchange_rate = st.slider('Kurs wymiany USD/PLN', 1.5, 5.0, 4.0, 0.01)
    
    # Predykcja wartości cen paliw z wykorzystaniem modeli ML
    prediction_df = pd.DataFrame([[oil_price, exchange_rate]], columns=['ropa naftowa', 'USD/PLN'])
    diesel_pred = predict_model(DIESEL_MODEL, data=prediction_df)
    super95_pred = predict_model(SUPER95_MODEL, data=prediction_df)
    diesel = ((diesel_pred.loc[0, "prediction_label"]) / 1000).round(2)
    super95 = ((super95_pred.loc[0, "prediction_label"]) / 1000).round(2)
   
     # Wydruk przewidywanych cen w dwóch kolumnach
    col_a, col_b, col_c = st.columns([5,1,6])
    with col_a:
        st.markdown(f"<h2 style='text-align: center;'>{diesel}</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>cena hurtowa netto oleju napędowego</h4>",
                     unsafe_allow_html=True)
    with col_c:
        st.markdown(f"<h2 style='text-align: center;'>{super95}</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>cena hurtowa netto benzyny bezołowiowej</h4>",
                     unsafe_allow_html=True)