import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import plotly.express as px
from PIL import Image
import requests
import matplotlib.pyplot as plt
from urllib.request import urlopen
import urllib.request as url
from flask import Flask, request
import shap








# Telecharger le Data x_test
@st.cache(allow_output_mutation=True)
def load_df():
    df = pd.read_csv('client_list.csv')
    return df

df = load_df()
df = df.select_dtypes(include=np.number)
df = df.drop(df.columns[0], axis=1)

# Supprimer les 0 de la colonne 'SK_ID_CURR'
lst = []
for each in df['SK_ID_CURR']:
    lst.append(str(each).split('.')[0])

# all values converting to integer data type
df['SK_ID_CURR'] = [int(i) for i in lst]


# Telecharger le Data x_test
@st.cache
def load_df1():
    df1 = pd.read_csv('client_list1.csv')
    return df1

df1 = load_df1()
df1 = df1.select_dtypes(include=np.number)


# Telecharger le Data x_test
@st.cache
def load_df2():
    df2 = pd.read_csv('client_list_original_data.csv')
    return df2
df2 = load_df2()

# telechager l'algorithme
@st.cache
def load_model():
    model_local = pickle.load(open("model.pkl", "rb"))
    return model_local

model = load_model()







@st.cache(allow_output_mutation=True)
def creat_explainer(model, data):
    explainer = shap.LinearExplainer(model, masker=shap.maskers.Impute(data=data))
    return explainer

@st.cache(allow_output_mutation=True)
def creat_shap_values(data):
    shap_values = creat_explainer(model, df).shap_values(data)
    return shap_values


# Fonction pour afficher le tableau de chaque cleint
def shap_table(ind):
    explainer = creat_explainer(model, df)
    shap_values = creat_shap_values(df)
    shap_table = pd.DataFrame(shap_values,columns=df.columns)
    st.table(shap_table.iloc[ind])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0, showPyplotGlobalUse=False)
    return shap_table


# Fonction pour Créer un graphique force plot
import shap
shap.initjs()
def force_plot (index):
    explainer = creat_explainer(model, df)
    shap_values = creat_shap_values(df)
    shap.force_plot(explainer.expected_value, shap_values[index,:],
                   df.iloc[index,:], matplotlib=True, show=False, figsize=(16, 5))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0, showPyplotGlobalUse=False)
    plt.clf()


# Fonction pour Créer un graphique shap summary
def shap_value(data):
    explainer = creat_explainer(model, df)
    shap_values = creat_shap_values(df)
    shap.summary_plot(shap_values, data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight')


# Télécharger le logo
background = Image.open('logo.png')
col1, col2, col3 = st.columns([0.7, 1.2, 0.7])
col2.image(background, use_column_width=True, caption = 'prêt à dépenser')



# titre et description
title =  '<p style="font-family:Cormorant; color:#69b3f2;text-align: center; font-size: 42px;">Bienvenue au dashboard "Prêt à dépenser"</p>'
st.markdown(title, unsafe_allow_html=True)
st.markdown('La mission principale de ce projet est de prédire le risque de faillite d un client pour une société de crédit.'
    'Pour cela, on a créé ce Dashboard pour faciliter l interpretation de classification de chaque client')
#st.success('Veuillez choisir une page dans la liste de navigation dans le SideBar')








with st.sidebar.container():
    page = st.sidebar.selectbox('Page navigation', ['Veuillez choisir :','Accuiel', 'Exploration des données','Prédiction'])



if page == 'Accuiel':

    if st.checkbox('Show dataframe'):
        st.write(df)




if page == 'Exploration des données':
    st.write('Dans cette page, on va explorer les données des clients en utilisant des graphiques.')
    st.write('')
    st.write('')
    st.subheader('shap.summary_plot')
    st.write("Le graphique shap.summary_plot est conçu pour afficher "
             "les principales features d'un jeu de données qui ont un impact sur le résultat du modèle")

    # afficher les features les plus importants avec le shap_value
    if st.checkbox('afficher shap plot'):
        #import copy
        #cloned_output = copy.deepcopy(shap_value(df))
        shap_value(df)



        st.write("Dans le graphique ci-dessous, on a choisi d'afficher les dix features les plus importants dans notre modèle"
                 "vous trouvez la liste des features ci-dessous")
        st.info('DAYS_BIRTH,  CODE_GENDER,  DAYS_ID_PUBLISH,  FLAG_OWN_CAR, NAME_EDUCATION_TYPE_Secondary / secondary special,'
                 'NAME_EDUCATION_TYPE_Higher education, FLAG_DOCUMENT_3, REGION_RATING_CLIENT_W_CITY, NAME_CONTRACT_TYPE, REGION_RATING_CLIENT')


    # Créer trois bouton univariée bivariée multivariée
    col1, col2, col3 = st.columns([1, 1, 1])



    with col1:
        btn_univariee=st.button('Analyse univariée')
    with col2:
        btn_bivariee=st.button('Analyse bivariée')
    with col3:
        btn_miltivariee=st.button('multivariée')




    if 'btn_univariee_clicked' not in st.session_state:
        st.session_state.btn_univariee_clicked = False
    if btn_univariee  or st.session_state.btn_univariee_clicked:
        st.session_state.btn_univariee_clicked = True
        # selectiontion de la colonne


        colonne_select = st.selectbox("Selectionnez une colonnes: ",
                                      ('DAYS_BIRTH', 'CODE_GENDER', 'DAYS_ID_PUBLISH', 'FLAG_OWN_CAR',
                                       'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                       'NAME_EDUCATION_TYPE_Higher education',
                                       'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'NAME_CONTRACT_TYPE',
                                       'REGION_RATING_CLIENT'))

        st.write('Vous avez choisi la colonne:', colonne_select)
        # afficher le graphique bowplot
        #if st.checkbox("Afficher le box Plot"):
        data_colomn = df[[colonne_select]]
        fig1 = px.box(data_colomn, width=800, height=400)
        barplot_chart = st.write(fig1)




    # Le boutton bivariee
    if 'btn_bivariee_clicked' not in st.session_state:
        st.session_state.btn_bivariee_clicked = False

    if btn_bivariee or st.session_state.btn_bivariee_clicked:

        st.session_state.btn_bivariee_clicked = True
        bichoice = st.multiselect("La liste des colonnes: ",
                                      ['DAYS_BIRTH', 'CODE_GENDER', 'DAYS_ID_PUBLISH', 'FLAG_OWN_CAR',
                                       'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                       'NAME_EDUCATION_TYPE_Higher education',
                                       'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'NAME_CONTRACT_TYPE',
                                       'REGION_RATING_CLIENT'])
        if len(bichoice) == 2:
            st.write("la premiere colonne choisi est : ", bichoice[0])
            st.write("la deuxieme colonne choisi est : ", bichoice[1])

            fig2 = px.bar(df, x=bichoice[0], y=bichoice[1], width=800, height=400)
            barplot_chart = st.write(fig2)


        else:
            st.write("Veuillez selectionner deux colonnes")



    # Le boutton multivariee

    if 'btn_multivariee_clicked' not in st.session_state:
        st.session_state.btn_multivariee_clicked = False

    if btn_miltivariee or st.session_state.btn_multivariee_clicked:
        st.session_state.btn_multivariee_clicked = True

        multi_choice = st.multiselect("Selectionnez Deux colonnes ou plus: ",
                                      ['DAYS_BIRTH', 'CODE_GENDER', 'DAYS_ID_PUBLISH', 'FLAG_OWN_CAR',
                                       'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                       'NAME_EDUCATION_TYPE_Higher education',
                                       'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'NAME_CONTRACT_TYPE',
                                       'REGION_RATING_CLIENT'])
        st.write("Veuillez choisir au minimum deux colonnes")
        if st.checkbox("Afficher la matrice de correlation"):
            # Creer un nouveau tableau qui contient une seulle colonne et on va ajouter les colonnes selectionner
            data_corela = pd.DataFrame(np.random.randint(0, 1000, size=(1000, 1)), columns=list('A'))
            for i in multi_choice:
                data_corela = data_corela.join(df[i])

            # Supprimer la premier colonne qu'on a creer pouravoir le nouveau tableau
            data_corela = data_corela.drop('A', 1)
            st.write(data_corela)

            fig3 = px.imshow(data_corela.corr(), width=750, height=750)
            barplot_chart = st.write(fig3)







if page == 'Prédiction':
    st.write("Dans cette page, on va prédire la lisibilité des clients, et comprendre le choix grâce au graphique ci-dessous.")

    # Fonction pour predire la legibilité du client
    def get_model_predictions(input):
        mdl_url = 'http://127.0.0.1:5000/predict'
        data_json = {'data': input}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        prediction = requests.post(mdl_url, json=data_json, headers=headers)
        predicted = json.loads(prediction.content.decode("utf-8"))
        return predicted




    # chercher l'dentifiant client
    identifiant = st.sidebar.selectbox('Choisi un ID client:', df.astype('int32'))
    st.write(identifiant)

    btn_pred = st.button("Predict")
    # créer un tableau qui contient que la ligne de client selectionné
    client_row = df.index[df['SK_ID_CURR'] == identifiant].tolist()
    client_row = df[df['SK_ID_CURR'] == identifiant]




    if btn_pred:
        # On récupère les résultats via l'API
        cli_json = json.loads(df[df['SK_ID_CURR'] == identifiant].to_json(orient='records'))[0]
        results_api = get_model_predictions(cli_json)


        st.write(f"la prediction de client qui a l'id: {identifiant} est:  {results_api['Prediction'][0]} " )
        st.write('Pour rappel 0 est solvable et 1 n est pas solvable ')

        #Client sovable
        if results_api['Prediction'][0] == 0:
            st.write('')
            # success
            st.success("Ce client est solvable")

        # Client non solvable
        else:
            st.write('')
            # error
            st.error("Ce client n'est pas solvable")


        # Scatter  plot pour afficher le client choisi dans l'ensemble des clients
        client_data = df[df['SK_ID_CURR'] == identifiant]
        fig = plt.figure(figsize = (10, 8))
        ax = plt.gca()
        ax.scatter(df['DAYS_ID_PUBLISH'], df['DAYS_BIRTH'], color="r", alpha=0.2)
        ax.scatter(client_data['DAYS_ID_PUBLISH'], client_data['DAYS_BIRTH'], color="b",s=60)
        plt.xlabel('DAYS_ID_PUBLISH')
        plt.ylabel('DAYS_BIRTH')
        plt.title('Les clients de la banque')
        st.pyplot(fig)


    # Créer deux bouton pour afficher le tableau et le forceplot de chaque client
    left, right = st.columns(2)

    with left:
        btn_left=st.button('afficher le Force Plot')
    with right:
        btn_right=st.button('Afficher les données du client')


    if btn_left:
        st.write("Une valeur élevée signifie que la probabilité d'une évaluation négative est plus grande."
                 "Ainsi, dans les graphiques ci-dessous, les caractéristiques rouges contribuent en fait à augmenter les chances d'une évaluation positive,"
                 " tandis que les caractéristiques négatives diminuent ces chances."
                 "Rappelez-vous que les valeurs des caractéristiques sont des valeurs TF-IDF.")

        force_plot([df.index[df['SK_ID_CURR'] == identifiant].tolist()][0])

    if btn_right:
        shap_table(df.index[df['SK_ID_CURR'] == identifiant].tolist()[0])









