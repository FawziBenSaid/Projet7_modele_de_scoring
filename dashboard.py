import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import plotly.express as px
from PIL import Image
import requests
import matplotlib.pyplot as plt
import shap








# Telecharger le Data x_test
@st.cache(allow_output_mutation=True)
def load_df():
    df = pd.read_csv('client_list3.csv')
    return df

df = load_df()
df3 = df
df = df.select_dtypes(include=np.number)
df = df.drop(df.columns[[0, -1]], axis=1)

# Supprimer les 0 de la colonne 'SK_ID_CURR'
lst = []
for each in df['SK_ID_CURR']:
    lst.append(str(each).split('.')[0])

# all values converting to integer data type
df['SK_ID_CURR'] = [int(i) for i in lst]


# Creer un tableau qui est une copie du premier tableau mais avec la colonne target
df3 = df3.select_dtypes(include=np.number)
df3 = df3.drop(df3.columns[0], axis=1)
# Supprimer les 0 de la colonne 'SK_ID_CURR'
lst = []
for each in df3['SK_ID_CURR']:
    lst.append(str(each).split('.')[0])

# all values converting to integer data type
df3['SK_ID_CURR'] = [int(i) for i in lst]





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


# Fonction pour Cr??er un graphique force plot
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


# Fonction pour Cr??er un graphique shap summary
def shap_summary(data):
    explainer = creat_explainer(model, df)
    shap_values = creat_shap_values(df)
    shap.summary_plot(shap_values, data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight')


# T??l??charger le logo
background = Image.open('logo.png')
col1, col2, col3 = st.columns([0.7, 1.2, 0.7])
col2.image(background, use_column_width=True, caption = 'pr??t ?? d??penser')



# titre et description
title =  '<p style="font-family:Cormorant; color:#69b3f2;text-align: center; font-size: 42px;">Bienvenue au dashboard "Pr??t ?? d??penser"</p>'
st.markdown(title, unsafe_allow_html=True)
st.write('---')







with st.sidebar.container():
    page = st.sidebar.selectbox('Veuillez choisir une page', ['Veuillez choisir :','Accuiel', 'Exploration des donn??es','Pr??diction'])



if page == 'Accuiel':

    st.info('La mission principale de ce projet est de pr??dire le risque de faillite d\'un client pour une soci??t?? de cr??dit.'
        ' '
        ' Pour cela, on a cr???? ce Dashboard pour faciliter l\'interpr??tation de classification de chaque client')

    st.markdown('Les clients sont de plus en plus demandeurs de transparences vis-??-vis aux d??cisions,'
                 'Grace ?? cet outil le charg?? de relation client peut expliquer la d??cision et les facteurs qui ont conduit ?? la prendre.')

    st.write(' - Pour afficher la liste des clients et leurs donn??es, veuillez appuyer sur le bouton ci-dessous.')

    if st.checkbox('Afficher le tableau de donn??e'):
        #st.write(df)
        st.write(df3)


st.sidebar.write('---')

if page == 'Exploration des donn??es':

    st.info('Dans cette page, on va explorer les donn??es des clients ')

    st.write('Veuillez commencer par afficher le SHAP PLot.')
    st.write('')
    #st.subheader('shap.summary_plot')

    # afficher les features les plus importants avec le shap_value
    if st.checkbox('afficher shap plot'):
        st.write('')
        st.write("Le graphique shap.summary_plot est con??u pour afficher "
                 "les principales features d'un jeu de donn??es qui ont un impact sur le r??sultat du mod??le")
        shap_summary(df)



        st.write("Le graphique Shap Plot nous a permis de trouver les dix features les plus importants qui impactent notre mod??le."
                 " Vous trouvez la liste des features ci-dessous")
        st.info('DAYS_BIRTH,  CODE_GENDER,  DAYS_ID_PUBLISH,  FLAG_OWN_CAR, NAME_EDUCATION_TYPE_Secondary / secondary special,'
                 'NAME_EDUCATION_TYPE_Higher education, FLAG_DOCUMENT_3, REGION_RATING_CLIENT_W_CITY, NAME_CONTRACT_TYPE, REGION_RATING_CLIENT')




    # Cr??er trois bouton univari??e bivari??e multivari??e
    col1, col2, col3 = st.columns([1, 1, 1])

    st.write('')

    with col1:
        btn_univariee=st.button('Analyse univari??e')
    with col2:
        btn_bivariee=st.button('Analyse bivari??e')
    with col3:
        btn_miltivariee=st.button('Analyse multivari??e')




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
            st.write("La premiere colonne choisi est : ", bichoice[0])
            st.write("La deuxi??me colonne choisi est : ", bichoice[1])

            fig2 = px.bar(df3, x=bichoice[0], y=bichoice[1],color="TARGET" ,width=800, height=400)
            barplot_chart = st.write(fig2)


        else:
            st.write("Veuillez selectionner deux colonnes")



    # Le boutton multivariee

    if 'btn_multivariee_clicked' not in st.session_state:
        st.session_state.btn_multivariee_clicked = False

    if btn_miltivariee or st.session_state.btn_multivariee_clicked:
        st.session_state.btn_multivariee_clicked = True

        multi_choice = st.multiselect("La liste des colonnes ",
                                      ['DAYS_BIRTH', 'CODE_GENDER', 'DAYS_ID_PUBLISH', 'FLAG_OWN_CAR',
                                       'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                       'NAME_EDUCATION_TYPE_Higher education',
                                       'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'NAME_CONTRACT_TYPE',
                                       'REGION_RATING_CLIENT'])
        st.write("Veuillez choisir au moins deux colonnes")
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







if page == 'Pr??diction':
    st.info("Dans cette page, on va pr??dire L'??ligibilit?? des clients.")
    st.write('')
    st.write('')
    st.write('Pour commencer Veuillez choisir un identifiant de client dans le Sidebar')

    # Fonction pour predire la legibilit?? du client
    def get_model_predictions(input):
        #mdl_url = 'http://127.0.0.1:5000/predict'
        mdl_url = 'http://fawzibensaid.pythonanywhere.com/predict'
        data_json = {'data': input}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        prediction = requests.post(mdl_url, json=data_json, headers=headers)
        predicted = json.loads(prediction.content.decode("utf-8"))
        return predicted




    # chercher l'dentifiant client
    identifiant = st.sidebar.selectbox('Choisissez un ID client:', df.astype('int32'))
    #st.write(identifiant)

    btn_pred2 = st.button("Predict")
    # cr??er un tableau qui contient que la ligne de client selectionn??
    #client_row = df.index[df['SK_ID_CURR'] == identifiant].tolist()
    #client_row = df[df['SK_ID_CURR'] == identifiant]




    if 'btn_pred2_clicked' not in st.session_state:
        st.session_state.btn_pred2_clicked = False

    if btn_pred2 or st.session_state.btn_pred2_clicked:
        st.session_state.btn_pred2_clicked = True
        # On r??cup??re les r??sultats via l'API
        cli_json = json.loads(df[df['SK_ID_CURR'] == identifiant].to_json(orient='records'))[0]
        results_api = get_model_predictions(cli_json)


        st.write(f"La prediction de client qui a l'id: {identifiant} est:  {results_api['Prediction'][0]} " )


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

        st.write(' - Pour rappel 0 est solvable et 1 n est pas solvable ')
        st.write('')
        st.write('')
        st.write('')

        st.write("Veuillez choisir deux colonnes comme axes de graphique scatter plot pour afficher le client dans"
                  "l'ensemble des clients")


        # Scatter  plot pour afficher le client choisi dans l'ensemble des clients

        scater_graph = st.multiselect("La liste des colonnes: ",
                                  ['DAYS_BIRTH','DAYS_ID_PUBLISH', 'CODE_GENDER', 'FLAG_OWN_CAR',
                                   'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                   'NAME_EDUCATION_TYPE_Higher education',
                                   'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'NAME_CONTRACT_TYPE',
                                   'REGION_RATING_CLIENT'])
        if len(scater_graph) == 2:
            st.write("la premiere colonne choisi est : ", scater_graph[0])
            st.write("la deuxieme colonne choisi est : ", scater_graph[1])


            client_data = df[df['SK_ID_CURR'] == identifiant]
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.scatter(df[scater_graph[0]], df[scater_graph[1]], color="r", alpha=0.2)
            ax.scatter(client_data[scater_graph[0]], client_data[scater_graph[1]], color="b", s=60)
            plt.xlabel(scater_graph[0])
            plt.ylabel(scater_graph[1])
            plt.title('Les clients de la banque')
            st.pyplot(fig)
        else:
            st.write("Veuillez selectionner deux colonnes")


    # Cr??er deux bouton pour afficher le tableau et le forceplot de chaque client
    left, right = st.columns(2)

    with left:
        btn_left=st.button('Afficher le Force Plot')
    with right:
        btn_right=st.button('Afficher les donn??es du client')


    if btn_left:
        force_plot([df.index[df['SK_ID_CURR'] == identifiant].tolist()][0])

        st.write("Une valeur ??lev??e signifie que la probabilit?? d'une ??valuation n??gative est plus grande."
                 "Ainsi, dans les graphiques ci-dessous, les caract??ristiques rouges contribuent en fait ?? augmenter "
                 "les chances d'une ??valuation positive, tandis que les caract??ristiques n??gatives diminuent ces chances."
                 " "
                 "Rappelez-vous que les valeurs des caract??ristiques sont des valeurs TF-IDF.")



    if btn_right:
        shap_table(df.index[df['SK_ID_CURR'] == identifiant].tolist()[0])
