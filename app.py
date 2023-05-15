import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from altair.vegalite.v4.api import Chart

st.set_page_config(page_title="RI-5 Mockup", layout='wide')

df = pd.read_csv('ae_imdrf.csv')
df5 = pd.read_csv('imdrf.csv')
st.sidebar.title('RI-5 Mockup')

rev = ['P060040']

selected = st.sidebar.selectbox('Review Number:', rev)

s_exp = st.sidebar.expander("Business Requirements")
s_exp.write("RI5 Business Requirements goes here")

s_exp2 = st.sidebar.expander("User Story 1")
s_exp2.write("User Story 1 goes here")

s_exp3 = st.sidebar.expander("User Story 2")
s_exp3.write("User Story 2 goes here")

s_exp4 = st.sidebar.expander("User Story 3")
s_exp4.write("User Story 3 goes here")

col1, col2, col3 = st.columns((5,1,5))

with col1:

    search = st.text_input('Search Clinical AE')

with col3:
    if search:

        #search = search.to_lower()
        #ricksdf = pd.read_csv('ricksdf.csv')
        #   preprocess it here
        # REMEMBER to create a tuple of code and scores and sort the tuple by the score retrieve the top 3
        
        #for i, j in zip(ricksdf.ae_term, index):
         #   if search == i:
          #      imdrf_code = df.at[index, 'imdrf_codes']
          #      scores = (df.at[index, 'scores'])
                



        # Create a TfidfVectorizer to transform the text data
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df5['term'])

        # Transform the user input into a vector
        user_vector = vectorizer.transform([search])

        # Compute the cosine similarity between the user vector and all the text vectors
        similarity_scores = cosine_similarity(X, user_vector)

        # Get the indices of the top 3 scores
        top_indices = similarity_scores.argsort(axis=0)[-3:].flatten()

        # Get the top 3 terms, rows, and scores
        top_terms = vectorizer.get_feature_names()
        top_rows = df5.iloc[top_indices]['term']
        top_scores = similarity_scores[top_indices].flatten()

        disp = pd.DataFrame(columns=('IMDRF Code', 'IMDRF Term', 'Score'))

        top_scores_percent = (top_scores * 100).round(2)
        top_scores_percent_str = [str(i)+'%' for i in top_scores_percent]
        top_scores_percent_str.reverse()
        codes = [df5.at[i, 'code'] for i in top_indices]
        codes.reverse()
        terms = [df5.at[i, 'term'] for i in top_indices]
        terms.reverse()
        disp['IMDRF Code'] = codes
        disp['IMDRF Term'] = terms
        disp['Score'] = top_scores_percent_str
        st.table(disp)

st.write("________________________________________________________________________________")
pre_spacer1, pre_row1, pre_row2, pre_spacer2, pre_row3, pre_row4 = st.columns((2.8,0.2,4,4,0.2,4))

with pre_row1:
    st.write("📋")
with pre_row2:
    st.markdown('<p><b>Comparisson Type</b></p>', unsafe_allow_html=True)

with pre_row3:
    st.write(":calendar:")
with pre_row4:
    st.markdown('<p><b>Date Range</b></p>', unsafe_allow_html=True)

col4, col4a, col5, col6, col6a = st.columns((2.5,2.5,1,2.5,2.5))
with col4:
    st.markdown("</hr>", unsafe_allow_html=True)
    btn1 = st.button('Clinical Study Comparison')
with col4a:
    st.markdown("</hr>", unsafe_allow_html=True)
    btn2 = st.button('MDR Comparison')

with col6:
    beg = st.date_input('Start date')
with col6a:
    end = st.date_input('End date')
st.write("________________________________________________________________________________")

button_style = """
        <style>
        .stButton > button {
            color: blue;
            width: 250px;
            height: 50px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)
        
    

spacerc7, col7, col9, spacerc9= st.columns((3,2.5,2.5,3))

pmas = pd.read_csv('PMA_AE_Manual.csv', encoding='ISO-8859-1')


mn = pmas.query("PMA == 'P060040'")
sp = pd.read_csv('cp.csv', encoding='ISO-8859-1')
ot = pmas.query("PMA == 'P100047'")
ot.dropna(inplace=True)
ot.reset_index(inplace=True, drop=True)

mn_rates = ["{:.2%}".format(i/1304) for i in mn['# MDRs'][:10]]
ot_rates = ["{:.2%}".format(i/1304) for i in ot['# MDRs'][:10]]
sp_rates = ["{:.2%}".format(i/1304) for i in sp['# MDRs'][:10]]

# Clear session state if the dropdown value changes
if 'selected_value' not in st.session_state or st.session_state.selected_value != selected:
    st.session_state.clear()
    st.session_state.selected_value = selected
   
# Check if choices are already stored in session state
if 'choices' not in st.session_state:
    samp = list(mn['AE_Terms'][:10])
    # Generate random choices and store them in session state
    st.session_state.choices = samp

# Retrieve choices from session state
choices = st.session_state.choices

# Check if percentages are already stored in session state
if 'percents' not in st.session_state:
    # Generate random percentages and store them in session state
    rt = mn['Affected / at Risk (%)'][:10]
    st.session_state.percents = rt

# Retrieve percentages from session state
percents = st.session_state.percents

# Check if percentages_sup are already stored in session state
if 'percents_sup' not in st.session_state:
    # Generate random percentages_sup and store them in session state
    rt_s = sp['Affected / at Risk (%)'][:10]
    st.session_state.percents_sup = rt_s

# Retrieve percentages_sup from session state
percents_sup = st.session_state.percents_sup


st.markdown("</hr>", unsafe_allow_html=True)
st.markdown("</hr>", unsafe_allow_html=True)
with col7:
    sel_choices = []
    exp = st.expander("Pre-loaded AE's/IMDRFs")
    
    sel0 = exp.checkbox(f'{choices[0]}', value=True)
    if sel0:
        selected0 = choices[0]
        sel_choices.append(choices[0])

    sel1 = exp.checkbox(f'{choices[1]}', value=True)
    if sel1:
        selected1 = choices[1]
        sel_choices.append(choices[1])
 
    sel2 = exp.checkbox(f'{choices[2]}', value=True)
    if sel2:
        selected2 = choices[2]
        sel_choices.append(choices[2])
 
    sel3 = exp.checkbox(f'{choices[3]}', value=True)   
    if sel3:
        selected3 = choices[3]
        sel_choices.append(choices[3])

    sel4 = exp.checkbox(f'{choices[4]}', value=True)         
    if sel4:
        selected4 = choices[4]
        sel_choices.append(choices[4])
   
    sel5 = exp.checkbox(f'{choices[5]}', value=True)
    if sel5:
        selected5 = choices[5]
        sel_choices.append(choices[5])

    sel6 = exp.checkbox(f'{choices[6]}', value=True)
    if sel6:
        selected6 = choices[6]
        sel_choices.append(choices[6])
 
    sel7 = exp.checkbox(f'{choices[7]}', value=True)
    if sel7:
        selected7 = choices[7]
        sel_choices.append(choices[7])
 
    sel8 = exp.checkbox(f'{choices[8]}', value=True)   
    if sel8:
        selected8 = choices[8]
        sel_choices.append(choices[8])

    sel9 = exp.checkbox(f'{choices[9]}', value=True)         
    if sel9:
        selected9 = choices[9]
        sel_choices.append(choices[9])

st.markdown("</hr>", unsafe_allow_html=True)    
st.markdown("</hr>", unsafe_allow_html=True)

with col9:

    submissions =[['P060040','P060040/SUP-HeartMate', 'P100047-HeartWare']]


    exp2 = st.expander("Submission Select")

    
    for i in submissions:
        if selected in i:
            sbs1 = exp2.checkbox(f'{i[1]}', value=False)
            sbs1_name = i[1]
            sbs2 = exp2.checkbox(f'{i[2]}', value=False)
            sbs2_name = i[2]

spacer0, col10, spacer1, col11, spacer2, col12, spacer3 = st.columns((0.4,4,0.15,4,0.15,4,0.4))

with col10:

    # Create an empty DataFrame
    df = pd.DataFrame(columns=(f'{selected}', 'Subject Submission'))

    # Populate the DataFrame
    df.loc[0] = ['Event', 'Rate Indicator']
    #choices = random.sample(samp1, k=4)
    for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, percents):
        df.loc[i] = [choice, percent]
    
    st.table(df)

if sbs1:
    with col11:

        # Create an empty DataFrame
        df2 = pd.DataFrame(columns=(f'{sbs1_name}', 'Subject Submission'))

        # Populate the DataFrame
        df2.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice2, percent2 in zip(range(1, len(sel_choices)+1), sel_choices, percents_sup):
            df2.loc[i] = [choice2, percent2]
        
        st.table(df2)

if sbs2:
    with col12:

        # Create an empty DataFrame
        df3 = pd.DataFrame(columns=(f'{sbs2_name}', 'Subject Submission'))

        # Populate the DataFrame
        df3.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        terms = list(ot['AE_Terms'][:10])
        ratess = list(ot['Affected / at Risk (%)'][:10])
        for i, ter, rat in zip(range(1,11), terms, ratess):
            df3.loc[i] = [ter, rat]


        
        st.table(df3)

st.write("________________________________________________________________________________")
st.subheader('MDR Rates')
if btn2:

    spacer4, col13, spacer5, col14, spacer6, col15, spacer7 = st.columns((0.4,4,0.15,4,0.15,4,0.4))

    with col13:

        # Create an empty DataFrame
        df = pd.DataFrame(columns=(f'{selected}', 'Subject Submission'))

        # Populate the DataFrame
        df.loc[0] = ['Event', 'MDR Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, mn_rates):
            df.loc[i] = [choice, percent]
        
        st.table(df)

    if sbs1:
        with col14:

            # Create an empty DataFrame
            df2 = pd.DataFrame(columns=(f'{sbs1_name}', 'Subject Submission'))

            # Populate the DataFrame
            df2.loc[0] = ['Event', 'MDR Rate Indicator']
            #choices = random.sample(samp1, k=4)
            for i, choice2, percent2 in zip(range(1, len(sel_choices)+1), sel_choices, sp_rates):
                df2.loc[i] = [choice2, percent2]
            
            st.table(df2)

    if sbs2:
        with col15:

            # Create an empty DataFrame
            df3 = pd.DataFrame(columns=(f'{sbs2_name}', 'Subject Submission'))

            # Populate the DataFrame
            df3.loc[0] = ['Event', 'MDR Rate Indicator']
            #choices = random.sample(samp1, k=4)
            terms = list(ot['AE_Terms'][:10])
            ratess = list(ot['Affected / at Risk (%)'][:10])
            for i, ter, rat in zip(range(1,11), terms, ot_rates):
                df3.loc[i] = [ter, rat]


            
            st.table(df3)