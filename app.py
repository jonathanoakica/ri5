import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RI-5 Mockup", layout='wide')

df = pd.read_csv('imdrf.csv')

st.sidebar.title('RI-5 Mockup')

rev = ['P1254/S012', 'P6489/S010', 'P9876/S008', 'P3574/S006']
selected = st.sidebar.selectbox('Review Number:', rev)

col1, col2, col3 = st.columns((5,1,5))

with col1:

    search = st.text_input('Search Clinical AE')

with col3:
    if search:
        # Create a TfidfVectorizer to transform the text data
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['term'])

        # Transform the user input into a vector
        user_vector = vectorizer.transform([search])

        # Compute the cosine similarity between the user vector and all the text vectors
        similarity_scores = cosine_similarity(X, user_vector)
        # Get the indices of the top 3 scores
        top_indices = similarity_scores.argsort(axis=0)[-3:].flatten()

        # Get the top 3 terms, rows, and scores
        top_terms = vectorizer.get_feature_names()
        top_rows = df.iloc[top_indices]['term']
        top_scores = similarity_scores[top_indices].flatten()

        disp = pd.DataFrame(columns=('IMDRF Code', 'IMDRF Term', 'Score'))

        top_scores_percent = (top_scores * 100).round(2)
        top_scores_percent_str = [str(i)+'%' for i in top_scores_percent]
        top_scores_percent_str.reverse()
        codes = [df.at[i, 'code'] for i in top_indices]
        codes.reverse()
        terms = [df.at[i, 'term'] for i in top_indices]
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
            width: 300px;
            height: 50px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)
        
    

spacerc7, col7, col9, spacerc9= st.columns((3,2.5,2.5,3))

# Clear session state if the dropdown value changes
if 'selected_value' not in st.session_state or st.session_state.selected_value != selected:
    st.session_state.clear()
    st.session_state.selected_value = selected
   
# Check if choices are already stored in session state
if 'choices' not in st.session_state:
    samp = ['Infection', 'Bleeding', 'Hypertension', 'Neurological Dysfunction', 'Respiratory Failure', 
            'Pericardial Fluid', 'Cardiac Arrhythmia', 'Missed Dose', 'Underdose', 'Sedation']
    # Generate random choices and store them in session state
    st.session_state.choices = random.sample(samp, k=5)

# Retrieve choices from session state
choices = st.session_state.choices

# Check if percentages are already stored in session state
if 'percents' not in st.session_state:
    # Generate random percentages and store them in session state
    st.session_state.percents = sorted([str(random.randint(10, 99)) + '%' for i in range(5)], reverse=True)

# Retrieve percentages from session state
percents = st.session_state.percents

# Check if percentages_sup are already stored in session state
if 'percents_sup' not in st.session_state:
    # Generate random percentages_sup and store them in session state
    st.session_state.percents_sup = sorted([str(random.randint(10, 99)) + '%' for i in range(5)], reverse=True)

# Retrieve percentages_sup from session state
percents_sup = st.session_state.percents_sup

# Check if percentages_sup are already stored in session state
if 'percents_sup1' not in st.session_state:
    # Generate random percentages_sup and store them in session state
    st.session_state.percents_sup1 = sorted([str(random.randint(10, 99)) + '%' for i in range(5)], reverse=True)

# Retrieve percentages_sup from session state
percents_sup1 = st.session_state.percents_sup1

# Check if percentages_other are already stored in session state
if 'percents_other' not in st.session_state:
    # Generate random percentages_other and store them in session state
    st.session_state.percents_other = sorted([str(random.randint(10, 99)) + '%' for i in range(5)], reverse=True)

# Retrieve percentages_other from session state
percents_other = st.session_state.percents_other

# Check if percentages_other are already stored in session state
if 'percents_other1' not in st.session_state:
    # Generate random percentages_other and store them in session state
    st.session_state.percents_other1 = sorted([str(random.randint(10, 99)) + '%' for i in range(5)], reverse=True)

# Retrieve percentages_other from session state
percents_other1 = st.session_state.percents_other1

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

st.markdown("</hr>", unsafe_allow_html=True)    
st.markdown("</hr>", unsafe_allow_html=True)

with col9:

    submissions =[['P1254/S012','P1254/S011-Supplement', 'P1254/S010-Supplement', 'P1111-Other', 'P2354-Other'],
    ['P6489/S010','P6489/S009-Supplement', 'P6489/S008-Supplement', 'P2222-Other', 'P3456-Other'],
    ['P9876/S008','P9876/S007-Supplement', 'P9876/S006-Supplement', 'P3333-Other', 'P4567-Other'],
    ['P3574/S006', 'P3574/S005-Supplement', 'P3574/S004-Supplement', 'P4444-Other', 'P5678-Other']]


    exp2 = st.expander("Submission Select")

    
    for i in submissions:
        if selected in i:
            sbs1 = exp2.checkbox(f'{i[1]}', value=False)
            sbs1_name = i[1]
            sbs2 = exp2.checkbox(f'{i[2]}', value=False)
            sbs2_name = i[2]
            sbs3 = exp2.checkbox(f'{i[3]}', value=False)
            sbs3_name = i[3]
            sbs4 = exp2.checkbox(f'{i[4]}', value=False)
            sbs4_name = i[4]

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
        df = pd.DataFrame(columns=(f'{selected}', f'{sbs1_name}'))

        # Populate the DataFrame
        df.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, percents_sup):
            df.loc[i] = [choice, percent]
        
        st.table(df)
if sbs2:
    with col12:

        # Create an empty DataFrame
        df = pd.DataFrame(columns=(f'{selected}', f'{sbs2_name}'))

        # Populate the DataFrame
        df.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, percents_sup1):
            df.loc[i] = [choice, percent]
        
        st.table(df)

spacer4, col13, spacer5, col14, spacer6, col15, spacer7 = st.columns((0.4,4,0.15,4,0.15,4,0.4))

if sbs3:
    with col13:

        # Create an empty DataFrame
        df = pd.DataFrame(columns=(f'{selected}', f'{sbs3_name}'))

        # Populate the DataFrame
        df.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, percents_other):
            df.loc[i] = [choice, percent]
        
        st.table(df)

if sbs4:
    with col14:

        # Create an empty DataFrame
        df = pd.DataFrame(columns=(f'{selected}', f'{sbs4_name}'))

        # Populate the DataFrame
        df.loc[0] = ['Event', 'Rate Indicator']
        #choices = random.sample(samp1, k=4)
        for i, choice, percent in zip(range(1, len(sel_choices)+1), sel_choices, percents_other1):
            df.loc[i] = [choice, percent]
        
        st.table(df)