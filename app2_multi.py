import streamlit as st
from PIL import Image
from io import BytesIO
import pandas as pd
import fitz  # PyMuPDF
import sqlite3
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, GridUpdateMode
import cv2
import numpy as np
import pytesseract
import google.generativeai as genai
import streamlit_authenticator as stauth
import os
import yaml
from yaml.loader import SafeLoader

#hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create an authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login
fields = {
    'username': 'Username',
    'password': 'Password'
}
name, authentication_status, username = authenticator.login()

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')


GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the GenerativeAI model
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 256,
}

# Configure safety settings for the model
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Load the GenerativeAI model
model = genai.GenerativeModel(model_name="gemini-1.5-flash-001", generation_config=MODEL_CONFIG, safety_settings=safety_settings)


def get_user_db(username):
    db_path = f"{username}_invoices.db"
    conn = sqlite3.connect(db_path)
    return conn

# Initialize session state variables if they don't already exist
if 'responses_deleted' not in st.session_state:
    st.session_state.responses_deleted = False

#Quest mapping
quest_back_front={'Tout extraire':'Tout extraire' , 'Assujetti à la TVA ?' : 'TVA', 
            'N° Etab Secondaire.Code Catégorie.Code TVA.Matricule Fiscal':'Matricule Fiscal', 
            'Nom et Prénom première ligne' : 'Nom et Prénom', 'Raison Sociale deuxième ligne': 'Raison Sociale', 
            'Activité': 'Activité', 'Date de début de l\'activité' : 'Date de début de l\'activité', 
            'Adresse': "Adresse", 'Other': "Question Personnalisée"}

quest_front_back={'Tout extraire':'Tout extraire','TVA':'Assujetti à la TVA ?',
                  'Matricule Fiscal':'N° Etab Secondaire.Code Catégorie.Code TVA.Matricule Fiscal', 
                  'Nom et Prénom': 'Nom et Prénom première ligne','Raison Sociale':'Raison Sociale deuxième ligne',
                  'Activité':'Activité','Date de début de l\'activité':'Date de début de l\'activité',
                  "Adresse":"Adresse","Question Personnalisée":"Other"}

def extract_images_from_pdf(pdf_file):
    images = []
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
    return images

def perform_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    horizontal_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, None, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, None, iterations=1)
    lines = cv2.add(horizontal_lines, vertical_lines)
    no_lines_image = cv2.bitwise_and(thresh_image, cv2.bitwise_not(lines))
    resized_image = cv2.resize(no_lines_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised_image = cv2.fastNlMeansDenoising(resized_image, h=30)
    text = pytesseract.image_to_string(denoised_image, lang='fra')
    return text

def process_image_with_geminiai(image_bytes, system_prompt, user_prompt):
    input_prompt = [system_prompt, {"mime_type": "image/png", "data": image_bytes}, user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

def save_response_to_db(cursor, invoice_name, question, response_text):
    cursor.execute('INSERT INTO invoices (invoice_name, question, response) VALUES (?, ?, ?)', (invoice_name, question, response_text))
    cursor.connection.commit()

def fetch_all_responses(connection):
    connection.execute('SELECT * FROM invoices')
    rows = connection.fetchall()
    return rows

def handle_grid_update(grid_update_event):
    updated_data = grid_update_event['data']
    deleted_rows = grid_update_event['deleted_rows']
    
    if deleted_rows:
        for row in deleted_rows:
            c.execute('DELETE FROM invoices WHERE id = ?', (row['ID'],))
            conn.commit()
    
    if updated_data:
        for item in updated_data:
            c.execute('''
                UPDATE invoices
                SET invoice_name = ?, question = ?, response = ?
                WHERE id = ?
            ''', (item['Invoice Name'], item['Question'], item['Response'], item['ID']))
            conn.commit()
    
    st.success("Responses updated successfully.")

def answer_question(quest, image, uploaded_file_name):
    if quest == 'Other':
        user_prompt = st.text_input("Enter your question:", key=f'{uploaded_file_name}_{quest}')
    else:
        user_prompt = quest

    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')

    ocr_text = perform_ocr(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    extended_system_prompt = system_prompt + f"{user_prompt}\nThe extracted text from the invoice image is:\n{ocr_text}"

    gemini_response = process_image_with_geminiai(image_bytes.getvalue(), extended_system_prompt, user_prompt)
    return gemini_response

def process_file(uploaded_file, selected_questions,cursor):
    try:
        global system_prompt
        system_prompt = """
        You are an expert in understanding invoices.
        Invoice images will be provided to you,
        and your task is to answer questions based on the content of the input image.
        Provide concise answers, not full sentences.
        If the question is about a specific field, provide only the value of that field.
        For example, if asked for the client name, provide only the name.
        And provide the answer from one line, not from multiple lines
        Here are some examples :
        Question: Assujetti à la TVA ?
        Response: "Non assujetti à la TVA" but it depends on the uploaded invoices.
        """
        image_placeholder = st.empty()
        
        if uploaded_file.type == 'application/pdf':
            pdf_images = extract_images_from_pdf(uploaded_file)
            if pdf_images:
                image_placeholder.image(pdf_images[0], caption=f'Extracted Image from {uploaded_file.name}', use_column_width=True)
                image = pdf_images[0]
            else:
                st.error(f"No images found in {uploaded_file.name}")
                return
        
        elif uploaded_file.type in ['image/jpeg', 'image/png']:
            image = Image.open(uploaded_file)
            image_placeholder.image(image, caption=f'Uploaded Image {uploaded_file.name}', use_column_width=True)

        for selected_question in selected_questions:
            if selected_question == 'Tout extraire':
                for quest in questions_back[1:-1]:
                    gemini_response = answer_question(quest, image, uploaded_file.name)
                    response_text = gemini_response if gemini_response else 'None'
                    save_response_to_db(cursor, uploaded_file.name, quest_back_front[quest], response_text)
                break
            else:
                gemini_response = answer_question(selected_question, image, uploaded_file.name)
                response_text = gemini_response if gemini_response else 'None'
                save_response_to_db(cursor, uploaded_file.name, quest_back_front[selected_question], response_text)
        st.success(f"Response for {uploaded_file.name} saved successfully.")
        image_placeholder.empty()
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        
# Get the database connection for the logged-in user
if authentication_status:
    conn = get_user_db(username)
    c = conn.cursor()
    st.write(f"Current working directory: {os.getcwd()}")

    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS invoices
        (id INTEGER PRIMARY KEY, invoice_name TEXT, question TEXT, response TEXT)
    ''')
    conn.commit()
    # Streamlit UI
    #st.title('')
    st.markdown("<h1 style='text-align: center;'>Automate the entry <br> of your clients' tax data</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 '>Upload your invoice images or PDFs and select questions to process them</h4>", unsafe_allow_html=True)
    #st.subheader('')
    
    questions_back = ['Tout extraire', 'Assujetti à la TVA ?', 'N° Etab Secondaire.Code Catégorie.Code TVA.Matricule Fiscal', 'Nom et Prénom première ligne', 'Raison Sociale deuxième ligne', 'Activité', 'Date de début de l\'activité', 'Adresse', 'Other']
    
    questions_front=['Tout extraire' , 'TVA', 'Matricule Fiscal', 'Nom et Prénom', 'Raison Sociale','Activité', 'Date de début de l\'activité', 
                "Adresse", "Question Personnalisée"]
    
    select_questions = st.multiselect('Select questions:', questions_front)
    
    selected_questions= [quest_front_back[quest] for quest in select_questions]
    
    uploaded_files = st.file_uploader("Upload Invoice Images/Pdfs", type=['jpg', 'png', 'pdf'], accept_multiple_files=True) 
    # Initialize lists to track processed and unprocessed files
    processed_files = []
    unprocessed_files = [file.name for file in uploaded_files] if uploaded_files else []
    
    if uploaded_files:
        # Display a "waiting" sign for files that are not yet processed
        if st.button('Process All Files'):   
            for uploaded_file in uploaded_files:
                placeholders = [st.empty() for _ in unprocessed_files] if unprocessed_files else []
                for i, file in enumerate(unprocessed_files):
                    placeholders[i].write(f'Waiting to process: {file}')
                with st.spinner(f'Processing {uploaded_file.name}...'):
                    process_file(uploaded_file, selected_questions,c)
                processed_files.append(uploaded_file.name)
                for i, file in enumerate(unprocessed_files):
                    placeholders[i].empty()
                unprocessed_files.remove(uploaded_file.name)
        


    # Displaying responses
    show_responses = st.button('Show all responses')
    if show_responses:
        rows = fetch_all_responses(c)
        if rows:
            df = pd.DataFrame(rows, columns=['ID', 'Invoice Name', 'Question', 'Response'])
            st.dataframe(df)
        else:
            st.write("No responses found.")
    
    # Button to delete records without confirmation
    if st.button('Delete all responses'):
        try:
            c.execute('DELETE FROM invoices')
            conn.commit()
            st.success("All responses deleted successfully.")
            st.session_state.responses_deleted = True
        except Exception as e:
            st.error(f"Error deleting responses: {e}")
    
    if not st.session_state.responses_deleted:
        st.write("Responses have not been deleted.")
    
    if show_responses:
        invoices = ['patente','contrat']
        for invoice_type in invoices:
            df_invoice = df[df['Invoice Name'].str.contains(invoice_type, case=False)]
            st.write(f"DataFrame for {invoice_type}:")
    
            gb = GridOptionsBuilder.from_dataframe(df_invoice)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_default_column(editable=True, groupable=True)
    
            grid_options = gb.build()
    
            grid_response = AgGrid(
                df_invoice,
                gridOptions=grid_options,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                reload_data=True,
                on_grid_update=handle_grid_update
            )
