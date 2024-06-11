from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session,flash
import os
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
import xlrd
import openpyxl
import shutil

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'  # Set to a random secret value

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls','txt'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
BASE_PATH = "C:/Users/Administrator/Desktop/MCB_PROJECT/Dashboard"
# Define a simple access control for the Management department
ACCESS_CODES = {
    'Management': 'director123'  # Example access code
}



@app.route('/')
def index():
    return render_template('home.html')
# BERT setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def determine_campus_profile_path(excel_filename):
    campus_profiles = {
        'Chemistry and New Material.xlsx': 'C:/Users/Administrator/Desktop/MCB_PROJECT/Brightlands Chemelot Campus profile/Brightlands Chemelot Campus profile.txt',
        'Agri Business and Food.xlsx': 'C:/Users/Administrator/Desktop/MCB_PROJECT/Brightlands Campus Greenport Venlo profile/Brightlands Campus Greenport Venlo profile.txt',
        'Medical and Life Sciences.xlsx': 'C:/Users/Administrator/Desktop/MCB_PROJECT/Brightlands Maastricht Health Campus BV profile/Brightlands Maastricht Health Campus BV profile.txt',
        'Data Sciences and Smart Services.xlsx': 'C:/Users/Administrator/Desktop/MCB_PROJECT/Brightlands_Smart_Services_Campus_profile/Brightlands_Smart_Services_Campus_profile.txt'
    }
    # Return the campus profile path for the given Excel filename
    return campus_profiles.get(excel_filename)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
def get_bert_embedding(text):
    tokens = tokenizer.tokenize(text)
    # Split tokens into chunks of size 510 (to account for [CLS] and [SEP] tokens)
    token_chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)]
    
    chunk_embeddings = []
    for token_chunk in token_chunks:
        # Add [CLS] and [SEP] tokens
        tokens = ['[CLS]'] + token_chunk + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = model(input_tensor)
        # Get the embedding of the [CLS] token for the chunk
        chunk_embeddings.append(last_hidden_states[0][:,0,:].numpy())
    
    # Average the embeddings of the chunks to get a single representation
    avg_embedding = sum(chunk_embeddings) / len(chunk_embeddings)
    return avg_embedding
def update_similarity_score(filename, rowIndex, similarity_score):
    filepath = os.path.join(BASE_PATH, 'BDM', filename)  # Adjust according to your file structure
    df = pd.read_excel(filepath)
    df.at[rowIndex, 'Similarity(campus with association)'] = similarity_score
    df.to_excel(filepath, index=False)
def calculate_similarity(org_filepath, campus_profile_path):

    with open(campus_profile_path, 'r', encoding='utf-8') as campus_file:
        campus_text = campus_file.read()
    
    with open(org_filepath, 'r', encoding='utf-8') as org_file:
        org_text = org_file.read()
    
    org_embedding = get_bert_embedding(org_text)
    campus_embedding = get_bert_embedding(campus_text)
    
    similarity = cosine_similarity(org_embedding.reshape(1, -1), campus_embedding.reshape(1, -1))[0][0]
    return similarity

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_marketing', methods=['GET', 'POST'])
def upload_marketing_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Clear the upload folder before saving the new file
            clear_folder(app.config['UPLOAD_FOLDER'])
            # Clear the MarketingUpload folder before saving the new file
            clear_folder(os.path.join(BASE_PATH, 'MarketingUpload'))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            normalized_df = process_marketing_file(file_path)
            # Save the normalized data to a new Excel file at this location "C:/Users/Administrator/Desktop/MCB_PROJECT/Dashboard/MarketingUpload"     
            normalized_filename = secure_filename(file.filename).rsplit('.', 1)[0] + '.xlsx'
            normalized_file_path = os.path.join(BASE_PATH, 'MarketingUpload', normalized_filename)
            normalized_df.to_excel(normalized_file_path, index=False)
            return redirect(f'/dashboard/MarketingUpload/{normalized_filename}')
    return render_template('upload_marketing.html')

def process_marketing_file(file_path):
    file_extension = file_path.rsplit('.', 1)[1].lower()
    if file_extension == 'xls':
        df = pd.read_excel(file_path, sheet_name='All posts', engine='xlrd')
    else:
        df = pd.read_excel(file_path, sheet_name='All posts', engine='openpyxl')
    
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)
    df['hashtags'] = df['Post title'].str.extract(r'#(\w+)')
    columns = ['Likes', 'Comments', 'Clicks', 'Impressions']
    weights = {'Likes': 0.2, 'Comments': 0.3, 'Clicks': 0.3, 'Impressions': 0.2}

    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns])
    normalized_df = pd.DataFrame(normalized_data, columns=columns)

    normalized_df['Weighted_Engagement_Score'] = (normalized_df['Likes'] * weights['Likes'] + 
                                                  normalized_df['Comments'] * weights['Comments'] + 
                                                  normalized_df['Clicks'] * weights['Clicks'] + 
                                                  normalized_df['Impressions'] * weights['Impressions'])

    normalized_df['Hashtags'] = df['hashtags']
    return normalized_df


@app.route('/upload_org_profile/<string:department>/<string:filename>/<int:rowIndex>/<int:colIndex>', methods=['POST'])
def upload_org_profile(department, filename, rowIndex, colIndex):
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        temp_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(temp_filepath)
        print(f"File saved to: {temp_filepath}")
        # Determine the correct campus profile based on the department and filename
        campus_profile_path = determine_campus_profile_path(filename)
        if not campus_profile_path:
            os.remove(temp_filepath)  # Clean up the uploaded file
            return jsonify({'status': 'error', 'message': 'Campus profile not found for the provided filename'}), 400

        # Calculate the similarity score
        similarity_score = calculate_similarity(temp_filepath, campus_profile_path)
        
        # Ensure similarity_score is JSON serializable
        similarity_score_json_serializable = float(similarity_score)

        # Load the Excel file that needs updating
        excel_filepath = os.path.join(BASE_PATH, department, filename)
        # Update the Excel file with the new similarity score for the specific row
        update_similarity_score(filename, rowIndex, similarity_score)
            
        os.remove(temp_filepath)  # Clean up the uploaded file

        return jsonify({'status': 'success', 'similarity_score': similarity_score_json_serializable}), 200
    return jsonify({'status': 'error', 'message': 'Invalid file extension'}), 400

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_file/<string:department>/<string:filename>')
def select_file_with_filename(department, filename):
    # Your code here
    return redirect(f'/dashboard/{department}/{filename}')



@app.route('/auth/<string:department>', methods=['GET', 'POST'])
def auth(department):
    if request.method == 'POST':
        user_code = request.form.get('accessCode')
        if user_code == ACCESS_CODES.get(department):
            session['temp_authenticated'] = True
            return redirect(url_for('department_files', department=department))
        else:
            flash('Invalid access code.', 'error')
    return render_template('auth.html', department=department)
@app.route('/auth_edit/<string:department>/<string:filename>', methods=['GET', 'POST'])
def auth_edit(department,filename):
    if request.method == 'POST':
        user_code = request.form.get('accessCode')
        if user_code == ACCESS_CODES.get(department):
            session['temp_authenticated'] = True
            return redirect(url_for('view_excel', department=department,filename=filename))
        else:
            flash('Invalid access code.', 'error')
    return render_template('auth.html', department=department,filename=filename)
@app.route('/upload/<string:department>', methods=['GET', 'POST'])
def upload_file(department):
    if request.method == 'POST':
        # Check if the post request has the file part
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], department + '.xlsx')
            file.save(filename)
            return redirect(url_for('redirect_to', department=department))
    return render_template('upload.html', department=department)

@app.route('/view_data/<string:department>')
def department_files(department):
    # Check if department requires an access code and if the user is authenticated
    if department == 'Management':
        if not session.pop('temp_authenticated',None):
            # Redirect to authentication page if not authenticated
            return redirect(url_for('auth', department=department))

    if department == 'BDM':
        file_dir = os.path.join(BASE_PATH, 'BDM')
        files = [f for f in os.listdir(file_dir) if f.endswith('.xlsx')]
        return render_template('bdm_files.html', files=files, department=department)
    if department == 'Linkedin':
        file_dir = os.path.join(BASE_PATH, 'Linkedin')
        files = [f for f in os.listdir(file_dir) if f.endswith('.xlsx')]
        return render_template('linkedin_files.html', files=files, department=department)
    elif department == 'Marketing':
        file_dir = os.path.join(BASE_PATH, 'Marketing')
        files = [f for f in os.listdir(file_dir) if f.endswith('.xlsx')]
        return render_template('marketing_files.html', files=files, department=department)
    else:
        filepath = os.path.join(BASE_PATH, department, department + '.xlsx')
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            table_html = df.to_html(classes='table table-striped', border=0)
            return render_template(f'{department}_page.html', table=table_html, department=department)
        else:
            return render_template('error.html', message=f'File for {department} not found.')
        


@app.route('/view_excel/<string:department>/<string:filename>')
def view_excel(department, filename):
    if department == 'Management':
        if not session.pop('temp_authenticated', None):
            # Redirect to authentication page if not authenticated
            return redirect(url_for('auth_edit', department=department, filename=filename))
    
    filepath = os.path.join(BASE_PATH, department, filename if department in ['BDM', 'Marketing'] else department + '.xlsx')
    
    if not os.path.exists(filepath):
        return render_template('error.html', message=f'File {filename} not found in {department}.')

    df = pd.read_excel(filepath)

    table_html = df.to_html(classes='table table-striped', border=0, table_id="excelTable", escape=False, index=False)
    soup = BeautifulSoup(table_html, 'html.parser')

    for th in soup.find_all('th'):
        th['contenteditable'] = 'false'
    for tr in soup.find_all('tr'):
        first_cell = tr.find('td')
        if first_cell:
            first_cell['contenteditable'] = 'false'

    table_html_modified = str(soup)

    # Use different templates based on the department
    if department == 'BDM':
        return render_template('edit_data.html', table=table_html_modified, department=department, filename=filename)
    elif department == 'Marketing' or department == 'Management':
        return render_template('edit_data_Marketing.html', table=table_html_modified, department=department, filename=filename)



@app.route('/view_file/<string:department>/<string:filename>')
def view_file(department, filename):
    # Similar logic to view_excel but intended for viewing rather than editing
    if department == 'Linkedin':
        filepath = os.path.join(BASE_PATH, department, department + '.xlsx')
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            df = df.applymap(lambda x: '{:.0f}'.format(x) if pd.notna(x) and isinstance(x, (int, float)) and x == int(x) else x)
            return render_template('view_linkedin_data.html', table=df.to_html(classes='data', header="true"))
    if department in ['BDM', 'Marketing']:
        filepath = os.path.join(BASE_PATH, department, filename)
    else:
        filepath = os.path.join(BASE_PATH, department, department + '.xlsx')
    
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        # Format numerical data to display integers without decimal points
        df = df.applymap(lambda x: '{:.0f}'.format(x) if pd.notna(x) and isinstance(x, (int, float)) and x == int(x) else x)
        # Convert the DataFrame to an HTML table string without editable attributes
        table_html = df.to_html(classes='table table-striped', border=0,index=False)
        return render_template('view_data.html', table=table_html, department=department, filename=filename)
    else:
        return render_template('error.html', message=f'File {filename} not found in {department}.')

@app.route('/select_file/<string:department>')
def select_file(department):
    if department == 'BDM' or department == 'Marketing':
        file_dir = os.path.join(BASE_PATH, department)
        # List all Excel files in the directory
        files = [f for f in os.listdir(file_dir) if f.endswith('.xlsx')]
        # Render a template displaying these files as links for editing
        return render_template('select_file_for_edit.html', files=files, department=department)

@app.route('/update_excel/<string:department>/<string:filename>', methods=['POST'])
def update_excel(department, filename):
    filepath = os.path.join(BASE_PATH, department, filename if department in ['BDM', 'Marketing'] else department + '.xlsx')
    
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "File does not exist."})
    
    data = request.json.get('tableData', [])
    df = pd.read_excel(filepath)
    similarity_index = df.columns.get_loc("Similarity(campus with association)") if "Similarity(campus with association)" in df.columns else None

    recalculated_rows = set()  # Keep track of rows where we recalculated Spinoff

    for cell in data:
        row, col, value = cell['row'], cell['col'], cell['value']
        if 0 <= row < len(df) and 0 <= col < len(df.columns):
            column_name = df.columns[col]

            if col == similarity_index and (value == "" or pd.isnull(value)):
                continue  # Skip similarity update if value is not valid
            
            if value == "" or pd.isnull(value):
                continue  # Skip empty values

            try:
                if column_name in ['Duration of event', 'Average number of delegates']:
                    # Ensure value is treated as a string for isdigit check
                    str_value = str(value)
                    if str_value.isdigit():
                        value = int(value)
                    else:
                        continue  # Skip if the value is not a valid digit string
                    
                    df.at[row, column_name] = value

                    # Schedule recalculation for Spinoff
                    recalculated_rows.add(row)

                elif column_name == 'Bidding deadline':
                    if value:
                        df.at[row, column_name] = datetime.strptime(value, '%Y-%m-%d').date()
                elif column_name == 'Shortlist':
                    df.at[row, column_name] = value.lower() == 'true'
                elif column_name == 'Status':
                    df.at[row, column_name] = value.lower()
                elif column_name == 'Initiatief MCB/MECC':
                    df.at[row, column_name] = value.lower()
                elif column_name == 'BDM Manager':
                    df.at[row, column_name] = value.lower()    
                else:
                    df.at[row, column_name] = value  # Handle other columns normally
            except ValueError as e:
                print(f"Error processing {column_name} for row {row}: {e}")

    # Recalculate Spinoff for rows that had relevant changes
    for row in recalculated_rows:
        if pd.notnull(df.at[row, 'Duration of event']) and pd.notnull(df.at[row, 'Average number of delegates']):
            duration = df.at[row, 'Duration of event']
            delegates = df.at[row, 'Average number of delegates']
            df.at[row, 'Spinoff'] = duration * delegates * 365  # Calculate Spinoff

    df.to_excel(filepath, index=False)
    if department == 'BDM':
        update_overall_file()
    if department == 'Marketing' and ('2022' in filename or '2023' in filename or '2024' in filename or '2025' in filename or '2026' in filename or '2027' in filename or '2028' in filename or '2029' in filename or '2030' in filename):
        update_marketing_overall(filepath, filename, department)
    return jsonify({"status": "success", "message": "Excel file updated successfully."})



def update_overall_file():
    overall_path = os.path.join(BASE_PATH, 'BDM', 'Overall.xlsx')
    file_dir = os.path.join(BASE_PATH, 'BDM')

    # Create a list to hold dataframes
    dataframes = []

    # Iterate over individual files, excluding 'Overall.xlsx'
    for filename in os.listdir(file_dir):
        if filename.endswith('.xlsx') and filename != 'Overall.xlsx':
            file_path = os.path.join(file_dir, filename)
            df = pd.read_excel(file_path)
            dataframes.append(df)

    # Concatenate all dataframes
    overall_df = pd.concat(dataframes, ignore_index=True)
    overall_df.to_excel(overall_path, index=False)
    print("Overall file updated successfully.")


def update_marketing_overall(filepath, filename, department):
    print(f"Updating overall for {filename}")  # Debugging line
    year = filename.split(' ')[1].split('.')[0]  # Adjusted to remove file extension before extracting the year
    print(f"Year extracted: {year}")  # Debugging line

    overall_path = os.path.join(BASE_PATH, department, 'Marketing Overall.xlsx')
    if os.path.exists(overall_path):
        df_overall = pd.read_excel(overall_path)
        df_year = pd.read_excel(filepath)

        for column in ['1st Quarter', '2nd Quarter','3rd Quarter', '4th Quarter','Overall']:
            if column in df_year:
                df_overall[f'{column}({year})'] = df_year[column]
            else:
                print(f"Column {column} not found in {filename}")  # Debugging line

        df_overall.to_excel(overall_path, index=False)
        print("Overall file updated successfully.")  # Confirmation message
    else:
        print(f"{overall_path} does not exist.")  # Error message




@app.route('/plot/<string:department>')
def plot_data(department):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], department + '.xlsx')
    df = pd.read_excel(filepath)
    plt.figure()
    df.plot()
    plot_path = os.path.join('static', department + '_plot.png')
    plt.savefig(plot_path)
    return send_from_directory('static', department + '_plot.png')

def append_linkedin_data(file_path):
    file_extension = file_path.rsplit('.', 1)[1].lower()
    
    if file_extension == 'xls':
        new_data = pd.read_excel(file_path, engine='xlrd', skiprows=1)
    elif file_extension == 'xlsx':
        new_data = pd.read_excel(file_path, engine='openpyxl', skiprows=1)
    else:
        raise ValueError("Unsupported file format")
    
    #print(new_data) to review the data
    print(new_data)
    new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce')

    existing_file_path = os.path.join(BASE_PATH, 'Linkedin', 'Linkedin.xlsx')
    
    # Load the existing data
    if os.path.exists(existing_file_path):
        existing_data = pd.read_excel(existing_file_path)
        existing_data['Date'] = pd.to_datetime(existing_data['Date'], errors='coerce')
        
        # Append only new rows
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset='Date', keep='last').sort_values('Date')
    else:
        combined_data = new_data
    
    # Save the combined data
    combined_data.to_excel(existing_file_path, index=False)

@app.route('/upload_linkedin', methods=['GET', 'POST'])
def upload_linkedin_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Clear the upload folder before saving the new file
            clear_folder(app.config['UPLOAD_FOLDER'])            
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            temp_filename = f'Linkedin_temp.{file_extension}'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            file.save(file_path)
            append_linkedin_data(file_path)
            return redirect(url_for('view_file', department='Linkedin', filename='Linkedin.xlsx'))
    return render_template('upload_linkedin.html')

if __name__ == '__main__':
    from dashboard_app  import init_dash
    app = init_dash(app)

    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)