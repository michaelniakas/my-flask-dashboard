# MCB Dashboard Application

## Overview

The MCB Dashboard Application is designed to provide a comprehensive view of data across different departments. Users can upload, view, and visualize data with ease. This application is built using Flask and Dash, offering a responsive and interactive user experience.

## Features

### View Data
- Users can view data for different departments, including `BDM`, `Marketing`, and `Linkedin`.
- Displays data in a table format with search functionality.

### Edit Data
- Users can upload new data files for `Marketing` and `Linkedin` departments.
- The uploaded data is appended to existing data and displayed accordingly.

### Visualizations
- Time series plots for various metrics in the `Linkedin` department.
- Interactive plots for the `BDM` department, allowing filtering by `Status` and `Shortlist`.

### Filtering
- For the `BDM` department, users can filter visualizations by `Status` and `Shortlist` values.

### Responsive Design
- The application has a responsive design, ensuring it works well on different devices.

### Data Processing
- Includes data normalization and processing for uploaded files.
- Handles different file formats (`.xls` and `.xlsx`).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mcb-dashboard.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mcb-dashboard
    ```
3. Create a virtual environment:
    ```bash
    python -m venv env
    ```
4. Activate the virtual environment:
    ```bash
    # On Windows
    .\env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    ```
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    python app.py
    ```
2. Open a web browser and go to `http://0.0.0.0:5000` to access the application.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
