from flask import Flask, request, jsonify, render_template
import requests
import re
from bs4 import BeautifulSoup
import json
from datetime import datetime as dt
import csv
import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class VolcanoLSTMForecaster:
    def __init__(self, sequence_length=30, forecast_days=7):
        self.sequence_length = sequence_length
        self.forecast_days = forecast_days
        self.scalers = {}
        self.label_encoders = {}
        self.model = None
        self.feature_columns = None
        self.target_columns = None
        
    def load_and_prepare_data(self, csv_path):
        """Load and prepare volcano data for LSTM training"""
        print("Loading volcano activity data...")
        df = pd.read_csv(csv_path)
        
        # Convert date and sort chronologically (oldest first for time series)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Define key target variables for forecasting
        self.target_columns = [
            'Alert_Level', 'Eruption_Count', 'Eruption_Severity_Score',
            'Volcanic_Earthquakes', 'Volcanic_Tremors', 'SO2_Flux_tpd',
            'Plume_Height_m', 'Plume_Strength'
        ]
        
        # Encode categorical features
        categorical_cols = ['Plume_Drift_Direction']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Select numeric features for modeling
        numeric_cols = [
            'Alert_Level', 'Acidity_pH', 'Crater_Temperature_C', 'SO2_Flux_tpd',
            'Plume_Height_m', 'Plume_Strength', 'Plume_Drift_Direction',
            'Eruption_Count', 'Eruption_Severity_Score',
            'Total_Eruption_Duration_Min', 'Avg_Eruption_Duration_Min',
            'Volcanic_Earthquakes', 'Volcanic_Tremors', 'Total_Tremor_Duration_Min',
            'Has_Long_Tremor', 'Has_Weak_Tremor', 'Caldera_Trend', 'TVI_Trend',
            'North_Trend', 'SE_Trend', 'LT_Inflation', 'LT_Deflation',
            'ST_Inflation', 'ST_Deflation'
        ]
        
        self.feature_columns = numeric_cols
        
        # Handle missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Scale features
        for col in numeric_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        print(f"Prepared {len(df)} samples with {len(numeric_cols)} features")
        return df[['Date'] + numeric_cols]
    
    def create_sequences(self, data, columns):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.forecast_days + 1):
            # Input sequence (past 30 days)
            X.append(data[i-self.sequence_length:i, :])  # Exclude date column
            
            # Target sequence (next 7 days for key variables)
            target_indices = [columns.get_loc(col) - 1 for col in self.target_columns]
            y.append(data[i:i+self.forecast_days, target_indices])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, output_shape):
        """Build multi-output LSTM model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(np.prod(output_shape), activation='linear'),
            tf.keras.layers.Reshape(output_shape)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, csv_path, validation_split=0.2, epochs=100, batch_size=32):
        """Train the LSTM model"""
        # Prepare data
        df = self.load_and_prepare_data(csv_path)
        data = df.drop(columns=['Date']).values
        
        # Create sequences
        X, y = self.create_sequences(data, df.columns)
        print(f"Created {len(X)} sequences")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=(self.forecast_days, len(self.target_columns))
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\nTraining Loss: {train_loss[0]:.4f}")
        print(f"Validation Loss: {val_loss[0]:.4f}")
        
        # Save model
        self.model.save('volcano_lstm_model.h5')
        print("Model saved as 'volcano_lstm_model.h5'")
        
        return history
    
    def predict_next_7_days(self, csv_path, plot_results=True):
        """Predict volcano activity for next 7 days"""
        if self.model is None:
            print("Loading saved model...")
            self.model = tf.keras.models.load_model('volcano_lstm_model.h5')
        
        # Prepare recent data
        df = self.load_and_prepare_data(csv_path)
        recent_data = df.iloc[-self.sequence_length:, 1:].values  # Last 30 days
        
        assert recent_data.shape[1] == self.model.input_shape[2], \
            f"Feature mismatch: expected {self.model.input_shape[2]}, got {recent_data.shape[1]}"

        # Make prediction
        prediction = self.model.predict(recent_data.reshape(1, *recent_data.shape))
        prediction = prediction.reshape(self.forecast_days, len(self.target_columns))
        
        # Inverse transform predictions
        predictions_dict = {}
        for i, col in enumerate(self.target_columns):
            if col in self.scalers:
                pred_values = self.scalers[col].inverse_transform(
                    prediction[:, i].reshape(-1, 1)
                ).flatten()
                predictions_dict[col] = pred_values
        
        # Create forecast dataframe
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            **predictions_dict
        })
        
        # Display results
        print("\n=== 7-DAY VOLCANO ACTIVITY FORECAST ===")
        print(forecast_df.round(2))
        
        # Risk assessment
        self._assess_volcanic_risk(forecast_df)
        
        if plot_results:
            self._plot_forecast(df, forecast_df)
        
        return forecast_df
    
    def _assess_volcanic_risk(self, forecast_df):
        """Assess volcanic risk based on predictions"""
        print("\n=== RISK ASSESSMENT ===")
        
        # Alert level analysis
        max_alert = forecast_df['Alert_Level'].max()
        if max_alert >= 3:
            risk_level = "HIGH"
        elif max_alert >= 2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        print(f"Overall Risk Level: {risk_level}")
        
        # Key indicators
        max_eruptions = forecast_df['Eruption_Count'].max()
        max_earthquakes = forecast_df['Volcanic_Earthquakes'].max()
        max_so2 = forecast_df['SO2_Flux_tpd'].max()
        max_plume = forecast_df['Plume_Height_m'].max()
        
        print(f"Max Predicted Eruptions: {max_eruptions:.1f}")
        print(f"Max Volcanic Earthquakes: {max_earthquakes:.1f}")
        print(f"Max SO2 Flux: {max_so2:.0f} tonnes/day")
        print(f"Max Plume Height: {max_plume:.0f} meters")
        
        # Warnings
        warnings = []
        if max_eruptions > 1:
            warnings.append("Multiple eruptions predicted")
        if max_earthquakes > 20:
            warnings.append("High seismic activity expected")
        if max_so2 > 2000:
            warnings.append("Elevated gas emissions")
        if max_plume > 1000:
            warnings.append("Significant ash plume possible")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"⚠️  {warning}")
        else:
            print("\n✅ No immediate warnings")

    def _plot_forecast(self, historical_df, forecast_df):
        """Plot historical data and forecast"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Taal Volcano 7-Day Activity Forecast', fontsize=16)
        
        # Plot key indicators
        indicators = [
            ('Eruption_Count', 'Eruption Count'),
            ('Volcanic_Earthquakes', 'Volcanic Earthquakes'),
            ('SO2_Flux_tpd', 'SO2 Flux (tonnes/day)'),
            ('Plume_Height_m', 'Plume Height (meters)')
        ]
        
        for idx, (col, title) in enumerate(indicators):
            ax = axes[idx // 2, idx % 2]
            
            # Historical data (last 60 days)
            hist_data = historical_df.tail(60)
            if col in self.scalers:
                hist_values = self.scalers[col].inverse_transform(
                    hist_data[col].values.reshape(-1, 1)
                ).flatten()
            else:
                hist_values = hist_data[col].values
            
            ax.plot(hist_data['Date'], hist_values, 'b-', label='Historical', alpha=0.7)
            ax.plot(forecast_df['Date'], forecast_df[col], 'r-', 
                   label='Forecast', linewidth=2, marker='o')
            
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('volcano_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_csv_data')
def get_csv_data():
    """Get CSV data for display on webpage"""
    try:
        csv_path = 'reversed_bulletin_forecast.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return jsonify({
                'success': True,
                'headers': df.columns.tolist(),
                'data': df.values.tolist()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'CSV file not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/bulk_scrape', methods=['POST'])
def bulk_scrape():
    """Bulk scrape Taal Volcano data from paginated URLs and save to CSV"""
    try:
        # Initialize CSV file with headers
        csv_filename = 'taal_volcano_bulletin_data.csv'
        csv_headers = ['Date', 'Alert_Level', 'Eruption', 'Seismicity', 'Acidity', 'Temperature',
                       'Sulfur_Dioxide_Flux', 'Plume', 'Ground_Deformation', 'Iframe_Source']
        
        # Check if CSV exists and get the latest date
        latest_date_in_csv = None
        csv_exists = os.path.exists(csv_filename)
        
        if csv_exists:
            try:
                with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header
                    dates = []
                    rows = list(reader)  # Read all rows into memory
                    
                    for row in rows:
                        if row and row[0] != '0':  # Skip empty or placeholder dates
                            try:
                                # Parse date in format "28 July 2025"
                                parsed_date = dt.strptime(row[0], '%d %B %Y')
                                dates.append(parsed_date)
                            except ValueError:
                                continue
                    
                    if dates:
                        latest_date_in_csv = max(dates)
                        print(f"Latest date in CSV: {latest_date_in_csv.strftime('%d %B %Y')}")
            except Exception as e:
                print(f"Error reading existing CSV: {str(e)}")
                latest_date_in_csv = None
        
        # Get current date
        current_date = dt.now().date()
        
        # Check if we need to scrape
        if latest_date_in_csv and latest_date_in_csv.date() >= current_date:
            return jsonify({
                'success': True,
                'message': f'Data is up to date. Latest date in CSV: {latest_date_in_csv.strftime("%d %B %Y")}',
                'csv_filename': csv_filename,
                'total_pages_processed': 0,
                'total_data_entries': 0
            })
        
        # Create or prepare CSV file if not exists
        if not csv_exists:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_headers)
        
        total_processed = 0
        total_data_entries = 0
        new_data_rows = []
        
        # Iterate through paginated URLs
        for n in range(0, 3411, 10):  # n=0 to n=3410, increment by 10
            base_url = f"https://www.phivolcs.dost.gov.ph/index.php/volcano-hazard/volcano-bulletin2/taal-volcano?start={n}"
            
            try:
                print(f"Processing page: {base_url}")
                
                # Fetch the page
                response = requests.get(base_url, verify=False, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                base_domain = '/'.join(base_url.split('/')[:3])
                
                # Find all Taal Volcano bulletin links
                bulletin_links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    text = a_tag.get_text(strip=True)
                    
                    # Check for Taal Volcano Summary pattern
                    normalized_text = text.lower()
                    if all(term in normalized_text for term in ['taal', 'volcano', 'summary', '24hr', 'observation']):
                        # Convert to absolute URL
                        if href.startswith('http'):
                            full_url = href
                        else:
                            full_url = base_domain + ('/' if not href.startswith('/') else '') + href
                        
                        bulletin_links.append({
                            'url': full_url,
                            'text': text
                        })
                
                print(f"Found {len(bulletin_links)} bulletin links on page {n}")
                
                # Process each bulletin link with deep scraping
                for link in bulletin_links:
                    try:
                        print(f"Deep scraping: {link['url']}")
                        
                        # Extract date from the bulletin text or URL
                        date_extracted = extract_date_from_text(link['text']) or extract_date_from_url(link['url'])
                        
                        # Skip if we already have this date or if it's older than our latest date
                        if date_extracted and latest_date_in_csv:
                            try:
                                extracted_date_obj = dt.strptime(date_extracted, '%d %B %Y')
                                if extracted_date_obj <= latest_date_in_csv:
                                    print(f"Skipping {date_extracted} - already have newer or equal data")
                                    continue
                            except ValueError:
                                pass  # Continue processing if date parsing fails
                        
                        # Perform deep scraping on the bulletin link
                        link_response = requests.get(link['url'], verify=False, timeout=15)
                        link_response.raise_for_status()
                        
                        link_soup = BeautifulSoup(link_response.text, 'html.parser')
                        link_base_url = '/'.join(link['url'].split('/')[:3])
                        
                        # Find iframe content
                        iframe_src = None
                        iframe_data = None
                        
                        # Look for iframe
                        target_div = link_soup.find('div', class_='sppb-addon-content')
                        if not target_div:
                            for div in link_soup.find_all('div'):
                                if div.has_attr('class') and 'sppb-addon-content' in div.get('class'):
                                    target_div = div
                                    break
                        
                        iframe = None
                        if target_div:
                            iframe = target_div.find('iframe')
                        if not iframe:
                            iframe = link_soup.find('iframe')
                        
                        if iframe and iframe.has_attr('src'):
                            iframe_src = iframe['src']
                            if not iframe_src.startswith(('http://', 'https://')):
                                iframe_src = link_base_url + ('/' if not iframe_src.startswith('/') else '') + iframe_src
                            
                            try:
                                # Fetch iframe content
                                iframe_response = requests.get(iframe_src, verify=False, timeout=15)
                                iframe_response.raise_for_status()
                                iframe_soup = BeautifulSoup(iframe_response.text, 'html.parser')
                                
                                # NEW PARSING METHOD: Look for PARAMETERS section
                                volcanic_data = parse_unified_volcanic_data(iframe_soup)
                                
                                # Prepare CSV row data
                                csv_row = [
                                    date_extracted or '0',
                                    volcanic_data.get('Alert_Level') or '0',
                                    volcanic_data.get('Eruption') or '0',
                                    volcanic_data.get('Seismicity') or '0',
                                    volcanic_data.get('Acidity') or '0',
                                    volcanic_data.get('Temperature') or '0',
                                    volcanic_data.get('Sulfur_Dioxide_Flux') or '0',
                                    volcanic_data.get('Plume') or '0',
                                    volcanic_data.get('Ground_Deformation') or '0',
                                    iframe_src or '0'
                                ]
                                
                                # Append new data row to the list
                                new_data_rows.append(csv_row)
                                
                                total_data_entries += 1
                                print(f"Data saved for: {date_extracted}")
                                
                            except Exception as e:
                                print(f"Error processing iframe {iframe_src}: {str(e)}")
                        
                        # Small delay between requests to be respectful
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error processing bulletin link {link['url']}: {str(e)}")
                        continue
                
                total_processed += 1
                
                # Delay between pages to be respectful to the server
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing page {base_url}: {str(e)}")
                continue
        
        # Read existing data from the CSV to avoid duplicates
        all_rows = []
        if csv_exists:
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                all_rows = list(reader)
        
        # Combine existing data with the new data
        all_rows.extend(new_data_rows)
        
        # Sort all rows by the date column (latest to earliest)
        all_rows.sort(key=lambda row: dt.strptime(row[0], '%d %B %Y'), reverse=True)
        
        # Rewrite the CSV file with updated data
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)  # Write headers
            writer.writerows(all_rows)  # Write sorted rows
        
        return jsonify({
            'success': True,
            'message': f'Bulk scraping completed. Processed {total_processed} pages, saved {total_data_entries} data entries.',
            'csv_filename': csv_filename,
            'total_pages_processed': total_processed,
            'total_data_entries': total_data_entries
        })
        
    except Exception as e:
        return jsonify({'error': f'Bulk scraping failed: {str(e)}'}), 500


def parse_parameters_table(soup):
    """Legacy wrapper for backward compatibility"""
    return parse_unified_volcanic_data(soup)

def parse_unified_volcanic_data(soup):
    """Unified method to parse both PARAMETERS section and Alert Level from row-one div
    
    Args:
        soup (BeautifulSoup): The parsed HTML content
        
    Returns:
        dict: Dictionary containing extracted volcanic parameters
    """
    result = {
        'Alert_Level': '0',
        'Eruption': '0',
        'Seismicity': '0',
        'Acidity': '0',
        'Temperature': '0',
        'Sulfur_Dioxide_Flux': '0',
        'Plume': '0',
        'Ground_Deformation': '0'
    }
    
    try:
        # Method 1: Try to extract from PARAMETERS section
        parameters_data = extract_from_parameters_section(soup)
        if parameters_data:
            result.update(parameters_data)
            print("Successfully extracted data from PARAMETERS section")
        
        # Method 2: Try to extract Alert Level from row-one div structure
        # Only if Alert Level wasn't found in PARAMETERS section
        if result['Alert_Level'] == '0':
            alert_data = extract_from_row_one_section(soup)
            if alert_data:
                result.update(alert_data)
                print("Successfully extracted Alert Level from row-one section")
        
        # Method 3: Fallback - search all tables for any missing parameters
        missing_params = [key for key, value in result.items() if value == '0']
        if missing_params:
            fallback_data = extract_from_all_tables(soup, missing_params)
            if fallback_data:
                result.update(fallback_data)
                print("Successfully extracted additional data from fallback method")
    
    except Exception as e:
        print(f"Error in unified volcanic data parsing: {str(e)}")
    
    return result

def extract_from_parameters_section(soup):
    """Extract data from PARAMETERS section using existing logic"""
    result = {}
    
    try:
        # Find the PARAMETERS section
        parameters_header = soup.find('p', class_='title1 bold', string='PARAMETERS')
        if not parameters_header:
            parameters_header = soup.find('p', string=re.compile(r'PARAMETERS', re.IGNORECASE))
        
        if parameters_header:
            table = find_table_after_element(parameters_header, soup)
            if table:
                extracted_data = parse_table_data(table)
                result.update(extracted_data)
    
    except Exception as e:
        print(f"Error extracting from PARAMETERS section: {str(e)}")
    
    return result

def extract_from_row_one_section(soup):
    """Extract Alert Level from row-one div structure using unified logic"""
    result = {}
    
    try:
        # Find the div with class="row-one"
        row_one_div = soup.find('div', class_='row-one')
        
        if row_one_div:
            print("Found row-one div")
            
            # Look for div with class="col-two" within or after row-one
            col_two_div = row_one_div.find('div', class_='col-two')
            
            # If not found within row-one, search more broadly
            if not col_two_div:
                parent = row_one_div.parent
                if parent:
                    col_two_div = parent.find('div', class_='col-two')
                if not col_two_div:
                    col_two_div = soup.find('div', class_='col-two')
            
            if col_two_div:
                print("Found col-two div")
                table = col_two_div.find('table')
                
                if table:
                    print("Found table in col-two")
                    # Use the same table parsing logic as PARAMETERS section
                    extracted_data = parse_table_data(table, focus_on_alert=True)
                    result.update(extracted_data)
    
    except Exception as e:
        print(f"Error extracting from row-one section: {str(e)}")
    
    return result

def find_table_after_element(element, soup):
    """Find table that comes after a specific element"""
    current_element = element
    table = None
    
    # Search for table in the following siblings or parent's siblings
    while current_element:
        if current_element.name == 'table':
            table = current_element
            break
        
        # Check if current element contains a table
        if hasattr(current_element, 'find') and current_element.find('table'):
            table = current_element.find('table')
            break
        
        # Move to next sibling
        current_element = current_element.find_next_sibling()
        if not current_element:
            # If no more siblings, try looking in the parent's next siblings
            parent = element.parent
            if parent:
                current_element = parent.find_next_sibling()
    
    return table

def parse_table_data(table, focus_on_alert=False):
    """Unified table parsing logic for both PARAMETERS and row-one sections"""
    result = {}
    
    try:
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                first_cell = cells[0]
                second_cell = cells[1]
                
                # Extract parameter name from first cell
                param_name = extract_parameter_name(first_cell)
                
                # Extract data from second cell using unified logic
                if param_name:
                    # Special handling for seismicity to capture description
                    if 'seismic' in param_name.lower():
                        data_value = extract_seismicity_with_description(second_cell)
                    else:
                        data_value = extract_cell_data(second_cell)
                    
                    # Map parameter name to result key
                    param_key, processed_value = map_parameter_to_key(param_name, data_value, focus_on_alert)
                    
                    # Store the extracted data
                    if param_key and processed_value:
                        result[param_key] = processed_value
                        print(f"Extracted {param_key}: {processed_value}")
    
    except Exception as e:
        print(f"Error parsing table data: {str(e)}")
    
    return result

def extract_parameter_name(cell):
    """Extract parameter name from table cell"""
    param_name = None
    
    # Try to find parameter name in bold element
    param_element = cell.find('b')
    if param_element:
        param_name = param_element.get_text(strip=True)
    else:
        # Try to find parameter name in any text within the cell
        cell_text = cell.get_text(strip=True)
        # Look for known parameter names
        for param in ['Alert Level', 'Eruption', 'Seismicity', 'Acidity', 'Temperature', 
                    'Sulfur Dioxide Flux', 'Plume', 'Ground Deformation']:
            if param.lower() in cell_text.lower():
                param_name = param
                break
    
    return param_name

def extract_cell_data(cell):
    """Extract data from table cell using unified logic"""
    # First try to find data in specific class element
    data_element = cell.find('p', class_='bold txtleft newfont')
    if data_element:
        return data_element.get_text(strip=True)
    
    # Try to find data in txt-no-eq class (for seismicity)
    txt_no_eq_element = cell.find('p', class_='txt-no-eq bold newfont')
    if txt_no_eq_element:
        return txt_no_eq_element.get_text(strip=True)
    
    # Try to get any text from the cell
    return cell.get_text(strip=True)

def extract_seismicity_with_description(cell):
    """Extract seismicity data with description from span tag"""
    # Look for the specific seismicity structure
    txt_no_eq_element = cell.find('p', class_='txt-no-eq bold newfont')
    if txt_no_eq_element:
        # Extract the numeric value (text before the span)
        full_text = txt_no_eq_element.get_text(strip=True)
        
        # Look for span with class containing 'txt-vq'
        span_element = txt_no_eq_element.find('span', class_=re.compile(r'txt-vq'))
        if span_element:
            # Get the description from the span
            description = span_element.get_text(strip=True)
            
            # Extract numeric value by removing the span text from full text
            numeric_part = full_text.replace(description, '').strip()
            
            # Ensure proper formatting: "numeric_value description"
            if numeric_part and description:
                return f"{numeric_part} {description}"
            elif numeric_part:
                return numeric_part
            else:
                return description
        else:
            # Fallback: try to extract number and remaining text with proper regex
            # Look for pattern: number + optional whitespace + text
            match = re.match(r'(\d+)\s*(.+)', full_text)
            if match:
                numeric_value = match.group(1)
                description = match.group(2).strip()
                # Format as requested: "numeric_value description"
                return f"{numeric_value} {description}"
            else:
                # If no number found, try to find just the number at the beginning
                number_match = re.match(r'(\d+)', full_text)
                if number_match:
                    numeric_value = number_match.group(1)
                    remaining_text = full_text[len(numeric_value):].strip()
                    if remaining_text:
                        return f"{numeric_value} {remaining_text}"
                    else:
                        return f"{numeric_value} volcanic earthquakes"  # Default description
    
    # Fallback to regular extraction with proper formatting
    cell_text = cell.get_text(strip=True)
    # Try to extract number and description from any cell text
    match = re.match(r'(\d+)\s*(.+)', cell_text)
    if match:
        numeric_value = match.group(1)
        description = match.group(2).strip()
        return f"{numeric_value} {description}"
    
    return cell_text

def map_parameter_to_key(param_name, data_value, focus_on_alert=False):
    """Map parameter name to result key with unified logic"""
    param_key = None
    param_lower = param_name.lower()
    
    if 'alert' in param_lower and 'level' in param_lower:
        param_key = 'Alert_Level'
        # Extract numeric value from alert level
        alert_match = re.search(r'(\d+)', data_value)
        if alert_match:
            return param_key, alert_match.group(1)
    elif 'eruption' in param_lower:
        param_key = 'Eruption'
    elif 'seismic' in param_lower:
        param_key = 'Seismicity'
        # For seismicity, return the full formatted string (number + description)
        return param_key, data_value
    elif 'acid' in param_lower:
        param_key = 'Acidity'
    elif 'temperature' in param_lower:
        param_key = 'Temperature'
    elif 'sulfur' in param_lower or 'dioxide' in param_lower:
        param_key = 'Sulfur_Dioxide_Flux'
        # Return the full text content including description and date
        # Instead of extracting just the numeric value, return the complete data
        return param_key, data_value
    elif 'plume' in param_lower:
        param_key = 'Plume'
    elif 'ground' in param_lower and 'deformation' in param_lower:
        param_key = 'Ground_Deformation'
    
    return param_key, data_value if param_key else (None, None)

def extract_from_all_tables(soup, missing_params):
    """Fallback method to search all tables for missing parameters"""
    result = {}
    
    try:
        tables = soup.find_all('table')
        for table in tables:
            # Check if this table contains parameter-related content
            table_text = table.get_text().lower()
            if any(param.lower().replace('_', ' ') in table_text for param in missing_params):
                extracted_data = parse_table_data(table)
                # Only update missing parameters
                for key, value in extracted_data.items():
                    if key in missing_params and value != '0':
                        result[key] = value
    
    except Exception as e:
        print(f"Error in fallback extraction: {str(e)}")
    
    return result

def extract_date_from_text(text):
    """Extract date from bulletin text"""
    if not text:
        return None
    
    # Look for date patterns in the text
    date_patterns = [
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None

def extract_date_from_url(url):
    """Extract date from URL if present"""
    if not url:
        return None
    
    # Look for date patterns in URL
    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', url)
    if date_match:
        return date_match.group(0)
    
    return None

def extract_alert_level_from_row_one(soup):
    """Legacy wrapper for backward compatibility"""
    return extract_from_row_one_section(soup)

@app.route('/process_data', methods=['POST'])
def process_data():
    """Process raw volcano data using dm_v0.5.py logic"""
    try:
        # Check if input file exists
        input_file = 'taal_volcano_bulletin_data.csv'
        if not os.path.exists(input_file):
            return jsonify({
                'success': False,
                'error': 'Input CSV file not found. Please run bulk scraping first.'
            })
        
        # Load and process data
        df = pd.read_csv(input_file)
        
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d %B %Y')
        
        # Process eruption data
        def parse_eruption_info(text):
            text = str(text).lower()
            result = {
                "Eruption_Count": 0,
                #"Eruption_Type": None,
                "Eruption_Severity_Score": 0,
                "Total_Eruption_Duration_Min": 0,
                "Avg_Eruption_Duration_Min": 0,
            }

            # FIXED: More precise eruption counting - avoid capturing unrelated "events"
            # Look specifically for eruption-related patterns
            eruption_patterns = [
                r'(\d+)\s+(?:phreatomagmatic|phreatic|magmatic)?\s*eruptions?',
                r'(\d+)\s+(?:minor|major|small)?\s*(?:phreatomagmatic|phreatic|magmatic)\s+eruptions?',
                r'(\d+)\s+eruption\s+events?'  # Only capture "eruption event", not just "event"
            ]
            
            count_matches = []
            for pattern in eruption_patterns:
                matches = re.findall(pattern, text)
                count_matches.extend(matches)
            
            result["Eruption_Count"] = sum(map(int, count_matches)) if count_matches else 0

            # FIXED: Enhanced type determination with better severity scoring
            if "phreatomagmatic" in text:
                #result["Eruption_Type"] = "phreatomagmatic"
                result["Eruption_Severity_Score"] = 3  # Higher severity for phreatomagmatic
            elif "phreatic" in text:
                #result["Eruption_Type"] = "phreatic"
                result["Eruption_Severity_Score"] = 2  # Lower severity for phreatic
            elif result["Eruption_Count"] > 0:
                #result["Eruption_Type"] = "other"
                result["Eruption_Severity_Score"] = 1

            # FIXED: Adjust severity based on descriptive keywords
            if "minor" in text or "small" in text:
                result["Eruption_Severity_Score"] = max(0, result["Eruption_Severity_Score"] - 0.5)
            elif "major" in text or "large" in text:
                result["Eruption_Severity_Score"] += 1

            # FIXED: Better duration extraction to avoid overlaps
            total_duration = 0
            duration_instances = 0
            
            # Create a copy of text to modify as we process
            text_for_duration = text

            # 1. Handle ranges like "2–5 minutes" first and remove them from text
            range_durations = re.findall(r'(\d+)\s*[-–]\s*(\d+)\s*minutes?', text_for_duration)
            for low, high in range_durations:
                avg = (int(low) + int(high)) / 2
                total_duration += avg
                duration_instances += 1
            
            # Remove processed ranges from text
            text_for_duration = re.sub(r'\d+\s*[-–]\s*\d+\s*minutes?', '', text_for_duration)

            # 2. Handle exact minute durations from remaining text
            minute_durations = re.findall(r'(\d+)\s*minutes?', text_for_duration)
            total_duration += sum(map(int, minute_durations))
            duration_instances += len(minute_durations)

            # 3. Handle seconds (convert to minutes) - process original text
            second_matches = re.findall(r'(\d+)\s*seconds?', text)
            total_seconds = sum(map(int, second_matches))
            total_duration += total_seconds / 60
            if second_matches:
                duration_instances += 1

            # Final results
            result["Total_Eruption_Duration_Min"] = round(total_duration, 2)
            result["Avg_Eruption_Duration_Min"] = round(total_duration / duration_instances, 2) if duration_instances else 0

            return pd.Series(result)


        # --- FIXED: Seismicity: Count total earthquakes + tremors ---
        def parse_seismicity_advanced(text):
            text = str(text).lower()
            # Initialize defaults
                
            result = {
                "Volcanic_Earthquakes": 0,
                "Volcanic_Tremors": 0,
                "Total_Tremor_Duration_Min": 0,
                "Has_Long_Tremor": 0,
                "Has_Weak_Tremor": 0,
            }
            
            # Parse volcanic earthquakes
            match_eq = re.search(r'(\d+)\s+volcanic earthquakes?', text)
            if match_eq:
                result["Volcanic_Earthquakes"] = int(match_eq.group(1))

            # Parse volcanic tremors
            match_tremor = re.search(r'(\d+)\s+volcanic tremors?', text)
            if match_tremor:
                result["Volcanic_Tremors"] = int(match_tremor.group(1))

            # FIXED: Avoid double-counting durations
            text_for_duration = text
                
            # Extract duration ranges like "3-608 minutes long" first
            durations = re.findall(r'(\d+)\s*[-–]\s*(\d+)\s*minutes', text_for_duration)
            for low_str, high_str in durations:
                low, high = int(low_str), int(high_str)
                avg = (low + high) / 2
                result["Total_Tremor_Duration_Min"] += avg
                if high > 60:
                    result["Has_Long_Tremor"] = 1
    
            # Remove processed ranges
            text_for_duration = re.sub(r'\d+\s*[-–]\s*\d+\s*minutes', '', text_for_duration)

            # Extract individual tremor durations from remaining text
            solo_durations = re.findall(r'(\d+)\s*minutes', text_for_duration)
            for d in solo_durations:
                minutes = int(d)
                result["Total_Tremor_Duration_Min"] += minutes
                if minutes > 60:
                    result["Has_Long_Tremor"] = 1

            # Weak tremor check
            if 'weak volcanic tremor' in text:
                result["Has_Weak_Tremor"] = 1

            return pd.Series(result)

        # --- Acidity: Extract numeric acidity (pH) ---
        def parse_acidity(val):
            if pd.isna(val) or str(val).strip() == '0':
                return None
            try:
                return float(re.search(r"[\d.]+", str(val)).group())
            except:
                return None

        df["Acidity_pH"] = df["Acidity"].apply(parse_acidity)
        
        # --- Temperature: Extract value in Celsius ---
        def parse_temperature(val):
            if pd.isna(val) or str(val).strip() == '0':
                return None
            try:
                return float(re.search(r"([\d.]+)\s*℃", str(val)).group(1))
            except:
                return None
        
        df["Crater_Temperature_C"] = df["Temperature"].apply(parse_temperature)

        # --- FIXED: Sulfur Dioxide Flux (SO2) ---
        def parse_so2(val):
            if pd.isna(val) or 'below detection limit' in str(val).lower():
                return 0
            try:
                # FIXED: Handle commas properly
                val_str = str(val)
                match = re.search(r"([\d,]+)\s*tonnes\s*/\s*day", val_str)
                if match:
                    return int(match.group(1).replace(",", ""))
                return None
            except:
                return None
        
        df["SO2_Flux_tpd"] = df["Sulfur_Dioxide_Flux"].apply(parse_so2)

        # --- Plume: Estimate plume height and strength ---
        def parse_plume_height(val):
            if pd.isna(val) or str(val).strip() == '0':
                return 0
            try:
                match = re.search(r"([\d,]+)\s*meters", str(val).replace(",", ""))
                return int(match.group(1)) if match else 0
            except:
                return 0

        # --- Plume: Estimate plume drift direction ---
        def parse_plume_drift(val):
            val = str(val).lower()
            directions = ["north", "northeast", "northwest", "east", "southeast", "south", "southwest", "west"]
            for d in directions:
                if d in val:
                    return d
            return "none"

        # --- Plume: Estimate plume strength ---
        def parse_plume_strength(val):
            val = str(val).lower()
            if "voluminous" in val:
                return 3
            elif "moderate" in val:
                return 2
            elif "weak" in val:
                return 1
            return 0
        
        df["Plume_Height_m"] = df["Plume"].apply(parse_plume_height)
        df["Plume_Drift_Direction"] = df["Plume"].apply(parse_plume_drift)
        df["Plume_Strength"] = df["Plume"].apply(parse_plume_strength)

        # --- FIXED: Ground Deformation: Advanced NLP-style Parsing ---
        def analyze_ground_deformation(text):
            text = str(text).lower()
            result = {
                'Caldera_Trend': 0,
                'TVI_Trend': 0,
                'North_Trend': 0,
                'SE_Trend': 0,
                'LT_Inflation': 0,
                'LT_Deflation': 0,
                'ST_Inflation': 0,
                'ST_Deflation': 0
            }

            # FIXED: Better logic to handle both inflation and deflation
            # Check for specific patterns rather than overwriting
            # Caldera trends
            if 'caldera' in text:
                if 'long-term deflation of the taal caldera' in text or 'caldera deflation' in text:
                    result['Caldera_Trend'] = -1
                elif 'long-term inflation of the taal caldera' in text or 'caldera inflation' in text:
                    result['Caldera_Trend'] = 1
                elif 'deflation' in text and 'caldera' in text:
                    result['Caldera_Trend'] = -1
                elif 'inflation' in text and 'caldera' in text:
                    result['Caldera_Trend'] = 1

            # TVI trends
            if 'tvi' in text or 'taal volcano island' in text:
                if 'deflation' in text:
                    result['TVI_Trend'] = -1
                elif 'inflation' in text:
                    result['TVI_Trend'] = 1
            
            # Regional trends
            if 'northern flank' in text or ('north' in text and 'flank' in text):
                if 'inflation' in text:
                    result['North_Trend'] = 1
                elif 'deflation' in text:
                    result['North_Trend'] = -1
            # Southeast flank
            if 'southeastern flank' in text or ('southeastern' in text and 'flank' in text):
                if 'inflation' in text:
                    result['SE_Trend'] = 1
                elif 'deflation' in text:
                    result['SE_Trend'] = -1

            # Temporal trends
            if 'long-term inflation' in text:
                result['LT_Inflation'] = 1
            if 'long-term deflation' in text:
                result['LT_Deflation'] = 1
            if 'short-term inflation' in text:
                result['ST_Inflation'] = 1
            if 'short-term deflation' in text:
                result['ST_Deflation'] = 1

            return pd.Series(result)


        # --- Apply eruption analysis ---
        eruption_features = df["Eruption"].apply(parse_eruption_info)
        df = pd.concat([df, eruption_features], axis=1)

        # --- Apply seismicity analysis ---
        seismicity_features = df["Seismicity"].apply(parse_seismicity_advanced)
        df = pd.concat([df, seismicity_features], axis=1)

        # Apply deformation analysis
        deformation_features = df["Ground_Deformation"].apply(analyze_ground_deformation)
        df = pd.concat([df, deformation_features], axis=1)

        # --- Drop original non-numeric columns not needed for modeling ---
        cols_to_drop = [
            "Eruption",
            "Seismicity",
            "Acidity",
            "Temperature",
            "Sulfur_Dioxide_Flux",
            "Plume",
            "Ground_Deformation",
            "Iframe_Source"
        ]

        df_cleaned = df.drop(columns=cols_to_drop)

        # --- Save the cleaned dataset ---
        df_cleaned.to_csv('taal_cleaned_forecast_ready.csv', index=False)
        
    except Exception as e:
        return jsonify({'error': f'Data processing failed: {str(e)}'}), 500

@app.route('/train_forecast', methods=['POST'])
def train_forecast():
    """Train LSTM model and generate forecast"""
    try:
        # Check if cleaned data exists
        input_file = 'taal_cleaned_forecast_ready.csv'
        if not os.path.exists(input_file):
            return jsonify({
                'success': False,
                'error': 'Cleaned data file not found. Please run data processing first.'
            })
        
        # Simple LSTM training simulation (simplified for web demo)
        df = pd.read_csv(input_file)
        

        # Run training and forecasting
        forecaster = VolcanoLSTMForecaster(sequence_length=30, forecast_days=7)
        history = forecaster.train(input_file, epochs=50, batch_size=16)
        forecast_df = forecaster.predict_next_7_days(input_file)
        forecast_df.to_csv('volcano_7day_forecast.csv', index=False)
        print("\nForecast saved to 'volcano_7day_forecast.csv'")

        return jsonify({
            'success': True,
            'message': 'LSTM model training completed and 7-day forecast generated.',
            'forecast_file': 'volcano_7day_forecast.csv',
            'forecast_days': 7
        })
        
    except Exception as e:
        return jsonify({'error': f'LSTM training failed: {str(e)}'}), 500

@app.route('/reverse_forecast', methods=['POST'])
def reverse_forecast():
    """Reverse forecast data back to bulletin format"""
    try:
        # Check if forecast data exists
        forecast_file = 'volcano_7day_forecast.csv'
        if not os.path.exists(forecast_file):
            return jsonify({
                'success': False,
                'error': 'Forecast file not found. Please run LSTM training first.'
            })
        
        # Load forecast data
        df = pd.read_csv(forecast_file)
        
        # Initialize bulletin format DataFrame
        bulletin_df = pd.DataFrame()
        
        # Convert Date format
        df['Date'] = pd.to_datetime(df['Date'])
        bulletin_df['Date'] = df['Date'].dt.strftime('%d %B %Y')
        
        # Reverse Alert_Level
        bulletin_df['Alert_Level'] = df['Alert_Level'].round().astype(int)
        
        # Reverse Eruption data
        def reverse_eruption(row):
            count = max(0, round(row['Eruption_Count']))
            severity = row['Eruption_Severity_Score']
            
            if count == 0:
                return "0"
            
            if severity >= 2.5:
                eruption_type = "Phreatomagmatic"
            elif severity >= 1.5:
                eruption_type = "Phreatic"
            elif severity >= 0.5:
                eruption_type = "Minor Phreatic"
            else:
                eruption_type = "Phreatic"
            
            if count == 1:
                return f"{count} {eruption_type} Eruption event (2 minutes long)"
            else:
                return f"{count} {eruption_type} Eruption events"
        
        bulletin_df['Eruption'] = df.apply(reverse_eruption, axis=1)
        
        # Reverse Seismicity data
        def reverse_seismicity(row):
            earthquakes = max(0, round(row['Volcanic_Earthquakes']))
            tremors = max(0, round(row['Volcanic_Tremors']))
            
            if earthquakes == 0 and tremors == 0:
                return "0 volcanic earthquakes"
            
            parts = []
            if earthquakes > 0:
                if earthquakes == 1:
                    parts.append(f"{earthquakes} volcanic earthquake")
                else:
                    parts.append(f"{earthquakes} volcanic earthquakes")
            
            if tremors > 0:
                if tremors == 1:
                    parts.append(f"{tremors} volcanic tremor (2 minutes long)")
                else:
                    parts.append(f"{tremors} volcanic tremors (2-3 minutes long)")
            
            if earthquakes == 0 and tremors > 0:
                return " including ".join(parts)
            elif earthquakes > 0 and tremors > 0:
                return parts[0] + " including " + parts[1]
            else:
                return parts[0]
        
        bulletin_df['Seismicity'] = df.apply(reverse_seismicity, axis=1)
        
        # Add other fields with realistic values
        bulletin_df['Acidity'] = df['Alert_Level'].apply(lambda x: f"{0.2 + x * 0.1:.1f} (19 February 2025)")
        bulletin_df['Temperature'] = df['Alert_Level'].apply(lambda x: f"{68.0 + x * 2 + np.random.normal(0, 1.5):.1f} ℃ (15 April 2025)")
        bulletin_df['Sulfur_Dioxide_Flux'] = df['SO2_Flux_tpd'].apply(
            lambda x: f"{max(1, round(x)):,} tonnes / day (15 July 2025)" if x > 0 else "Below detection limit"
        )
        
        # Reverse Plume data
        def reverse_plume(row):
            height = max(0, round(row['Plume_Height_m']))
            strength = round(row['Plume_Strength'])
            
            parts = []
            
            if height > 0:
                parts.append(f"{height} meters tall")
            
            strength_map = {0: None, 1: "Weak emission", 2: "Moderate emission", 3: "Voluminous emission"}
            if strength in strength_map and strength_map[strength]:
                parts.append(strength_map[strength])
            
            directions = ["northeast", "north", "south", "east", "west", "southeast", "southwest", "northwest"]
            drift = np.random.choice(directions)
            if parts:
                parts.append(f"{drift} drift")
            
            return "; ".join(parts) if parts else "None observed"
        
        bulletin_df['Plume'] = df.apply(reverse_plume, axis=1)
        
        # Ground deformation
        bulletin_df['Ground_Deformation'] = "Long-term deflation of the Taal Caldera; short-term inflation of the southeastern flank of the Taal Volcano Island"
        
        # Save to CSV
        output_file = 'reversed_bulletin_forecast.csv'
        bulletin_df.to_csv(output_file, index=False)
        
        return jsonify({
            'success': True,
            'message': f'Reverse engineering completed. Generated {len(bulletin_df)} rows with {len(bulletin_df.columns)} columns.',
            'output_file': output_file,
            'rows': len(bulletin_df),
            'columns': len(bulletin_df.columns)
        })
        
    except Exception as e:
        return jsonify({'error': f'Reverse forecast failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
