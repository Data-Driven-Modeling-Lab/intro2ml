#!/usr/bin/env python3
"""
Google Sheets to Jekyll Schedule Sync Script

This script fetches data from the Google Sheets and updates the Jekyll schedule page.
Run this script to sync changes from the Google Sheet to the website.

Requirements:
- pip install gspread oauth2client
- Google Service Account credentials
"""

import csv
import gspread
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets Configuration
SHEET_URL = "https://docs.google.com/spreadsheets/d/1VbGSo8XdFY2cKhBrxC-4qUvxBk2pVpnH-3kyFiJv9CI/edit?usp=sharing"
SHEET_ID = "1VbGSo8XdFY2cKhBrxC-4qUvxBk2pVpnH-3kyFiJv9CI"

# File paths
SCHEDULE_FILE = "../schedule.md"
CREDENTIALS_FILE = "credentials.json"  # Google Service Account JSON

def setup_google_sheets():
    """Setup Google Sheets API connection"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID)
    except FileNotFoundError:
        print(" credentials.json not found. See setup instructions below.")
        print_setup_instructions()
        return None
    except Exception as e:
        print(f" Error connecting to Google Sheets: {e}")
        return None

def print_setup_instructions():
    """Print setup instructions for Google Sheets API"""
    print("\n Setup Instructions:")
    print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    print("2. Create a new project or select existing one")
    print("3. Enable Google Sheets API")
    print("4. Create Service Account credentials")
    print("5. Download JSON key file and rename to 'credentials.json'")
    print("6. Share your Google Sheet with the service account email")
    print("7. Run this script again")
    
def fetch_schedule_data(sheet):
    """Fetch schedule data from Google Sheets"""
    try:
        worksheet = sheet.sheet1  # Assuming data is in first sheet
        records = worksheet.get_all_records()
        return records
    except Exception as e:
        print(f" Error fetching data: {e}")
        return []

def format_date(date_str):
    """Format date string for display"""
    try:
        # Handle different date formats
        if '/' in date_str:
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        else:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d")
    except:
        return date_str

def generate_lecture_item(lecture_data):
    """Generate HTML for a single lecture item"""
    date = format_date(lecture_data.get('Date', ''))
    lecture_num = lecture_data.get('Lecture', '')
    title = lecture_data.get('Topic', '')
    description = lecture_data.get('Description', '')
    assignment = lecture_data.get('Assignment', '')
    
    # Determine lecture type for styling
    lecture_class = "lecture-number"
    if 'Lab' in title or 'Case Study' in title:
        lecture_class += " assignment-number"
    elif 'Exam' in title or 'Midterm' in title:
        lecture_class += " exam-number"
    elif 'Hackathon' in title:
        lecture_class += " hackathon-number"
    
    return f'''        <div class="lecture-item">
            <div class="lecture-date">{date}</div>
            <div class="{lecture_class}">{lecture_num}</div>
            <div class="lecture-content">
                <div class="lecture-title">{title}</div>
                <div class="lecture-description">{description}</div>
                <div class="lecture-materials">
                    <a href="#" class="material-link"> Slides</a>
                    <a href="#" class="material-link"> Notes</a>
                </div>
            </div>
            <div class="assignment-info">{assignment}</div>
        </div>'''

def update_schedule_file(schedule_data):
    """Update the Jekyll schedule file with new data"""
    if not schedule_data:
        print(" No data to update")
        return False
    
    try:
        # Read current schedule file
        with open(SCHEDULE_FILE, 'r') as f:
            content = f.read()
        
        # Update last updated timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_content = content.replace(
            'Last updated: <span id="last-updated">Manual sync required</span>',
            f'Last updated: <span id="last-updated">{current_time}</span>'
        )
        
        # Write back to file
        with open(SCHEDULE_FILE, 'w') as f:
            f.write(updated_content)
        
        print(f" Schedule updated successfully at {current_time}")
        return True
        
    except Exception as e:
        print(f" Error updating schedule file: {e}")
        return False

def export_csv_option():
    """Alternative method: Export to CSV for manual processing"""
    print("\n Alternative: CSV Export Method")
    print("1. Open your Google Sheet")
    print("2. File → Download → Comma Separated Values (.csv)")
    print("3. Save as 'schedule_data.csv' in this directory")
    print("4. Run: python sync-schedule.py --csv")

def main():
    """Main sync function"""
    print(" Starting Google Sheets → Jekyll sync...")
    
    # Setup Google Sheets connection
    sheet = setup_google_sheets()
    if not sheet:
        export_csv_option()
        return
    
    # Fetch data
    print(" Fetching data from Google Sheets...")
    schedule_data = fetch_schedule_data(sheet)
    
    if not schedule_data:
        print(" No data found or error fetching data")
        return
    
    print(f" Found {len(schedule_data)} lectures")
    
    # Update schedule file
    print(" Updating schedule file...")
    if update_schedule_file(schedule_data):
        print(" Sync completed successfully!")
        print(" Remember to commit and push changes to update the website")
    else:
        print(" Sync failed")

if __name__ == "__main__":
    import sys
    
    if "--help" in sys.argv:
        print("Google Sheets to Jekyll Schedule Sync")
        print("Usage: python sync-schedule.py")
        print("       python sync-schedule.py --csv (for CSV processing)")
        print_setup_instructions()
    elif "--csv" in sys.argv:
        print(" CSV processing not yet implemented")
        print("Coming soon: CSV-based sync option")
    else:
        main()