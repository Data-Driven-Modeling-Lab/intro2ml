# Google Sheets Integration Setup

This guide helps you set up automatic synchronization between your Google Sheets schedule and the Jekyll website.

##  Overview

The website can automatically sync with your Google Sheets schedule using:
1. **Manual sync** - Run script locally when needed
2. **Automatic sync** - GitHub Actions runs daily to check for updates
3. **CSV export** - Simple export/import workflow

##  Quick Setup (5 minutes)

### Step 1: Google Cloud Console Setup

1. **Go to [Google Cloud Console](https://console.cloud.google.com/)**
2. **Create a new project** or select existing one
3. **Enable Google Sheets API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

### Step 2: Service Account Creation

1. **Go to "IAM & Admin" → "Service Accounts"**
2. **Click "Create Service Account"**
3. **Fill in details**:
   - Name: `schedule-sync`
   - Description: `Sync course schedule from Google Sheets`
4. **Click "Create and Continue"**
5. **Skip roles for now** (click "Continue" then "Done")

### Step 3: Generate Credentials

1. **Click on your new service account**
2. **Go to "Keys" tab**
3. **Click "Add Key" → "Create new key"**
4. **Select "JSON" format**
5. **Download the file** and rename it to `credentials.json`

### Step 4: Share Your Google Sheet

1. **Open your [course schedule Google Sheet](https://docs.google.com/spreadsheets/d/1VbGSo8XdFY2cKhBrxC-4qUvxBk2pVpnH-3kyFiJv9CI/edit?usp=sharing)**
2. **Click "Share" button**
3. **Add the service account email** (found in credentials.json as "client_email")
4. **Give "Viewer" permissions**

##  Local Setup

### Install Dependencies

```bash
pip install gspread oauth2client
```

### Place Credentials

```bash
# Move credentials.json to the _scripts directory
mv ~/Downloads/credentials.json website/_scripts/
```

### Run Sync Script

```bash
cd website/_scripts
python sync-schedule.py
```

##  Automatic GitHub Actions Setup

### Step 1: Add Credentials to GitHub Secrets

1. **Go to your GitHub repository**
2. **Settings → Secrets and variables → Actions**
3. **Click "New repository secret"**
4. **Name**: `GOOGLE_SHEETS_CREDENTIALS`
5. **Value**: Copy entire contents of `credentials.json` file
6. **Click "Add secret"**

### Step 2: Enable Actions (if needed)

1. **Go to "Actions" tab in your repository**
2. **Enable GitHub Actions** if prompted
3. **The workflow will run automatically**

### Step 3: Manual Trigger

1. **Go to Actions → "Sync Schedule from Google Sheets"**
2. **Click "Run workflow"**
3. **Check results in the workflow run**

##  Google Sheets Format

Your Google Sheet should have these columns:

| Date | Lecture | Topic | Description | Assignment | Materials |
|------|---------|-------|-------------|------------|-----------|
| 8/26/2025 | 1 | Introduction + Ethics | Course overview... | Pre-course Survey | slides.pdf |
| 8/28/2025 | 2 | Data, Models, Python | Practical foundations... | | notes.md |

### Required Columns:
- **Date**: Date in MM/DD/YYYY or YYYY-MM-DD format
- **Lecture**: Lecture number or identifier
- **Topic**: Main lecture title
- **Description**: Brief description
- **Assignment**: Assignment info (optional)
- **Materials**: Links or file names (optional)

##  Sync Options

### 1. Automatic Daily Sync
- Runs every day at 6 AM UTC
- No action required
- Changes are committed automatically

### 2. Manual GitHub Trigger
- Go to Actions → "Sync Schedule from Google Sheets"
- Click "Run workflow"
- Updates immediately

### 3. Local Manual Sync
```bash
cd website/_scripts
python sync-schedule.py
git add schedule.md
git commit -m "Update schedule from Google Sheets"
git push
```

### 4. Simple CSV Export (No API needed)
```bash
# 1. Export Google Sheet as CSV
# 2. Save as schedule_data.csv
# 3. Run:
python sync-schedule.py --csv
```

##  Troubleshooting

### Common Issues:

** "credentials.json not found"**
- Make sure file is in `_scripts/` directory
- Check file permissions

** "Permission denied" on Google Sheets**
- Verify service account email is shared on the sheet
- Check service account has "Viewer" permissions

** "API not enabled"**
- Enable Google Sheets API in Google Cloud Console
- Wait a few minutes for propagation

** GitHub Actions failing**
- Check `GOOGLE_SHEETS_CREDENTIALS` secret is set correctly
- Verify JSON format is valid

### Debug Commands:

```bash
# Test credentials
python -c "import json; json.load(open('credentials.json'))"

# Test API connection
python -c "
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
print(' Connection successful')
"
```

##  Security Notes

- **Never commit `credentials.json`** to version control
- Service account only needs "Viewer" access to the sheet
- Credentials are stored securely in GitHub Secrets
- Consider rotating credentials periodically

##  Advanced Configuration

### Custom Sync Intervals

Edit `.github/workflows/sync-schedule.yml`:

```yaml
on:
  schedule:
    # Every 6 hours
    - cron: '0 */6 * * *'
    # Weekly on Mondays at 9 AM
    - cron: '0 9 * * 1'
    # Before each class (Tues/Thurs at 7 AM)
    - cron: '0 7 * * 2,4'
```

### Multiple Sheets Support

Modify `sync-schedule.py` to handle multiple worksheets:

```python
# Access specific worksheet
worksheet = sheet.worksheet("Fall 2025")
# Or by index
worksheet = sheet.get_worksheet(1)
```

### Custom Formatting

The script can be extended to:
- Parse different date formats
- Handle multiple material links
- Generate custom HTML layouts
- Add validation and error handling

---

##  Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review GitHub Actions logs for error details
3. Test credentials manually using debug commands
4. Contact the instructor for assistance

**Happy syncing! **