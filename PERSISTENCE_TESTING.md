# Database Persistence Testing Guide

The dashboard now uses SQLite to persist job information across server restarts. This means you won't lose any progress during testing!

## What Was Added

### Backend Changes
- **SQLite Database**: Jobs are stored in `backend/jobs.db`
- **Database Schema**: Stores all job fields including artifacts and notes
- **CRUD Operations**: All job operations now use the database instead of in-memory storage
- **Auto-initialization**: Database tables are created automatically on server startup

### Frontend Changes
- **Persistence Info Banner**: Shows database status and job statistics at the top of the jobs page
- **Job Count**: Displays total number of jobs in the database
- **Oldest Job Timestamp**: Shows when the oldest job was created (helpful to verify old jobs persist)
- **Last Fetch Time**: Shows when the job list was last refreshed
- **Manual Refresh Button**: Click the refresh icon to reload jobs from the database
- **Empty State Message**: Improved messaging when no jobs exist

## How to Test Persistence

### Method 1: Automated Test Script

Run the provided test script:

```bash
cd backend
./test_persistence.sh
```

This script will:
1. Upload a test file and create a job
2. Wait for OCR processing
3. Restart the backend server
4. Verify the job still exists after restart

### Method 2: Manual Testing

1. **Start the backend** (if not already running):
   ```bash
   cd backend
   uvicorn backend_demo:app --reload
   ```

2. **Start the frontend** (if not already running):
   ```bash
   cd inspector-frontend
   npm run dev
   ```

3. **Upload a job**:
   - Go to http://localhost:3000/upload
   - Upload a .360 file
   - Note the job ID

4. **View the job**:
   - Go to http://localhost:3000/jobs
   - You should see the persistence info banner at the top
   - Your job should be listed in the table

5. **Restart the backend server**:
   ```bash
   # In the backend directory
   pkill -f uvicorn
   uvicorn backend_demo:app --reload
   ```

6. **Verify persistence**:
   - Refresh the jobs page in your browser (or wait 5 seconds for auto-refresh)
   - Your job should still be there!
   - The "Oldest" timestamp in the info banner should show your original job creation time

## Database Location

The SQLite database is stored at:
```
backend/jobs.db
```

You can inspect it directly using the `sqlite3` command:

```bash
cd backend
sqlite3 jobs.db "SELECT id, whiteboard_id, status, datetime(created_at, 'unixepoch') as created FROM jobs;"
```

## Testing Different Scenarios

### Test 1: Simple Upload and Restart
1. Upload one job
2. Restart server
3. Verify job persists

### Test 2: Multiple Jobs
1. Upload several jobs
2. Wait for OCR processing
3. Restart server
4. Verify all jobs persist

### Test 3: Full Processing Pipeline
1. Upload a job
2. Wait for OCR → PUSHED_TO_DRIVE status
3. Trigger agent processing: `curl -X POST http://127.0.0.1:8000/agent/process/{whiteboard_id}`
4. Wait for PROCESSED status with artifacts
5. Restart server
6. Verify job still shows PROCESSED status with all artifacts

### Test 4: Database Survives Multiple Restarts
1. Create some jobs
2. Restart server multiple times
3. Jobs should persist across all restarts

## Current Database Contents

Check current jobs:
```bash
curl http://127.0.0.1:8000/jobs | jq '.data | length'
```

View all jobs:
```bash
curl http://127.0.0.1:8000/jobs | jq '.data'
```

## Clear Database (for fresh testing)

If you want to start with an empty database:

```bash
cd backend
rm jobs.db
# Restart the server - it will create a new empty database
pkill -f uvicorn
uvicorn backend_demo:app --reload
```

## Troubleshooting

**Jobs not persisting?**
- Check if `backend/jobs.db` file exists
- Check backend logs for database errors
- Verify the server is starting successfully

**Frontend not showing persistence banner?**
- Make sure you have at least one job in the database
- Try refreshing the page
- Check browser console for errors

**Test script fails?**
- Make sure backend is running before running the script
- Check that port 8000 is not blocked
- Verify you have `jq` installed (`brew install jq` on macOS)
