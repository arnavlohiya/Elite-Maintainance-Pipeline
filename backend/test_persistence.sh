#!/bin/bash

# Test script for database persistence
# This script helps verify that jobs persist across server restarts

set -e

echo "🧪 Testing SQLite Database Persistence"
echo "======================================"
echo ""

# Check if backend is running
if ! curl -s http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
    echo "❌ Backend is not running. Please start it first:"
    echo "   uvicorn backend_demo:app --reload"
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Create a test file
TEST_FILE="test_persistence_$(date +%s).360"
echo "test video content for persistence check" > "$TEST_FILE"
echo "📤 Uploading test file: $TEST_FILE"

# Upload the file
RESPONSE=$(curl -s -F "file=@$TEST_FILE" http://127.0.0.1:8000/upload)
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

echo "✅ Job created: $JOB_ID"
echo ""

# Wait for OCR to complete
echo "⏳ Waiting for OCR processing..."
sleep 3

# Check job status
JOB_DATA=$(curl -s http://127.0.0.1:8000/jobs/$JOB_ID)
STATUS=$(echo "$JOB_DATA" | jq -r '.status')
WHITEBOARD_ID=$(echo "$JOB_DATA" | jq -r '.whiteboard_id')

echo "✅ Job status: $STATUS"
echo "✅ Whiteboard ID: $WHITEBOARD_ID"
echo ""

# Count total jobs before restart
JOBS_BEFORE=$(curl -s http://127.0.0.1:8000/jobs | jq '.data | length')
echo "📊 Total jobs in database: $JOBS_BEFORE"
echo ""

echo "🔄 Now restarting the backend server..."
echo "   (This will test if jobs persist across restarts)"
echo ""

# Find and kill the uvicorn process
if pkill -f "uvicorn backend_demo:app"; then
    echo "✅ Stopped backend server"
    sleep 2
else
    echo "⚠️  Could not stop server (may not be running)"
fi

# Restart the server
echo "🚀 Starting backend server..."
cd "$(dirname "$0")"
uvicorn backend_demo:app --reload --host 127.0.0.1 --port 8000 > /dev/null 2>&1 &

# Wait for server to start
echo "⏳ Waiting for server to restart..."
sleep 4

# Verify server is running
if ! curl -s http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
    echo "❌ Failed to restart server"
    exit 1
fi

echo "✅ Backend server restarted"
echo ""

# Check if jobs still exist
JOBS_AFTER=$(curl -s http://127.0.0.1:8000/jobs | jq '.data | length')
JOB_STILL_EXISTS=$(curl -s http://127.0.0.1:8000/jobs/$JOB_ID | jq -r '.id' 2>/dev/null || echo "null")

echo "📊 Jobs after restart: $JOBS_AFTER"
echo ""

if [ "$JOB_STILL_EXISTS" = "$JOB_ID" ]; then
    echo "✅ SUCCESS! Job $JOB_ID persisted across server restart!"
    echo "✅ Database persistence is working correctly!"
else
    echo "❌ FAILED! Job $JOB_ID not found after restart"
    echo "❌ Database persistence may not be working"
    exit 1
fi

echo ""
echo "🎉 Persistence test completed successfully!"
echo ""
echo "You can view the job in the frontend at:"
echo "   http://localhost:3000/jobs"
echo ""

# Cleanup
rm -f "$TEST_FILE"
