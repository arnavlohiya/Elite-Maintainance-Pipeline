// frontend/lib/api.js
const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export async function upload360(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData
  });
  
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${res.status} ${text}`);
  }
  
  return res.json();
}

export async function listJobs() {
  const res = await fetch(`${BASE_URL}/jobs`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch jobs');
  const data = await res.json();
  return data.data;
}

export async function getJob(id) {
  const res = await fetch(`${BASE_URL}/jobs/${id}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Job ${id} not found`);
  return res.json();
}

export async function triggerAgent(whiteboardId) {
  const res = await fetch(`${BASE_URL}/agent/process/${whiteboardId}`, {
    method: 'POST'
  });
  
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Agent failed: ${text}`);
  }
  
  return res.json();
}
