'use client';

import React, { useState, useRef } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Button,
  Typography,
  Stack,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
} from '@mui/material';

export default function UploadCard() {
  const [uploads, setUploads] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const onFileChange = (e) => {
    const selected = Array.from(e.target.files || []).map((file, index) => ({
      key: `${file.name}-${file.lastModified}-${index}`,
      file,
      name: file.name,
      size: file.size,
      status: 'pending',
      progress: 0,
      jobId: null,
      message: null,
    }));
    setUploads(selected);
    setError(null);
  };

  const updateUpload = (key, updates) => {
    setUploads((current) =>
      current.map((item) =>
        item.key === key ? { ...item, ...updates } : item
      )
    );
  };

  const uploadSingleFile = (item) => {
    return new Promise((resolve) => {
      const formData = new FormData();
      formData.append('file', item.file);

      const xhr = new XMLHttpRequest();
      const baseUrl =
        process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
      xhr.open('POST', `${baseUrl}/upload`);

      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const pct = Math.round((event.loaded / event.total) * 100);
        updateUpload(item.key, { progress: pct });
      };

      xhr.onload = () => {
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            const response = JSON.parse(xhr.responseText);
            updateUpload(item.key, {
              status: 'success',
              progress: 100,
              jobId: response.job_id,
            });
          } else {
            updateUpload(item.key, {
              status: 'error',
              message: `Upload failed: ${xhr.status} ${xhr.statusText}`,
            });
          }
        } catch (err) {
          updateUpload(item.key, {
            status: 'error',
            message: 'Upload failed: invalid response from server',
          });
        }
        resolve();
      };

      xhr.onerror = () => {
        updateUpload(item.key, {
          status: 'error',
          message: 'Network error during upload.',
        });
        resolve();
      };

      xhr.send(formData);
    });
  };

  const onUpload = async () => {
    if (!uploads.length) return;

    const invalid = uploads.find(
      (item) => !item.name.toLowerCase().endsWith('.360')
    );
    if (invalid) {
      setError('All files must have the .360 extension.');
      return;
    }

    setIsUploading(true);
    setError(null);
    const queueSnapshot = [...uploads];

    for (const item of queueSnapshot) {
      updateUpload(item.key, { status: 'uploading', progress: 0 });
      await uploadSingleFile(item);
    }

    setIsUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Upload .360 videos to Drive
        </Typography>
        <Typography variant="body2" paragraph>
          Select one or more .360 files to send to the Drive inbox. Admins will
          take care of processing after the uploads finish.
        </Typography>

        <Stack direction="column" spacing={2}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".360"
            multiple
            onChange={onFileChange}
            style={{ display: 'none' }}
          />
          <Button
            variant="outlined"
            onClick={() => fileInputRef.current?.click()}
            sx={{ alignSelf: 'flex-start' }}
          >
            Choose files
          </Button>

          {uploads.length > 0 && (
            <List dense>
              {uploads.map((item) => (
                <ListItem key={item.key}>
                  <Stack sx={{ width: '100%' }}>
                    <ListItemText
                      primary={item.name}
                      secondary={`${(item.size / (1024 * 1024)).toFixed(1)} MB`}
                    />
                    {item.status === 'uploading' && (
                      <LinearProgress
                        variant="determinate"
                        value={item.progress}
                        sx={{ mt: 1 }}
                      />
                    )}
                    {item.status === 'success' && (
                      <Typography
                        variant="body2"
                        sx={{ mt: 1 }}
                        color="success.main"
                      >
                        Uploaded (job id: {item.jobId})
                      </Typography>
                    )}
                    {item.status === 'error' && (
                      <Typography
                        variant="body2"
                        sx={{ mt: 1 }}
                        color="error"
                      >
                        {item.message}
                      </Typography>
                    )}
                  </Stack>
                </ListItem>
              ))}
            </List>
          )}

          {error && (
            <Typography color="error" variant="body2">
              {error}
            </Typography>
          )}

        </Stack>
      </CardContent>

      <CardActions>
        <Button
          onClick={onUpload}
          disabled={uploads.length === 0 || isUploading}
          variant="contained"
        >
          {isUploading ? 'Uploading…' : 'Upload to Drive'}
        </Button>
      </CardActions>
    </Card>
  );
}
