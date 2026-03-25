'use client';

import { useState, useCallback, useEffect, useMemo, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { uploadModel, listModels } from '@/lib/api';
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  List,
  ListItemButton,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
  AlertTitle,
  Chip,
  Stack,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';

const ModelViewer = dynamic(() => import('@/components/ModelViewer'), {
  ssr: false,
  loading: () => (
    <Box
      sx={{
        width: '100%',
        height: 520,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: '#0f1117',
        borderRadius: 2,
      }}
    >
      <CircularProgress />
    </Box>
  ),
});

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

function ViewerContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [activeModelUrl, setActiveModelUrl] = useState(null);
  const [activeModelName, setActiveModelName] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [modelLoadError, setModelLoadError] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showAllModels, setShowAllModels] = useState(false);

  const jobId = searchParams.get('jobId');

  const { data: modelsData, refetch: refetchModels } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  });

  // Handle direct linking via query params
  useEffect(() => {
    const modelUrlParam = searchParams.get('model');
    const modelNameParam = searchParams.get('name');

    if (modelUrlParam) {
      const timestamp = new Date().getTime();
      const separator = modelUrlParam.includes('?') ? '&' : '?';
      const cacheBustedUrl = modelUrlParam.startsWith('http')
        ? `${modelUrlParam}${separator}t=${timestamp}`
        : `${BASE_URL}${modelUrlParam}${separator}t=${timestamp}`;

      setActiveModelUrl(cacheBustedUrl);
      setActiveModelName(modelNameParam || modelUrlParam.split('/').pop());
      setModelLoadError(false); // Reset error state on new model
    }
  }, [searchParams]);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      const ext = file.name.split('.').pop().toLowerCase();
      if (!['glb', 'gltf'].includes(ext)) {
        setUploadError('Only .glb and .gltf files are supported');
        return;
      }
      setUploading(true);
      setUploadError(null);
      try {
        const result = await uploadModel(file);
        setActiveModelUrl(`${BASE_URL}${result.url}`);
        setActiveModelName(result.original_name);
        await refetchModels();
      } catch (e) {
        setUploadError(e.message || 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [refetchModels],
  );

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const allModels = modelsData ?? [];
  
  // Filter models based on the current job if applicable
  const filteredModels = useMemo(() => {
    if (showAllModels || !jobId) return allModels;
    
    // In the demo, the model filename typically contains 
    // the whiteboard ID (which we use as name) or parts
    // of the jobId. For now, we'll filter  by the model 
    // currently loaded if it came from a job.
    return allModels.filter(m => {
      const modelName = searchParams.get('name');
      return modelName && m.filename.includes(modelName);
    });
  }, [allModels, showAllModels, jobId, searchParams]);

  return (
    <Container maxWidth="xl" sx={{ mt: 4, pb: 6 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 3 }}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <ViewInArIcon color="primary" fontSize="large" />
          <Typography variant="h4" fontWeight="bold">
            3D Model Viewer
          </Typography>
        </Stack>
        {jobId && (
          <Chip 
            label={`Job: ${jobId}`} 
            color="primary" 
            variant="outlined" 
            onDelete={() => router.push('/viewer')}
          />
        )}
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Upload .glb or .gltf models exported from Agisoft Metashape to view them here.
      </Typography>

      {/* Upload drop zone */}
      <Paper
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        sx={{
          p: 4,
          mb: 3,
          textAlign: 'center',
          border: '2px dashed',
          borderColor: isDragging ? 'primary.main' : 'divider',
          bgcolor: isDragging ? 'action.hover' : 'background.paper',
          transition: 'border-color 0.2s, background-color 0.2s',
        }}
      >
        <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
        <Typography variant="h6" gutterBottom>
          Drag & drop a .glb or .gltf file here
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Models generated by Agisoft Metashape — export as GLB before uploading
        </Typography>
        <Button
          variant="contained"
          component="label"
          disabled={uploading}
          startIcon={
            uploading ? <CircularProgress size={16} color="inherit" /> : <FolderOpenIcon />
          }
        >
          {uploading ? 'Uploading…' : 'Choose File'}
          <input type="file" hidden accept=".glb,.gltf" onChange={onFileChange} />
        </Button>
      </Paper>

      {uploadError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setUploadError(null)}>
          {uploadError}
        </Alert>
      )}

      {/* Viewer + sidebar */}
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
        {/* 3D canvas */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {activeModelUrl ? (
            <>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <ViewInArIcon color="primary" fontSize="small" />
                  <Typography variant="subtitle1" fontWeight="bold" noWrap>
                    {activeModelName}
                  </Typography>
                  <Chip label="Live" color="success" size="small" />
                </Stack>
                {modelLoadError && (
                  <Chip label="Loading Failed" color="error" size="small" variant="outlined" />
                )}
              </Stack>
              
              {modelLoadError ? (
                <Paper
                  sx={{
                    height: 520,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: '#1a1d24',
                    borderRadius: 2,
                    gap: 2,
                    border: '1px solid #3f4451'
                  }}
                >
                  <Alert severity="error" sx={{ width: '80%' }}>
                    <AlertTitle>Incompatible Model File</AlertTitle>
                    This file could not be parsed as a valid 3D model. This often happens if the file is empty or corrupted during processing.
                  </Alert>
                  <Typography variant="body2" sx={{ color: 'grey.500' }}>
                    Technical details: GLTF/GLB parse error
                  </Typography>
                </Paper>
              ) : (
                <>
                  <ModelViewer url={activeModelUrl} onError={() => setModelLoadError(true)} />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Left-drag to rotate &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Right-drag to pan
                  </Typography>
                </>
              )}
            </>
          ) : (
            <Paper
              sx={{
                height: 520,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: '#0f1117',
                borderRadius: 2,
                gap: 1,
              }}
            >
              <ViewInArIcon sx={{ fontSize: 72, color: '#374151' }} />
              <Typography color="#6b7280" variant="body1">
                No model loaded
              </Typography>
              <Typography color="#4b5563" variant="body2">
                Upload a .glb file above to get started
              </Typography>
            </Paper>
          )}
        </Box>

        {/* Previously uploaded models sidebar */}
        {(filteredModels.length > 0 || jobId) && (
          <Paper sx={{ width: 280, flexShrink: 0 }}>
            <Box sx={{ p: 2, pb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                {showAllModels || !jobId ? `All Models (${allModels.length})` : `Job Models (${filteredModels.length})`}
              </Typography>
              {jobId && (
                <Button 
                  size="small" 
                  onClick={() => setShowAllModels(!showAllModels)}
                  sx={{ fontSize: '0.65rem', py: 0 }}
                >
                  {showAllModels ? 'Show Job Only' : 'Show All'}
                </Button>
              )}
            </Box>
            <Divider />
            <List dense sx={{ maxHeight: 520, overflow: 'auto', p: 0 }}>
              {filteredModels.length > 0 ? (
                filteredModels.map((model) => (
                  <ListItemButton
                    key={model.filename}
                    selected={activeModelUrl?.includes(model.url)}
                    onClick={() => {
                      const timestamp = new Date().getTime();
                      setActiveModelUrl(`${BASE_URL}${model.url}?t=${timestamp}`);
                      setActiveModelName(model.filename);
                      setModelLoadError(false);
                    }}
                    sx={{ py: 1.5 }}
                  >
                    <ListItemText
                      primary={
                        <Typography noWrap sx={{ fontSize: '0.8rem' }}>
                          {model.filename}
                        </Typography>
                      }
                      secondary={
                        <Typography sx={{ fontSize: '0.72rem', color: 'text.secondary' }}>
                          {`${(model.size_bytes / 1024).toFixed(1)} KB · ${new Date(
                            model.uploaded_at * 1000,
                          ).toLocaleDateString()}`}
                        </Typography>
                      }
                    />
                  </ListItemButton>
                ))
              ) : (
                <Box sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    No models found for this job.
                  </Typography>
                </Box>
              )}
            </List>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default function ViewerPage() {
  return (
    <Suspense fallback={
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography sx={{ mt: 2 }}>Loading viewer...</Typography>
      </Box>
    }>
      <ViewerContent />
    </Suspense>
  );
}
