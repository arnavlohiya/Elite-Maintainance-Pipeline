'use client';

import React, { useState, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getJob, triggerAgent } from '@/lib/api';
import {
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  Stack,
  Chip,
  Stepper,
  Step,
  StepLabel,
  Divider,
  Link as MuiLink,
} from '@mui/material';
import StatusPill from '@/components/StatusPill';

const PIPELINE_STEPS = [
  'UPLOADED',
  'OCR_OK',          // or NEEDS_REVIEW
  'PUSHED_TO_DRIVE',
  'PROCESSING',
  'PROCESSED',
];

function stepIndexForStatus(status) {
  const idx = PIPELINE_STEPS.indexOf(status);
  if (idx >= 0) return idx;
  if (status === 'NEEDS_REVIEW') return 1;
  return 0;
}

export default function JobDetail({ jobId }) {
  const queryClient = useQueryClient();
  const [agentError, setAgentError] = useState(null);

  const { data: job, isLoading, error } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: 3000,
  });

  const activeStep = useMemo(
    () => (job ? stepIndexForStatus(job.status) : 0),
    [job],
  );

  if (isLoading) return <Typography>Loading…</Typography>;
  if (error || !job) return <Typography color="error">Job not found</Typography>;

  const canTriggerAgent = job.whiteboard_id && job.status !== 'PROCESSED';

  const onSimulateProcess = async () => {
    if (!job.whiteboard_id) return;
    setAgentError(null);
    try {
      await triggerAgent(job.whiteboard_id);
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    } catch (e) {
      setAgentError(e.message || 'Agent failed');
    }
  };

  const modelArtifacts = job.artifacts.filter((a) => a.kind === 'MODEL');
  const reportArtifacts = job.artifacts.filter((a) => a.kind === 'REPORT');
  const logArtifacts = job.artifacts.filter((a) => a.kind === 'LOG');

  return (
    <Stack spacing={2}>
      {/* Summary card */}
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Job {job.id}
          </Typography>

          <Stack direction="row" spacing={1} sx={{ mb: 1 }} alignItems="center">
            <StatusPill status={job.status} />
            {job.whiteboard_id && (
              <Chip label={`WB: ${job.whiteboard_id}`} size="small" />
            )}
          </Stack>

          <Typography variant="body2">
            Original file: <strong>{job.original_name}</strong>
          </Typography>
          <Typography variant="body2">
            Created: {new Date(job.created_at * 1000).toLocaleString()}
          </Typography>
          <Typography variant="body2">
            Updated: {new Date(job.updated_at * 1000).toLocaleString()}
          </Typography>
          {job.ocr_confidence != null && (
            <Typography variant="body2">
              OCR confidence: {job.ocr_confidence.toFixed(2)}
            </Typography>
          )}

          {canTriggerAgent && (
            <Button
              sx={{ mt: 2 }}
              variant="outlined"
              onClick={onSimulateProcess}
            >
              Simulate processing (dev)
            </Button>
          )}
          {agentError && (
            <Typography color="error" variant="body2" sx={{ mt: 1 }}>
              {agentError}
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Timeline / pipeline */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Pipeline status
          </Typography>
          <Stepper activeStep={activeStep} alternativeLabel>
            {PIPELINE_STEPS.map((step) => (
              <Step key={step}>
                <StepLabel>{step}</StepLabel>
              </Step>
            ))}
          </Stepper>
          <Typography variant="body2" sx={{ mt: 1 }}>
            This timeline shows where the job is in the inspector → backend → drive
            → agent → model pipeline.
          </Typography>
        </CardContent>
      </Card>

      {/* Artifacts */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Artifacts
          </Typography>

          {job.artifacts.length === 0 && (
            <Typography variant="body2">No artifacts yet.</Typography>
          )}

          {modelArtifacts.length > 0 && (
            <>
              <Typography variant="subtitle1">3D model</Typography>
              <List dense>
                {modelArtifacts.map((a, i) => (
                  <ListItem key={`model-${i}`}>
                    <ListItemText
                      primary={a.path}
                      secondary="GLB model produced by the agent. For now, open it with a local GLB viewer."
                    />
                  </ListItem>
                ))}
              </List>
              <Divider sx={{ my: 1 }} />
            </>
          )}

          {reportArtifacts.length > 0 && (
            <>
              <Typography variant="subtitle1">Report</Typography>
              <List dense>
                {reportArtifacts.map((a, i) => (
                  <ListItem key={`report-${i}`}>
                    <ListItemText
                      primary={a.path}
                      secondary="Processing report (demo text file standing in for a PDF)."
                    />
                  </ListItem>
                ))}
              </List>
              <Divider sx={{ my: 1 }} />
            </>
          )}

          {logArtifacts.length > 0 && (
            <>
              <Typography variant="subtitle1">Logs</Typography>
              <List dense>
                {logArtifacts.map((a, i) => (
                  <ListItem key={`log-${i}`}>
                    <ListItemText
                      primary={a.path}
                      secondary="Agent run logs. Open locally on the processing machine."
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}

          {/* Fallback raw list */}
          {job.artifacts.length > 0 && (
            <>
              <Divider sx={{ my: 1 }} />
              <Typography variant="caption">
                NOTE: paths above are local filesystem paths on the backend side in
                this demo. In a real deployment they would be URLs pointing to
                cloud storage (S3 / GCS / Drive).
              </Typography>
            </>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
}
