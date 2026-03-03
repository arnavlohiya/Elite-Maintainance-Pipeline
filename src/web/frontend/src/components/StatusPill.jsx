// frontend/components/StatusPill.jsx
import React from 'react';
import { Chip } from '@mui/material';

const colorByStatus = {
  UPLOADED: 'default',
  OCR_OK: 'primary',
  NEEDS_REVIEW: 'warning',
  PUSHED_TO_DRIVE: 'primary',
  PROCESSING: 'primary',
  PROCESSED: 'success',
  FAILED: 'error'
};

export default function StatusPill({ status }) {
  const color = colorByStatus[status] || 'default';
  return <Chip label={status} color={color} size="small" />;
}
