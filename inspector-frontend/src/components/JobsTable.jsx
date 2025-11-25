

'use client';

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { listJobs } from '@/lib/api';
import { useRouter } from 'next/navigation';
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Paper,
  TableContainer,
  CircularProgress,
  Typography,
  TablePagination,
  TextField,
  MenuItem,
  Stack,
} from '@mui/material';
import StatusPill from '@/components/StatusPill';

const STATUS_OPTIONS = [
  'ALL',
  'UPLOADED',
  'OCR_OK',
  'NEEDS_REVIEW',
  'PUSHED_TO_DRIVE',
  'PROCESSING',
  'PROCESSED',
  'FAILED',
];

export default function JobsTable() {
  const router = useRouter();
  const { data, isLoading, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: 5000,
  });

  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [search, setSearch] = useState('');

  const filteredJobs = useMemo(() => {
    if (!data) return [];

    return data
      .filter((job) => {
        if (statusFilter !== 'ALL' && job.status !== statusFilter) return false;
        if (!search.trim()) return true;

        const s = search.trim().toLowerCase();
        return (
          job.id.toLowerCase().includes(s) ||
          (job.whiteboard_id && job.whiteboard_id.toLowerCase().includes(s))
        );
      })
      .sort((a, b) => b.created_at - a.created_at);
  }, [data, statusFilter, search]);

  const handleChangePage = (_, newPage) => setPage(newPage);

  const handleChangeRowsPerPage = (e) => {
    setRowsPerPage(parseInt(e.target.value, 10));
    setPage(0);
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Typography color="error">Failed to load jobs</Typography>;
  if (!data || data.length === 0)
    return <Typography>No jobs yet. Upload a .360 file to get started.</Typography>;

  const start = page * rowsPerPage;
  const end = start + rowsPerPage;
  const pageRows = filteredJobs.slice(start, end);

  return (
    <Paper>
      <Stack direction="row" spacing={2} sx={{ p: 2, pb: 0 }} alignItems="center">
        <TextField
          size="small"
          label="Search jobs"
          placeholder="job id or WB id"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <TextField
          size="small"
          select
          label="Status"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(0);
          }}
          sx={{ minWidth: 150 }}
        >
          {STATUS_OPTIONS.map((s) => (
            <MenuItem key={s} value={s}>
              {s}
            </MenuItem>
          ))}
        </TextField>
      </Stack>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Job ID</TableCell>
              <TableCell>Whiteboard ID</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pageRows.map((job) => (
              <TableRow
                key={job.id}
                hover
                style={{ cursor: 'pointer' }}
                onClick={() => router.push(`/jobs/${job.id}`)}
              >
                <TableCell>{job.id}</TableCell>
                <TableCell>{job.whiteboard_id || '—'}</TableCell>
                <TableCell>
                  <StatusPill status={job.status} />
                </TableCell>
                <TableCell>
                  {new Date(job.created_at * 1000).toLocaleString()}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={filteredJobs.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[5, 10, 25, 50]}
      />
    </Paper>
  );
}
