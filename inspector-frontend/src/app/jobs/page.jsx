'use client';

import React from 'react';
import { useSession, signIn } from 'next-auth/react';
import { Container, Typography, Button, Box } from '@mui/material';
import JobsTable from '@/components/JobsTable';

export default function JobsPage() {
  const { data: session, status } = useSession();
  const loading = status === 'loading';

  return (
    <Container sx={{ mt: 4 }}>
      {loading && <Typography>Loading…</Typography>}

      {!loading && !session && (
        <Box>
          <Typography variant="h5" gutterBottom>
            Sign in required
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            You must sign in with Google to view jobs.
          </Typography>
          <Button variant="contained" onClick={() => signIn('google')}>
            Sign in with Google
          </Button>
        </Box>
      )}

      {!loading && session && <JobsTable />}
    </Container>
  );
}
