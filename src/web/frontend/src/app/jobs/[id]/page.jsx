// src/app/jobs/[id]/page.jsx
import React from 'react';
import JobDetail from '@/components/JobDetail';

export default function JobDetailPage({ params }) {
  const resolvedParams = React.use(params);
  return (
    <main style={{ padding: 24 }}>
      <JobDetail jobId={resolvedParams.id} />
    </main>
  );
}
