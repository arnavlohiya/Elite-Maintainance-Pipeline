// frontend/app/page.jsx
import React from 'react';
import Link from 'next/link';
import JobsTable from '@/components/JobsTable';

const highlights = [
  {
    title: 'Rapid Intake',
    description:
      'Drag-and-drop bulk uploads get footage into Drive within minutes so administrators can stay focused on review.',
  },
  {
    title: 'Field Friendly',
    description:
      'The upload experience stays lightweight and mobile-ready, perfect for crews capturing work on site.',
  },
  {
    title: 'Admin Oversight',
    description:
      'Operations teams monitor every job from the dashboard and trigger processing once uploads arrive.',
  },
];

const stats = [
  { label: 'Fleet Sites', value: '120+' },
  { label: 'Jobs Processed', value: '8.6K' },
  { label: 'Avg. Turnaround', value: '< 24h' },
];

export default function HomePage() {
  return (
    <main
      style={{
        padding: '40px 24px 80px',
        display: 'flex',
        flexDirection: 'column',
        gap: 32,
        background:
          'radial-gradient(circle at top, rgba(0, 150, 255, 0.08), transparent 60%)',
        minHeight: '100vh',
      }}
    >
      <section
        style={{
          borderRadius: 24,
          padding: '48px 32px',
          background:
            'linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #415a77 100%)',
          color: '#f4f7fb',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 32,
          boxShadow: '0 30px 80px rgba(9, 29, 63, 0.35)',
        }}
      >
        <div>
          <p
            style={{
              letterSpacing: '0.3em',
              fontSize: 12,
              textTransform: 'uppercase',
              color: 'rgba(255,255,255,0.7)',
              marginBottom: 16,
            }}
          >
            Elite Maintenance Group
          </p>
          <h1 style={{ fontSize: 40, margin: 0, lineHeight: 1.1 }}>
            Welcome to your operations command center.
          </h1>
          <p style={{ marginTop: 16, fontSize: 16, color: 'rgba(255,255,255,0.85)' }}>
            Stay ahead of every inspection, push bulk videos straight to Drive,
            and give admins the visibility to keep projects moving.
          </p>
          <div style={{ marginTop: 32, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <Link
              href="/upload"
              style={{
                background: '#f97316',
                color: '#0d1b2a',
                padding: '12px 28px',
                borderRadius: 999,
                fontWeight: 600,
                textDecoration: 'none',
              }}
            >
              Upload Footage
            </Link>
            <Link
              href="/jobs"
              style={{
                border: '1px solid rgba(255,255,255,0.3)',
                color: '#f4f7fb',
                padding: '12px 28px',
                borderRadius: 999,
                fontWeight: 600,
                textDecoration: 'none',
              }}
            >
              Review Jobs
            </Link>
          </div>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: 16,
          }}
        >
          {stats.map((stat) => (
            <div
              key={stat.label}
              style={{
                background: 'rgba(255,255,255,0.1)',
                borderRadius: 20,
                padding: '24px 20px',
                textAlign: 'center',
                backdropFilter: 'blur(6px)',
              }}
            >
              <div style={{ fontSize: 32, fontWeight: 700 }}>{stat.value}</div>
              <div style={{ fontSize: 14, opacity: 0.7 }}>{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 20,
        }}
      >
        {highlights.map((item) => (
          <div
            key={item.title}
            style={{
              background: '#fff',
              borderRadius: 20,
              padding: 24,
              boxShadow: '0 15px 35px rgba(15, 23, 42, 0.08)',
              border: '1px solid rgba(15,23,42,0.05)',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>{item.title}</h3>
            <p style={{ color: '#475569', marginBottom: 0 }}>{item.description}</p>
          </div>
        ))}
      </section>

      <section
        style={{
          background: '#fff',
          borderRadius: 28,
          padding: 24,
          boxShadow: '0 25px 60px rgba(15,23,42,0.1)',
          border: '1px solid rgba(226, 232, 240, 0.9)',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 16,
            flexWrap: 'wrap',
            gap: 12,
          }}
        >
          <div>
            <h2 style={{ margin: 0, color: '#0f172a' }}>Live Jobs Overview</h2>
            <p style={{ margin: 0, color: '#475569' }}>
              Every upload appears here the moment it hits Drive so admins can pick it up.
            </p>
          </div>
          <Link
            href="/jobs"
            style={{
              fontWeight: 600,
              textDecoration: 'none',
              color: '#2563eb',
            }}
          >
            View full queue →
          </Link>
        </div>
        <JobsTable />
      </section>
    </main>
  );
}
