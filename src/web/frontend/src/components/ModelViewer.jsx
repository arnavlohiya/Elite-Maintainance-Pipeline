'use client';

import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, Center, Grid } from '@react-three/drei';
import { Box } from '@mui/material';

function Model({ url }) {
  const { scene } = useGLTF(url);
  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function LoadingBox() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#374151" wireframe />
    </mesh>
  );
}

export default function ModelViewer({ url }) {
  return (
    <Box
      sx={{
        width: '100%',
        height: '520px',
        bgcolor: '#0f1117',
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      <Canvas camera={{ position: [0, 1.5, 4], fov: 50 }} gl={{ antialias: true }}>
        <color attach="background" args={['#0f1117']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 3, -5]} intensity={0.4} />

        <Suspense fallback={<LoadingBox />}>
          <Model url={url} />
          <Environment preset="city" />
        </Suspense>

        <Grid
          position={[0, -2, 0]}
          infiniteGrid
          cellSize={0.5}
          cellThickness={0.5}
          sectionSize={3}
          sectionThickness={1}
          sectionColor="#1e3a5f"
          cellColor="#1a2535"
          fadeDistance={25}
        />

        <OrbitControls
          autoRotate
          autoRotateSpeed={0.8}
          enablePan
          enableZoom
          enableRotate
          minDistance={0.01}
          maxDistance={500}
        />
      </Canvas>
    </Box>
  );
}
