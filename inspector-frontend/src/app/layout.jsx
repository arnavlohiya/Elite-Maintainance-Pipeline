// src/app/layout.jsx
import './globals.css';
import { Providers } from './providers';
import NavBar from '@/components/NavBar';
import AppEmotionCacheProvider from '@/components/AppEmotionCacheProvider';

export const metadata = {
  title: 'Inspector Dashboard',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning={true}>
        <AppEmotionCacheProvider>
          <Providers>
            <NavBar />
            {children}
          </Providers>
        </AppEmotionCacheProvider>
      </body>
    </html>
  );
}
