'use client'

import * as React from 'react'
import {
  ThemeProvider as NextThemesProvider,
  ThemeProvider as NextThemesProviderComponent,
} from 'next-themes'

interface CustomThemeProviderProps {
  children: React.ReactNode;
  attribute?: string;
  defaultTheme?: string;
  enableSystem?: boolean;
}

export function ThemeProvider({ children, ...props }: CustomThemeProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>
}
