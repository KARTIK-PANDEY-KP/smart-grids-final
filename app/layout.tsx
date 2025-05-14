import type { Metadata } from 'next'
import './globals.css'
import { AssistantPopup } from '@/components/assistant-ui/assistant-popup'
import { TooltipProvider } from '@/components/ui/tooltip'

export const metadata: Metadata = {
  title: 'v0 App',
  description: 'Created with v0',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
        <TooltipProvider>
          {children}
          <AssistantPopup />
        </TooltipProvider>
      </body>
    </html>
  )
}
