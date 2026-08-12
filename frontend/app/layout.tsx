import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { API_URL } from "@/lib/api";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

/**
 * Origin of the backend, for the connection preload.
 *
 * The page's first act after hydration is a health request whose whole purpose
 * is to start a suspended service booting. Opening the TCP and TLS connection
 * while the document is still parsing takes that setup off the critical path,
 * so the request leaves as soon as there is a request to send. Falls back to
 * the raw value if it is somehow not a parseable URL, which only costs the
 * preconnect.
 */
const API_ORIGIN: string = (() => {
  try {
    return new URL(API_URL).origin;
  } catch {
    return API_URL;
  }
})();

export const metadata: Metadata = {
  title: "QuickBite Analytics",
  description:
    "Natural-language business analytics powered by a multi-agent AI system.",
};

/**
 * Root layout.
 *
 * @param children - The routed page content.
 * @returns The application shell.
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="preconnect" href={API_ORIGIN} crossOrigin="anonymous" />
        <link rel="dns-prefetch" href={API_ORIGIN} />
      </head>
      <body className="min-h-screen bg-canvas font-sans text-ink antialiased">
        {children}
      </body>
    </html>
  );
}
