import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

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
      <body className="min-h-screen bg-base font-sans text-ink antialiased">
        {children}
      </body>
    </html>
  );
}
