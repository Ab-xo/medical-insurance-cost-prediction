import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "MedPredict.ai",
    template: "%s | MedPredict.ai",
  },
  description:
    "Advanced medical insurance cost prediction dashboard powered by machine learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex bg-background text-foreground" suppressHydrationWarning>
        <Sidebar />
        {/* offset for desktop sidebar; top offset for mobile bar */}
        <main className="flex-1 md:ml-64 mt-14 md:mt-0 p-5 md:p-8 lg:p-10 min-h-screen overflow-x-hidden">
          {children}
        </main>
      </body>
    </html>
  );
}
