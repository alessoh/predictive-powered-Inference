import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PPI Engine — DOT Prototype",
  description:
    "Prediction-powered inference over live state DOT work-zone feeds: rectified estimators, active label selection, honest confidence intervals.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col text-[14px] leading-relaxed">
        <header className="border-b" style={{ borderColor: "var(--edge)" }}>
          <div className="mx-auto flex max-w-[1400px] items-center gap-6 px-4 py-3">
            <Link href="/" className="text-[16px] font-semibold tracking-tight">
              PPI Engine
              <span className="ml-2 text-[12px] font-normal" style={{ color: "var(--ink-3)" }}>
                DOT prototype
              </span>
            </Link>
            <nav aria-label="Primary" className="flex gap-4 text-[14px]">
              <Link href="/" className="hover:underline">
                Experiment Lab
              </Link>
              <Link href="/runs" className="hover:underline">
                Runs
              </Link>
              <Link href="/methodology" className="hover:underline">
                Methodology
              </Link>
            </nav>
          </div>
        </header>
        <div className="mx-auto w-full max-w-[1400px] grow px-4 py-4">{children}</div>
        <footer
          className="border-t px-4 py-3 text-[12px]"
          style={{ borderColor: "var(--edge)", color: "var(--ink-3)" }}
        >
          <div className="mx-auto max-w-[1400px]">
            Every number originates in the Python inference core or the agent layer; methods,
            targets, and oracles are labeled. Feeds: MS / UT / MO / KY WZDx (docs/02-feeds.md).
          </div>
        </footer>
      </body>
    </html>
  );
}
