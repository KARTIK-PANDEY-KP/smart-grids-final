import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-md bg-blue-600"></div>
            <span className="text-xl font-bold">SmartGrid AI</span>
          </div>
          <nav className="flex items-center gap-4">
            <Link href="/app" className="text-sm font-medium hover:underline">
              Features
            </Link>
            <Link href="/app" className="text-sm font-medium hover:underline">
              Pricing
            </Link>
            <Link href="/app" className="text-sm font-medium hover:underline">
              Docs
            </Link>
            <Button asChild variant="outline" size="sm">
              <Link href="/app">Login</Link>
            </Button>
            <Button asChild size="sm">
              <Link href="/app">Try Demo</Link>
            </Button>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <section className="container mx-auto px-4 py-20 text-center">
          <h1 className="mb-4 text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">Enrich your data with AI</h1>
          <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground">
            Transform your spreadsheets with AI-powered insights. No coding required.
          </p>
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button asChild size="lg" className="gap-2">
              <Link href="/app">
                Try Demo <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/app">Login with Google</Link>
            </Button>
          </div>
        </section>

        <section className="container mx-auto px-4 py-12">
          <h2 className="mb-8 text-center text-3xl font-bold">How It Works</h2>
          <div className="grid gap-8 md:grid-cols-3">
            <div className="rounded-lg border p-6 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                1
              </div>
              <h3 className="mb-2 text-xl font-medium">Input Your Data</h3>
              <p className="text-muted-foreground">
                Add or import your structured data into the spreadsheet interface.
              </p>
            </div>
            <div className="rounded-lg border p-6 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                2
              </div>
              <h3 className="mb-2 text-xl font-medium">Add AI Column</h3>
              <p className="text-muted-foreground">Specify what insights you want to extract from your data.</p>
            </div>
            <div className="rounded-lg border p-6 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                3
              </div>
              <h3 className="mb-2 text-xl font-medium">Get AI Insights</h3>
              <p className="text-muted-foreground">Watch as AI analyzes your data and provides valuable insights.</p>
            </div>
          </div>
        </section>

        <section className="bg-slate-50 py-12">
          <div className="container mx-auto px-4">
            <h2 className="mb-8 text-center text-3xl font-bold">Key Features</h2>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <div className="rounded-lg border bg-white p-6">
                <h3 className="mb-2 text-xl font-medium">AI Enrichment</h3>
                <p className="text-muted-foreground">
                  Enhance your data with AI-generated insights and classifications.
                </p>
              </div>
              <div className="rounded-lg border bg-white p-6">
                <h3 className="mb-2 text-xl font-medium">Smart Columns</h3>
                <p className="text-muted-foreground">
                  Create columns that automatically analyze and transform your data.
                </p>
              </div>
              <div className="rounded-lg border bg-white p-6">
                <h3 className="mb-2 text-xl font-medium">No Coding Required</h3>
                <p className="text-muted-foreground">
                  Simple interface that anyone can use without technical knowledge.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="border-t py-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <p className="text-sm text-muted-foreground">© 2025 SmartGrid AI. All rights reserved.</p>
            <div className="flex gap-4">
              <Link href="#" className="text-sm text-muted-foreground hover:underline">
                Terms
              </Link>
              <Link href="#" className="text-sm text-muted-foreground hover:underline">
                Privacy
              </Link>
              <Link href="#" className="text-sm text-muted-foreground hover:underline">
                GitHub
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
