import { HeroSection } from '@/components/landing/hero-section';
import { ParticlesBackground } from '@/components/ui/particles-background';
import { PreprocessingWrapper } from '@/components/preprocessing/preprocessing-wrapper';
import Link from 'next/link';
import { buttonVariants } from '@/components/ui/button';
import { MessageSquareCode } from 'lucide-react';

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col relative overflow-hidden">
      <ParticlesBackground />
      <HeroSection />
      
      <section className="container mx-auto px-4 py-16 relative z-10">
        <div className="flex items-center justify-between mb-8">
          <div className="space-y-1">
            <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary via-chart-1 to-chart-2">
              AutoML Workspace
            </h2>
            <p className="text-muted-foreground">
              Choose your preferred way to interact with our AutoML platform
            </p>
          </div>
          
          <Link 
            href="/chatbot"
            className={buttonVariants({
              size: "lg",
              className: "gap-2 bg-gradient-to-r from-chart-2 to-chart-3 hover:opacity-90"
            })}
          >
            <MessageSquareCode className="w-5 h-5" />
            Try AI Assistant
          </Link>
        </div>
        
        <PreprocessingWrapper />
      </section>
    </main>
  );
}